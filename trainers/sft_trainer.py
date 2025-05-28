import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import wandb
import logging
from dataclasses import dataclass
import os
from accelerate import Accelerator  # Import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    wandb_project: str = "qwen-sft"
    wandb_run: str = "instruction-following-smoltalk-sft"
    model_id: str = "Qwen/Qwen2.5-0.5B"
    train_split: str = "train"
    test_split: str = "test"
    train_batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 1000
    checkpoint_steps: int = 20000
    output_dir: str = "./sft_model"
    # Turn on mixed precision training to reduce memory usage and speed up training.
    fp16: bool = True
    # Add gradient accumulation steps
    gradient_accumulation_steps: int = 32


class CustomSFTTrainer:
    """
    Our custom trainer for SFT fine-tuning.
    Loosely inspired by the HuggingFace SFTTrainer.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        args: SFTTrainingArguments,
    ):
        self.args = args

        # Initialize Accelerator for distributed training and mixed precision
        self.accelerator = Accelerator(
            mixed_precision="fp16" if args.fp16 else "no",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # Load model and tokenizer from model_id
        logger.info(f"Loading model and tokenizer from {args.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(args.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Initialize wandb so we can monitor training. Only on the main process.
        if self.accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project, name=args.wandb_run, config=args.__dict__
            )
            wandb.watch(self.model, log_freq=args.logging_steps)

        # Initialize the optimizer. We use AdamW.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate
        )

        # Initialize the learning rate scheduler. We use a linear scheduler.
        # Calculate total training steps considering gradient accumulation
        num_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_training_steps = num_steps_per_epoch * args.epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare everything for training with Accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

        # The device is now managed by Accelerator
        self.device = self.accelerator.device
        logger.info(f"Using device {self.device}")

    def train(self):
        """
        Kicks off SFT training.
        """
        logger.info("\n*******Running training**********\n")
        self.model.train()

        global_steps = 0
        for epoch in range(self.args.epochs):
            # Wrap dataloader with tqdm only on the main process
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for step, batch in enumerate(progress_bar):
                # Gradient accumulation context
                with self.accelerator.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    # 1. Run forward pass. Accelerator handles mixed precision.
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                    # Scale loss for gradient accumulation
                    loss = loss / self.args.gradient_accumulation_steps

                    # 2. Run backward pass.
                    self.accelerator.backward(loss)

                    # 3. Clip gradients and step the optimizer/scheduler
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                # Log the training loss and current learning rate.
                # Only log on the main process
                if self.accelerator.is_main_process:
                    global_steps += (
                        1  # Increment global_steps only on main process for logging
                    )
                    if global_steps % self.args.logging_steps == 0:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        logs = {
                            "train_loss": loss.item()
                            * self.args.gradient_accumulation_steps,  # Unscale loss for logging
                            "learning_rate": current_lr,
                            "global_step": global_steps,
                        }
                        wandb.log(logs, step=global_steps)

                    # Log the evaluation loss.
                    if global_steps % self.args.eval_steps == 0:
                        eval_loss = self.evaluate()
                        wandb.log({"eval_loss": eval_loss}, step=global_steps)
                        logger.info(
                            f"Running evaluation at {global_steps}, got loss: {eval_loss}"
                        )

                        # Set model back to train mode after doing evaluation.
                        self.model.train()

                    if global_steps % self.args.checkpoint_steps == 0:
                        logger.info(f"Saving checkpoint after {global_steps}")
                        output_dir = os.path.join(
                            self.args.output_dir, f"checkpoint-{global_steps}"
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        # Use accelerator.save_model for proper saving in distributed setup
                        self.accelerator.save_model(self.model, output_dir)
                        self.tokenizer.save_pretrained(output_dir)

        # Save the final model after training is finished. Only on the main process.
        if self.accelerator.is_main_process:
            output_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{global_steps}"
            )
            os.makedirs(output_dir, exist_ok=True)
            self.accelerator.save_model(self.model, output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"\nTraining complete! Model saved to {output_dir}")
            wandb.finish()

    @torch.no_grad()
    def evaluate(self):
        logger.info("\n*******Running evaluation**********\n")
        self.model.eval()

        total_eval_loss = 0
        num_eval_batches = 0
        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        ):
            # Accelerator already handled device placement for dataloader batches
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            total_eval_loss += loss.item()
            num_eval_batches += 1

        # Gather losses from all processes and average
        total_eval_loss = (
            self.accelerator.gather(torch.tensor(total_eval_loss).to(self.device))
            .sum()
            .item()
        )
        num_eval_batches = (
            self.accelerator.gather(torch.tensor(num_eval_batches).to(self.device))
            .sum()
            .item()
        )

        avg_eval_loss = total_eval_loss / num_eval_batches
        logger.info(f"Evaluation Loss: {avg_eval_loss:.4f}")
        return avg_eval_loss
