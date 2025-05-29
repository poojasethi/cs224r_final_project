import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import logging
import os
from dataclasses import dataclass
from accelerate import Accelerator  # Import Accelerator
# from trainers.sft_trainer import CustomSFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./sft_model/checkpoint-20000/" 
BETA = 0.2

@dataclass
class DPOTrainingArguments:
    wandb_project: str = "qwen-dpo"
    wandb_run: str = "instruction-following-ultrafeedback-dpo"
    model_id: str = "Qwen/Qwen2.5-0.5B"
    train_split: str = "train"
    test_split: str = "test"
    train_batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    checkpoint_steps: int = 20000
    sft_output_dir: str = "./sft_model"
    dpo_output_dir: str = "./dpo_model"
    # Turn on mixed precision training to reduce memory usage and speed up training.
    fp16: bool = False # True
    # Add gradient accumulation steps
    gradient_accumulation_steps: int = 32

class DPOTrainer:   # NOTE: Copied from Pooja's sft_trainer
    def __init__(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        args: DPOTrainingArguments,
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


    def dpo_loss(
        self,  
        pref_outputs, 
        dispref_outputs, 
        preferred_ids, 
        preferred_a_masks, 
        dispreferred_ids, 
        dispreferred_a_masks,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: str = "auto"
        ):

        checkpoint_path = CHECKPOINT_PATH
        beta = BETA

        # logger.info(f"Loading tokenizer from {checkpoint_path}...")
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Loading model from {checkpoint_path}...")
        sft_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        sft_model.to(device)
        sft_model.eval()

        with torch.no_grad():
            ref_pref_outputs = sft_model( # .generate
                input_ids=preferred_ids,
                attention_mask=preferred_a_masks,
                # max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # do_sample=do_sample,
                # pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # num_beams=1,
                # repetition_penalty=1.1,
            )
            ref_dispref_outputs = sft_model( # .generate
                input_ids=dispreferred_ids,
                attention_mask=dispreferred_a_masks,
                # max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # do_sample=do_sample,
                # pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # num_beams=1,
                # repetition_penalty=1.1,
            )
        print("pref", pref_outputs.shape)
        print("ref pref", ref_pref_outputs.logits.shape)
        wins = pref_outputs / ref_pref_outputs.logits
        losses = dispref_outputs / ref_dispref_outputs.logits
        inside_sig = beta * torch.log(wins) - beta * torch.log(losses)
        inside_expect = torch.log(torch.sigmoid(inside_sig))
        loss = - inside_expect.mean()
        return loss

    def train(self):      # NOTE: Heavily copied from Pooja's SFT Train function
        """
        Kicks off DPO training.
        """
        logger.info("\n*******Running training**********\n")
        self.model.train()

        global_steps = 0
        for epoch in range(self.args.epochs):
            progress_bar = tqdm(self.train_dataloader,
                                desc=f"Epoch {epoch + 1}/{self.args.epochs}")

            for step, batch in enumerate(progress_bar):
                preferred_ids = batch["preferred_ids"].to(self.device)
                preferred_a_masks = batch["preferred_a_masks"].to(self.device)
                dispreferred_ids = batch['dispreferred_ids'].to(self.device)
                dispreferred_a_masks = batch['dispreferred_a_masks'].to(self.device)
                # score_chosen = batch['score_chosen'].to(self.device)            # TODO: When do I use these?
                # score_rejected = batch['score_rejected'].to(self.device)        # TODO: When do I use these? 

                
                # 1. Run forward pass (with mixed precision).
                with torch.amp.autocast("cuda", enabled=self.args.fp16):
                    pref_outputs = self.model(
                        input_ids=preferred_ids,
                        attention_mask=preferred_a_masks
                    )
                    dispref_outputs = self.model(
                        input_ids=dispreferred_ids,
                        attention_mask=dispreferred_a_masks
                    )
                    loss = self.dpo_loss(pref_outputs.logits, dispref_outputs.logits, preferred_ids, 
                                    preferred_a_masks, dispreferred_ids, dispreferred_a_masks, self.tokenizer)  

                # 2. Run backward pass.
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 3. Step the optimizer.
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients for stability during training.
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients for stability during training.
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0)
                    self.optimizer.step()

                # 4. Step the learning rate scheduler.
                self.lr_scheduler.step()

                # 5. Clear gradients.
                self.optimizer.zero_grad()

                # Log the training loss and current learning rate.
                global_steps += 1
                if global_steps % self.args.logging_steps == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    logs = {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "global_step": global_steps,
                    }
                    wandb.log(logs, step=global_steps)

                # Log the evaluation loss.
                if global_steps % self.args.eval_steps == 0:
                    eval_loss = self.evaluate()
                    wandb.log({"eval_loss": eval_loss}, step=global_steps)

                     # Set model back to train mode after doing evaluation.
                    self.model.train() 

        # Save the final model after training is finished.
        output_dir = os.path.join(
            self.args.dpo_output_dir, f"checkpoint-{global_steps}")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"\nTraining complete! Model saved to {output_dir}")
        wandb.finish()


    @torch.no_grad()
    def evaluate(self):         # Copied from Pooja's Sft trainer file with minimal changes
        logger.info("\n*******Running evaluation**********\n")
        self.model.eval()

        total_eval_loss = 0
        num_eval_batches = 0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.args.fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = outputs.loss

            total_eval_loss += loss.item()
            num_eval_batches += 1

        avg_eval_loss = total_eval_loss / num_eval_batches
        logger.info(f"Evaluation Loss: {avg_eval_loss:.4f}")
        return avg_eval_loss

