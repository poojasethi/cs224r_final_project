import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import logging
from dataclasses import dataclass
from accelerate import Accelerator 
from torchtune.rlhf.loss import DPOLoss
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./checkpoints/sft_model_original/checkpoint-100000/"
BETA = 0.1

@dataclass
class DPOTrainingArguments:
    wandb_project: str = "qwen-dpo"
    wandb_run: str = "instruction-following-ultrafeedback-dpo"
    model_id: str = # "Qwen/Qwen2.5-0.5B"
    train_split: str = "train"
    test_split: str = "test"
    train_batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 1e-6
    warmup_steps: int = 200
    logging_steps: int = 100
    eval_steps: int = 1000
    checkpoint_steps: int = 10000
    output_dir: str = "./dpo_model"
    # Turn on mixed precision training to reduce memory usage and speed up training.
    fp16: bool = False # Set to True for mixed precision
    # Add gradient accumulation steps
    gradient_accumulation_steps: int = 16

class DPOTrainer:
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

        logger.info(f"Loading sft model from {CHECKPOINT_PATH}...")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32()
        self.sft_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH, device_map="auto", torch_dtype=torch_dtype)
        self.sft_model.eval() # Ensure SFT model is in evaluation mode

        # Load policy model and tokenizer from model_id
        logger.info(f"Loading policy model and tokenizer from {args.model_id}")
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

        # Calculate total training steps considering gradient accumulation
        # Ensure num_steps_per_epoch is at least 1.
        num_steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
        num_training_steps = num_steps_per_epoch * args.epochs

        # Initialize the learning rate scheduler. We use a linear scheduler.
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
        )

        # Prepare things on accelerator
        (
            self.model,
            self.sft_model, # Prepare sft_model as well
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.sft_model, # Prepare sft_model
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

        # The device is now managed by Accelerator
        self.device = self.accelerator.device
        logger.info(f"Using device {self.device}")

    def get_log_probs(self, logits, labels, attention_mask):
        """
        Calculates the log probabilities of a sequence given model logits.
        Args:
            logits (torch.Tensor): Logits from the model (batch_size, sequence_length, vocab_size).
            labels (torch.Tensor): Token IDs of the sequence (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask (batch_size, sequence_length).
        Returns:
            torch.Tensor: Sum of log probabilities for each sequence in the batch.
        """
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probabilities for the true labels
        # log_probs_labels shape: (batch_size, sequence_length - 1)
        log_probs_labels = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Apply attention mask to only consider valid tokens
        # Sum log probabilities for each sequence
        sequence_log_probs = (log_probs_labels * shift_attention_mask).sum(dim=-1)
        return sequence_log_probs

    def dpo_loss(
        self,
        pref_outputs_logits,
        dispref_outputs_logits,
        preferred_ids,
        preferred_a_masks,
        dispreferred_ids,
        dispreferred_a_masks,
    ):
        beta = BETA
        
         # Get reference model logits
        with torch.no_grad():
            ref_pref_outputs = self.sft_model(
                input_ids=preferred_ids,
                attention_mask=preferred_a_masks,
            )
            ref_dispref_outputs = self.sft_model(
                input_ids=dispreferred_ids,
                attention_mask=dispreferred_a_masks,
            )

        # Calculate log probs for policy model
        policy_log_probs_preferred = self.get_log_probs(pref_outputs_logits, preferred_ids, preferred_a_masks)
        policy_log_probs_dispreferred = self.get_log_probs(dispref_outputs_logits, dispreferred_ids, dispreferred_a_masks)

        # Calculate log probs for reference model
        ref_log_probs_preferred = self.get_log_probs(ref_pref_outputs.logits, preferred_ids, preferred_a_masks)
        ref_log_probs_dispreferred = self.get_log_probs(ref_dispref_outputs.logits, dispreferred_ids, dispreferred_a_masks)

        # Calculate the log-ratios, which are the "wins" and "losses" terms
        # log_ratio_preferred corresponds to log(pi_policy(y_w|x) / pi_ref(y_w|x))
        log_ratio_preferred = policy_log_probs_preferred - ref_log_probs_preferred

        # log_ratio_dispreferred corresponds to log(pi_policy(y_l|x) / pi_ref(y_l|x))
        log_ratio_dispreferred = policy_log_probs_dispreferred - ref_log_probs_dispreferred

        # beta * (log_ratio_preferred - log_ratio_dispreferred)
        dpo_score = beta * (log_ratio_preferred - log_ratio_dispreferred)

        # Clamp the DPO score for numerical stability.
        dpo_score = torch.clamp(dpo_score, -50, 50)

        # The final DPO loss, analogous to original 'inside_expect' and final loss calculation
        # L_DPO = - log(sigmoid(dpo_score))
        loss = -F.logsigmoid(dpo_score).mean() 

        # Sanity check our loss value matches the torchtune DPO loss. 
        # We do not use the below version for training!!
        # loss_fn = DPOLoss(beta=BETA)
        # dpo_loss = loss_fn(
        #     policy_log_probs_preferred,
        #     policy_log_probs_dispreferred,
        #     ref_log_probs_preferred,
        #     ref_log_probs_dispreferred
        # )

        return loss

    def train(self):
        """
        Kicks off DPO training.
        """
        logger.info("\n*******Running training**********\n")
        self.model.train()

        global_steps = 0
        for epoch in range(self.args.epochs):
            progress_bar = tqdm(self.train_dataloader,
                                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                                disable=not self.accelerator.is_main_process) # Disable tqdm for non-main processes

            for step, batch in enumerate(progress_bar):
                # Gradient accumulation context
                with self.accelerator.accumulate(self.model):
                    # 1. Run forward pass. 
                    pref_outputs = self.model(
                        input_ids=batch["preferred_ids"],
                        attention_mask=batch["preferred_a_masks"]
                    )
                    dispref_outputs = self.model(
                        input_ids=batch["dispreferred_ids"],
                        attention_mask=batch["dispreferred_a_masks"]
                    )

                    loss = self.dpo_loss(
                        pref_outputs.logits,
                        dispref_outputs.logits,
                        batch["preferred_ids"],
                        batch["preferred_a_masks"],
                        batch["dispreferred_ids"],
                        batch["dispreferred_a_masks"],
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.args.gradient_accumulation_steps

                    # 2. Run backward pass.
                    self.accelerator.backward(loss)

                    # 3. Clip gradients and step the optimizer/scheduler
                    if self.accelerator.sync_gradients:
                        logger.info(f"Running backward pass after global step: {global_steps}")
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

                    # Log and evaluate only when gradients are synced (i.e., after an actual optimizer step)
                    if global_steps % self.args.logging_steps == 0:
                        current_lr = self.lr_scheduler.get_last_lr()[0] # Get LR after scheduler step
                        logs = {
                            "train_loss": loss.item() * self.args.gradient_accumulation_steps, # Unscale loss for logging
                            "learning_rate": current_lr,
                            "global_step": global_steps,
                        }
                        wandb.log(logs, step=global_steps)
                        progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

    
                    # Log the evaluation loss.
                    if global_steps % self.args.eval_steps == 0:
                        eval_loss = self.evaluate()
                        wandb.log({"eval_loss": eval_loss}, step=global_steps)
                        logger.info(
                            f"Running evaluation at {global_steps}, got loss: {eval_loss}"
                        )

                        # Set model back to train mode after doing evaluation.
                        self.model.train()

                    # Save the checkpoint.
                    if global_steps % self.args.checkpoint_steps == 0:
                        logger.info(f"Saving checkpoint after {global_steps}")
                        output_dir = os.path.join(
                            self.args.output_dir, f"checkpoint-{global_steps}"
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        # Use accelerator.save_model for proper saving in distributed setup
                        self.accelerator.unwrap_model(self.model).save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

        # Save the final model after training is finished.
        # Use accelerator.save_model for proper distributed saving
        if self.accelerator.is_main_process:
            output_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{global_steps}")

            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"\nTraining complete! Model saved to {output_dir}")
            wandb.finish()


    @torch.no_grad()
    def evaluate(self):
        logger.info("\n*******Running evaluation**********\n")
        self.model.eval()

        total_eval_loss = 0
        num_eval_batches = 0
        # Disable tqdm for non-main processes during evaluation
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
        
            # Run forward pass for policy model
            pref_outputs = self.model(
                input_ids=batch["preferred_ids"],
                attention_mask=batch["preferred_a_masks"]
            )
            dispref_outputs = self.model(
                input_ids=batch["dispreferred_ids"],
                attention_mask=batch["dispreferred_a_masks"]
            )

            # Calculate DPO loss for evaluation
            loss = self.dpo_loss(
                pref_outputs.logits,
                dispref_outputs.logits,
                batch["preferred_ids"],
                batch["preferred_a_masks"],
                batch["dispreferred_ids"],
                batch["dispreferred_a_masks"],
            )

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
