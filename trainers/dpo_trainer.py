import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import logging
from dataclasses import dataclass
from trainers.sft_trainer import CustomSFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DPOTrainingArguments:
    wandb_project: str
    wandb_run: str
    epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    sft_output_dir: str = "./sft_model"
    dpo_output_dir: str = "./dpo_model"
    # Turn on mixed precision training to reduce memory usage and speed up training.
    fp16: bool = True

class DPOTrainer:   # NOTE: Copied from Pooja's sft_trainer
    def __init__(self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        args: DPOTrainingArguments,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args

        # Initialize wandb so we can monitor training.
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run, config=args.__dict__)
        wandb.watch(self.model, log_freq=args.logging_steps)

        # Set the device.
        device = None
        if torch.cuda.is_available():
            device = "cuda" 
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
        self.device = torch.device(device)
        
        logger.info(f"Using device {self.device}")
        self.model.to(self.device)

        # Initialize the optimizer. We use Adam.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate)

        # Initialize the learning rate scheduler. We use a linear scheduler.
        num_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_steps_per_epoch * args.epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
        )

        # Enable mixed precision training.
        self.scaler = torch.amp.GradScaler("cuda") if self.args.fp16 else None

    def dpo_loss(
        self, 
        beta, 
        pref_outputs, 
        dispref_outputs, 
        preferred_ids, 
        preferred_a_masks, 
        dispreferred_ids, 
        dispreferred_a_masks):

        sft_model = model.load_state_dict(torch.load(self.args.sft_output_dir))
        ref_pref_outputs = sft_model.generate(
            input_ids=dispreferred_ids,
            attention_mask=dispreferred_a_masks
        )
        ref_dispref_outputs = sft_model.generate(
            input_ids=preferred_ids,
            attention_mask=preferred_a_masks
        )
        wins = pref_outputs / ref_pref_outputs
        losses = dispref_outputs / ref_dispref_outputs
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
                score_chosen = batch['score_chosen'].to(self.device)            # TODO: When do I use these?
                score_rejected = batch['score_rejected'].to(self.device)        # TODO: When do I use these? 

                
                # 1. Run forward pass (with mixed precision).
                with torch.amp.autocast("cuda", enabled=self.args.fp16):
                    pref_outputs = self.model(
                        input_ids=dispreferred_ids,
                        attention_mask=dispreferred_a_masks
                    )
                    dispref_outputs = self.model(
                        input_ids=preferred_ids,
                        attention_mask=preferred_a_masks
                    )
                    loss = dpo_loss(beta, pref_outputs, dispref_outputs, preferred_ids, 
                                    preferred_a_masks, dispreferred_ids, dispreferred_a_masks)  

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

