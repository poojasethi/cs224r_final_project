from trainers.dpo_trainer import DPOTrainer, DPOTrainingArguments
from data.dataloader_utils import get_dataloader
from data.utils import get_tokenizer
from transformers import AutoModelForCausalLM
import argparse
import logging
import datetime
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_instruction_following_dpo(args: DPOTrainingArguments):
    logger.info(f"Training model {args.model_id}")

    train_dataloader = get_dataloader(
        dataset_name="ultrafeedback",
        split=args.train_split,  # Note: Using a smaller dataset for debugging.
        batch_size=args.train_batch_size,
    )
    # print_cuda_memory("gotten train dataloader")

    eval_dataloader = get_dataloader(
        dataset_name="ultrafeedback",
        split=args.test_split,
        batch_size=args.eval_batch_size,
    )

    trainer = DPOTrainer(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=args,
    )
    # print_cuda_memory("DPOTrainer est")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    if args.debug:
        # Use a small model and dataset that can run on cpu for debugging locally.
        experiment_args = DPOTrainingArguments(
            wandb_project="pythia-dpo",
            wandb_run="instruction-following-ultrafeedback-dpo",
            model_id="EleutherAI/pythia-70m",
            train_split="train_prefs[:1%]",
            test_split="test_prefs[:1%]"
        )
        run_instruction_following_dpo(experiment_args)
    else: 
        now = datetime.datetime.now()
        date_time_string = now.strftime("%y-%m-%d-%H%M%S")
        output_dir = f"checkpoints/dpo_model_{date_time_string}"
        os.makedirs(output_dir)
        logger.info(f"Kicking off dpo training and saving results to {output_dir}")

        # Kick off a full experiment
        experiment_args = DPOTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-ultrafeedback-dpo",
            train_split="train_prefs",
            test_split="test_prefs[:1%]",
            dpo_output_dir=output_dir,
            train_batch_size=1
        )
        run_instruction_following_dpo(experiment_args)
