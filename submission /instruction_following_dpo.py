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
    print(f"Training model {args.model_id}")

    train_dataloader = get_dataloader(
        dataset_name="ultrafeedback",
        split=args.train_split,  # Note: Using a smaller dataset for debugging.
        batch_size=args.train_batch_size,
    )

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
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    if args.debug:
        # debug on 10% of dataset
        now = datetime.datetime.now()
        date_time_string = now.strftime("%y-%m-%d-%H%M%S")
        output_dir = f"checkpoints/dpo_model_debug_{date_time_string}"
        os.makedirs(output_dir)
        print(f"Kicking off dpo training and saving results to {output_dir}")

        # Kick off a full experiment
        experiment_args = DPOTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-ultrafeedback-dpo",
            train_split="train_prefs[:1%]",
            test_split="test_prefs[:1%]",
            output_dir=output_dir,
            train_batch_size=2
        )
        run_instruction_following_dpo(experiment_args) 
    else: 
        # run training on full dataset
        now = datetime.datetime.now()
        date_time_string = now.strftime("%y-%m-%d-%H%M%S")
        output_dir = f"checkpoints/dpo_model_{date_time_string}"
        os.makedirs(output_dir)
        print(f"Kicking off dpo training and saving results to {output_dir}")

        # Kick off a full experiment
        experiment_args = DPOTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-ultrafeedback-dpo",
            train_split="train_prefs",
            test_split="test_prefs[:10%]",
            output_dir=output_dir,
            train_batch_size=2
        )
        run_instruction_following_dpo(experiment_args)
