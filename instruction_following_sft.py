from trainers.sft_trainer import CustomSFTTrainer, SFTTrainingArguments
from data.dataloader_utils import get_dataloader
from data.utils import get_tokenizer
from transformers import AutoModelForCausalLM
import argparse
import logging
import datetime
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_instruction_following_sft(args: SFTTrainingArguments):
    logger.info(f"Training model {args.model_id}")

    train_dataloader = get_dataloader(
        dataset_name="smoltalk",
        split=args.train_split,  # Note: Using a smaller dataset for debugging.
        batch_size=args.train_batch_size,
    )

    eval_dataloader = get_dataloader(
        dataset_name="smoltalk",
        split=args.test_split,
        batch_size=args.eval_batch_size,
    )

    trainer = CustomSFTTrainer(
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
        now = datetime.datetime.now()
        date_time_string = now.strftime("%y-%m-%d-%H%M%S")
        output_dir = f"checkpoints/sft_model_debug_{date_time_string}"
        os.makedirs(output_dir)
        logger.info(f"Kicking off sft training and saving results to {output_dir}")

        # Kick off a full experiment
        experiment_args = SFTTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-smoltalk-sft",
            train_split="train[:1%]",
            test_split="test[:1%]",
            output_dir=output_dir,
        )
        run_instruction_following_sft(experiment_args) 
    else: 
        now = datetime.datetime.now()
        date_time_string = now.strftime("%y-%m-%d-%H%M%S")
        output_dir = f"checkpoints/sft_model_{date_time_string}"
        os.makedirs(output_dir)
        logger.info(f"Kicking off sft training and saving results to {output_dir}")

        # Kick off a full experiment
        experiment_args = SFTTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-smoltalk-sft",
            test_split="test[:1%]",
            output_dir=output_dir,
        )
        run_instruction_following_sft(experiment_args)
