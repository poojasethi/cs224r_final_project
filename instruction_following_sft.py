from trainers.sft_trainer import CustomSFTTrainer, SFTTrainingArguments
from data.dataloader_utils import get_dataloader
from data.utils import get_tokenizer
from transformers import AutoModelForCausalLM
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_instruction_following_sft(args: SFTTrainingArguments):
    # Initialize the tokenizer and the base model. 
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")

    logger.info(f"Training model {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # load in bfloat16 if hardware support is available, otherwise float32
        # torch_dtype=torch.auto,
    )

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
        model=model,
        tokenizer=tokenizer,
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
        # Use a small model and dataset that can run on cpu for debugging locally.
        experiment_args = SFTTrainingArguments(
            wandb_project="pythia-sft",
            wandb_run="instruction-following-smoltalk-sft",
            model_id="EleutherAI/pythia-70m",
            train_split="train[:1%]",
            test_split="test[:1%]"
        )
        run_instruction_following_sft(experiment_args)
    else: 
        # Kick off a full experiment
        experiment_args = SFTTrainingArguments(
            wandb_project="qwen-sft",
            wandb_run="instruction-following-smoltalk-sft",
            test_split="test[:1%]",
        )
        run_instruction_following_sft(experiment_args)
