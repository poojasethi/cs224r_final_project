from trainers.sft_trainer import CustomSFTTrainer, SFTTrainingArguments
from data.test_dataloaders import get_dataloader
from data.utils import get_tokenizer
from transformers import AutoModelForCausalLM


def run_instruction_following_sft(model_id: str = "Qwen/Qwen2.5-0.5B"):
    args = SFTTrainingArguments(
        wandb_project=f"{model_id}-sft",
        wandb_run="instruction-following-smoltalk-sft"
    )

    # Initialize the tokenizer and the base model.
    tokenizer = get_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # load in bfloat16 if hardware support is available, otherwise float32
        torch_dtype="auto",
    )

    train_dataloader = get_dataloader(
        dataset_name="smoltalk",
        split="train",
    )

    eval_dataloader = get_dataloader(
        dataset_name="smoltalk",
        split="test",
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=args,
    )

    # trainer.train()


if __name__ == "__main__":
    run_instruction_following_sft()
