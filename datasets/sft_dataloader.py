import torch
import sys
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List
from utils import get_tokenizer, HFDatasetWrapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def tokenize_sft(
    examples: Dict[str, List], max_length: int = 1024
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of examples for SFT.
    """
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")

    # Format the text for SFT fine-tuning. 
    texts = tokenizer.apply_chat_template(examples["messages"], tokenize=False)
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # For causal language modeling (SFT), the labels are typically the next tokens of the input sequence.
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def get_smoltok_dataset(split: str):
    """
    Prepares the SmolTok dataset for SFT.
    """
    smoltok_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)

    logger.info(
        "SmolTok dataset loaded successfully with %s examples.", len(smoltok_dataset)
    )
    logger.info("Original dataset columns: %s", smoltok_dataset.column_names)

    logger.info("Applying tokenization...")
    smoltok_tokenized_dataset = smoltok_dataset.map(
        tokenize_sft,
        batched=True,  # Process in batches for efficiency
        remove_columns=[
            col
            for col in smoltok_dataset.column_names
            if col not in ["input_ids", "attention_mask", "labels"]
        ],  # Remove original text columns
        desc="Tokenizing SmolTok dataset",  # Description for progress bar
    )
    logger.info("Tokenized dataset columns: %s", smoltok_tokenized_dataset.column_names)
    logger.info("First tokenized example: %s", smoltok_tokenized_dataset[0])

    smoltok_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return smoltok_tokenized_dataset

def get_smoltok_dataloader(
    split: str = "train[:1%]",
    batch_size: int = 8,
):
    """
    Creates the SmolTok dataloader.
    """
    smoltok_tokenized_dataset = get_smoltok_dataset(split)

    # Create PyTorch dataloader.
    logger.info("Creating SmolTok DataLoader...")
    smoltok_dataloader = DataLoader(
        smoltok_tokenized_dataset,
        batch_size=batch_size,  # Use the defined batch size
        shuffle=True,  # Shuffle data for training
        num_workers=2,  # Use multiple workers for faster data loading (adjust based on your system)
    )

    logger.info(
        f"SmolTok DataLoader created successfully with batch size {batch_size}."
    )
    logger.info(f"Number of batches per epoch: {len(smoltok_dataloader)}")
    return smoltok_dataloader


def test_smoltok_dataloader():
    """
    Instantiates the SmolTok dataloader and iterates through one batch for testing.
    """
    smoltok_dataloader = get_smoltok_dataloader()
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")
    logger.info("\nExample batch from SmolTok DataLoader:")
    try:
        for i, batch in enumerate(smoltok_dataloader):
            logger.info(f"Batch {i+1}:")
            logger.info("Input IDs shape: %s", batch["input_ids"].shape)
            logger.info("Attention Mask shape: %s", batch["attention_mask"].shape)
            logger.info("Labels shape: %s", batch["labels"].shape)
            # Decode a sample to check
            logger.info(f"Decoded Input IDs (first item): {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
            break  # Just show the first batch
    except Exception as e:
        logger.warning(f"Error iterating through SmolTok DataLoader: {e}")


# TODO (malisha): Add WarmStart dataset.
def get_warmstart_dataset(split: str):
    """
    Prepares the WarmStart dataset for SFT.
    """
    raise NotImplementedError()


# TODO(malisha): Add WarmStart dataloader.
def get_warmstart_dataloader(split: str = "train[:1%]", batch_size: int = 8):
    """
    Prepares the WarmStart data loader.
    """
    raise NotImplementedError()

# TODO(malisha): Test WarmStart dataloader.
def test_warmstart_dataloader():
    """
    Instantiates the WarmStart dataloader and iterates through one batch for testing.
    """
    raise NotImplementedError()

if __name__ == "__main__":
    test_smoltok_dataloader()