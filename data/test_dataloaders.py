from data.preference_datasets import SmolTokDataset
from torch.utils.data import DataLoader
import logging
import sys
from data.utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_dataloader(
    dataset_name: str,
    split: str = "train[:1%]",
    batch_size: int = 8,
):
    """
    Creates the SmolTok dataloader.
    """
    dataset = None
    if dataset_name == "smoltok":
        dataset = SmolTokDataset(split=split)
    else:
        raise ValueError(f"Unrecognized dataset {dataset_name}")

    # Create PyTorch dataloader.
    logger.info("Creating SmolTok DataLoader...")
    smoltok_dataloader = DataLoader(
        dataset,
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
    smoltok_dataloader = get_dataloader("smoltok")
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")
    logger.info("\nExample batch from SmolTok DataLoader:")
    try:
        for i, batch in enumerate(smoltok_dataloader):
            logger.info(f"Batch {i+1}:")
            logger.info("Input IDs shape: %s", batch["input_ids"].shape)
            logger.info("Attention Mask shape: %s", batch["attention_mask"].shape)
            logger.info("Labels shape: %s", batch["labels"].shape)
            # Decode a sample to check
            logger.info(
                f"Decoded Input IDs (first item): {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}"
            )
            break  # Just show the first batch
    except Exception as e:
        logger.warning(f"Error iterating through SmolTok DataLoader: {e}")


if __name__ == "__main__":
    test_smoltok_dataloader()