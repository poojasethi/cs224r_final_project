from data.preference_datasets import SmolTalkDataset, UltraFeedbackDataset
from torch.utils.data import DataLoader
import logging
import sys
from data.utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataloader(
    dataset_name: str,
    split: str = "train[:1%]",
    batch_size: int = 8,
):
    """
    Creates the SmolTok dataloader.
    """
    dataset = None
    if dataset_name == "smoltalk":
        dataset = SmolTalkDataset(split=split)
        # Create PyTorch dataloader.
        logger.info("Creating Smoltalk DataLoader...")
        dataset_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,  # Use the defined batch size
            shuffle=True,  # Shuffle data for training
            num_workers=2,  # Use multiple workers for faster data loading (adjust based on your system)
        )
    elif dataset_name == "ultrafeedback":
        dataset = UltraFeedbackDataset()
        # Create PyTorch dataloader.
        logger.info("Creating UltraFeedback DataLoader...")
        dataset_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,  # Use the defined batch size
            shuffle=True,  # Shuffle data for training
            num_workers=2,  # Use multiple workers for faster data loading (adjust based on your system)
            collate_fn=dataset.collate
        )

    else:
        raise ValueError(f"Unrecognized dataset {dataset_name}")

    logger.info(
        f"{dataset_name} DataLoader created successfully with batch size {batch_size}."
    )
    logger.info(f"Number of batches per epoch: {len(dataset_dataloader)}")
    return dataset_dataloader