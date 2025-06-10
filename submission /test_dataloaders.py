from data.preference_datasets import SmolTalkDataset, UltraFeedbackDataset
from torch.utils.data import DataLoader
import logging
import sys
from data.utils import get_tokenizer
from data.dataloader_utils import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test_dataset_dataloader(dataset):
    """
    Instantiates the SmolTok dataloader and iterates through one batch for testing.
    """
    dataset_dataloader = get_dataloader(dataset)
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")
    logger.info(f"\nExample batch from {dataset} DataLoader:")
    try:
        for i, batch in enumerate(dataset_dataloader):
            logger.info(f"Batch {i+1}:")
            logger.info("Input IDs shape: %s", batch["input_ids"].shape)
            logger.info("Attention Mask shape: %s", batch["attention_mask"].shape)
            if dataset == "smoltok":
                logger.info("Labels shape: %s", batch["labels"].shape)
            # Decode a sample to check
            logger.info(
                f"Decoded Input IDs (first item in batch): {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}"
            )
            break  # Just show the first batch
    except Exception as e:
        logger.warning(f"Error iterating through {dataset} DataLoader: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        test_dataset_dataloader("smoltok")
        test_dataset_dataloader("ultrafeedback")
    else:
        if sys.argv[1] == "ultrafeedback":
            test_dataset_dataloader("ultrafeedback")
        elif sys.argv[1] == "smoltok":
            test_dataset_dataloader("smoltok")
        else:
            print("Options are 'ultrafeedback' or 'smoltok'")