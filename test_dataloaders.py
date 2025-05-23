from data.preference_datasets import SmolTalkDataset
from torch.utils.data import DataLoader
import logging
import sys
from data.utils import get_tokenizer
from data.dataloader_utils import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test_smoltok_dataloader():
    """
    Instantiates the SmolTok dataloader and iterates through one batch for testing.
    """
    smoltok_dataloader = get_dataloader("smoltalk")
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