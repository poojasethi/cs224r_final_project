import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict
from datasets import load_dataset, Dataset as HFDataset

logger = logging.getLogger(__name__)

def get_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info(f"Tokenizer for {model_id} loaded successfully.")

    # Set padding token if the tokenizer doesn't have one set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Setting tokenizer.pad_token to tokenizer.eos_token ({tokenizer.pad_token_id})."
        )

    return tokenizer

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset: HFDataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        # Ensure all items are torch tensors. The map function should handle this
        # if return_tensors='pt' was used. If not, you might need to convert here.
        return item
