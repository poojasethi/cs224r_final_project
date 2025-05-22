import torch
import sys
from torch.utils.data import Dataset
from data import load_dataset
from typing import Dict, List
from utils import get_tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512

"""
Datasets for verifier task.
"""
class CogBehaveDataset(Dataset):
    """
    CogBehave dataset for SFT.
    """
    def __init__(
        self,
        path="Asap7772/cog_behav_all_strategies",
        split="train[:1%]",
        tokenizer="Qwen/Qwen2.5-0.5B",
        max_length=MAX_LENGTH,
    ):
        logger.info(f"Loading data from {path}")
        self.dataset = load_dataset(path, split=split)
        logger.info(f"Loaded {len(self.dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length

        # Pre-tokenize all the data so that training is faster.
        self.dataset = self._tokenize_dataset(self.dataset)
    
    def _tokenize_dataset(self, dataset):
        # TODO: Add tokenization.
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

class CountDownDataset(Dataset):
    """
    CountDown dataset for DPO and RLOO.
    """
    def __init__(
        self,
        path="Jiayi-Pan/Countdown-Tasks-3to4",
        split="train[:1%]",
        tokenizer="Qwen/Qwen2.5-0.5B",
        max_length=MAX_LENGTH,
    ):
        logger.info(f"Loading data from {path}")
        self.dataset = load_dataset(path, split=split)
        logger.info(f"Loaded {len(self.dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length

        # Pre-tokenize all the data so that training is faster.
        self.dataset = self._tokenize_dataset(self.dataset)
    
    def _tokenize_dataset(self, dataset):
        # TODO: Add tokenization
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
