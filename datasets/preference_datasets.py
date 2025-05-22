import torch
import sys
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List
from utils import get_tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 1024

MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH

"""
Datasets for preference learning task.
"""

class SmolTokDataset(Dataset):
    """
    SmolTok dataset for SFT.
    """
    def __init__(
        self,
        path="HuggingFaceTB/smol-smoltalk",
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
        self.dataset = self._tokenize_dataset(self.dataset, max_length)

    def _tokenize_dataset(self, dataset, max_length: int):
        def tokenize_sft(examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
            """
            Tokenizes a batch of examples for SFT.
            """
            # Format the text for SFT fine-tuning.

            # TODO: Consider truncating the chat to be at most 2 in length (one user + one assistant).
            texts = self.tokenizer.apply_chat_template(
                examples["messages"], tokenize=False
            )
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        # TODO: Mask out the query tokens.
        output_cols = ["input_ids", "attention_mask", "labels"]
        smoltok_tokenized_dataset = dataset.map(
            tokenize_sft,
            batched=True,  # Process in batches for efficiency
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in output_cols
            ],
            desc="Tokenizing SmolTok dataset",
        )
        smoltok_tokenized_dataset.set_format(
            type="torch", columns=output_cols
        )
        return smoltok_tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item


class UltraFeedbackDataset(Dataset):
    """
    UltraFeedback dataset for DPO and RLOO.
    """
    def __init__(
        self,
        path="HuggingFaceH4/ultrafeedback_binarized",
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
        self.dataset = self._tokenize_dataset(self.dataset, max_length)

    def _tokenize_dataset(self, dataset, max_length: int):
        def tokenize_dicts(batch, tokenizer):
            # ls_of_stringified_dicts = []
            # for dicts_list in batch:
            #     ex = []
            #     for dict in dicts_list:
            #         c = dict["content"]
            #         role = dict["role"]
            #         s = f"content: {c}, role: {role}"
            #         ex.append(s)
            #     joined = " | ".join(ex)
            #     ls_of_stringified_dicts.append(joined)
            texts = tokenizer.apply_chat_template(batch[0], tokenize=False)
            return tokenizer(texts, padding='max_length', truncation=True)

        output_cols = [
            "prompt",
            "prompt_id",
            "chosen",
            "rejected",
            "messages",
            "score_chosen",
            "score_rejected",
            "input_ids",
            "token_type_ids",
            "attention_mask",
        ]
        
        dataset = dataset.map(lambda e: self.tokenizer(e['prompt'], truncation=True, padding='max_length'), batched=True) 
        dataset = dataset.map(lambda e: self.tokenizer(e['prompt_id'], truncation=True, padding='max_length'), batched=True)
        dataset = dataset.map(lambda batch: tokenize_dicts(batch['chosen'], self.tokenizer), batched=True)
        dataset = dataset.map(lambda batch: tokenize_dicts(batch['rejected'], self.tokenizer), batched=True)
        dataset = dataset.map(lambda batch: tokenize_dicts(batch['messages'], self.tokenizer), batched=True)
        dataset = dataset.map(lambda batch: self.tokenizer([str(x) for x in batch['score_chosen']], truncation=True, padding='max_length'), batched=True)
        dataset = dataset.map(lambda batch: self.tokenizer([str(x) for x in batch['score_rejected']], truncation=True, padding='max_length'), batched=True)
        dataset.set_format(type="torch", columns=output_cols)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
