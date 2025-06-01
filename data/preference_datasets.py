import torch
import sys
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List
from data.utils import get_tokenizer
import logging
from tqdm.auto import tqdm
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 1024
MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH

"""
Datasets for preference learning task.
"""
class SmolTalkDataset(Dataset):
    """
    SmolTalk dataset for SFT.
    """
    def __init__(
        self,
        path="HuggingFaceTB/smol-smoltalk",
        split="train[:1%]", # Consider using an even smaller split for initial testing
        tokenizer="Qwen/Qwen2.5-0.5B",
        max_length=MAX_LENGTH,
    ):
        logger.info(f"Loading data from {path}")
        self.dataset = load_dataset(path, split=split)
        logger.info(f"Loaded {len(self.dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length

    def _apply_chat_template_and_tokenize(self, messages: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Applies chat template, tokenizes, and creates labels for SFT for a single example.
        """
        # Truncate the chat to be at most 2 in length (one user + one assistant).
        truncated_messages = []
        if len(messages) > 2:
            user_msg = None
            assistant_msg = None
            # Go through messages in reverse to find the last user-assistant pair
            for msg in reversed(messages):
                if msg["role"] == "assistant" and assistant_msg is None:
                    assistant_msg = msg
                elif msg["role"] == "user" and user_msg is None and assistant_msg is not None:
                    user_msg = msg
                    break
            # If both found, use them; otherwise use the original messages
            if user_msg and assistant_msg:
                truncated_messages.append([user_msg, assistant_msg])
            else:
                truncated_messages.append(messages[-2:] if len(messages) >= 2 else messages)
        else:
            truncated_messages.append(messages)
            
        # Use chat template to truncated messages
        texts = self.tokenizer.apply_chat_template(
            truncated_messages, tokenize=False
        )
        
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Create labels for SFT training
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def _mask_query_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask out the query tokens so that loss is only computed on assistant responses for a single example.
        """
        # Remove the batch dimension added by the tokenizer when processing a single item.
        current_input_ids = input_ids.squeeze(0)
        current_labels = labels.squeeze(0)

        # Method 1: Look for Qwen's chat template markers
        decoded = self.tokenizer.decode(current_input_ids, skip_special_tokens=False)
        assistant_marker = "<|im_start|>assistant"
        assistant_start_idx = None

        if assistant_marker in decoded:
            # Find the position after the assistant marker
            marker_pos = decoded.find(assistant_marker)
            if marker_pos != -1:
                # Get text up to and including the marker
                prefix = decoded[:marker_pos + len(assistant_marker)]
                # Add newline if it's typically there
                if decoded[marker_pos + len(assistant_marker):marker_pos + len(assistant_marker) + 1] == '\n':
                    prefix += '\n'
                # Encode prefix to find where assistant content starts
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                assistant_start_idx = len(prefix_tokens)
        
        # Second method: if assistant marker not found
        if assistant_start_idx is None:
            # Look for "assistant" token sequence
            assistant_token = self.tokenizer.encode("assistant", add_special_tokens=False)
            if assistant_token:
                tokens = current_input_ids.tolist()
                for j in range(len(tokens) - len(assistant_token) + 1):
                    if tokens[j:j+len(assistant_token)] == assistant_token:
                        assistant_start_idx = j + len(assistant_token)
                        # Skip any formatting tokens after "assistant"
                        while (assistant_start_idx < len(tokens) and 
                                tokens[assistant_start_idx] in [self.tokenizer.pad_token_id, 
                                                                 self.tokenizer.eos_token_id]):
                            assistant_start_idx += 1
                        break
        
        # If assistant start found, mask everything before it
        if assistant_start_idx is not None and assistant_start_idx < len(current_labels):
            current_labels[:assistant_start_idx] = -100
        else:
            # Fallback: if assistant start not found, mask the first half
            mask_length = len(current_labels) // 2
            current_labels[:mask_length] = -100
        
        # Mask padding tokens
        current_labels[current_input_ids == self.tokenizer.pad_token_id] = -100
        
        return current_labels.unsqueeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        tokenized_item = self._apply_chat_template_and_tokenize(item["messages"])
        masked_labels = self._mask_query_tokens(tokenized_item["input_ids"], tokenized_item["labels"])
        
        return {
            "input_ids": tokenized_item["input_ids"].squeeze(0),
            "attention_mask": tokenized_item["attention_mask"].squeeze(0),
            "labels": masked_labels.squeeze(0)
        }



class UltraFeedbackDataset(Dataset):
    """
    UltraFeedback dataset for DPO and RLOO.
    """
    def __init__(
        self,
        path="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs[:1%]",
        tokenizer="Qwen/Qwen2.5-0.5B",
        max_length=MAX_LENGTH,
    ):
        logger.info(f"Loading data from {path}")
        raw_dataset = load_dataset(path, split=split)
        logger.info(f"Loaded {len(raw_dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length

        self.dataset = []
        for example in raw_dataset:
            tokenized = self._tokenize_example(example)
            self.dataset.append(tokenized)

    def _tokenize_example(self, example):
        prompts = self.tokenizer(
            "Prompt: " + example["prompt"] + "\n",
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        chosen = self.tokenizer(
            self.tokenizer.apply_chat_template(example["chosen"], tokenize=False),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        rejected = self.tokenizer(
            self.tokenizer.apply_chat_template(example["rejected"], tokenize=False),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        preferred_ids = torch.cat([prompts.input_ids, chosen.input_ids], dim=-1).squeeze(0)
        preferred_a_masks = torch.cat([prompts.attention_mask, chosen.attention_mask], dim=-1).squeeze(0)
        dispreferred_ids = torch.cat([prompts.input_ids, rejected.input_ids], dim=-1).squeeze(0)
        dispreferred_a_masks = torch.cat([prompts.attention_mask, rejected.attention_mask], dim=-1).squeeze(0)

        return {
            'preferred_ids' : preferred_ids,
            'preferred_a_masks' : preferred_a_masks,
            'dispreferred_ids' : dispreferred_ids,
            'dispreferred_a_masks' : dispreferred_a_masks,
        }

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    



 