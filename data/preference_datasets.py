import torch
import sys
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List
from data.utils import get_tokenizer
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 256
MAX_RESPONSE_LENGTH = 512
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
        self.dataset = load_dataset(path, split=split)
        logger.info(f"Loaded {len(self.dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length

        # Pre-tokenize all the data so that training is faster.
        self.collate = partial(self._tokenize_dataset, self.tokenizer, self.max_length)

    def _tokenize_dataset(self, batch, tokenizer, max_length: int):     # NOTE: Heavily inspired by https://github.com/0xallam/Direct-Preference-Optimization/blob/main/src/train.py  
        def tokenize_dicts(batch, tokenizer):
            texts = tokenizer.apply_chat_template(batch, tokenize=False)
            return tokenizer(texts, padding='max_length', truncation=True)

        # output_cols = [
        #     "prompt",
        #     "prompt_id",
        #     "chosen",
        #     "rejected",
        #     "messages",
        #     "score_chosen",
        #     "score_rejected",
        #     "input_ids",
        #     "attention_mask",
        # ]
        

        prompts = tokenizer(
            ["Prompt: " + elem['prompt'] + '\n' for elem in batch],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        chosen = tokenizer(
            [tokenizer.apply_chat_template(elem['chosen'], tokenize=False) for elem in batch],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        rejected = tokenizer(
            [tokenizer.apply_chat_template(elem['rejected'], tokenize=False) for elem in batch],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        preferred_ids = torch.cat([prompts.input_ids, chosen.input_ids], dim=-1)
        preferred_a_masks = torch.cat([prompts.attention_mask, chosen.attention_mask], dim=-1)
        dispreferred_ids = torch.cat([prompts.input_ids, rejected.input_ids], dim=-1)
        dispreferred_a_masks = torch.cat([prompts.attention_mask, rejected.attention_mask], dim=-1)

        return {
            'preferred_ids' : preferred_ids,
            'preferred_a_masks' : preferred_a_masks,
            'dispreferred_ids' : dispreferred_ids,
            'dispreferred_a_masks' : dispreferred_a_masks,
            'score_chosen' : batch['score_chosen'],
            'score_rejected' : batch['score_rejected'],
        }

        # chosen_data = dataset.map(lambda example: {
        #     "prompt": example['prompt'],
        #     "answer": example['chosen'], 
        #     "score": example['score_chosen']
        # })
        # chosen_data = dataset.map(lambda example: {
        #     "prompt": example['prompt'],
        #     "prompt_id": example['prompt_id'], 
        #     "answer": example['chosen'], 
        #     "messages": example['messages'],
        #     "score": example['score_chosen']
        # })
        # rejected_data = dataset['prompt'] + dataset['prompt_id'] + dataset['rejected'] + dataset['messages'] + dataset['score_rejected']


        # dataset = dataset.map(lambda e: self.tokenizer(e['prompt'], truncation=True, padding='max_length'), batched=True) 
        # dataset = dataset.map(lambda e: self.tokenizer(e['prompt_id'], truncation=True, padding='max_length'), batched=True)
        # dataset = dataset.map(lambda batch: tokenize_dicts(batch['chosen'], self.tokenizer), batched=True)
        # dataset = dataset.map(lambda batch: tokenize_dicts(batch['rejected'], self.tokenizer), batched=True)
        # dataset = dataset.map(lambda batch: tokenize_dicts(batch['messages'], self.tokenizer), batched=True)
        # dataset = dataset.map(lambda batch: self.tokenizer([str(x) for x in batch['score_chosen']], truncation=True, padding='max_length'), batched=True)
        # dataset = dataset.map(lambda batch: self.tokenizer([str(x) for x in batch['score_rejected']], truncation=True, padding='max_length'), batched=True)
        # dataset.set_format(type="torch", columns=output_cols)
        # return dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item