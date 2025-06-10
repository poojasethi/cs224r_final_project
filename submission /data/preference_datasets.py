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

# Maxium lengths in terms of # of tokens.
MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = 1024
MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH

# Use the approximation that each token is ~3/4 of a word.
# Slightly less to account for special tokens we might add.
TOKEN_TO_WORD_RATIO = 0.70

# Calculate maxium lengths in terms of # of words.
MAX_PROMPT_WORD_COUNT = int(MAX_PROMPT_LENGTH * TOKEN_TO_WORD_RATIO) 
MAX_RESPONSE_WORD_COUNT = int(MAX_RESPONSE_LENGTH * TOKEN_TO_WORD_RATIO)

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
        print(f"Loading data from {path}")
        raw_dataset = load_dataset(path, split=split)
        print(f"Loaded {len(raw_dataset)} samples.")

        self.max_length = max_length
        self.tokenizer = get_tokenizer(tokenizer)
        self.dataset = []
        
        # Tokenize
        skipped_examples = 0
        progress_bar = tqdm(raw_dataset)
        for example in progress_bar:
            tokenized_example = self._tokenize_example(example)
            if tokenized_example:
                self.dataset.append(tokenized_example)
            else: 
                skipped_examples += 1
        print(f"Tokenized {len(self.dataset)} valid samples.")
        print(f"Skipped {skipped_examples} examples due to prompt being too long.")

    def _tokenize_example(self, example: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Applies chat template, tokenizes, and creates labels for SFT for a single example.
        """
        # Truncate the chat to be at most 2 in length (one user + one assistant).
        truncated_messages = []
        user_msg = None
        assistant_msg = None

        messages = example["messages"]

        # Go through messages to find the first user-assistant pair.
        for msg in messages:
            if msg["role"] == "user" and user_msg is None:
                user_msg = msg
            if msg["role"] == "assistant" and assistant_msg is None:
                assistant_msg = msg
    
        prompt = user_msg["content"]
        prompt_split = prompt.split()
        if len(prompt_split) > MAX_PROMPT_WORD_COUNT:
            # Skip examples that have super long prompts. Our test set only has short prompts.
            print(f"Prompt was longer than expected ({len(prompt_split)} words), can be no more than {MAX_PROMPT_WORD_COUNT}.")
            return None

        # Truncate the chosen response to a reasonable maximum number of words.
        response = assistant_msg["content"]
        response_split = response.split()
        if len(response_split) > MAX_RESPONSE_WORD_COUNT:
            print(f"Chosen reponse was longer than expected ({len(response_split)} words), trimming to {MAX_RESPONSE_WORD_COUNT}.")
            response_truncated = " ".join(response_split[:MAX_RESPONSE_WORD_COUNT])

            # Update the examples to use the truncated response.
            response = response_truncated 
        
        # Add EOS tokens to prevent model from rambling.
        response = response + self.tokenizer.eos_token 
        assistant_msg["content"] = response

        truncated_messages.append(user_msg)
        truncated_messages.append(assistant_msg)
 
        # Use chat template to truncated messages
        templated_messages= self.tokenizer.apply_chat_template(
            truncated_messages, tokenize=False
        )

        tokenized = self.tokenizer(
            templated_messages,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Create labels for SFT training
        tokenized["labels"] = tokenized["input_ids"].clone()

        ##### The below steps mask the query tokens so that they don't contribute to the loss. ###
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False) 

        templated_prompt = self.tokenizer.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )

        prompt_tokenized = self.tokenizer(
            templated_prompt,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        ) 
        prompt_input_id_len = len(prompt_tokenized["input_ids"].squeeze(0)) 

        # Update the labels so that the prompt token ids don't contribute to the loss.
        tokenized["labels"][:, :prompt_input_id_len] = -100

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["labels"].squeeze(0)
        }
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UltraFeedbackDataset(Dataset):
    """
    UltraFeedback dataset for DPO and RLOO.
    Fixed version of the original dataset to correctly truncate.
    """
    def __init__(
        self,
        path="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs[:1%]",
        tokenizer_path="Qwen/Qwen2.5-0.5B",
    ):
        print(f"Loading data from {path}")
        raw_dataset = load_dataset(path, split=split)
        print(f"Loaded {len(raw_dataset)} samples.")

        self.tokenizer = get_tokenizer(tokenizer_path)
        
        # Tokenize
        self.dataset = []
        skipped_examples = 0
        progress_bar = tqdm(raw_dataset)
        for example in progress_bar:
            tokenized_example = self._tokenize_example(example)
            if tokenized_example:
                self.dataset.append(tokenized_example)
            else: 
                skipped_examples += 1
        print(f"Tokenized {len(self.dataset)} valid samples.")
        print(f"Skipped {skipped_examples} examples due to prompt being too long.")

    def _tokenize_example(self, example):
        example_chosen = example["chosen"]
        example_rejected = example["rejected"]

        prompt_chosen = example_chosen[0]["content"]
        prompt_rejected = example_rejected[0]["content"]
        assert prompt_chosen == prompt_rejected, "Prompts between chosen and reject should match."
        prompt = prompt_chosen

        prompt_split = prompt.split()
        if len(prompt_split) > MAX_PROMPT_WORD_COUNT:
            # Skip examples that have super long prompts. Our test set only has short prompts.
            print(f"Prompt was longer than expected ({len(prompt_split)} words), can be no more than {MAX_PROMPT_WORD_COUNT}.")
            return None

        # Truncate the chosen response to a reasonable maximum number of words.
        chosen_response = example_chosen[1]["content"]
        chosen_response_split = chosen_response.split()
        if len(chosen_response_split) > MAX_RESPONSE_WORD_COUNT:
            print(f"Chosen reponse was longer than expected ({len(chosen_response_split)} words), trimming to {MAX_RESPONSE_WORD_COUNT}.")
            chosen_truncated = " ".join(chosen_response_split[:MAX_RESPONSE_WORD_COUNT])

            # Update the examples to use the truncated response.
            example_chosen[1]["content"] = chosen_truncated

        # Truncate the rejected response to a reasonable maximum number of words.
        rejected_response = example_rejected[1]["content"]
        rejected_response_split = rejected_response.split()
        if len(rejected_response_split) > MAX_RESPONSE_WORD_COUNT:
            print(f"Rejected reponse was longer than expected ({len(rejected_response_split)} words), trimming to {MAX_RESPONSE_WORD_COUNT}.")
            rejected_truncated = " ".join(rejected_response_split[:MAX_RESPONSE_WORD_COUNT])

            # Update the examples to use the truncated response.
            example_rejected[1]["content"] = rejected_truncated
        
        # Add EOS tokens to prevent model from rambling.
        example_chosen[1]["content"] = example_chosen[1]["content"]  + self.tokenizer.eos_token
        example_rejected[1]["content"] = example_rejected[1]["content"] + self.tokenizer.eos_token 

        chosen = self.tokenizer(
            self.tokenizer.apply_chat_template(example_chosen, tokenize=False, add_generation_prompt=False),
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
            padding_side="right",
        )
        rejected = self.tokenizer(
            self.tokenizer.apply_chat_template(example_rejected, tokenize=False, add_generation_prompt=False),
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
            padding_side="right",
        )
        
        preferred_ids = chosen.input_ids.squeeze(0)
        preferred_a_masks = chosen.attention_mask.squeeze(0)
        dispreferred_ids = rejected.input_ids.squeeze(0)
        dispreferred_a_masks = rejected.attention_mask.squeeze(0)

        return {
            'preferred_ids' : preferred_ids,
            'preferred_a_masks' : preferred_a_masks,
            'dispreferred_ids' : dispreferred_ids,
            'dispreferred_a_masks' : dispreferred_a_masks,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



 