import torch
import sys
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List
from data.utils import get_tokenizer
import logging

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

            #truncate the chat to be at most 2 in length (one user + one assistant).
            truncated_messages = []
            for messages in examples["messages"]:
                #keep only last user message and assistant response (max 2 messages)
                if len(messages) > 2:
                    #find the last user message and corresponding assistant response
                    user_msg = None
                    assistant_msg = None
                    #go through messages in reverse to find the last user-assistant pair
                    for msg in reversed(messages):
                        if msg["role"] == "assistant" and assistant_msg is None:
                            assistant_msg = msg
                        elif msg["role"] == "user" and user_msg is None and assistant_msg is not None:
                            user_msg = msg
                            break
                    #if both found, use them; otherwise use the original messages
                    if user_msg and assistant_msg:
                        truncated_messages.append([user_msg, assistant_msg])
                    else:
                        truncated_messages.append(messages[-2:] if len(messages) >= 2 else messages)
                else:
                    truncated_messages.append(messages)
            
            #use chat template to truncated messages
            texts = self.tokenizer.apply_chat_template(
                truncated_messages, tokenize=False
            )
    
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            #create labels for SFT training
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        def mask_query_tokens(examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            Mask out the query tokens so that loss is only computed on assistant responses
            """
            input_ids = examples["input_ids"]
            labels = examples["labels"]
            
            for i in range(len(input_ids)):
                current_input_ids = input_ids[i]
                current_labels = labels[i]
                #convert to list for easier processing
                tokens = current_input_ids.tolist()

                #find where assistant response starts
                assistant_start_idx = None
                
                #Method 1: Look for Qwen's chat template markers
                decoded = self.tokenizer.decode(current_input_ids, skip_special_tokens=False)
                assistant_marker = "<|im_start|>assistant"
                if assistant_marker in decoded:
                    # Find the position after the assistant marker
                    marker_pos = decoded.find(assistant_marker)
                    if marker_pos != -1:
                        #get text up to and including the marker
                        prefix = decoded[:marker_pos + len(assistant_marker)]
                        #add newline if it's typically there
                        if decoded[marker_pos + len(assistant_marker):marker_pos + len(assistant_marker) + 1] == '\n':
                            prefix += '\n'
                        #encode prefix to find where assistant content starts
                        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                        assistant_start_idx = len(prefix_tokens)
                
                #second method: if assistant marker not found
                if assistant_start_idx is None:
                    #look for "assistant" token sequence
                    assistant_token = self.tokenizer.encode("assistant", add_special_tokens=False)
                    if assistant_token:
                        for j in range(len(tokens) - len(assistant_token) + 1):
                            if tokens[j:j+len(assistant_token)] == assistant_token:
                                assistant_start_idx = j + len(assistant_token)
                                #skip any formatting tokens after "assistant"
                                while (assistant_start_idx < len(tokens) and 
                                    tokens[assistant_start_idx] in [self.tokenizer.pad_token_id, 
                                                                    self.tokenizer.eos_token_id]):
                                    assistant_start_idx += 1
                                break
                #if assistant start found, mask everything before it
                if assistant_start_idx is not None and assistant_start_idx < len(current_labels):
                    current_labels[:assistant_start_idx] = -100
                else:
                    #mask first half
                    mask_length = len(current_labels) // 2
                    current_labels[:mask_length] = -100
                #mask padding tokens
                current_labels[current_input_ids == self.tokenizer.pad_token_id] = -100
                labels[i] = current_labels
            examples["labels"] = labels
            return examples
        
        #apply processing steps in sequence
        output_cols = ["input_ids", "attention_mask", "labels"]
        
        #first tokenize data
        tokenized_dataset = dataset.map(
            tokenize_sft,
            batched=True,
            remove_columns=[
                col for col in dataset.column_names 
                if col not in output_cols
            ],
            desc="Tokenizing SmolTalk dataset",
        )
        tokenized_dataset.set_format(
            type="torch", columns=output_cols
        )
        
        #then apply masking to the tokenized data
        masked_dataset = tokenized_dataset.map(
            mask_query_tokens,
            batched=True,
            desc="Masking query tokens",
        )
        
        masked_dataset.set_format(
            type="torch", columns=output_cols
        )
        
        return masked_dataset

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
