import torch
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import logging


# Look at Pooja's pushed code for SFT. Tokenize formatted text 

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
        

def tokenize_ultra(ds, tokenizer):
    print("Tokenizing!")
    dataset = ds.map(lambda e: tokenizer(e['prompt'], truncation=True, padding='max_length'), batched=True) 
    dataset = dataset.map(lambda e: tokenizer(e['prompt_id'], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda batch: tokenize_dicts(batch['chosen'], tokenizer), batched=True)
    dataset = dataset.map(lambda batch: tokenize_dicts(batch['rejected'], tokenizer), batched=True)
    dataset = dataset.map(lambda batch: tokenize_dicts(batch['messages'], tokenizer), batched=True)
    dataset = dataset.map(lambda batch: tokenizer([str(x) for x in batch['score_chosen']], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda batch: tokenizer([str(x) for x in batch['score_rejected']], truncation=True, padding='max_length'), batched=True)

    dataset.set_format(type='torch', columns=['prompt', 'prompt_id', 'chosen', 'rejected', 'messages', 'score_chosen', 'score_rejected', 'input_ids', 'token_type_ids', 'attention_mask'])

    train_data = dataset['train_prefs'] # preference modeling eg for dpo
    test_data = dataset['test_prefs']

    train_ultra_dataloader, test_ultra_dataloader = make_dataloader(train_data, test_data)

    return train_ultra_dataloader, test_ultra_dataloader




def tokenize_nums(batch, tokenizer):
    ls_of_stringified_nums = []
    for output in batch['nums']:
        str_numbers = []
        for num in output:
            str_numbers.append(str(num))
        
        one_str = " ".join(str_numbers)
        ls_of_stringified_nums.append(one_str)

    tokenized = tokenizer(ls_of_stringified_nums, padding='max_length', truncation=True)

    return tokenized


def tokenize_count(datse, tokenizer):
    print("Tokenizing!")
    dataset_math = datse.map(lambda e: tokenizer([str(x) for x in e['target']], truncation=True, padding='max_length', return_tensors="pt"), batched=True)
    dataset_math = dataset_math.map(lambda batch: tokenize_nums(batch, tokenizer), batched=True) 

    dataset_math.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

    lengths = [int(dataset_math.shape[0]*0.8)+1, int(dataset_math.shape[0]*0.2)]
    train_set, val_set = torch.utils.data.random_split(dataset_math, lengths)

    train_math_dataloader, test_math_dataloader = make_dataloader(train_set, val_set)
    return train_math_dataloader, test_math_dataloader



def make_dataloader(train_set, test_set):
    print("Making dataloader")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    next(iter(train_dataloader))
    next(iter(test_dataloader))

    return train_dataloader, test_dataloader



def est_dataset(input):
    if input == "ultrafeedback":
        print("Loading UltraFeedback")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
        train_dataloader, test_dataloader = tokenize_ultra(ds, tokenizer)
    
    elif input == "countdown":
        print("Loading Countdown")
        tokenizer = AutoTokenizer.from_pretrained("AnReu/math_albert")
        datse = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")['train']
        train_dataloader, test_dataloader = tokenize_count(datse, tokenizer)
    
    return train_dataloader, test_dataloader



if __name__ == "__main__":
    if len(sys.argv) != 2:
        est_dataset("ultrafeedback")
        est_dataset("countdown")
    else:
        if sys.argv[1] == "ultrafeedback":
            est_dataset("ultrafeedback")
        elif sys.argv[1] == "countdown":
            est_dataset("countdown")
        else:
            print("Options are 'ultrafeedback' or 'countdown'")