import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")




tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Look at Pooja's pushed code for SFT. Tokenize formatted text 
# sequence = "Using a Transformer network is simple"
# tokens = tokenizer.tokenize(sequence, padding=True)

# print(tokens)

# TBD by task
# chat = [
#   {"role": "user", "content": "Hello, how are you?"},
#   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]

# tokenizer.apply_chat_template(chat, tokenize=False)
def tokenize_dicts(batch):
    ls_of_stringified_dicts = []
    for dicts_list in batch:
        ex = []
        for dict in dicts_list:
            c = dict["content"]
            role = dict["role"]
            s = f"content: {c}, role: {role}"
            ex.append(s)
        joined = " | ".join(ex)
        ls_of_stringified_dicts.append(joined)

    return tokenizer(ls_of_stringified_dicts, padding='max_length', truncation=True)
print("Tokenizing UltraFeedback")
dataset = ds.map(lambda e: tokenizer(e['prompt'], truncation=True, padding='max_length'), batched=True) # fix so tokenize everything
dataset = dataset.map(lambda e: tokenizer(e['prompt_id'], truncation=True, padding='max_length'), batched=True)
dataset = dataset.map(lambda batch: tokenize_dicts(batch['chosen']), batched=True)
dataset = dataset.map(lambda batch: tokenize_dicts(batch['rejected']), batched=True)
dataset = dataset.map(lambda batch: tokenize_dicts(batch['messages']), batched=True)
dataset = dataset.map(lambda batch: tokenizer([str(x) for x in batch['score_chosen']], truncation=True, padding='max_length'), batched=True)
dataset = dataset.map(lambda batch: tokenizer([str(x) for x in batch['score_rejected']], truncation=True, padding='max_length'), batched=True)

dataset.set_format(type='torch', columns=['prompt', 'prompt_id', 'chosen', 'rejected', 'messages', 'score_chosen', 'score_rejected', 'input_ids', 'token_type_ids', 'attention_mask'])

train_data = dataset['train_prefs'] # preference modeling eg for dpo
test_data = dataset['test_prefs']
train_fdbk_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_fdbk_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)
next(iter(train_fdbk_dataloader))
next(iter(test_fdbk_dataloader))



tokenizer = AutoTokenizer.from_pretrained("AnReu/math_albert")

datse = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")['train']

def tokenize_nums(batch):
    # breakpoint()
    ls_of_stringified_nums = []
    for output in batch['nums']:
        str_numbers = []
        for num in output:
            str_numbers.append(str(num))
        
        one_str = " ".join(str_numbers)
        ls_of_stringified_nums.append(one_str)

    tokenized = tokenizer(ls_of_stringified_nums, padding='max_length', truncation=True)

    return tokenized

print("Tokenizing Countdown")
dataset_math = datse.map(lambda e: tokenizer([str(x) for x in e['target']], truncation=True, padding='max_length', return_tensors="pt"), batched=True) # tokenize everything (input AND output)
# breakpoint()
dataset_math = dataset_math.map(lambda batch: tokenize_nums(batch), batched=True) # tokenize everything (input AND output)

# breakpoint()
# dataset_math.set_format(type='torch', columns=['target', 'nums'])
dataset_math.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

lengths = [int(dataset_math.shape[0]*0.8)+1, int(dataset_math.shape[0]*0.2)]
# lengths = dataset_math.train_test_split(test_size=0.1)

train_set, val_set = torch.utils.data.random_split(dataset_math, lengths)
# breakpoint() 

train_math_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)
test_math_dataloader = torch.utils.data.DataLoader(val_set, batch_size=32)


next(iter(train_math_dataloader))
next(iter(test_math_dataloader))