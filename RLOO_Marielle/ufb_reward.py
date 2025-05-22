import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dataset_management
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import implementation

def loss_bt(outputs):
    prompt = dataset_management.train_dataloader['prompt']
    y_w = dataset_management.train_dataloader['chosen']
    y_l = dataset_management.train_dataloader['rejected']

    reward_w = R(y_w)
    reward_l = R(y_l)

    inside = torch.log(torch.sigmoid(reward_w - reward_l))
    reward_ultrafeedback = torch.mean(inside)
    return reward_ultrafeedback



def reward_model_ultrafeedback():
    reward_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    optimizer = AdamW(reward_model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(dataset_management.train_dataloader)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reward_model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    reward_model.train()
    for epoch in range(num_epochs):
        for batch in dataset_management.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = reward_model(**batch)
            loss = loss_bt(outputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)


    metric= load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute(),