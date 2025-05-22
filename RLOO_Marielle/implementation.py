import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import dataset_management
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import ufb_reward
import re
import random
import ast
import operator
import ctdn_reward

k = 10


def big_train(dataset):
    if dataset == 'ultrafeedback':
        R = ufb_reward.reward_model_ultrafeedback()
    elif dataset == "countdown":
        R = ctdn_reward.compute_score()
    else:
        print("This is the wrong input for a dataset and the reward model isn't specified.")

    main_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    optimizer = AdamW(main_model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(dataset_management.train_dataloader)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    main_model.to(device)
    progress_bar = tqdm(range(num_training_steps))

    samples = main_model.generate(k)
    rewards = R(samples)

    loss = 0
    for i in range(k):
        reward = rewards[i]
        ex = rewards
        ex.remove(rewards[i])
        average_one_out = ex.mean()
        inside = reward - average_one_out
        grad = torch.log_prob(samples[i]).with_grad()
        loss += inside*grad

    loss = loss/k

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    progress_bar.update(1)



