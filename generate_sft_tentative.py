import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import logging
from dataclasses import dataclass
from trainers.sft_trainer import CustomSFTTrainer



sft_model = model.load_state_dict(torch.load(self.args.sft_output_dir))
        ref_pref_outputs = sft_model.generate(
            input_ids=dispreferred_ids,
            attention_mask=dispreferred_a_masks
        )
        ref_dispref_outputs = sft_model.generate(
            input_ids=preferred_ids,
            attention_mask=preferred_a_masks
        )