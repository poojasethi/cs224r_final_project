import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
from trl import SFTTrainer

@dataclass
class TrainingArguments:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    output_dir: str = "./sft_model"
    fp16: bool = True # Turns on mixed precision training if GPU supports it
    project_name: str = "qwen-smoltalk-sft"
    run_name: str = "custom-sft-run"

class CustomSFTTrainer:
    """
    Our custom trainer for SFT fine-tuning. 
    Loosely inspired by the HuggingFace SFTTrainer.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        args: TrainingArguments,
    ):
        # TODO Initialize models, wandb, etc.
        raise NotImplementedError() 
        
    def train(self):
        """
        Kicks off SFT training.
        """
        raise NotImplementedError()