import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    wandb_project: str
    wandb_run: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    output_dir: str = "./sft_model"
    # Turn on mixed precision training to reduce memory usage and speed up training.
    fp16: bool = True
    # Use gradient accumulation to effectively increase batch size without increasing memory.
    gradient_accumulation_steps: int = 15 


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
        args: SFTTrainingArguments,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args

        # Set the device.
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device {self.device}")
        self.model.to(self.device)

        # Initialize the optimizer. We use Adam.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Initialize the learning rate scheduler. We use a linear scheduler.
        num_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_training_steps = num_steps_per_epoch * args.num_train_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
        )

        # Enable mixed precision training.
        self.scaler = torch.amp.GradScaler("cuda") if self.args.fp16 else None

        # Initialize WandB so we can monitor training.
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=args.__dict__)
        wandb.watch(self.model, log_freq=args.logging_steps)

    def train(self):
        """
        Kicks off SFT training.
        """
        raise NotImplementedError()
