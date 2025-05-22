import torch
import sys
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List
from utils import get_tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 1024

"""
Datasets for verifier task.
"""