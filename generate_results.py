import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from data.utils import get_tokenizer
import json
import logging
from tqdm.auto import tqdm
from data.dataloader_utils import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./sft_model/checkpoint-80000/" 

def generate_from_checkpoint(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch: dict,
    sample_idx: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "auto"
):
    """
    Generates text using a model loaded from a saved checkpoint.
    """
    if sample_idx >= batch['input_ids'].shape[0]:
        raise ValueError(f"Error: sample_idx {sample_idx} is out of bounds for batch size {batch['input_ids'].shape[0]}")
    
    input_ids = batch['input_ids'][sample_idx].unsqueeze(0).to(model.device)
    attention_mask = batch['attention_mask'][sample_idx].unsqueeze(0).to(model.device) 
    labels = batch['labels'][sample_idx]

    # 1. Decode the original input to the model (user query + assistant prompt)
    original_input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    logger.info(f"Origal input to model:\n{original_input_text}\n")

    # 2. Decode the ground truth response (what the model was trained on)
    valid_label_ids = [label_id for label_id in labels.tolist() if label_id != -100]
    if valid_label_ids:
        ground_truth_response = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
        logger.info(f"Ground truth assistant response (from dataset):\n{ground_truth_response}\n")
    else:
        logger.info("\nNo ground truth labels found for this sample.")

    # 3. Generate the fine-tuned model's response.
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # num_beams=1,
                # repetition_penalty=1.1,
            )
        generated_tokens = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.info(f"Generated assistant response (from model):\n{generated_text}\n")
    except Exception as e:
        logger.info(f"Error during generation: {e}")
        
    logger.info("*"*80)

if __name__ == "__main__":
    checkpoint_path = CHECKPOINT_PATH

    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Error: Checkpoint path '{checkpoint_path}' does not exist or is not a directory.")

    logger.info(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda")
    logger.info(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    eval_dataloader = get_dataloader(
        dataset_name="smoltalk",
        split="test[:1%]",
        batch_size=1,
    )

    progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    for step, batch in enumerate(progress_bar):
        messages = generate_from_checkpoint(
            tokenizer,
            model,
            batch,
        )
        if step >= 5:
            break