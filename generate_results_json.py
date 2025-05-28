import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from data.utils import get_tokenizer
import json
import logging
from tqdm.auto import tqdm
from data.dataloader_utils import get_dataloader
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./sft_model/checkpoint-80000/" 
INPUT_JSON_PATH = "evaluation/input/ultrafeedback.json"
OUTPUT_JSON_PATH = "evaluation/output/ultrafeedback_checkpoint.json"

def load_tokenizer_and_model(
    checkpoint_path = CHECKPOINT_PATH     
):
    # Set the device.
    device = None
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Error: Checkpoint path '{checkpoint_path}' does not exist or is not a directory.")

    logger.info(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    return tokenizer, model

def generate_from_checkpoint(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_length: int = 512,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "auto"
) -> str:
    """
    Generates text using a model loaded from a saved checkpoint.
    """
    message = [
        {"role": "user", "content": prompt}
    ]

    # Use chat template to truncated messages
    text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_pompt=True
    )
    
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
 
    input_ids = tokenized['input_ids'][0].unsqueeze(0).to(model.device)
    attention_mask = tokenized['attention_mask'][0].unsqueeze(0).to(model.device) 

    # 1. Decode the original input to the model (user query + assistant prompt)
    original_input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    logger.info(f"Origal input to model:\n{original_input_text}\n")

    # 2. Generate the fine-tuned model's response.
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
        
    return generated_text

if __name__ == "__main__":
    output = []

    tokenizer, model = load_tokenizer_and_model()

    input_df = pd.read_json(INPUT_JSON_PATH, lines=True)
    prompts = input_df["prompt"].to_list()
    progress_bar = tqdm(prompts, desc="Evaluating")
    
    for step, prompt in enumerate(progress_bar):
        response = generate_from_checkpoint(
            tokenizer,
            model,
            prompt,
        )
        output.append({"prompt": prompt, "response": response})

    with open(OUTPUT_JSON_PATH, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
