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

# Don't change this.
INPUT_JSON_PATH = "evaluation/input/ultrafeedback_heldout_prompt.json"

# Update below to use the right model.
# CHECKPOINT_PATH = "./checkpoints/dpo_model_25-06-07-102513/checkpoint-40000/" 
# CHECKPOINT_PATH = "./checkpoints/sft_model_original/checkpoint-100000/"
CHECKPOINT_PATH = "./checkpoints/sft_model_25-06-08-201600/checkpoint-35000/"
OUTPUT_JSON_PATH = "evaluation/output/sft/ultrafeedback_heldout_prompts_sft_model_25-06-08-201600_35000.json"

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
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 20,
    num_beams=1,
    repetition_penalty=1.3,            # Helps avoid tail-end babbling
    no_repeat_ngram_size=2,            # Prevents repeating phrases
    early_stopping=False,               # Ends when EOS is likely
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
        message, tokenize=False, add_generation_prompt=True
    )
    
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
 
    input_ids = tokenized['input_ids'][0].unsqueeze(0).to(model.device)
    attention_mask = tokenized['attention_mask'][0].unsqueeze(0).to(model.device) 

    # 1. Decode the original input to the model (user query + assistant prompt)
    original_input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
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
                # top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping
            )
        
        generated_tokens = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # generated_text = generated_text.split("<|im_end|>")[0].strip()

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
