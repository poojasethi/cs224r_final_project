import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # Reverted to AutoModelForCausalLM
from openai import OpenAI
import os
import json
import logging
from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints/sft_model_original/checkpoint-100000/"
INPUT_JSON_PATH = "evaluation/input/ultrafeedback.json"
OUTPUT_ITERATIVE_RESULTS_PATH = "evaluation/output/ultrafeedback_iterative_refinement.json" # Detailed results
OUTPUT_FINAL_RESPONSES_PATH = "evaluation/output/ultrafeedback_final_responses.json" # New file for just prompt and final response

# Nemotron Reward Model configuration
REWARD_MODEL_API_KEY = os.getenv("REWARD_MODEL_API_KEY", "")
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"
REWARD_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Configuration for iterative refinement
MAX_ITERATIONS = 3
SCORE_IMPROVEMENT_THRESHOLD = 0.05
MIN_ACCEPTABLE_SCORE = 0.7

# --- Helper Functions ---

def load_model_tokenizer_and_device(
    model_path: str = CHECKPOINT_PATH
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Loads the model and tokenizer using transformers.
    """
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not os.path.isdir(model_path):
        raise ValueError(f"Error: Model path '{model_path}' does not exist or is not a directory.")

    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return tokenizer, model, device

def get_reward_score(client: OpenAI, model_name: str, user_content: str, assistant_content: str) -> Optional[float]:
    """
    Gets a reward score for a prompt-response pair using the Nemotron 70B Reward Model.
    Returns a single float score (typically between 0 and 1, higher is better).
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        )
        score_str = completion.choices[0].message.content.removeprefix("reward:")
        score = float(score_str)
        return score
    except Exception as e:
        logger.error(f"Error getting reward score from Nemotron API: {e}")
        return None

def generate_and_refine(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM, # Changed back to AutoModelForCausalLM
    reward_client: OpenAI,
    prompt: str,
    device: str, # Device is needed for transformers model
    max_length: int = 512,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Tuple[str, Optional[float], int, Optional[float]]:
    """
    Generates and iteratively refines a response using the model and Nemotron reward model.
    """
    current_response = ""
    current_overall_score = -float('inf')
    initial_overall_score = None
    num_iterations = 0

    for i in range(MAX_ITERATIONS):
        num_iterations = i + 1
        messages = [{"role": "user", "content": prompt}]
        
        if i > 0:
            feedback_instruction = (
                f"Previous Response: {current_response}\n\n"
                f"Refine the previous response to achieve a higher quality score. "
                f"Prioritize instruction following, truthfulness, helpfulness, and accuracy. "
                f"Ensure the refined response directly answers the user's original prompt, "
                f"is comprehensive, factually correct, and avoids any redundant or unnecessary information. "
                f"Focus on concise and impactful improvements to the previous response."
            )
            messages.append({"role": "assistant", "content": current_response})
            messages.append({"role": "user", "content": feedback_instruction})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = tokenized['input_ids'].to(model.device)
        attention_mask = tokenized['attention_mask'].to(model.device)

        try:
            with torch.no_grad():
                output_ids = model.generate( # Using model.generate from transformers
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    repetition_penalty=1.3,
                )
            
            start_index = input_ids.shape[1]
            generated_tokens = output_ids[0, start_index:]
            new_response_segment = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            current_response = new_response_segment
            logger.info(f"Iteration {i+1} - Generated assistant response:\n{current_response}\n")

            new_overall_score = get_reward_score(reward_client, REWARD_MODEL_NAME, prompt, current_response)

            if new_overall_score is None:
                logger.warning(f"Could not get reward score for iteration {i+1}. Stopping refinement for this prompt.")
                break
                
            if i == 0:
                initial_overall_score = new_overall_score
            
            logger.info(f"Iteration {i+1} - Nemotron Reward Score: {new_overall_score:.4f}")

            if new_overall_score is not None and \
               (new_overall_score > current_overall_score + SCORE_IMPROVEMENT_THRESHOLD or \
                new_overall_score >= MIN_ACCEPTABLE_SCORE):
                current_overall_score = new_overall_score
                if new_overall_score >= MIN_ACCEPTABLE_SCORE and i > 0:
                    logger.info(f"Stopping early: Satisfactory score reached ({new_overall_score:.4f}).")
                    break
            else:
                logger.info(f"Stopping early: No significant improvement (current score {new_overall_score:.4f} vs previous best {current_overall_score:.4f}) or score decreased.")
                break

        except Exception as e:
            logger.error(f"Error during generation in iteration {i+1}: {e}")
            break
            
    return current_response, current_overall_score, num_iterations, initial_overall_score

if __name__ == "__main__":
    if not REWARD_MODEL_API_KEY:
        logger.error("REWARD_MODEL_API_KEY environment variable is not set. Please set it before running the script.")
        exit()

    detailed_output = []
    final_responses_output = []
    
    total_responses_processed = 0
    responses_improved_count = 0
    total_score_improvement = 0.0

    tokenizer, model, device = load_model_tokenizer_and_device() # Load transformers model

    reward_client = OpenAI(
        base_url=REWARD_MODEL_BASE_URL,
        api_key=REWARD_MODEL_API_KEY
    )
    logger.info(f"Initialized OpenAI client for Nemotron Reward Model: {REWARD_MODEL_NAME}")

    input_df = pd.read_json(INPUT_JSON_PATH, lines=True)
    prompts = input_df["prompt"].to_list()
    progress_bar = tqdm(prompts, desc="Evaluating and Refining Responses")
    
    for step, prompt in enumerate(progress_bar):
        total_responses_processed += 1
        
        final_response, final_score, num_iterations, initial_score = generate_and_refine(
            tokenizer,
            model,
            reward_client,
            prompt,
            device
        )
        
        improvement_for_this_response = 0.0
        if final_score is not None and initial_score is not None:
             improvement_for_this_response = final_score - initial_score
             if improvement_for_this_response > SCORE_IMPROVEMENT_THRESHOLD:
                 responses_improved_count += 1
                 total_score_improvement += improvement_for_this_response

        detailed_output.append({
            "prompt": prompt,
            "initial_score": initial_score,
            "final_response": final_response,
            "final_score": final_score,
            "num_iterations": num_iterations,
            "score_improvement": improvement_for_this_response
        })

        final_responses_output.append({
            "prompt": prompt,
            "response": final_response
        })

    with open(OUTPUT_ITERATIVE_RESULTS_PATH, 'w') as f:
        for item in detailed_output:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_FINAL_RESPONSES_PATH, 'w') as f:
        for item in final_responses_output:
            f.write(json.dumps(item) + "\n")

    avg_score_improvement_for_improved = total_score_improvement / responses_improved_count if responses_improved_count > 0 else 0.0

    logger.info(f"\n**** Iterative Refinement Summary***")
    logger.info(f"Total responses processed: {total_responses_processed}")
    logger.info(f"Responses significantly improved: {responses_improved_count}")
    logger.info(f"Percentage of responses significantly improved: {(responses_improved_count / total_responses_processed * 100):.2f}%")
    logger.info(f"Average score improvement for significantly improved responses: {avg_score_improvement_for_improved:.4f}")
    logger.info(f"Detailed results saved to {OUTPUT_ITERATIVE_RESULTS_PATH}")
    logger.info(f"Final prompts and responses saved to {OUTPUT_FINAL_RESPONSES_PATH}")