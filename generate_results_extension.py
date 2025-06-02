import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import logging
from tqdm.auto import tqdm
import pandas as pd
from openai import OpenAI # For the reward model
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./checkpoints/sft_model_original/checkpoint-100000/"
INPUT_JSON_PATH = "evaluation/input/ultrafeedback_tiny.json"
OUTPUT_JSON_PATH = "evaluation/output/ultrafeedback_checkpoint-100000_refinement_extension.json"

# Reward model specific configuration
REWARD_MODEL_API_KEY = os.getenv("REWARD_MODEL_API_KEY", "")
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward" # Assuming this is accessible via your NVIDIA API key

# Test-time inference parameters
MAX_REFINEMENT_STEPS = 3 # How many times to try and refine the response
IMPROVEMENT_THRESHOLD = 0.05 # Minimum score improvement to continue refining

# --- Utility Functions ---

def load_tokenizer_and_model(checkpoint_path=CHECKPOINT_PATH):
    """
    Loads the tokenizer and language model from the specified checkpoint path.
    """
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
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on device: {device}")

    return tokenizer, model

def get_reward_score(client, model_name, user_content, assistant_content):
    """
    Gets a reward score for a prompt-response pair using the specified reward model.
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            # Ensure the reward model outputs only the score
            max_tokens=10 # Score is typically short
        )
        score_str = completion.choices[0].message.content.removeprefix("reward:").strip()
        score = float(score_str)
        return score
    except Exception as e:
        logger.error(f"Error getting reward score from {model_name}: {e}")
        return None

def generate_from_checkpoint(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_length: int = 512, # Max length for tokenization of input
    max_new_tokens: int = 512, # Max tokens to generate
    temperature: float = 0.7, # Increased temperature for more diverse initial responses
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1, # Slightly reduced to encourage variety
) -> str:
    """
    Generates text using a model loaded from a saved checkpoint.
    """
    message = [{"role": "user", "content": prompt}]

    # Use chat template to format messages
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

    input_ids = tokenized['input_ids'].to(model.device)
    attention_mask = tokenized['attention_mask'].to(model.device)

    logger.debug(f"Original input to model (debug):\n{tokenizer.decode(input_ids[0], skip_special_tokens=False)}\n")

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
                num_beams=1,
                repetition_penalty=repetition_penalty, 
            )
        generated_tokens = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        logger.debug(f"Generated assistant response (debug):\n{generated_text}\n")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        generated_text = "" # Return empty string on error

    return generated_text

def refine_response(
    original_prompt: str,
    current_response: str,
    reward_client: OpenAI,
    reward_model_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_refinement_steps: int = MAX_REFINEMENT_STEPS,
    improvement_threshold: float = IMPROVEMENT_THRESHOLD
) -> Tuple[str, bool]:
    """
    Iteratively refines a response based on reward model feedback.
    """
    best_response = current_response
    best_score = get_reward_score(reward_client, reward_model_name, original_prompt, current_response)
    logger.info(f"Initial score for prompt '{original_prompt[:50]}...': {best_score}")

    if best_score is None:
        logger.warning("Could not get initial reward score. Skipping refinement.")
        return best_response

    response_improved = False
    for step in range(max_refinement_steps):
        logger.info(f"Refinement step {step + 1}/{max_refinement_steps} for prompt '{original_prompt[:50]}...'")

        # Get feedback from the reward model (or a specific critic model if available)
        refinement_instruction = (
            "The previous response did not fully meet the quality criteria. "
            "Please revise it to be more precise, helpful, honest, and truthful"
            "Ensure you strictly follow the original instruction and avoid any rambling or irrelevant information. "
            "Focus on providing a concise and direct answer to the user's request. "
            "Do not start or cut off your response in the middle of a sentence."
            "Respond ONLY with the revised answer."
        )

        # TODO: we could also prompt an LLM reward model for explicit critique.
        # critic_prompt = f"Critique the following response to the user's query:\nUser: {original_prompt}\nAssistant: {current_response}\n\nProvide specific feedback on its instruction following, honesty, helpfulness, truthfulness, and conciseness. Highlight areas for improvement."
        # critique = generate_from_critic_model(critic_prompt)

        # Construct the conversation history for refinement
        # The model sees the original prompt, its previous attempt, and the refinement instruction.
        refinement_messages = [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": current_response},
            {"role": "user", "content": refinement_instruction}
        ]

        # Apply chat template for the refinement prompt
        refinement_text = tokenizer.apply_chat_template(
            refinement_messages, tokenize=False, add_generation_prompt=True
        )

        # Generate a refined response
        new_response = generate_from_checkpoint(
            tokenizer,
            model,
            refinement_text,
            max_new_tokens=512, # Allow sufficient tokens for refined response
            temperature=0.3,
            repetition_penalty=1.4 # Keep some penalty
        )

        # Remove any unncessary whitespace.
        new_response = new_response.strip(".").strip()

        if not new_response:
            logger.warning(f"No new response generated in refinement step {step+1}. Stopping refinement.")
            break

        new_score = get_reward_score(reward_client, reward_model_name, original_prompt, new_response)
        logger.info(f"Refinement step {step + 1} score: {new_score}")

        if new_score is None:
            logger.warning("Could not get new reward score. Stopping refinement.")
            break

        # Check for improvement
        if new_score > best_score + improvement_threshold:
            logger.info(f"Response improved! Old score: {best_score:.4f}, New score: {new_score:.4f}")
            best_response = new_response
            best_score = new_score
            response_improved = True
        else:
            logger.info(f"No significant improvement (or score decreased). Old score: {best_score:.4f}, New score: {new_score:.4f}. Stopping refinement.")
            break # Stop if no significant improvement

        current_response = new_response # Use the new response for the next iteration

    logger.info(f"Final score after refinement: {best_score}")
    return best_response, response_improved

if __name__ == "__main__":
    output = []

    # Initialize reward model client
    if not REWARD_MODEL_API_KEY:
        logger.error("Reward model API key is required. Please set the 'REWARD_MODEL_API_KEY' environment variable.")
        exit()
    reward_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=REWARD_MODEL_API_KEY
    )
    logger.info(f"Initialized OpenAI client for reward model: {REWARD_MODEL_NAME}")

    tokenizer, model = load_tokenizer_and_model()

    input_df = pd.read_json(INPUT_JSON_PATH, lines=True)
    prompts = input_df["prompt"].to_list()
    progress_bar = tqdm(prompts, desc="Processing prompts with iterative refinement")

    num_responses_improved = 0
    for step, prompt in enumerate(progress_bar):
        logger.info(f"\n****** Running prompt {step + 1} ******")
        logger.info(f"Prompt: {prompt}")

        # 1. Generate initial response
        initial_response = generate_from_checkpoint(
            tokenizer,
            model,
            prompt,
            temperature=0.7 # Start with a slightly higher temperature for diversity
        )
        logger.info(f"Initial response:\n{initial_response}")

        # 2. Refine the response iteratively
        final_response, response_improved = refine_response(
            original_prompt=prompt,
            current_response=initial_response,
            reward_client=reward_client,
            reward_model_name=REWARD_MODEL_NAME,
            tokenizer=tokenizer,
            model=model
        )
        logger.info(f"Final refined response:\n{final_response}")
        output.append({"prompt": prompt, "response": final_response})

        if response_improved:
            num_responses_improved += 1
    
    # Log the number of responses that were improved by iterative refinment:
    num_responses_improved += 1
    logger.info(f"Fraction of responses improved: {num_responses_improved}/{len(prompts)}")

    # Save results
    with open(OUTPUT_JSON_PATH, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Generated responses saved to {OUTPUT_JSON_PATH}")