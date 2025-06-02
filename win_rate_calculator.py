import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_responses_file(file_path):
    """
    Loads prompts and fine-tuned model responses from jsonl file.
    Expected format: {"prompt": "...", "response": "..."}
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Successfully loaded {len(data)} entries from {file_path}")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        exit()
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        exit()
    return data

def generate_base_model_response(model, tokenizer, prompt, max_new_tokens=100):
    """
    Generates a response from the base model for a given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Use sampling for more diverse responses
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def get_reward_score(client, model_name, user_content, assistant_content):
    """
    Gets a reward score for a prompt-response pair using the Nemotron 70B Reward Model.
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
        logger.error(f"Error getting reward score: {e}")
        return None

def calculate_win_rate(
    fine_tuned_responses_file: str,
    base_model_id: str,
    reward_model_api_key: str,
    reward_model_name: str = "nvidia/llama-3.1-nemotron-70b-reward"
):
    """
    Calculates the win rate of the fine-tuned model against a base model.
    """
    # Load fine-tuned model responses
    fine_tuned_data = load_responses_file(fine_tuned_responses_file)
    if not fine_tuned_data:
        logger.error("No data loaded from the fine-tuned responses file. Exiting.")
        return

    # Initialize base model and tokenizer
    logger.info(f"Loading base model and tokenizer from {base_model_id}")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16) # Use bfloat16 for efficiency
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        # Move base model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        base_model.eval() # Set to evaluation mode
        logger.info(f"Base model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Error loading base model or tokenizer: {e}")
        exit()

    # Initialize OpenAI client for the reward model
    if not reward_model_api_key:
        logger.error("Reward model API key is required. Please set the 'REWARD_MODEL_API_KEY' environment variable or pass it as an argument.")
        exit()

    reward_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=reward_model_api_key
    )
    logger.info(f"Initialized OpenAI client for reward model: {reward_model_name}")

    win_labels = []

    logger.info("Starting win rate calculation...")
    for entry in tqdm(fine_tuned_data, desc="Processing prompts"):
        prompt = entry["prompt"]
        fine_tuned_response = entry["response"]

        # 1. Get response from the base model
        base_model_response = generate_base_model_response(base_model, base_tokenizer, prompt)

        # 2. Get reward scores
        fine_tuned_score = get_reward_score(reward_client, reward_model_name, prompt, fine_tuned_response)
        base_model_score = get_reward_score(reward_client, reward_model_name, prompt, base_model_response)

        if fine_tuned_score is None or base_model_score is None:
            logger.warning(f"Skipping prompt '{prompt}' due to missing reward score.")
            continue

        # 3. Construct per-prompt win-rate binary label
        if fine_tuned_score > base_model_score:
            win_labels.append(1)
        else:
            win_labels.append(0)

    # 4. Calculate the win rate
    if not win_labels:
        logger.warning("No valid comparisons were made. Win rate cannot be calculated.")
        return 0.0

    win_rate = sum(win_labels) / len(win_labels)
    logger.info(f"Win rate")
    logger.info(f"Total prompts processed: {len(fine_tuned_data)}")
    logger.info(f"Valid comparisons: {len(win_labels)}")
    logger.info(f"Fine-tuned model wins: {sum(win_labels)}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    return win_rate

if __name__ == "__main__":
    FINE_TUNED_RESPONSES_FILE = "evaluation/output/ultrafeedback_checkpoint.json"

    # ID of the base model to compare against (e.g., "Qwen/Qwen2.5-0.5B")
    # This should be the same model_id you used for the base model in training.
    BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B" # Replace with your base model ID

    # Your NVIDIA API key for the Nemotron 70B Reward Model
    # export REWARD_MODEL_API_KEY="YOUR_API_KEY_HERE"
    REWARD_MODEL_API_KEY = os.getenv("REWARD_MODEL_API_KEY", "")

    # --- Run the calculation ---
    win_rate = calculate_win_rate(
        fine_tuned_responses_file=FINE_TUNED_RESPONSES_FILE,
        base_model_id=BASE_MODEL_ID,
        reward_model_api_key=REWARD_MODEL_API_KEY
    )

    if win_rate is not None:
        logger.info(f"Final Win Rate: {win_rate:.2%}")

