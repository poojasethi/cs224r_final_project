import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import os
import json
import logging
from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHECKPOINT_PATH = "./checkpoints/dpo_model_25-06-06-224951/checkpoint-60000/"
INITIAL_RESULTS_PATH = "evaluation/output/ultrafeedback_checkpoint_dpo.json"
OUTPUT_ITERATIVE_RESULTS_PATH = "evaluation/output/ultrafeedback_dpo_teacher_model.json" # Detailed results
OUTPUT_FINAL_RESPONSES_PATH = "evaluation/output/ultrafeedback_dpo_teacher_model_final_responses.json" # New file for just prompt and final response

# Nemotron Reward Model configuration
REWARD_MODEL_API_KEY = os.getenv("REWARD_MODEL_API_KEY", "")
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"
REWARD_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Teacher Model Configuration (Optional)
# Using Qwen/Qwen2.5-1.5B-Instruct as the default teacher model
TEACHER_MODEL_CHECKPOINT_PATH = "Qwen/Qwen2.5-1.5B-Instruct" 
TEACHER_MODEL_GEN_TEMPERATURE = 0.7
TEACHER_MODEL_GEN_TOP_P = 0.95

# Configuration for iterative refinement
MAX_ITERATIONS = 1
SCORE_IMPROVEMENT_THRESHOLD = 0.05
MIN_ACCEPTABLE_SCORE = 0.7

def load_model_tokenizer_and_device(
    model_path: str = CHECKPOINT_PATH,
    teacher_model_path: Optional[str] = None
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str, Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]:
    """
    Loads the student and optionally the teacher model and their tokenizers using transformers.
    """
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not os.path.isdir(model_path):
        raise ValueError(f"Error: Student model path '{model_path}' does not exist or is not a directory.")

    logger.info(f"Loading student tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading student model from {model_path}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype)
    model.eval()

    teacher_tokenizer, teacher_model = None, None
    if teacher_model_path:
        logger.info(f"Attempting to load teacher tokenizer from {teacher_model_path}...")
        try:
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
            if teacher_tokenizer.pad_token is None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            logger.info(f"Loading teacher model from {teacher_model_path}...")
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            teacher_model.eval()
            logger.info("Teacher model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load teacher model from '{teacher_model_path}': {e}. Continuing without teacher model for refinement.")
            teacher_tokenizer, teacher_model = None, None

    return tokenizer, model, device, teacher_tokenizer, teacher_model

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
    model: AutoModelForCausalLM,
    reward_client: OpenAI,
    prompt: str,
    response: str,
    device: str, # This 'device' is now primarily for logging, as device_map handles placement
    teacher_tokenizer: Optional[AutoTokenizer] = None,
    teacher_model: Optional[AutoModelForCausalLM] = None,
    use_teacher_model: bool = False, # New parameter for refinement
    max_length: int = 1024,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Tuple[str, Optional[float], int, Optional[float]]:
    """
    Generates and iteratively refines a response using the model and Nemotron reward model.
    The student model generates the initial response. If enabled, the teacher model is used for refinement steps.
    """
    current_response = response.strip()
    num_iterations = 0

    initial_overall_score = get_reward_score(reward_client, REWARD_MODEL_NAME, prompt, current_response)
    logger.info(f"Initial score: {initial_overall_score}")

    # logger.info(f"Prompt:\n{prompt}\n")
    # logger.info(f"Initial Response:\n{current_response}\n")

    current_best_score = initial_overall_score

    if use_teacher_model and teacher_model and teacher_tokenizer:
        logger.info(f"Refining response using teacher model ({TEACHER_MODEL_CHECKPOINT_PATH}).")
        model = teacher_model
        tokenizer = teacher_tokenizer
        temperature = TEACHER_MODEL_GEN_TEMPERATURE
        top_p = TEACHER_MODEL_GEN_TOP_P
    else: # i > 0 but no teacher model for refinement, or not using it
        logger.info(f"Refining response using student model.")

    for i in range(MAX_ITERATIONS):
        num_iterations = i + 1
        messages = []
        
         # Get feedback from the reward model (or a specific critic model if available)Add commentMore actions
        refinement_instruction = (
            "The response below may have not fully meet the quality criteria. "
            "Please revise the response to be more precise, helpful, honest, and truthful. "
            "Ensure you strictly follow the instruction and avoid any rambling or irrelevant information. "
            "Please only output the new response. Do not provide any explanations of changes that were made. \n"
            f"Here is the original instruction:\n{prompt}\n"
            f"Here is the response to revise:\n{current_response}\n"
        )

        # TODO: we could also prompt an LLM reward model for explicit critique.
        # critic_prompt = f"Critique the following response to the user's query:\nUser: {original_prompt}\nAssistant: {current_response}\n\nProvide specific feedback on its instruction following, honesty, helpfulness, truthfulness, and conciseness. Highlight areas for improvement."
        # critique = generate_from_critic_model(critic_prompt)

        # Construct the conversation history for refinement
        messages = [
            {"role": "user", "content": refinement_instruction}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move input_ids and attention_mask to the generation model's device
        input_ids = tokenized['input_ids'].to(model.device)
        attention_mask = tokenized['attention_mask'].to(model.device)

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

            current_response = generated_text
            # logger.info(f"Iteration {i+1} Refined response:\n{current_response}\n")

            new_overall_score = get_reward_score(reward_client, REWARD_MODEL_NAME, prompt, current_response)
            if new_overall_score > current_best_score:
                current_best_score = new_overall_score
                logger.info(f"Score improved! New score: {current_best_score}.")
            else: 
                logger.info(f"Score did not improve. New score: {new_overall_score}.")

        except Exception as e:
            logger.error(f"Error during generation in iteration {i+1}: {e}")
            break
            
    return current_response, current_best_score, num_iterations, initial_overall_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative response refinement script with optional teacher model for refinement.")
    parser.add_argument("--use-teacher-model", action="store_true", 
                        help="Whether to use a teacher model for subsequent refinement steps (after the initial generation by the student model). Defaults to Qwen/Qwen2.5-1.5B-Instruct if enabled.")
    parser.add_argument("--teacher-model-path", type=str, default=TEACHER_MODEL_CHECKPOINT_PATH,
                        help="Path or HuggingFace ID for the teacher model. Only used if --use_teacher_model is set.")
    args = parser.parse_args()

    if args.teacher_model_path:
        TEACHER_MODEL_CHECKPOINT_PATH = args.teacher_model_path

    if not REWARD_MODEL_API_KEY:
        logger.error("REWARD_MODEL_API_KEY environment variable is not set. Please set it before running the script.")
        exit()

    detailed_output = []
    final_responses_output = []
    
    total_responses_processed = 0
    responses_improved_count = 0
    total_score_improvement = 0.0

    # Load models and tokenizers
    if args.use_teacher_model:
        tokenizer, model, device, teacher_tokenizer, teacher_model = load_model_tokenizer_and_device(
            model_path=CHECKPOINT_PATH, 
            teacher_model_path=TEACHER_MODEL_CHECKPOINT_PATH
        )
        if not teacher_model or not teacher_tokenizer:
            raise ValueError("Teacher model was requested for refinement but could not be loaded. Proceeding without it.")
    else:
        tokenizer, model, device, _, _ = load_model_tokenizer_and_device(model_path=CHECKPOINT_PATH)
        teacher_tokenizer, teacher_model = None, None # Ensure these are None if not used

    reward_client = OpenAI(
        base_url=REWARD_MODEL_BASE_URL,
        api_key=REWARD_MODEL_API_KEY
    )
    logger.info(f"Initialized OpenAI client for Nemotron Reward Model: {REWARD_MODEL_NAME}")

    input_df = pd.read_json(INITIAL_RESULTS_PATH, lines=True)
    prompts = input_df["prompt"].to_list()
    responses = input_df["response"].to_list()

    progress_bar = tqdm(list(zip(prompts, responses)))
    for step, (prompt, response) in enumerate(progress_bar):
        total_responses_processed += 1
        
        final_response, final_score, num_iterations, initial_score = generate_and_refine(
            tokenizer,
            model,
            reward_client,
            prompt,
            response,
            device,
            teacher_tokenizer=teacher_tokenizer,
            teacher_model=teacher_model,
            use_teacher_model=args.use_teacher_model
        )
        
        improvement_for_this_response = 0.0
        if final_score is not None and initial_score is not None:
            improvement_for_this_response = final_score - initial_score
            if improvement_for_this_response > SCORE_IMPROVEMENT_THRESHOLD:
                logger.info(f"Score improved! Delta is: {improvement_for_this_response}")
                responses_improved_count += 1
                total_score_improvement += improvement_for_this_response

        score_improved = improvement_for_this_response > 0.0

        detailed_output.append({
            "prompt": prompt,
            "initial_score": initial_score,
            "final_response": final_response,
            "final_score": final_score,
            "num_iterations": num_iterations,
            "score_improved": score_improved,
            "score_improvement": improvement_for_this_response,
            "used_teacher_model_for_refinement": args.use_teacher_model
        })

        final_responses_output.append({
            "prompt": prompt,
            "response": final_response if score_improved else response
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
    logger.info(f"Teacher model for refinement was {'ENABLED' if args.use_teacher_model else 'DISABLED'}.")
    if args.use_teacher_model:
        logger.info(f"Teacher model used for refinement: {TEACHER_MODEL_CHECKPOINT_PATH}")