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
from statistics import mean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHECKPOINT_PATH = "./checkpoints/dpo_model_25-06-06-224951/checkpoint-60000/"
INITIAL_RESULTS_PATH = "evaluation/output/sft/ultrafeedback_heldout_prompts_sft_model_25-06-08-201600_35000.json"
OUTPUT_ITERATIVE_RESULTS_PATH = "evaluation/output/extension/ultrafeedback_heldout_teacher_40mini_model.json" # Detailed results
OUTPUT_FINAL_RESPONSES_PATH = "evaluation/output/extension/ultrafeedback_heldout_teacher_model_4omini_final_responses.json" # New file for just prompt and final response

# Nemotron Reward Model configuration
REWARD_MODEL_API_KEY = os.getenv("REWARD_MODEL_API_KEY", "")
REWARD_MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-reward"
REWARD_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Teacher Model is ChatGPT Configuration
# Marielle instruction: in any new terminal, run export TEACHER_MODEL_API_KEY="your_real_api_key_here", then python your_script.py
TEACHER_MODEL_API_KEY = os.getenv("TEACHER_MODEL_API_KEY", "")
TEACHER_MODEL_NAME = "gpt-4o-mini-2024-07-18"
TEACHER_MODEL_BASE_URL = "https://api.openai.com/v1"

# Configuration for iterative refinement
MAX_ITERATIONS = 1
SCORE_IMPROVEMENT_THRESHOLD = 0.05
MIN_ACCEPTABLE_SCORE = 0.7


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
    client: OpenAI,
    prompt: str,
    response: str,
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
    initial_response = response

    initial_overall_score = get_reward_score(reward_client, REWARD_MODEL_NAME, prompt, current_response)
    logger.info(f"Initial score: {initial_overall_score}")

    # logger.info(f"Prompt:\n{prompt}\n")
    # logger.info(f"Initial Response:\n{current_response}\n")

    # current_best_score = initial_overall_score

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
        
        # try:   

        response = client.responses.create(
            model=TEACHER_MODEL_NAME,
            input=refinement_instruction
        )

        # print(response.output_text)
        # breakpoint()
        current_response = response.output_text
            # logger.info(f"Iteration {i+1} Refined response:\n{current_response}\n")

        
        # except Exception as e:
        #     breakpoint()
        #     logger.error(f"Error during generation in iteration {i+1}: {e}")
        #     break
            
    return initial_response, initial_overall_score, current_response 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative response refinement script with optional teacher model for refinement.")
    parser.add_argument("--use-teacher-model", action="store_true", 
                        help="Whether to use a teacher model for subsequent refinement steps (after the initial generation by the student model). Defaults to Qwen/Qwen2.5-1.5B-Instruct if enabled.")
    # parser.add_argument("--teacher-model-path", type=str, default=TEACHER_MODEL_CHECKPOINT_PATH,
    #                     help="Path or HuggingFace ID for the teacher model. Only used if --use_teacher_model is set.")
    args = parser.parse_args()

    # if args.teacher_model_path:
    #     TEACHER_MODEL_CHECKPOINT_PATH = args.teacher_model_path

    if not REWARD_MODEL_API_KEY:
        logger.error("REWARD_MODEL_API_KEY environment variable is not set. Please set it before running the script.")
        exit()

    if not TEACHER_MODEL_API_KEY:
        logger.error("TEACHER_MODEL_API_KEY environment variable is not set. Please set it before running the script.")
        exit()

    detailed_output = []
    final_responses_output = []
    
    total_responses_processed = 0
    responses_improved_count = 0
    total_score_improvement = 0.0

    reward_client = OpenAI(
        base_url=REWARD_MODEL_BASE_URL,
        api_key=REWARD_MODEL_API_KEY
    )
    logger.info(f"Initialized OpenAI client for Nemotron Reward Model: {REWARD_MODEL_NAME}")

    teacher_client = OpenAI(api_key=TEACHER_MODEL_API_KEY)

    input_df = pd.read_json(INITIAL_RESULTS_PATH, lines=True)
    prompts = input_df["prompt"].to_list()[:3]
    responses = input_df["response"].to_list()[:3]

    initial_scores = []
    final_scores = []
    progress_bar = tqdm(list(zip(prompts, responses)))
    for step, (prompt, response) in enumerate(progress_bar):
        total_responses_processed += 1
        
        initial_response, initial_overall_score, new_response = generate_and_refine(
            teacher_client,
            prompt,
            response
        )

        initial_scores.append(initial_overall_score)
        new_overall_score = get_reward_score(reward_client, REWARD_MODEL_NAME, prompt, new_response)
        final_scores.append(new_overall_score)
        current_best_score = initial_overall_score
        if new_overall_score > current_best_score:
            current_best_score = new_overall_score
            logger.info(f"Score improved! New score: {current_best_score}.")
        else: 
            logger.info(f"Score did not improve. Original score used: {current_best_score}.")
        
        improvement_for_this_response = 0.0
        if current_best_score is not None and initial_overall_score is not None:
            improvement_for_this_response = current_best_score - initial_overall_score
            if improvement_for_this_response > SCORE_IMPROVEMENT_THRESHOLD:
                logger.info(f"Score improved! Delta is: {improvement_for_this_response}")
                responses_improved_count += 1
                total_score_improvement += improvement_for_this_response

        score_improved = improvement_for_this_response > 0.0

        detailed_output.append({
            "prompt": prompt,
            "initial_response": initial_response,
            "initial_score": initial_overall_score,
            "final_response": new_response,
            "final_score": current_best_score,
            "score_improved": score_improved,
            "score_improvement": improvement_for_this_response
        })

        final_responses_output.append({
            "prompt": prompt,
            "response": new_response if score_improved else initial_response
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
    logger.info(f"Overall Average Initial Score was {mean(initial_scores)}")
    logger.info(f"Overall Average of Final Score: {mean(final_scores)}")
    # if args.use_teacher_model:
    #     logger.info(f"Teacher model used for refinement: {TEACHER_MODEL_CHECKPOINT_PATH}")