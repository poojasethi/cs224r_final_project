# CS 224R Final Project
Final project for CS224R: Deep Reinforcement Learning. RL Fine-Tuning of Language Models. https://cs224r.stanford.edu/material/CS224R_Default_Project_Guidelines.pdf 

## Activate conda environment
```
conda env create --name rl_llm
conda activate rl_llm
pip install transformers torch datasets wandb 
```

## Run experiments
Log-in to wandb
```
wandb login
```

Start SFT training with SmolTok dataset.
```
python instruction_following_sft.py
```

Generate sample results from the fine-tuned checkpoint.
```
python generate_results.py
```