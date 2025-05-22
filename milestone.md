# Milestone Requirements

Your milestone report should be a one-page document that showcases the completion of the initial implementation in section 1.

This entails validating the data loading, implementation, and evaluation RL Fine-Tuning of Language Models of the algorithms referenced in section 1.

Think about what metrics would be useful to present here when evaluating on the two tasks.

In your report, you are required to show the quantiative and qualitative performance of your policy,
effectively reporting the quantitative metrics against the relevant baselines.

Finally, submit your initial checkpoints for SFT, DPO, and RLOO to the leaderboard for both the Countdown and Ultrafeedback tasks (anonymously if you wish) and share your unique id in the report for review.

## Data Loading and Construction

**Preference Datasets**
* [ ] SmolTok (dataset for SFT)
* [ ] UltraFeedback (dataset for DPO and RLOO)

**Verifier-Based Datasets**
* [ ] WarmStart Dataset (SFT)
* [ ] On-Policy Preference Dataset (DPO)
* [ ] Prompts Dataset (RLOO)

## Method Implementation
* [ ] SFT Model - Malisha
* [ ] DPO Model - Pooja
* [ ] RLOO Model - Marielle

## Evaluation Setup

## Milestone Deliverables
1. Verifier-Based Task
    a. SFT on the CogBehave dataset using Qwen2.5-0.5B. Win rate Evaluation is still on Countdown!
    b. DPO on the Countdown dataset using the SFT model checkpoint.
    c. Reward Model on the Countdown dataset using Qwen2.5-0.5B.
    d. RLOO on the Countdown dataset using the SFT model checkpoint.
2. Preference Task
    a. SFT on the SmolTalk dataset using Qwen2.5-0.5B. Win rate Evaluation is still on Ultrafeedback!
    b. DPO on the UltraFeedback dataset using the SFT model checkpoint.
    c. Reward Model on the UltraFeedback dataset using Qwen2.5-0.5B.
    c. RLOO on the UltraFeedback dataset using the SFT model checkpoint.

