import torch
import os

class DPOTrainer:
    def __init__():
        # Initialize TensorBoard
        self.log_dir = os.path.join("runs", f"dpo_{dataset_class.__name__}_{os.path.basename(output_dir)}")
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to: {log_dir}")

        policy_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True).to(device)
        ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True).to(device)
        ref_model.eval() # Reference model should be in evaluation mode and not updated
        
    def train():
        """
        Trains a model using Direct Preference Optimization (DPO) from scratch.
            max_length (int): Max sequence length for tokenization.
        """
        print(f"\n--- Starting DPO Training (from scratch) for {dataset_class.__name__} ---")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        train_dataset = dataset_class(split="train", max_length=max_length)

        # Training parameters
        num_train_epochs = 1
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 4
        learning_rate = 5e-6
        weight_decay = 0.01
        warmup_ratio = 0.03
        beta = 0.1 # DPO beta parameter

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=per_device_train_batch_size,
            collate_fn=preference_collate_fn,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        num_training_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps) * num_train_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        scaler = None
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()
        else:
            autocast_dtype = torch.float32

        policy_model.train()
        global_step = 0
        for epoch in range(num_train_epochs):
            print(f"Epoch {epoch + 1}/{num_train_epochs}")
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                chosen_input_ids = batch['chosen_input_ids'].to(device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(device)
                rejected_input_ids = batch['rejected_input_ids'].to(device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(device)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == 'cuda'):
                    # Calculate log probabilities for chosen responses
                    policy_chosen_log_probs = policy_model(
                        input_ids=chosen_input_ids,
                        attention_mask=chosen_attention_mask
                    ).logits.log_softmax(dim=-1)
                    
                    ref_chosen_log_probs = ref_model(
                        input_ids=chosen_input_ids,
                        attention_mask=chosen_attention_mask
                    ).logits.log_softmax(dim=-1)

                    # Calculate log probabilities for rejected responses
                    policy_rejected_log_probs = policy_model(
                        input_ids=rejected_input_ids,
                        attention_mask=rejected_attention_mask
                    ).logits.log_softmax(dim=-1)

                    ref_rejected_log_probs = ref_model(
                        input_ids=rejected_input_ids,
                        attention_mask=rejected_attention_mask
                    ).logits.log_softmax(dim=-1)

                    # DPO Loss calculation
                    # For simplicity, we calculate log_probs for the full sequence.
                    # In a more refined DPO, you'd mask out prompt tokens.
                    # Here, we sum over the sequence for each token's log prob.
                    
                    # Sum log probabilities over the sequence, considering attention mask
                    def get_sequence_log_probs(log_probs, input_ids, attention_mask):
                        # Gather log probabilities for the actual tokens
                        log_probs_for_tokens = torch.gather(log_probs[:, :-1, :], dim=2, index=input_ids[:, 1:].unsqueeze(2)).squeeze(2)
                        # Mask out padded tokens and prompt tokens (if applicable)
                        # For a basic DPO from scratch, we'll sum over the response part.
                        # This assumes the prompt is at the beginning and loss is calculated on the full sequence.
                        # A more robust implementation would require identifying prompt vs response tokens.
                        return (log_probs_for_tokens * attention_mask[:, 1:]).sum(dim=-1)

                    policy_chosen_log_probs_sum = get_sequence_log_probs(policy_chosen_log_probs, chosen_input_ids, chosen_attention_mask)
                    ref_chosen_log_probs_sum = get_sequence_log_probs(ref_chosen_log_probs, chosen_input_ids, chosen_attention_mask)
                    
                    policy_rejected_log_probs_sum = get_sequence_log_probs(policy_rejected_log_probs, rejected_input_ids, rejected_attention_mask)
                    ref_rejected_log_probs_sum = get_sequence_log_probs(ref_rejected_log_probs, rejected_input_ids, rejected_attention_mask)

                    # Calculate ratios
                    pi_log_ratio = policy_chosen_log_probs_sum - policy_rejected_log_probs_sum
                    ref_log_ratio = ref_chosen_log_probs_sum - ref_rejected_log_probs_sum

                    # DPO objective
                    logits = beta * (pi_log_ratio - ref_log_ratio)
                    loss = -torch.nn.functional.logsigmoid(logits).mean() / gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log to console and TensorBoard
                    current_loss = loss.item() * gradient_accumulation_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    if global_step % 10 == 0:
                        print(f"Global Step {global_step}, Loss: {current_loss:.4f}, LR: {current_lr:.6f}")
                    
                    writer.add_scalar('Loss/train', current_loss, global_step)
                    writer.add_scalar('LearningRate/train', current_lr, global_step)

        os.makedirs(output_dir, exist_ok=True)
        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"DPO model saved to {output_dir}")
        writer.close() # Close TensorBoard writer
        del policy_model, ref_model
        torch.cuda.empty_cache()`