import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import gc
import os
import random

class MyGRPOOptimizer:
    """Implements GRPO algorithm for TextWorld agent"""
    
    def __init__(self, config):
        """
        Initialize the GRPO optimizer
        
        Args:
            config: Configuration object with GRPO parameters
        """
        self.config = config
        
        # GRPO hyperparameters
        self.num_samples = config.num_samples if hasattr(config, 'num_samples') else 8  # G in the writeup
        self.epsilon = config.epsilon if hasattr(config, 'epsilon') else 0.2  # PPO clipping parameter
        self.beta = config.beta if hasattr(config, 'beta') else 0.01  # KL penalty coefficient
        self.learning_rate = config.learning_rate if hasattr(config, 'learning_rate') else 5e-5
        self.num_epochs = config.num_epochs if hasattr(config, 'num_epochs') else 3
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 2  # Smaller batch size since each example is larger
        self.max_grad_norm = config.max_grad_norm if hasattr(config, 'max_grad_norm') else 1.0
        
        # Reward function parameters
        self.format_reward = config.format_reward if hasattr(config, 'format_reward') else 0.5
        self.format_penalty = config.format_penalty if hasattr(config, 'format_penalty') else -1.0
        self.room_reward = config.room_reward if hasattr(config, 'room_reward') else 0.5
        self.room_penalty = config.room_penalty if hasattr(config, 'room_penalty') else -0.5
        
        # Discount factor
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Print hyperparameters
        print(f"Initialized GRPO optimizer with {self.num_samples} samples, epsilon={self.epsilon}, beta={self.beta}")
        print(f"Reward parameters: format_reward={self.format_reward}, format_penalty={self.format_penalty}, room_reward={self.room_reward}, room_penalty={self.room_penalty}")
        
    def compute_advantages(self, steps_data):
        """
        Compute advantages for each step in the trajectories
        
        Args:
            steps_data: Pre-collected gameplay data
            
        Returns:
            step_advantages: List of advantages for each step
        """
        step_advantages = []
        
        # First pass: collect all rewards
        for step in steps_data:
            step_rewards = step.get("rewards", [0.0])
            # Compute mean and std of rewards for normalization
            rewards_mean = sum(step_rewards) / len(step_rewards)
            rewards_std = (sum((r - rewards_mean) ** 2 for r in step_rewards) / len(step_rewards)) ** 0.5
            # Avoid division by zero
            if rewards_std < 1e-8:
                rewards_std = 1.0
            advantages = [(r - rewards_mean) / rewards_std for r in step_rewards]
            step_advantages.append(advantages)
        
        return step_advantages
    
    def _compute_ppo_loss(self, logprobs, old_logprobs, advantages):
        """
        Compute PPO loss
        
        Args:
            logprobs: Log probabilities of the current policy (already averaged over tokens)
            old_logprobs: Log probabilities of the old policy (already averaged over tokens)
            advantages: Advantages for each completion
            
        Returns:
            PPO loss (averaged over completions)
        """
        # Clip log probability differences to prevent extreme ratios
        # This helps with numerical stability
        log_ratio = logprobs - old_logprobs
        log_ratio = torch.clamp(log_ratio, min=-2.0, max=2.0)
        
        # Compute ratio between new and old policies
        ratio = torch.exp(log_ratio)
        
        # Additional safety clipping for ratios
        ratio = torch.clamp(ratio, min=0.1, max=10.0)
        
        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # Compute losses
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        
        # Take the minimum (pessimistic bound)
        # This implements the F term in the GRPO objective
        ppo_values = torch.min(surrogate1, surrogate2)
        
        # Average over completions (G in the GRPO paper)
        # This implements the 1/G sum_a term in the GRPO objective
        ppo_loss = -ppo_values.mean()
        
        # Debug information for extreme values
        if torch.isnan(ppo_loss).any() or torch.isinf(ppo_loss).any():
            print(f"Warning: NaN or Inf in PPO loss!")
            print(f"  logprobs: {logprobs}")
            print(f"  old_logprobs: {old_logprobs}")
            print(f"  log_ratio: {log_ratio}")
            print(f"  ratio: {ratio}")
            print(f"  advantages: {advantages}")
        
        return ppo_loss
    
    def _compute_kl_penalty(self, logprobs, old_logprobs):
        """
        Compute KL divergence penalty using the formula:
        D_KL = exp(old_logprobs - logprobs) - (old_logprobs - logprobs) - 1
        
        Args:
            logprobs: Log probabilities of the current policy (already averaged over tokens)
            old_logprobs: Log probabilities of the old policy (already averaged over tokens)
            
        Returns:
            KL divergence penalty (averaged over completions)
        """
        # Compute delta of log probabilities
        delta_lp = old_logprobs - logprobs
        
        # Allow for larger differences in log probabilities
        delta_lp = torch.clamp(delta_lp, min=-2.0, max=2.0)
        
        # Compute KL divergence: exp(delta_lp) - delta_lp - 1
        kl_div = torch.exp(delta_lp) - delta_lp - 1.0
        
        # Clip KL divergence to a reasonable range
        kl_div = torch.clamp(kl_div, min=0.0, max=5.0)
        
        # Average over completions (G in the GRPO paper)
        # This implements the 1/G sum_a term in the GRPO objective
        kl_penalty = kl_div.mean()
        
        # Debug information for extreme values
        if torch.isnan(kl_penalty).any() or torch.isinf(kl_penalty).any():
            print(f"Warning: NaN or Inf in KL penalty!")
            print(f"  logprobs: {logprobs}")
            print(f"  old_logprobs: {old_logprobs}")
            print(f"  delta_lp: {delta_lp}")
            print(f"  kl_div: {kl_div}")
        
        return kl_penalty
        
    def optimize(self, agent, steps_data, save_path=None, save_each_epoch=False):
        """
        Optimize the policy using GRPO
        
        Args:
            agent: TextWorldLLMAgent instance
            steps_data: Pre-collected gameplay data
            save_path: Path to save the model
            save_each_epoch: If True, save model after each epoch
            
        Returns:
            Dictionary with training metrics
        """
        print("Optimizing policy using GRPO...")
        
        # Make sure model is in training mode
        agent.model.train()
        agent.model.to(self.device)
        
        # Compute advantages for all trajectories
        step_advantages = self.compute_advantages(steps_data)
        
        # Prepare data for training, maintaining the structure of prompts and completions
        # Each entry in structured_data is a prompt with its G completions and advantages
        structured_data = []
        
        for step, advantages in zip(steps_data, step_advantages):
            state = step["state"]
            outputs = step["outputs"]
                
            # Create a data point with the prompt and all its completions
            structured_data.append({
                "state": state,
                "outputs": outputs,
                "advantages": advantages,
                "old_logprobs": None
            })
        
        # Shuffle the prompts (but keep completions together)
        random.shuffle(structured_data)
        
        # Create batches of B prompts
        batch_size = self.batch_size
        batches = [structured_data[i:i+batch_size] for i in range(0, len(structured_data), batch_size)]
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            [p for p in agent.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Training metrics
        metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": [],
            "ratio_mean": [],
            "ratio_min": [],
            "ratio_max": [],
            "advantage_mean": [],
            "advantage_min": [],
            "advantage_max": []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_ppo_loss = 0.0
            epoch_kl_penalty = 0.0
            epoch_ratio_mean = 0.0
            epoch_ratio_min = float('inf')
            epoch_ratio_max = float('-inf')
            epoch_advantage_mean = 0.0
            epoch_advantage_min = float('inf')
            epoch_advantage_max = float('-inf')

            # At the beginning of each epoch, compute old logprobs for all prompts
            print(f"Computing old logprobs for epoch {epoch+1}...")
            
            # Temporarily set model to eval mode for consistent old_logprobs
            agent.model.eval()
            with torch.no_grad():  # Explicitly use no_grad for the entire block
                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Computing old logprobs")):
                    for prompt_data in batch:
                        state = prompt_data["state"]
                        outputs = prompt_data["outputs"]
                        
                        # Tokenize the prompt once outside the completion loop
                        input_tokens = agent.tokenizer(
                            state,
                            padding="max_length",
                            truncation=True,
                            max_length=self.config.max_input_length if hasattr(self.config, 'max_input_length') else 512,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        old_logprobs_list = []
                        # Process all completions for this prompt
                        for output in outputs:
                            output_tokens = agent.tokenizer(
                                output,
                                padding="max_length",
                                truncation=True,
                                max_length=self.config.max_completion_length if hasattr(self.config, 'max_completion_length') else 128,
                                return_tensors="pt"
                            ).to(self.device)
                            
                            # We don't need with_grad=False here since we're in a no_grad block
                            old_logprobs = self._compute_logprobs(agent.model, input_tokens, output_tokens, 0)
                            old_logprobs_list.append(old_logprobs)
                            
                            # Optional: Clear cache periodically to prevent OOM
                            if len(old_logprobs_list) % 100 == 0:
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        prompt_data["old_logprobs"] = old_logprobs_list
                        
                    # Print progress every few batches
                    if batch_idx % 10 == 0:
                        print(f"  Computed old logprobs for {batch_idx}/{len(batches)} batches")

            # Set model back to train mode for the training loop
            agent.model.train()
            
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}")):
                batch_loss = 0.0
                batch_ppo_loss = 0.0
                batch_kl_penalty = 0.0
                batch_ratios = []
                batch_advantages = []
                
                # Process each prompt in the batch
                for prompt_data in batch:
                    state = prompt_data["state"]
                    outputs = prompt_data["outputs"]
                    advantages = torch.tensor(prompt_data["advantages"], dtype=torch.float32).to(self.device)
                    
                    # Track advantages for debugging
                    batch_advantages.extend(prompt_data["advantages"])
                    
                    # Tokenize the prompt once
                    input_tokens = agent.tokenizer(
                        state,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.max_input_length if hasattr(self.config, 'max_input_length') else 512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    old_logprobs_list = prompt_data["old_logprobs"]
                    old_logprobs = torch.cat(old_logprobs_list, dim=0)
                    
                    # Compute new log probabilities and policy loss
                    optimizer.zero_grad()
                    
                    new_logprobs_list = []
                    for output in outputs:
                        # Tokenize output
                        output_tokens = agent.tokenizer(
                            output,
                            padding="max_length",
                            truncation=True,
                            max_length=self.config.max_completion_length if hasattr(self.config, 'max_completion_length') else 128,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Get log probabilities for the output (with gradients)
                        new_logprobs = self._compute_logprobs(agent.model, input_tokens, output_tokens, 0, with_grad=True)
                        new_logprobs_list.append(new_logprobs)
                    
                    # Combine new log probabilities
                    new_logprobs = torch.cat(new_logprobs_list, dim=0)
                    
                    # Debug ratios
                    with torch.no_grad():
                        ratios = torch.exp(new_logprobs - old_logprobs)
                        batch_ratios.extend(ratios.cpu().numpy().tolist())                
                        print(f"\nStep ratios: {ratios}")
                    
                    # Compute PPO loss for this prompt (average over its completions)
                    prompt_ppo_loss = self._compute_ppo_loss(new_logprobs, old_logprobs, advantages)
                    
                    # Compute KL penalty for this prompt (average over its completions)
                    prompt_kl_penalty = self._compute_kl_penalty(new_logprobs, old_logprobs)
                    
                    # Compute total loss for this prompt
                    prompt_loss = prompt_ppo_loss + self.beta * prompt_kl_penalty
                    
                    # Accumulate losses for the batch
                    batch_loss += prompt_loss.item()
                    batch_ppo_loss += prompt_ppo_loss.item()
                    batch_kl_penalty += prompt_kl_penalty.item()
                    
                    # Backward pass for this prompt
                    prompt_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
                
                # Update model
                optimizer.step()
                
                # Compute batch statistics
                batch_size = len(batch)
                if batch_size > 0:
                    # Average losses over prompts in the batch
                    avg_batch_loss = batch_loss / batch_size
                    avg_batch_ppo_loss = batch_ppo_loss / batch_size
                    avg_batch_kl_penalty = batch_kl_penalty / batch_size
                    
                    # Update epoch metrics
                    epoch_loss += avg_batch_loss
                    epoch_ppo_loss += avg_batch_ppo_loss
                    epoch_kl_penalty += avg_batch_kl_penalty
                    
                    # Compute ratio statistics
                    if batch_ratios:
                        ratio_mean = sum(batch_ratios) / len(batch_ratios)
                        ratio_min = min(batch_ratios)
                        ratio_max = max(batch_ratios)
                        
                        epoch_ratio_mean += ratio_mean
                        epoch_ratio_min = min(epoch_ratio_min, ratio_min)
                        epoch_ratio_max = max(epoch_ratio_max, ratio_max)
                    
                    # Compute advantage statistics
                    if batch_advantages:
                        adv_mean = sum(batch_advantages) / len(batch_advantages)
                        adv_min = min(batch_advantages)
                        adv_max = max(batch_advantages)
                        
                        epoch_advantage_mean += adv_mean
                        epoch_advantage_min = min(epoch_advantage_min, adv_min)
                        epoch_advantage_max = max(epoch_advantage_max, adv_max)
                    
                    # Print batch metrics every 10 batches
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(batches)}: Loss={avg_batch_loss:.4f}, PPO={avg_batch_ppo_loss:.4f}, KL={avg_batch_kl_penalty:.4f}")
                        print(f"  Advantages: mean={adv_mean:.4f}, min={adv_min:.4f}, max={adv_max:.4f}")
                        print(f"  Ratios: mean={ratio_mean:.4f}, min={ratio_min:.4f}, max={ratio_max:.4f}")
                
                # Add this to your optimizer's training loop
                if epoch == 0 and batch_idx == 0:
                    print(f"Batch structure in optimizer:")
                    print(f"  Batch size: {len(batch)} prompts")
                    for i, prompt_data in enumerate(batch[:2]):
                        print(f"  Prompt {i}:")
                        print(f"    Outputs: {len(prompt_data['outputs'])} completions")
                        print(f"    Advantages: {len(prompt_data['advantages'])} values")
                
            
            # Compute average metrics for the epoch
            num_batches = len(batches)
            avg_loss = epoch_loss / num_batches
            avg_ppo_loss = epoch_ppo_loss / num_batches
            avg_kl_penalty = epoch_kl_penalty / num_batches
            avg_ratio_mean = epoch_ratio_mean / num_batches
            avg_ratio_min = epoch_ratio_min
            avg_ratio_max = epoch_ratio_max
            avg_advantage_mean = epoch_advantage_mean / num_batches
            avg_advantage_min = epoch_advantage_min
            avg_advantage_max = epoch_advantage_max
            
            # Add to metrics
            metrics["loss"].append(avg_loss)
            metrics["ppo_loss"].append(avg_ppo_loss)
            metrics["kl_penalty"].append(avg_kl_penalty)
            metrics["ratio_mean"].append(avg_ratio_mean)
            metrics["ratio_min"].append(avg_ratio_min)
            metrics["ratio_max"].append(avg_ratio_max)
            metrics["advantage_mean"].append(avg_advantage_mean)
            metrics["advantage_min"].append(avg_advantage_min)
            metrics["advantage_max"].append(avg_advantage_max)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, PPO Loss: {avg_ppo_loss:.4f}, KL Penalty: {avg_kl_penalty:.4f}")
            print(f"  Ratios: mean={avg_ratio_mean:.4f}, min={avg_ratio_min:.4f}, max={avg_ratio_max:.4f}")
            print(f"  Advantages: mean={avg_advantage_mean:.4f}, min={avg_advantage_min:.4f}, max={avg_advantage_max:.4f}")
            
            # Save model after each epoch if requested
            if save_each_epoch and save_path:
                epoch_save_path = f"{save_path}_epoch_{epoch+1}.pt"
                epoch_metrics = {k: metrics[k][:epoch+1] for k in metrics}  # Include metrics up to this epoch
                self._save_model(agent, epoch_save_path, epoch_metrics)
                print(f"Model saved after epoch {epoch+1} to {epoch_save_path}")
        
        # Set model back to evaluation mode
        agent.model.eval()
        
        return metrics
    
    def train(self, agent, env=None, num_iterations=5, num_episodes_per_iteration=5, max_steps=10, save_path=None, steps_data=None, save_each_epoch=False):
        """
        Train the agent using GRPO with pre-collected trajectories
        
        Args:
            agent: TextWorldLLMAgent instance
            env: Not used, kept for backward compatibility
            num_iterations: Not used, kept for backward compatibility
            num_episodes_per_iteration: Not used, kept for backward compatibility
            max_steps: Not used, kept for backward compatibility
            save_path: Path to save the trained model
            steps_data: Pre-collected gameplay data
            save_each_epoch: If True, save model after each epoch
            
        Returns:
            Dictionary with training metrics
        """
        print("Training agent using GRPO with pre-collected trajectories...")
        
        if steps_data is None:
            raise ValueError("Trajectories must be provided for training")
        
        all_metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": [],
            "rewards": []
        }
        
        # Compute average reward
        avg_reward = self._compute_average_reward(steps_data)
        all_metrics["rewards"].append(avg_reward)
        print(f"Average reward from pre-collected trajectories: {avg_reward:.4f}")
        
        # Optimize policy
        metrics = self.optimize(agent, steps_data, save_path=save_path, save_each_epoch=save_each_epoch)
        
        # Update metrics
        all_metrics["loss"].extend(metrics["loss"])
        all_metrics["ppo_loss"].extend(metrics["ppo_loss"])
        all_metrics["kl_penalty"].extend(metrics["kl_penalty"])
        
        # Save model if path is provided
        if save_path:
            final_save_path = f"{save_path}_final.pt"
            self._save_model(agent, final_save_path, all_metrics)
            print(f"Final model saved to {final_save_path}")
        
        return all_metrics
    
    def _compute_average_reward(self, steps_data):
        """
        Compute average reward across all trajectories
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Average reward
        """
        total_reward = 0.0
        total_steps = 0
        
        for step in steps_data:
            rewards = step.get("rewards", [0.0])
            total_reward += sum(rewards)
            total_steps += len(rewards)
        
        if total_steps == 0:
            return 0.0
        
        return total_reward / total_steps
    
    def _save_model(self, agent, path, metrics=None):
        """
        Save the model to disk
        
        Args:
            agent: TextWorldLLMAgent instance
            path: Path to save the model
            metrics: Optional metrics to save with the model
        """
        print(f"Saving model to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'metrics': metrics
        }, path)
    
    def _compute_logprobs(self, model, inputs, output_tokens, batch_idx=0, with_grad=False):
        """
        Compute log probabilities for a given output sequence
        
        Args:
            model: The model to use for computing log probabilities
            inputs: Tokenized input sequence
            output_tokens: Tokenized output sequence
            batch_idx: Index in the batch
            with_grad: Whether to compute gradients (True for new policy, False for old policy)
            
        Returns:
            Log probability for the output sequence (averaged over tokens)
        """
        # Get input for this example
        input_ids = inputs.input_ids[batch_idx:batch_idx+1]
        attention_mask = inputs.attention_mask[batch_idx:batch_idx+1] if hasattr(inputs, 'attention_mask') else None
        
        # Get output for this example
        output_ids = output_tokens.input_ids[0]
        
        # Check if we're using an encoder-decoder model (like T5) or a decoder-only model (like GPT)
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", True)
        
        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        # For encoder-decoder models, we need to provide decoder inputs
        if is_encoder_decoder:
            # For computing log probs, we need to shift the output_ids to create decoder_input_ids
            # The first token of decoder_input_ids is the decoder_start_token_id
            # The rest are the output_ids except the last one
            decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
            if decoder_start_token_id is None:
                decoder_start_token_id = getattr(model.config, "pad_token_id", 0)
            
            # Create decoder_input_ids by shifting output_ids right and prepending decoder_start_token_id
            decoder_input_ids = torch.cat([
                torch.tensor([[decoder_start_token_id]], device=self.device),
                output_ids[:-1].unsqueeze(0)
            ], dim=1)
            
            model_inputs["decoder_input_ids"] = decoder_input_ids
        
        # Forward pass through the model
        if with_grad:
            # Compute with gradients for the new policy
            outputs = model(**model_inputs)
        else:
            # Compute without gradients for the old policy
            with torch.no_grad():
                outputs = model(**model_inputs)
            
        # Get logits
        logits = outputs.logits
        
        # Apply temperature scaling for numerical stability
        temperature = 1.0
        scaled_logits = logits / temperature
        
        # Compute log probabilities
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        
        # Extract log probabilities for the actual output tokens
        token_log_probs = []
        
        if is_encoder_decoder:
            # For encoder-decoder models, we compare each position in the output with the next token
            for i in range(len(output_ids) - 1):  # -1 because we don't need the last token's prediction
                if output_ids[i+1] == model.config.pad_token_id:
                    continue  # Skip pad tokens
                token_log_prob = log_probs[0, i, output_ids[i+1]]
                token_log_probs.append(token_log_prob)
        else:
            # For decoder-only models, we need to handle differently
            # This part remains the same as before
            for i in range(len(output_ids) - 1):  # -1 because we don't need the last token's prediction
                if output_ids[i+1] == model.config.pad_token_id:
                    continue  # Skip pad tokens
                token_log_prob = log_probs[0, i, output_ids[i+1]]
                token_log_probs.append(token_log_prob)
        
        # Combine token log probabilities by averaging over tokens
        # This implements the 1/|o^a_t| factor in the GRPO objective
        if token_log_probs:
            # Clip log probabilities to prevent extreme values
            clipped_log_probs = torch.stack(token_log_probs)
            # Clip to a wider range that better matches observed values
            clipped_log_probs = torch.clamp(clipped_log_probs, min=-30.0, max=-1.0)
            return clipped_log_probs.mean().unsqueeze(0)
        else:
            return torch.tensor([-20.0], device=self.device)  # Return a reasonable default value



