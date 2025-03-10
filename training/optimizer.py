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
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 8
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
    
    def compute_advantages(self, trajectories):
        """
        Compute advantages for each step in the trajectories
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Dictionary mapping (trajectory_idx, step_idx) to a list of advantages
        """
        advantages_dict = {}
        
        for traj_idx, trajectory in enumerate(tqdm(trajectories, desc="Computing advantages")):
            for step_idx, step in enumerate(trajectory["steps"]):
                # Get rewards for this step
                rewards = step.get("rewards", [0.0])
                
                # Compute advantage as the reward (no baseline for now)
                advantages = rewards
                
                # Store advantages in the dictionary
                advantages_dict[(traj_idx, step_idx)] = advantages
        
        return advantages_dict
    
    def _compute_ppo_loss(self, logprobs, old_logprobs, advantages):
        """
        Compute PPO loss
        
        Args:
            logprobs: Log probabilities of the current policy
            old_logprobs: Log probabilities of the old policy
            advantages: Advantages
            
        Returns:
            PPO loss
        """
        # Compute ratio between new and old policies
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # Compute losses
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        
        # Take the minimum (pessimistic bound)
        ppo_loss = -torch.min(surrogate1, surrogate2).mean()
        
        return ppo_loss
    
    def _compute_kl_penalty(self, logprobs, old_logprobs):
        """
        Compute KL divergence penalty
        
        Args:
            logprobs: Log probabilities of the current policy
            old_logprobs: Log probabilities of the old policy
            
        Returns:
            KL divergence penalty
        """
        # Compute KL divergence
        kl = (old_logprobs - logprobs).mean()
        
        return kl
    
    def optimize(self, agent, trajectories):
        """
        Optimize the policy using GRPO
        
        Args:
            agent: TextWorldLLMAgent instance
            trajectories: List of trajectories
            
        Returns:
            Dictionary with training metrics
        """
        print("Optimizing policy using GRPO...")
        
        # Compute advantages for all trajectories
        print("Computing advantages for trajectories...")
        advantages_dict = self.compute_advantages(trajectories)
        
        # Prepare data for training
        train_data = []
        for trajectory in tqdm(trajectories, desc="Computing advantages"):
            for step in trajectory["steps"]:
                state = step["state"]
                outputs = step["outputs"]
                
                # Get the advantage for this step
                step_idx = step["step"]
                traj_idx = trajectory["episode"]
                advantages = advantages_dict.get((traj_idx, step_idx), [0.0] * len(outputs))
                
                # Add each output as a separate training example
                for i, output in enumerate(outputs):
                    advantage = advantages[i] if i < len(advantages) else 0.0
                    train_data.append({
                        "state": state,
                        "output": output,
                        "advantage": advantage
                    })
        
        # Shuffle the data
        random.shuffle(train_data)
        
        # Create batches
        batch_size = self.batch_size
        batches = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            [p for p in agent.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Training metrics
        metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": []
        }
        
        # Make sure model is in training mode
        agent.model.train()
        agent.model.to(self.device)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_ppo_loss = 0.0
            epoch_kl_penalty = 0.0
            
            for batch in tqdm(batches, desc=f"Epoch {epoch+1}"):
                # Prepare batch data
                states = [item["state"] for item in batch]
                outputs = [item["output"] for item in batch]
                advantages = torch.tensor([item["advantage"] for item in batch], dtype=torch.float32).to(self.device)
                
                # Tokenize inputs and outputs
                inputs = agent.tokenizer(
                    states,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.max_input_length if hasattr(self.config, 'max_input_length') else 512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get old log probabilities
                old_logprobs_list = []
                
                for i, output in enumerate(outputs):
                    # Tokenize output
                    output_tokens = agent.tokenizer(
                        output,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.max_completion_length if hasattr(self.config, 'max_completion_length') else 128,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get log probabilities for the output (without gradients)
                    old_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i, with_grad=False)
                    old_logprobs_list.append(old_logprobs)
                
                # Combine old log probabilities
                old_logprobs = torch.cat(old_logprobs_list, dim=0)
                
                # Compute new log probabilities and policy loss
                optimizer.zero_grad()
                
                new_logprobs_list = []
                for i, output in enumerate(outputs):
                    # Tokenize output
                    output_tokens = agent.tokenizer(
                        output,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.max_completion_length if hasattr(self.config, 'max_completion_length') else 128,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get log probabilities for the output (with gradients)
                    new_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i, with_grad=True)
                    new_logprobs_list.append(new_logprobs)
                
                # Combine new log probabilities
                new_logprobs = torch.cat(new_logprobs_list, dim=0)
                
                # Compute PPO loss
                ppo_loss = self._compute_ppo_loss(new_logprobs, old_logprobs, advantages)
                
                # Compute KL penalty
                kl_penalty = self._compute_kl_penalty(new_logprobs, old_logprobs)
                
                # Compute total loss
                loss = ppo_loss + self.beta * kl_penalty
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
                
                # Update model
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_ppo_loss += ppo_loss.item()
                epoch_kl_penalty += kl_penalty.item()
            
            # Compute average metrics for the epoch
            avg_loss = epoch_loss / len(batches)
            avg_ppo_loss = epoch_ppo_loss / len(batches)
            avg_kl_penalty = epoch_kl_penalty / len(batches)
            
            # Add to metrics
            metrics["loss"].append(avg_loss)
            metrics["ppo_loss"].append(avg_ppo_loss)
            metrics["kl_penalty"].append(avg_kl_penalty)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, PPO Loss: {avg_ppo_loss:.4f}, KL Penalty: {avg_kl_penalty:.4f}")
        
        # Set model back to evaluation mode
        agent.model.eval()
        
        return metrics
    
    def train(self, agent, env=None, num_iterations=5, num_episodes_per_iteration=5, max_steps=10, save_path=None, trajectories=None):
        """
        Train the agent using GRPO with pre-collected trajectories
        
        Args:
            agent: TextWorldLLMAgent instance
            env: Not used, kept for backward compatibility
            num_iterations: Not used, kept for backward compatibility
            num_episodes_per_iteration: Not used, kept for backward compatibility
            max_steps: Not used, kept for backward compatibility
            save_path: Path to save the trained model
            trajectories: Pre-collected trajectories
            
        Returns:
            Dictionary with training metrics
        """
        print("Training agent using GRPO with pre-collected trajectories...")
        
        if trajectories is None:
            raise ValueError("Trajectories must be provided for training")
        
        all_metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": [],
            "rewards": []
        }
        
        # Compute average reward
        avg_reward = self._compute_average_reward(trajectories)
        all_metrics["rewards"].append(avg_reward)
        print(f"Average reward from pre-collected trajectories: {avg_reward:.4f}")
        
        # Optimize policy
        metrics = self.optimize(agent, trajectories)
        
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
    
    def _compute_average_reward(self, trajectories):
        """
        Compute average reward across all trajectories
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Average reward
        """
        total_reward = 0.0
        total_steps = 0
        
        for trajectory in trajectories:
            for step in trajectory["steps"]:
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
            Log probabilities for the output sequence
        """
        # Get input for this example
        input_ids = inputs.input_ids[batch_idx:batch_idx+1]
        attention_mask = inputs.attention_mask[batch_idx:batch_idx+1] if hasattr(inputs, 'attention_mask') else None
        
        # Get output for this example
        output_ids = output_tokens.input_ids[0]
        
        # Check if we're using an encoder-decoder model (like T5) or a decoder-only model (like GPT)
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        
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
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
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
        
        # Combine token log probabilities
        if token_log_probs:
            return torch.stack(token_log_probs).mean().unsqueeze(0)
        else:
            return torch.tensor([0.0], device=self.device)