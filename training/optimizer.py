import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import gc
import os

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
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99  # Discount factor
        
        # Training parameters
        self.num_iterations = config.num_iterations if hasattr(config, 'num_iterations') else 3
        self.num_episodes_per_iteration = config.num_episodes_per_iteration if hasattr(config, 'num_episodes_per_iteration') else 5
        
        # Device
        self.device = config.device if hasattr(config, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initialized GRPO optimizer with {self.num_samples} samples, epsilon={self.epsilon}, beta={self.beta}")
        print(f"Reward parameters: format_reward={self.format_reward}, format_penalty={self.format_penalty}, room_reward={self.room_reward}, room_penalty={self.room_penalty}")
        print(f"Training for {self.num_epochs} epochs with batch size {self.batch_size} and learning rate {self.learning_rate}")
    
    def compute_advantages(self, trajectories):
        """
        Compute advantages from trajectories
        
        Args:
            trajectories: List of trajectories, where each trajectory contains:
                - states: List of states s_t
                - outputs: List of outputs o^a_t for each state
                - rewards: List of rewards R^a_t for each output
        
        Returns:
            List of trajectories with advantages added
        """
        print("Computing advantages for trajectories...")
        
        for trajectory in tqdm(trajectories, desc="Computing advantages"):
            # For each step in the trajectory
            for step_idx, step_data in enumerate(trajectory["steps"]):
                # Get rewards for all samples at this step
                rewards = step_data["rewards"]
                
                # Compute mean and std of rewards
                rewards_mean = np.mean(rewards)
                rewards_std = np.std(rewards)
                
                # Avoid division by zero
                if rewards_std == 0:
                    rewards_std = 1.0
                
                # Compute advantages: (R^a_t - mean(R^a_t)) / std(R^a_t)
                advantages = [(r - rewards_mean) / rewards_std for r in rewards]
                
                # Store advantages in the trajectory
                step_data["advantages"] = advantages
        
        return trajectories
    
    def _compute_ppo_loss(self, logprobs, old_logprobs, advantages):
        """
        Compute the PPO loss
        
        Args:
            logprobs: Log probabilities from current model
            old_logprobs: Log probabilities from old model
            advantages: Advantage estimates
            
        Returns:
            PPO loss
        """
        # Compute probability ratio
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        
        # Take the minimum of the two surrogate losses
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        return ppo_loss
    
    def _compute_kl_penalty(self, logprobs, old_logprobs):
        """
        Compute KL divergence penalty
        
        Args:
            logprobs: Log probabilities from current model
            old_logprobs: Log probabilities from old model
            
        Returns:
            KL divergence penalty
        """
        # Compute KL divergence: exp(old_logprobs - logprobs) - (old_logprobs - logprobs) - 1
        kl_div = torch.exp(old_logprobs - logprobs) - (old_logprobs - logprobs) - 1
        kl_penalty = self.beta * kl_div.mean()
        
        return kl_penalty
    
    def optimize(self, agent, trajectories):
        """
        Update policy using GRPO
        
        Args:
            agent: TextWorldLLMAgent instance
            trajectories: List of trajectories with advantages
            
        Returns:
            Dictionary with training metrics
        """
        print("Optimizing policy using GRPO...")
        
        # Compute advantages if not already computed
        if not trajectories[0]["steps"][0].get("advantages"):
            trajectories = self.compute_advantages(trajectories)
        
        # Create a copy of the agent's model for old policy
        old_model = copy.deepcopy(agent.model)
        old_model.eval()  # Set to evaluation mode
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            [p for p in agent.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Prepare training data
        train_data = []
        for trajectory in trajectories:
            for step_data in trajectory["steps"]:
                states = step_data["states"]  # Input prompts
                outputs = step_data["outputs"]  # Generated outputs
                advantages = step_data["advantages"]  # Computed advantages
                
                for i in range(len(outputs)):
                    train_data.append({
                        "state": states[i],
                        "output": outputs[i],
                        "advantage": advantages[i]
                    })
        
        # Training metrics
        metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": []
        }
        
        # Training loop
        agent.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Create batches
            batches = [train_data[i:i+self.batch_size] for i in range(0, len(train_data), self.batch_size)]
            
            epoch_loss = 0
            epoch_ppo_loss = 0
            epoch_kl_penalty = 0
            
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
                    max_length=agent.config.max_input_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get old log probabilities
                with torch.no_grad():
                    old_logprobs_list = []
                    
                    for i, output in enumerate(outputs):
                        # Tokenize output
                        output_tokens = agent.tokenizer.encode(output, add_special_tokens=False)
                        
                        # Get log probabilities from old model
                        old_outputs = old_model(
                            input_ids=inputs.input_ids[i:i+1],
                            attention_mask=inputs.attention_mask[i:i+1]
                        )
                        old_logits = old_outputs.logits
                        
                        # Calculate log probabilities for each token in the output
                        old_token_logprobs = []
                        for j, token_id in enumerate(output_tokens):
                            if j == 0:
                                # First token prediction based on input
                                token_logits = old_logits[0, -1, :]
                            else:
                                # Subsequent token predictions
                                # We need to run the model again with the input + generated tokens so far
                                context_tokens = torch.cat([
                                    inputs.input_ids[i:i+1], 
                                    torch.tensor([output_tokens[:j]]).to(self.device)
                                ], dim=1)
                                context_outputs = old_model(input_ids=context_tokens)
                                token_logits = context_outputs.logits[0, -1, :]
                            
                            # Get log probability of the actual token that was generated
                            token_logprob = F.log_softmax(token_logits, dim=0)[token_id]
                            old_token_logprobs.append(token_logprob)
                        
                        # Average log probability across all tokens in the output
                        old_logprobs_list.append(torch.mean(torch.stack(old_token_logprobs)))
                    
                    old_logprobs = torch.stack(old_logprobs_list)
                
                # Get current log probabilities
                logprobs_list = []
                
                # Zero gradients
                optimizer.zero_grad()
                
                for i, output in enumerate(outputs):
                    # Tokenize output
                    output_tokens = agent.tokenizer.encode(output, add_special_tokens=False)
                    
                    # Get log probabilities from current model
                    current_outputs = agent.model(
                        input_ids=inputs.input_ids[i:i+1],
                        attention_mask=inputs.attention_mask[i:i+1]
                    )
                    current_logits = current_outputs.logits
                    
                    # Calculate log probabilities for each token in the output
                    current_token_logprobs = []
                    for j, token_id in enumerate(output_tokens):
                        if j == 0:
                            # First token prediction based on input
                            token_logits = current_logits[0, -1, :]
                        else:
                            # Subsequent token predictions
                            context_tokens = torch.cat([
                                inputs.input_ids[i:i+1], 
                                torch.tensor([output_tokens[:j]]).to(self.device)
                            ], dim=1)
                            context_outputs = agent.model(input_ids=context_tokens)
                            token_logits = context_outputs.logits[0, -1, :]
                        
                        # Get log probability of the actual token that was generated
                        token_logprob = F.log_softmax(token_logits, dim=0)[token_id]
                        current_token_logprobs.append(token_logprob)
                    
                    # Average log probability across all tokens in the output
                    logprobs_list.append(torch.mean(torch.stack(current_token_logprobs)))
                
                logprobs = torch.stack(logprobs_list)
                
                # Compute PPO loss
                ppo_loss = self._compute_ppo_loss(logprobs, old_logprobs, advantages)
                
                # Compute KL penalty
                kl_penalty = self._compute_kl_penalty(logprobs, old_logprobs)
                
                # Total loss
                loss = ppo_loss + kl_penalty
                
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
            
            # Average metrics for the epoch
            avg_loss = epoch_loss / len(batches)
            avg_ppo_loss = epoch_ppo_loss / len(batches)
            avg_kl_penalty = epoch_kl_penalty / len(batches)
            
            # Store metrics
            metrics["loss"].append(avg_loss)
            metrics["ppo_loss"].append(avg_ppo_loss)
            metrics["kl_penalty"].append(avg_kl_penalty)
            
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, PPO Loss: {avg_ppo_loss:.4f}, KL Penalty: {avg_kl_penalty:.4f}")
        
        # Clean up
        del old_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return metrics
    
    def compute_rewards(self, trajectory, rollout):
        """
        Compute rewards for a trajectory based on the rollout
        
        Args:
            trajectory: Dictionary with state and output information
            rollout: Rollout object with reward information
            
        Returns:
            Total reward combining format, room prediction, and discounted game rewards
        """
        # Get format reward/penalty
        format_check_passed = rollout.format_check_passed
        format_reward = self.format_reward if format_check_passed else self.format_penalty
        
        # Get room prediction reward/penalty
        room_prediction = rollout.action_info.get("room_prediction")
        actual_room = rollout.next_room
        
        # If room prediction is None, apply penalty
        if room_prediction is None:
            room_reward = self.room_penalty
        else:
            # Check if room prediction matches actual room
            # Allow for some fuzzy matching (lowercase, strip whitespace)
            pred_room = room_prediction.lower().strip()
            actual_room = actual_room.lower().strip() if actual_room else ""
            
            room_reward = self.room_reward if pred_room == actual_room else self.room_penalty
        
        # Get discounted game rewards
        game_reward = rollout.total_reward
        
        # Combine rewards
        total_reward = format_reward + room_reward + game_reward
        
        return total_reward
    
    def collect_trajectories(self, agent, env, num_episodes=10, max_steps=10):
        """
        Collect trajectories by running the agent in the environment
        
        Args:
            agent: TextWorldLLMAgent instance
            env: TextWorld environment
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            
        Returns:
            List of trajectories
        """
        from training.rollout import Rollout
        
        print(f"Collecting {num_episodes} trajectories with {self.num_samples} samples per step...")
        
        trajectories = []
        
        for episode in tqdm(range(num_episodes), desc="Episodes"):
            # Reset environment
            obs, infos = env.reset()
            
            # Reset agent
            agent.reset()
            
            # Initialize trajectory
            trajectory = {
                "episode": episode,
                "steps": []
            }
            
            # Run episode
            for step in range(max_steps):
                # Get valid actions
                valid_actions = infos["admissible_commands"]
                
                # Format prompt
                room = agent._get_room_name(obs)
                prompt = agent.format_prompt(obs, valid_actions, room)
                
                # Generate multiple outputs
                outputs = []
                for _ in range(self.num_samples):
                    completion = agent.get_completion(prompt)
                    outputs.append(completion)
                
                # Run rollouts for each output
                rewards = []
                rollouts = []
                
                for output in outputs:
                    # Create a copy of the agent for rollout
                    rollout_agent = copy.deepcopy(agent)
                    
                    # Create rollout
                    rollout = Rollout(
                        model=agent.model,
                        tokenizer=agent.tokenizer,
                        device=self.device,
                        env=env,
                        agent=rollout_agent,
                        action_history=agent.action_history.copy(),
                        completion=output
                    )
                    
                    # Run rollout
                    rollout.run(max_steps=max_steps, gamma=self.gamma)
                    
                    # Compute reward
                    reward = self.compute_rewards(trajectory, rollout)
                    rewards.append(reward)
                    rollouts.append(rollout)
                
                # Store step data
                step_data = {
                    "step": step,
                    "state": prompt,
                    "states": [prompt] * self.num_samples,  # Same prompt for all samples
                    "outputs": outputs,
                    "rewards": rewards,
                    "rollouts": rollouts
                }
                
                trajectory["steps"].append(step_data)
                
                # Choose best output for next step
                best_idx = np.argmax(rewards)
                best_output = outputs[best_idx]
                best_rollout = rollouts[best_idx]
                
                # Update environment with best action
                action = best_rollout.action
                obs, reward, done, infos = env.step(action)
                
                # Update agent state
                agent.update_state_after_action(obs, reward, done, infos)
                
                # Break if episode is done
                if done:
                    break
            
            # Add trajectory to list
            trajectories.append(trajectory)
        
        return trajectories
    
    def train(self, agent, env, num_iterations=5, num_episodes_per_iteration=5, max_steps=10, save_path=None, trajectories=None):
        """
        Train the agent using GRPO
        
        Args:
            agent: TextWorldLLMAgent instance
            env: TextWorld environment (can be None if trajectories are provided)
            num_iterations: Number of training iterations
            num_episodes_per_iteration: Number of episodes to collect per iteration
            max_steps: Maximum steps per episode
            save_path: Path to save the trained model
            trajectories: Pre-collected trajectories (if None, will collect new ones)
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Training agent using GRPO for {num_iterations} iterations...")
        
        all_metrics = {
            "loss": [],
            "ppo_loss": [],
            "kl_penalty": [],
            "rewards": []
        }
        
        # If trajectories are provided, use them directly
        if trajectories:
            print("Using pre-collected trajectories")
            
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
        
        # Otherwise, collect trajectories and train iteratively
        # Check if environment is provided
        if env is None:
            raise ValueError("Environment must be provided when no pre-collected trajectories are available")
            
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")
            
            # Collect trajectories
            trajectories = self.collect_trajectories(
                agent=agent,
                env=env,
                num_episodes=num_episodes_per_iteration,
                max_steps=max_steps
            )
            
            # Compute average reward
            avg_reward = self._compute_average_reward(trajectories)
            all_metrics["rewards"].append(avg_reward)
            print(f"Average reward: {avg_reward:.4f}")
            
            # Optimize policy
            metrics = self.optimize(agent, trajectories)
            
            # Update metrics
            all_metrics["loss"].extend(metrics["loss"])
            all_metrics["ppo_loss"].extend(metrics["ppo_loss"])
            all_metrics["kl_penalty"].extend(metrics["kl_penalty"])
            
            # Save model if path is provided
            if save_path:
                iteration_save_path = f"{save_path}_iteration_{iteration+1}.pt"
                self._save_model(agent, iteration_save_path, metrics)
                print(f"Model saved to {iteration_save_path}")
        
        # Save final model
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
        total_reward = 0
        total_steps = 0
        
        for trajectory in trajectories:
            for step_data in trajectory["steps"]:
                # Get the reward of the chosen action (best reward)
                best_reward = max(step_data["rewards"])
                total_reward += best_reward
                total_steps += 1
        
        return total_reward / total_steps if total_steps > 0 else 0
    
    def _save_model(self, agent, path, metrics=None):
        """
        Save model and optimizer state
        
        Args:
            agent: TextWorldLLMAgent instance
            path: Path to save the model
            metrics: Training metrics to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'metrics': metrics
        }, path)