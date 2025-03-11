from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import Dataset
from environment.task_env import TaskEnvManager, TaskConfig
from agents.textworld_llm_agent import TextWorldLLMAgent
import numpy as np
from config.config import GameConfig, RewardType, GoalType, GameType, get_game_config
from training.rollout import Rollout
import json
import os
from datetime import datetime
from google.colab import drive
import copy
from collections import Counter

class TextWorldRLTrainer:
    def __init__(self, rl_config, main_config=None, model_path=None, use_map=True):
        """Initialize the RL trainer
        
        Args:
            rl_config: Configuration specific to RL training
            main_config: Main TextWorld configuration (optional)
            model_path: Path to pretrained model (optional)
            use_map: Whether to use the map tool in the agent
        """
        self.config = rl_config
        self.grpo_config = rl_config  # Set grpo_config to be the same as config for consistency
        
        # Ensure max_prompt_length and max_completion_length are set
        if not hasattr(self.grpo_config, 'max_prompt_length'):
            self.grpo_config.max_prompt_length = self.grpo_config.max_input_length if hasattr(self.grpo_config, 'max_input_length') else 512
        if not hasattr(self.grpo_config, 'max_completion_length'):
            self.grpo_config.max_completion_length = self.grpo_config.max_output_length if hasattr(self.grpo_config, 'max_output_length') else 128
        
        # Create or use main config
        if main_config is None:
            # Import necessary components
            from config.config import get_game_config, RewardType, GoalType
            
            # Create a default main config
            main_config = get_game_config(
                reward_type=RewardType.DENSE,
                goal_type=GoalType.DETAILED,
                max_history_actions=3
            )
        
        self.main_config = main_config
        
        # Create environment manager using just the task_config
        self.task_config = TaskConfig(
            max_steps=rl_config.max_steps,
            scale=rl_config.scale if hasattr(rl_config, 'scale') else 10
        )
        self.env_manager = TaskEnvManager(self.task_config)
        
        self.model_name = main_config.model_config.model_name if hasattr(main_config, 'model_config') and hasattr(main_config.model_config, 'model_name') else "google/flan-t5-large"
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens to tokenizer before loading the model
        special_tokens = {
            'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        if model_path:
            print(f"Loading model from {model_path}")
            try:
                # Check if model_path is a directory or a file
                if os.path.isdir(model_path):
                    # Load from Hugging Face model directory
                    print(f"Loading model from directory: {model_path}")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                else:
                    # Load from checkpoint file
                    print(f"Loading model from checkpoint file: {model_path}")
                
                # Initialize the model with the base architecture
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Resize model embeddings to match tokenizer with special tokens
                self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Load the checkpoint
                checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
                
                # Check if it's a nested checkpoint
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    print("Detected training checkpoint format. Loading model_state_dict.")
                    model_state_dict = checkpoint["model_state_dict"]
                else:
                    print("Using checkpoint directly as model_state_dict")
                    model_state_dict = checkpoint
                
                # Load the state dict
                self.model.load_state_dict(model_state_dict)
                
                # Explicitly move model to GPU if available
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")
                self.model = self.model.to(self.device)
                
                # Create agent for evaluation
                self.agent = TextWorldLLMAgent(self.main_config, training_mode=True, use_map=use_map)
                
                # Set the model and tokenizer directly instead of having the agent load them
                self.agent.model = self.model
                self.agent.tokenizer = self.tokenizer
                self.agent.device = self.device  # Explicitly set the agent's device
                self.agent.training_mode = False  # Set back to False for normal operation
                
                print("Successfully loaded model.")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        else:
            # Load default model
            print(f"Loading default model: {self.model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Resize model embeddings to match tokenizer with special tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Explicitly move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Create agent for evaluation
        self.agent = TextWorldLLMAgent(self.main_config, training_mode=True, use_map=use_map)

        # Set the model and tokenizer directly instead of having the agent load them
        self.agent.model = self.model
        self.agent.tokenizer = self.tokenizer
        self.agent.device = self.device  # Explicitly set the agent's device
        self.agent.training_mode = False  # Set back to False for normal operation
        
        # Initialize optimizer
        self.optimizer_type = getattr(self.config, 'optimizer_type', 'custom')  # 'custom' or 'huggingface'
        
        if self.optimizer_type == 'huggingface':
            try:
                from trl import PPOConfig, PPOTrainer
                print("Using Hugging Face's PPO implementation")
                
                # Initialize Hugging Face PPO trainer
                ppo_config = PPOConfig(
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    mini_batch_size=self.config.mini_batch_size if hasattr(self.config, 'mini_batch_size') else 4,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1,
                    optimize_cuda_cache=True,
                    early_stopping=self.config.early_stopping if hasattr(self.config, 'early_stopping') else False,
                    target_kl=self.config.target_kl if hasattr(self.config, 'target_kl') else 0.1,
                    ppo_epochs=self.config.ppo_epochs if hasattr(self.config, 'ppo_epochs') else 4,
                    clip_range=self.config.clip_range if hasattr(self.config, 'clip_range') else 0.2,
                    vf_coef=self.config.vf_coef if hasattr(self.config, 'vf_coef') else 0.1,
                    horizon=self.config.horizon if hasattr(self.config, 'horizon') else 10000,
                    target=self.config.target if hasattr(self.config, 'target') else 6,
                    init_kl_coef=self.config.init_kl_coef if hasattr(self.config, 'init_kl_coef') else 0.2,
                    adap_kl_ctrl=self.config.adap_kl_ctrl if hasattr(self.config, 'adap_kl_ctrl') else True,
                )
                
                self.ppo_trainer = PPOTrainer(
                    config=ppo_config,
                    model=self.agent.model,
                    ref_model=None,  # Will be set during training
                    tokenizer=self.agent.tokenizer,
                    dataset=None,  # Will be set during training
                    data_collator=None,  # Will be set during training
                )
                
            except ImportError:
                print("Warning: trl package not found. Falling back to custom GRPO implementation.")
                self.optimizer_type = 'custom'
        
        if self.optimizer_type == 'custom':
            from training.optimizer import MyGRPOOptimizer
            print("Using custom GRPO implementation")
            
            # Initialize custom GRPO optimizer
            self.grpo_optimizer = MyGRPOOptimizer(self.config)
        
        # Initialize gameplay data
        self.gameplay_data = []
        
        print(f"Initialized TextWorldRLTrainer with {self.optimizer_type} optimizer")

    def collect_gameplay_data(self, difficulties=None, episodes_per_difficulty=5):
        """Collect gameplay data for training"""
        if difficulties is None:
            difficulties = self.config.difficulties if hasattr(self.config, 'difficulties') and self.config.difficulties else [1, 5, 10]
            
        episodes_per_difficulty = self.config.episodes_per_difficulty if hasattr(self.config, 'episodes_per_difficulty') else episodes_per_difficulty

        # Store episode data as a list of dictionaries
        all_episode_data = []
        
        # Number of completions to generate per prompt
        num_generations = self.config.num_generations if hasattr(self.config, 'num_generations') else self.config.num_samples if hasattr(self.config, 'num_samples') else 6
        
        # Ensure model is on the correct device before starting
        self.model = self.model.to(self.device)

        # Track token statistics
        total_input_tokens = 0
        total_output_tokens = 0
        max_input_tokens = 0
        max_output_tokens = 0

        # Store token counts for each completion
        completion_token_counts = []

        # Iterate over difficulties
        for difficulty in difficulties:
            print(f"Collecting data for difficulty {difficulty}")
            # Create environment for this difficulty
            env = self.env_manager.get_or_create_env(difficulty)

            for episode in range(episodes_per_difficulty):
                print(f"Episode {episode+1}/{episodes_per_difficulty}")
                obs, infos = env.reset()
                done = False
                episode_success = False
                episode_data = []
                action_history = []

                # Reset the agent and initialize known_rooms
                self.agent.reset()
                self.agent.known_rooms = set()
                
                # Explicitly parse and set the goal from the initial observation
                if self.agent.goal is None or self.agent.goal == "Not set":
                    self.agent.goal = self.agent.parse_goal(obs)
                    print(f"Set goal: {self.agent.goal}")

                while not done and len(episode_data) < self.config.max_steps:
                    # Get valid actions
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]

                    # Format prompt
                    current_room = self.agent._get_room_name(obs)
                    prompt = self.agent.format_prompt(obs, valid_actions, current_room)
                    
                    # Count input tokens
                    input_tokens = self.tokenizer.encode(prompt)
                    input_token_count = len(input_tokens)
                    total_input_tokens += input_token_count
                    max_input_tokens = max(max_input_tokens, input_token_count)

                    # Log warning if input is close to or exceeds max length
                    if hasattr(self.config, 'max_prompt_length') and input_token_count >= self.config.max_prompt_length - 10:
                        print(f"⚠️ Input length ({input_token_count} tokens) close to max ({self.config.max_prompt_length})")

                    # Generate G completions
                    completions = []
                    completion_token_counts = []
                    
                    # Prepare inputs for generation
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    # Generate G completions
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_completion_length if hasattr(self.config, 'max_completion_length') else 128,
                            min_length=20,
                            num_return_sequences=num_generations,
                            num_beams=num_generations,  # Use beam search to get diverse completions
                            do_sample=True,  # Set to True to use temperature
                            temperature=0.7,
                            early_stopping=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    
                    # Process completions
                    for i in range(num_generations):
                        completion = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=False)
                        completion = completion.replace("<pad>", "").strip()
                        
                        # Ensure completion ends with EOS token
                        if not completion.endswith(self.tokenizer.eos_token):
                            completion += self.tokenizer.eos_token
                        
                        completions.append(completion)
                        print(f"Completion {i+1}: {completion}")
                        
                        # Count output tokens
                        output_tokens = self.tokenizer.encode(completion)
                        output_token_count = len(output_tokens)
                        completion_token_counts.append(output_token_count)
                        total_output_tokens += output_token_count
                        max_output_tokens = max(max_output_tokens, output_token_count)
                        
                        # Log warning if output is close to max length
                        if hasattr(self.config, 'max_completion_length') and output_token_count >= self.config.max_completion_length - 10:
                            print(f"⚠️ Output length ({output_token_count} tokens) close to max ({self.config.max_completion_length})")
                    
                    # Do rollouts for each completion to get rewards
                    rollout_rewards = []
                    
                    for completion_idx, completion in enumerate(completions):
                        # Create a rollout for this completion
                        rollout = Rollout(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=self.device,
                            env=env,
                            agent=self.agent,
                            action_history=action_history,
                            completion=completion
                        )
                        
                        # Run the rollout
                        max_rollout_steps = self.config.max_steps - len(action_history) if hasattr(self.config, 'max_steps') else 10
                        
                        # Run the rollout - no special case handling needed
                        rollout.run(max_steps=max_rollout_steps, gamma=self.config.gamma if hasattr(self.config, 'gamma') else 0.99)
                        
                        # Compute total reward including format and room prediction bonuses
                        total_reward = rollout.compute_total_reward(self.config)
                        rollout_rewards.append(total_reward)
                    
                    # Store step data
                    step_data = {
                        "prompt": prompt,
                        "completions": completions,
                        "rewards": rollout_rewards,
                        "completion_token_counts": completion_token_counts
                    }
                    
                    episode_data.append(step_data)
                    
                    # Check format and extract action directly without creating temporary rollouts
                    valid_command_indices = []
                    for i, completion in enumerate(completions):
                        format_check = self.agent.check_format(completion)
                        if format_check["has_command_tags"]:
                            valid_command_indices.append(i)
                    
                    # Strategy 1: If some completions have valid command tags, only sample from those
                    if valid_command_indices:
                        # Choose the completion with the highest reward among those with valid command tags
                        valid_rewards = [rollout_rewards[i] for i in valid_command_indices]
                        best_valid_idx = valid_command_indices[valid_rewards.index(max(valid_rewards))]
                        chosen_completion = completions[best_valid_idx]
                        print(f"Choosing from {len(valid_command_indices)} completions with valid command tags")
                    else:
                        # Strategy 2: If no completions have valid command tags, choose the one with highest reward
                        best_completion_idx = rollout_rewards.index(max(rollout_rewards))
                        chosen_completion = completions[best_completion_idx]
                        print("No completions with valid command tags, choosing based on reward")
                    
                    # Extract action directly without creating another rollout
                    action_info = self.agent.extract_action_from_completion(chosen_completion, valid_actions)
                    action = action_info.get('action', None)
                    
                    # If action is invalid, choose a random valid action
                    if action is None or action not in valid_actions:
                        action = np.random.choice(valid_actions)
                        print(f"Invalid action, using random action: {action}")
                    
                    # Reset environment to current state
                    obs, infos = env.reset()
                    
                    # Replay action history to get to current state
                    for past_action in action_history:
                        obs, _, _, infos = env.step(past_action)
                    
                    # Take the chosen action
                    next_obs, reward, done, next_infos = env.step(action)
                    
                    # Add action to history
                    action_history.append(action)
                    
                    # Check if episode was completed successfully
                    if done:
                        episode_success = (reward > 0)
                    
                    # Update for next step
                    obs, infos = next_obs, next_infos
                    self.agent.update_state_after_action(obs, reward, done, next_infos)

                    # Explicitly update known_rooms with the current room
                    current_room = self.agent._get_room_name(next_obs)
                    if current_room:
                        self.agent.known_rooms.add(current_room)
                
                # Add episode data to all episodes
                all_episode_data.append({
                    "steps": episode_data,
                    "success": episode_success,
                    "difficulty": difficulty,
                    "num_steps": len(episode_data)
                })
        
        # Prepare dataset for training
        dataset = {
            "prompt": [],
            "completion": [],
            "reward": [],
            "completion_token_count": []  # Add this new key to store token counts
        }
        
        # Flatten episode data into prompt-completion-reward triples
        for episode in all_episode_data:
            for step in episode["steps"]:
                prompt = step["prompt"]
                for i, completion in enumerate(step["completions"]):
                    dataset["prompt"].append(prompt)
                    dataset["completion"].append(completion)
                    dataset["reward"].append(step["rewards"][i])
                    
                    # Add token count for this completion
                    if "completion_token_counts" in step and i < len(step["completion_token_counts"]):
                        dataset["completion_token_count"].append(step["completion_token_counts"][i])
                    else:
                        # If token count not available, estimate it
                        token_count = len(self.tokenizer.encode(completion))
                        dataset["completion_token_count"].append(token_count)
        
        # Print statistics
        print(f"\nCollected {len(all_episode_data)} episodes with {len(dataset['prompt'])} prompt-completion pairs")
        print(f"Average input tokens: {total_input_tokens / len(dataset['prompt']):.1f}")
        print(f"Average output tokens: {total_output_tokens / len(dataset['completion']):.1f}")
        print(f"Max input tokens: {max_input_tokens}")
        print(f"Max output tokens: {max_output_tokens}")
        
        # Store episode data for later use
        self.episode_data = all_episode_data
        
        return dataset
    
    
    def reward_function(self, completions, reward=None, **kwargs):
        """Custom reward function for GRPO
        
        Args:
            completions: List of model outputs
            reward: Pre-computed rewards from rollouts (passed from dataset)
            **kwargs: Additional arguments passed from the dataset
        
        Returns:
            List of rewards for each completion
        """
        # Track token counts during training (keep this part for monitoring)
        if hasattr(self, 'token_tracking_step'):
            self.token_tracking_step += 1
        else:
            self.token_tracking_step = 0
            self.token_tracking_data = {
                'steps': [],
                'avg_output_tokens': [],
                'max_output_tokens': []
            }
        
        # Only log every 10 steps to avoid too much output
        should_log = (self.token_tracking_step % 10 == 0)
        
        # Calculate token statistics for this batch
        output_token_counts = [len(self.tokenizer.encode(completion)) for completion in completions]
        avg_output_tokens = sum(output_token_counts) / len(output_token_counts)
        max_output_tokens = max(output_token_counts)
        
        # Store for tracking
        if should_log:
            self.token_tracking_data['steps'].append(self.token_tracking_step)
            self.token_tracking_data['avg_output_tokens'].append(avg_output_tokens)
            self.token_tracking_data['max_output_tokens'].append(max_output_tokens)
            print(f"\n[Step {self.token_tracking_step}] Output tokens - Avg: {avg_output_tokens:.1f}, Max: {max_output_tokens}")
        
        # Check for potential truncation
        truncation_count = sum(1 for count in output_token_counts if count >= self.config.max_output_length * 0.9)
        if truncation_count > 0 and should_log:
            print(f"⚠️ {truncation_count}/{len(completions)} completions may be truncated (>= {self.config.max_output_length * 0.9} tokens)")
        
        # If we have pre-computed rewards, use them directly
        if reward is not None:
            return reward
        
        # FALLBACK ONLY: This code only runs if pre-computed rewards are missing
        # This should match the logic in Rollout.compute_total_reward
        print("WARNING: Using fallback reward calculation. Pre-computed rewards not found.")
        rewards = []
        for completion in completions:
            # Check format of the current completion
            format_check_result = self.agent.check_format(completion)
            
            # Apply format penalty ONLY for command and room tags
            format_reward = 0.0  # Start with no penalty
            
            # Check command tags
            if not format_check_result["has_command_tags"]:
                format_reward += self.config.format_penalty / 2  # Half penalty for missing command tags
            
            # Check room tags
            if not format_check_result["has_room_tags"]:
                format_reward += self.config.format_penalty / 2  # Half penalty for missing room tags
            
            rewards.append(format_reward)
        
        return rewards
    
    def collect_and_save_gameplay_data(self, difficulties=None, episodes_per_difficulty=5, save_path=None):
        """Collect gameplay data and save it to a JSON file
        
        Args:
            difficulties: List of difficulty levels to collect data from
            episodes_per_difficulty: Number of episodes to collect per difficulty
            save_path: Path to save the JSON file (default: Google Drive)
            
        Returns:
            Path to the saved JSON file
        """
        # Collect gameplay data
        dataset = self.collect_gameplay_data(difficulties, episodes_per_difficulty)
        
        # Convert dataset to a serializable format
        data_dict = {
            "data": {
                "prompt": dataset["prompt"],
                "completion": dataset["completion"],
                "reward": dataset["reward"],
                "completion_token_count": dataset["completion_token_count"]
            },
            "metadata": {
                "difficulties": difficulties,
                "episodes_per_difficulty": episodes_per_difficulty,
                "model": self.model.config.name_or_path,
                "timestamp": datetime.now().isoformat(),
                "config": {k: v for k, v in vars(self.config).items() if not callable(v) and not k.startswith('__')}
            }
        }
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine save path
        if save_path is None:
            # Mount Google Drive if not already mounted
            try:
                drive_mounted = os.path.exists('/content/drive')
                if not drive_mounted:
                    drive.mount('/content/drive')
                
                # Create directory if it doesn't exist
                save_dir = '/content/drive/MyDrive/textworld_rl_data'
                os.makedirs(save_dir, exist_ok=True)
                
                # Create filename with timestamp
                save_path = f"{save_dir}/gameplay_data_{timestamp}.json"
            except Exception as e:
                print(f"Error mounting Google Drive: {e}")
                # Fallback to local storage
                save_path = f"/content/gameplay_data_{timestamp}.json"
        
        # Save data to JSON file
        with open(save_path, 'w') as f:
            json.dump(data_dict, f)
        
        print(f"Gameplay data saved to {save_path}")
        
        # Store the path for later use
        self.last_saved_data_path = save_path
        
        return save_path
    
    
    def load_gameplay_data(self, file_path=None):
        """Load gameplay data from a JSON file
        
        Args:
            file_path: Path to the JSON file (if None, uses the last saved path)
            
        Returns:
            Dataset object with the loaded data
        """
        # Use last saved path if no path provided
        if file_path is None:
            if hasattr(self, 'last_saved_data_path'):
                file_path = self.last_saved_data_path
            else:
                raise ValueError("No file path provided and no previous save found")
        
        print(f"Loading gameplay data from {file_path}")
        
        # Load data from JSON file
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        
        # Extract data
        if "data" in data_dict:
            # New format with data and metadata
            dataset_dict = data_dict["data"]
        else:
            # Old format with direct data
            dataset_dict = data_dict
        
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Print dataset statistics
        print(f"Loaded {len(dataset)} examples")
        if "reward" in dataset.features:
            rewards = dataset["reward"]
            print(f"Reward min: {min(rewards)}, max: {max(rewards)}, avg: {sum(rewards)/len(rewards):.4f}")
        
        return dataset
    
    def train(self, use_saved_data=False, data_path=None, save_model_path=None, save_each_epoch=False):
        """
        Train the agent using RL
        
        Args:
            use_saved_data: Whether to use saved gameplay data
            data_path: Path to saved gameplay data
            save_model_path: Path to save the trained model
            save_each_epoch: If True, save model after each epoch
            
        Returns:
            Dictionary with training metrics
        """
        print("Starting RL training...")
        
        # Load or collect gameplay data
        if use_saved_data and data_path:
            print(f"Loading gameplay data from {data_path}")
            self.gameplay_data = self.load_gameplay_data(data_path)
        else:
            print("Collecting gameplay data...")
            difficulties = self.config.difficulties if hasattr(self.config, 'difficulties') else [1, 2, 3]
            episodes_per_difficulty = self.config.episodes_per_difficulty if hasattr(self.config, 'episodes_per_difficulty') else 5
            self.gameplay_data = self.collect_gameplay_data(difficulties, episodes_per_difficulty)
        
        print(f"Collected {len(self.gameplay_data)} gameplay episodes")
        
        # Train using the appropriate optimizer
        if self.optimizer_type == 'huggingface':
            return self._train_with_huggingface(save_model_path)
        else:
            return self._train_with_custom_grpo(save_model_path, save_each_epoch)
    
    def _train_with_huggingface(self, save_model_path=None):
        """
        Train the agent using Hugging Face's PPO implementation
        
        Args:
            save_model_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        from trl import PPOTrainer
        
        print("Training with Hugging Face's PPO implementation...")
        
        # Convert gameplay data to the format expected by Hugging Face's PPO trainer
        ppo_dataset = self._convert_data_for_huggingface()
        
        # Set the dataset for the PPO trainer
        self.ppo_trainer.dataset = ppo_dataset
        
        # Train the model
        metrics = self.ppo_trainer.train()
        
        # Save the model if a path is provided
        if save_model_path:
            self.ppo_trainer.save_model(save_model_path)
            print(f"Model saved to {save_model_path}")
        
        return metrics
    
    def _train_with_custom_grpo(self, save_model_path=None, save_each_epoch=False):
        """
        Train the agent using our custom GRPO implementation with pre-collected data
        
        Args:
            save_model_path: Path to save the trained model
            save_each_epoch: If True, save model after each epoch with format "{save_model_path}_epoch_{epoch}.pt"
            
        Returns:
            Dictionary with training metrics
        """
        print("Training with custom GRPO implementation...")
        
        # Convert gameplay data to trajectories format expected by our GRPO optimizer
        steps_data = self._convert_data_for_custom_grpo()
        
        if not steps_data:
            raise ValueError("No valid samples found. Please check your gameplay data.")
        
        # Print training parameters
        print(f"Training with pre-collected data:")
        print(f"  {len(steps_data)} total steps in data")
        print(f"  save_path: {save_model_path}")
        print(f"  save_each_epoch: {save_each_epoch}")
        
        # Train using our custom GRPO optimizer with pre-collected trajectories
        metrics = self.grpo_optimizer.train(
            agent=self.agent,
            steps_data=steps_data,  # Pass pre-collected trajectories
            save_path=save_model_path,
            save_each_epoch=save_each_epoch
        )
        
        return metrics
    
    def _convert_data_for_huggingface(self):
        """
        Convert gameplay data to the format expected by Hugging Face's PPO trainer
        
        Returns:
            Dataset in the format expected by Hugging Face's PPO trainer
        """
        # This is a placeholder - you'll need to implement the actual conversion
        # based on the specific requirements of Hugging Face's PPO trainer
        
        # Example structure (modify as needed):
        ppo_dataset = []
        
        for episode in self.gameplay_data:
            for step in episode['steps']:
                ppo_dataset.append({
                    'query': step['prompt'],
                    'response': step['completion'],
                    'reward': step['reward']
                })
        
        return ppo_dataset
    
    def _convert_data_for_custom_grpo(self):
        """
        Convert gameplay data to the format expected by our custom GRPO optimizer.
        Ensures that G completions for each step are kept together.
        
        Returns:
            List of step dictionaries, each containing:
            {
                "state": prompt,
                "outputs": [completion1, completion2, ..., completionG],
                "rewards": [reward1, reward2, ..., rewardG]
            }
        """
        print("Converting gameplay data to GRPO format...")
        steps_data = []  # Will hold all steps without episode structure
        
        # Handle different input formats
        if hasattr(self.gameplay_data, 'to_dict') and callable(getattr(self.gameplay_data, 'to_dict')):
            # Convert Dataset to dictionary
            print("Converting Dataset to dictionary...")
            data_dict = self.gameplay_data.to_dict()
        elif isinstance(self.gameplay_data, dict) and "data" in self.gameplay_data:
            # Already in the right format
            data_dict = self.gameplay_data["data"]
        else:
            print("Unsupported data format")
            return []
        
        # Verify we have the required fields
        required_fields = ["prompt", "completion", "reward"]
        if not all(field in data_dict for field in required_fields):
            print(f"Missing required fields. Found: {list(data_dict.keys())}")
            return []
        
        # Get the lists
        prompts = data_dict["prompt"]
        completions = data_dict["completion"]
        rewards = data_dict["reward"]
        
        # Get G from config
        G = self.config.num_generations if hasattr(self.config, 'num_generations') else \
            self.config.num_samples if hasattr(self.config, 'num_samples') else 6
        print(f"Using {G} completions per step from config")
        
        # Calculate number of steps
        total_examples = len(prompts)
        if total_examples % G != 0:
            print(f"Warning: Total examples ({total_examples}) is not divisible by G ({G})")
            # Truncate to nearest multiple of G
            total_examples = (total_examples // G) * G
            prompts = prompts[:total_examples]
            completions = completions[:total_examples]
            rewards = rewards[:total_examples]
        
        num_steps = total_examples // G
        print(f"Total examples: {total_examples}")
        print(f"Number of steps: {num_steps}")
        
        # Process each step (group of G completions)
        for step_idx in range(num_steps):
            start_idx = step_idx * G
            end_idx = start_idx + G
            
            # Get prompt (should be the same for all G completions)
            prompt = prompts[start_idx]
            
            # Verify all prompts in this group are the same
            group_prompts = prompts[start_idx:end_idx]
            if not all(p == prompt for p in group_prompts):
                print(f"Warning: Not all prompts in step {step_idx} are the same. This indicates a data alignment issue.")
                print("First prompt:", prompt[:100], "...")
                different_prompts = [(i, p[:100]) for i, p in enumerate(group_prompts) if p != prompt]
                print(f"Different prompts found at indices: {different_prompts}")
                continue
            
            # Get all G completions and rewards for this step
            step_completions = completions[start_idx:end_idx]
            step_rewards = rewards[start_idx:end_idx]
            
            # Verify we have exactly G completions and rewards
            if len(step_completions) != G or len(step_rewards) != G:
                print(f"Warning: Step {step_idx} has incorrect number of completions/rewards. Expected {G}, got {len(step_completions)}/{len(step_rewards)}")
                continue
            
            # Create step data with all G completions
            step_data = {
                "state": prompt,
                "outputs": step_completions,
                "rewards": step_rewards
            }
            
            steps_data.append(step_data)
        
        # Print debug information
        print(f"\nConverted data structure:")
        print(f"Number of steps: {len(steps_data)}")
        
        if steps_data:
            first_step = steps_data[0]
            print(f"\nFirst step details:")
            print(f"Number of completions: {len(first_step['outputs'])}")
            print(f"Number of rewards: {len(first_step['rewards'])}")
            print(f"First completion: {first_step['outputs'][0][:100]}...")
            print(f"Rewards: {first_step['rewards']}")
            
            # Verify all steps have G completions
            completion_counts = [len(step["outputs"]) for step in steps_data]
            if min(completion_counts) != max(completion_counts):
                print(f"Warning: Not all steps have the same number of completions!")
                print(f"Min: {min(completion_counts)}, Max: {max(completion_counts)}")
            elif min(completion_counts) != G:
                print(f"Warning: Steps have {min(completion_counts)} completions, expected {G}!")
        else:
            print("Warning: No valid steps found after filtering!")
        
        return steps_data