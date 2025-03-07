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
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Add special tokens to tokenizer before loading the model
        special_tokens = {
            'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Check if model_path is a .pt file or a directory
        if model_path and model_path.endswith('.pt'):
            # Load from .pt file
            print(f"Loading model from PyTorch checkpoint: {model_path}")
            try:
                # Try loading with weights_only=True to avoid security warnings
                checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
                
                # Initialize the model with the base architecture
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                
                # Resize model embeddings to match tokenizer with special tokens
                self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Load the state dict directly
                self.model.load_state_dict(checkpoint)
                print("Successfully loaded model weights.")
            except Exception as e:
                print(f"Error loading checkpoint with weights_only=True: {e}")
                print("Trying with default settings...")
                
                # Try loading with default settings as fallback
                checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # Initialize the model with the base architecture
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                
                # Resize model embeddings to match tokenizer with special tokens
                self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Check the structure of the checkpoint
                if "model_state_dict" in checkpoint:
                    # This is a training checkpoint with multiple components
                    print("Detected training checkpoint format. Loading model_state_dict.")
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # This is just a model state dict
                    print("Detected model state dict format.")
                    self.model.load_state_dict(checkpoint)
        else:
            # Load from Hugging Face model directory or hub
            print(f"Loading model from directory or hub: {model_path or 'google/flan-t5-base'}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path if model_path else "google/flan-t5-base"
            )
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
        
        # Configure GRPO
        self.grpo_config = GRPOConfig(
            learning_rate=rl_config.learning_rate,
            per_device_train_batch_size=rl_config.batch_size,
            gradient_accumulation_steps=rl_config.gradient_accumulation_steps if hasattr(rl_config, 'gradient_accumulation_steps') else 4,
            max_completion_length=rl_config.max_output_length if hasattr(rl_config, 'max_output_length') else 128,
            max_prompt_length=rl_config.max_input_length if hasattr(rl_config, 'max_input_length') else 512,
            num_train_epochs=rl_config.num_epochs,
            logging_steps=rl_config.log_steps if hasattr(rl_config, 'log_steps') else 10,
            save_steps=rl_config.save_steps if hasattr(rl_config, 'save_steps') else 100,
            output_dir=rl_config.checkpoint_dir,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            seed=42,
            # GRPO specific parameters
            beta=rl_config.beta if hasattr(rl_config, 'beta') else 0.1,  # KL penalty coefficient
            use_vllm=rl_config.use_vllm if hasattr(rl_config, 'use_vllm') else False,
            temperature=rl_config.temperature if hasattr(rl_config, 'temperature') else 0.7,
        )

    def collect_gameplay_data(self, difficulties=None, episodes_per_difficulty=5):
        """Collect gameplay data for training"""
        if difficulties is None:
            difficulties = self.config.difficulties if hasattr(self.config, 'difficulties') and self.config.difficulties else [1, 5, 10]
            
        episodes_per_difficulty = self.config.episodes_per_difficulty if hasattr(self.config, 'episodes_per_difficulty') else episodes_per_difficulty

        # Store episode data as a list of dictionaries
        # Each dictionary has keys "prompt" "completions" and "rewards", where the values for "completions" and "rewards are a list of strings
        all_episode_data = []
        
        # Number of completions to generate per prompt
        num_generations = self.config.num_generations if hasattr(self.config, 'num_generations') else 4
        
        # Ensure model is on the correct device before starting
        self.model = self.model.to(self.device)

        # Track token statistics
        total_input_tokens = 0
        total_output_tokens = 0
        max_input_tokens = 0
        max_output_tokens = 0

        # Import the Rollout class
        from training.rollout import Rollout

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
                    if input_token_count >= self.grpo_config.max_prompt_length - 10:
                        print(f"⚠️ Input length ({input_token_count} tokens) close to max ({self.grpo_config.max_prompt_length})")

                    # Generate G completions
                    completions = []
                    completion_token_counts = []
                    
                    # Prepare inputs for generation
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    # Generate G completions
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.grpo_config.max_completion_length,
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
                        completions.append(completion)
                        print(f"Completion {i+1}: {completion}")
                        
                        # Count output tokens
                        output_tokens = self.tokenizer.encode(completion)
                        output_token_count = len(output_tokens)
                        completion_token_counts.append(output_token_count)
                        total_output_tokens += output_token_count
                        max_output_tokens = max(max_output_tokens, output_token_count)
                        
                        # Log warning if output is close to max length
                        if output_token_count >= self.grpo_config.max_completion_length - 10:
                            print(f"⚠️ Output length ({output_token_count} tokens) close to max ({self.grpo_config.max_completion_length})")
                    
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
                    
                    # Choose action to take in the environment
                    # Option 1: Choose the completion with the highest reward
                    best_completion_idx = rollout_rewards.index(max(rollout_rewards))
                    chosen_completion = completions[best_completion_idx]
                    
                    # Create a rollout for the chosen completion to get the action
                    chosen_rollout = Rollout(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        env=env,
                        agent=self.agent,
                        action_history=action_history,
                        completion=chosen_completion
                    )
                    
                    # Extract the action (without running the rollout)
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]
                    action_info = chosen_rollout.extract_action_from_completion(chosen_completion, valid_actions)
                    action = action_info.get('action', None)
                    
                    # If action is invalid, choose a random valid action
                    if action is None or action not in valid_actions:
                        action = np.random.choice(valid_actions)
                    
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
                    "difficulty": difficulty
                })
        
        # Print token statistics
        print("\n===== Token Count Statistics =====")
        print(f"Total episodes: {len(all_episode_data)}")
        print(f"Input tokens - Total: {total_input_tokens}, Max: {max_input_tokens}")
        print(f"Output tokens - Total: {total_output_tokens}, Max: {max_output_tokens}")
        print("=================================\n")
        
        # Convert to dataset format expected by GRPO
        # We need to flatten the episode data into a list of prompts, completions, and rewards
        all_prompts = []
        all_completions = []
        all_rewards = []
        all_completion_token_counts = []
        
        for episode in all_episode_data:
            for step in episode["steps"]:
                # For each step, add G entries to the dataset (one for each completion)
                for i in range(len(step["completions"])):
                    all_prompts.append(step["prompt"])
                    all_completions.append(step["completions"][i])
                    all_rewards.append(step["rewards"][i])
                    all_completion_token_counts.append(step["completion_token_counts"][i])
        
        # Create dataset
        dataset_dict = {
            "prompt": all_prompts,
            "completion": all_completions,
            "reward": all_rewards,
            "completion_token_count": all_completion_token_counts
        }
        
        # Store the raw episode data for potential future use
        self.episode_data = all_episode_data
        
        # After collecting data
        avg_completion_tokens = sum(sum(step["completion_token_counts"]) / len(step["completion_token_counts"]) 
                                   for step in episode_data) / len(episode_data)
        max_completion_tokens = max(max(step["completion_token_counts"]) for step in episode_data)

        print(f"Average completion tokens: {avg_completion_tokens:.1f}")
        print(f"Maximum completion tokens: {max_completion_tokens}")

        # Check for potential truncation
        if max_completion_tokens > self.config.max_output_length * 0.9:
            print(f"WARNING: Some completions are approaching the maximum length ({max_completion_tokens}/{self.config.max_output_length})")
        
        return Dataset.from_dict(dataset_dict)
    
    
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
                format_reward += self.config.format_failure_penalty / 2  # Half penalty for missing command tags
            
            # Check room tags
            if not format_check_result["has_room_tags"]:
                format_reward += self.config.format_failure_penalty / 2  # Half penalty for missing room tags
            
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
    
    def save_enhanced_gameplay_data(self, data_path=None):
        """Save enhanced gameplay data with additional statistics
        
        Args:
            data_path: Path to the original gameplay data (if None, uses the last saved path)
            
        Returns:
            Path to the enhanced gameplay data
        """
        # Use last saved path if no path provided
        if data_path is None:
            if hasattr(self, 'last_saved_data_path'):
                data_path = self.last_saved_data_path
            else:
                raise ValueError("No data path provided and no previous save found")
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the enhanced data path
        save_dir = os.path.dirname(data_path)
        filename = os.path.basename(data_path)
        enhanced_data_path = f"{save_dir}/enhanced_{filename.replace('.json', '')}_{timestamp}.json"
        
        # Load the original data
        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        
        # Create enhanced metadata
        enhanced_metadata = {
            "enhancement_timestamp": timestamp,
            "original_data_path": data_path,
            "model_source": self.model.config.name_or_path,
            "format_failure_penalty": self.config.format_failure_penalty if hasattr(self.config, 'format_failure_penalty') else None,
            "room_prediction_penalty": self.config.room_prediction_penalty if hasattr(self.config, 'room_prediction_penalty') else None,
            "description": "Enhanced gameplay data with additional statistics"
        }
        
        # If we have episode data available, add statistics
        if hasattr(self, 'episode_data'):
            # Calculate format error rate
            format_errors = 0
            total_completions = 0
            
            for episode in self.episode_data:
                for step in episode["steps"]:
                    for completion in step["completions"]:
                        total_completions += 1
                        format_check = self.agent.check_format(completion)
                        if not (format_check["has_command_tags"] and format_check["has_room_tags"]):
                            format_errors += 1
            
            format_error_rate = format_errors / total_completions if total_completions > 0 else 0
            enhanced_metadata["format_error_rate"] = format_error_rate
            enhanced_metadata["total_completions"] = total_completions
            enhanced_metadata["format_errors"] = format_errors
            
            # Add episode success rate
            successful_episodes = sum(1 for episode in self.episode_data if episode["success"])
            total_episodes = len(self.episode_data)
            success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
            enhanced_metadata["episode_success_rate"] = success_rate
            enhanced_metadata["successful_episodes"] = successful_episodes
            enhanced_metadata["total_episodes"] = total_episodes
            
            # Add difficulty breakdown
            difficulty_stats = {}
            for difficulty in set(episode["difficulty"] for episode in self.episode_data):
                episodes_at_difficulty = [ep for ep in self.episode_data if ep["difficulty"] == difficulty]
                successful_at_difficulty = sum(1 for ep in episodes_at_difficulty if ep["success"])
                success_rate_at_difficulty = successful_at_difficulty / len(episodes_at_difficulty) if episodes_at_difficulty else 0
                
                difficulty_stats[str(difficulty)] = {
                    "episodes": len(episodes_at_difficulty),
                    "successful": successful_at_difficulty,
                    "success_rate": success_rate_at_difficulty
                }
            
            enhanced_metadata["difficulty_stats"] = difficulty_stats
        
        # Add enhanced metadata to the data dictionary
        data_dict["enhanced_metadata"] = enhanced_metadata
        
        # Save the enhanced data
        with open(enhanced_data_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"Enhanced gameplay data saved to {enhanced_data_path}")
        print(f"Format error rate: {enhanced_metadata.get('format_error_rate', 'N/A') * 100:.1f}%")
        print(f"Episode success rate: {enhanced_metadata.get('episode_success_rate', 'N/A') * 100:.1f}%")
        
        return enhanced_data_path
    
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
        
        # Load data from JSON file
        with open(file_path, 'r') as f:
            data_dict = json.load(f)

    def train(self, use_saved_data=False, data_path=None, save_model_path=None):
        """Train the model using GRPO
        
        Args:
            use_saved_data: Whether to use saved gameplay data instead of collecting new data
            data_path: Path to the saved gameplay data (if None, uses the last saved path)
            save_model_path: Path to save the model after training (default: checkpoint_dir/rl_model_timestamp)
        """
        # Get training data
        if use_saved_data:
            train_dataset = self.load_gameplay_data(data_path)
        else:
            train_dataset = self.collect_gameplay_data()
        
        # Import our custom trainer
        from training.custom_grpo_trainer import CustomGRPOTrainer
        
        # Initialize our custom GRPO trainer
        trainer = CustomGRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_function,
            args=self.grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Plot token trends
        self.plot_token_trends()
        
        # Save the model
        self.save_model(save_model_path)