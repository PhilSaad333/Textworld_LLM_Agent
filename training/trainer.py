from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import Dataset
from environment.task_env import TaskEnvManager, TaskConfig
from agents.textworld_llm_agent import TextWorldLLMAgent
import numpy as np
from config.config import GameConfig, RewardType, GoalType, GameType, get_game_config

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
        self.agent = TextWorldLLMAgent(self.main_config, use_map=use_map)
        
        # Set the model and tokenizer directly instead of having the agent load them
        self.agent.model = self.model
        self.agent.tokenizer = self.tokenizer
        self.agent.device = self.device  # Explicitly set the agent's device
        
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
            
        all_prompts = []
        all_completions = []
        all_rewards = []
        all_episode_completions = []  # Track if the episode was completed successfully
        all_format_checks = []  # Track format check results
        
        # Use the class device attribute instead of checking model parameters
        print(f"Using device for data collection: {self.device}")
        
        # Ensure model is on the correct device before starting
        self.model = self.model.to(self.device)
        
        for difficulty in difficulties:
            print(f"Collecting data for difficulty {difficulty}")
            env = self.env_manager.get_or_create_env(difficulty)
            
            for episode in range(episodes_per_difficulty):
                print(f"Episode {episode+1}/{episodes_per_difficulty}")
                obs, infos = env.reset()
                done = False
                episode_prompts = []
                episode_completions = []
                episode_rewards = []
                episode_format_checks = []
                episode_success = False  # Track if this episode was completed successfully
                
                while not done and len(episode_prompts) < self.config.max_steps:
                    # Get valid actions
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]
                    
                    # Format prompt
                    current_room = self.agent._get_room_name(obs)
                    prompt = self.agent.format_prompt(obs, valid_actions, current_room)
                    
                    # Get action from agent
                    action, action_info = self.agent.get_action(
                        env, obs, infos, valid_actions, len(episode_prompts)
                    )
                    
                    # Get model's raw completion - ensure everything is on the same device
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            min_length=20,
                            num_beams=1,  # Use greedy decoding for data collection
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    
                    completion = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
                    completion = completion.replace("<pad>", "").strip()
                    
                    # Take action in environment
                    next_obs, reward, done, next_infos = env.step(action)
                    
                    # Check if episode was completed successfully
                    if done and reward > 0:
                        episode_success = True
                    
                    # Store format check result but DON'T modify the reward here
                    # We'll handle format rewards in the reward_function
                    format_check = self.agent.true_state.get('format_check_passed', True)
                    episode_format_checks.append(format_check)
                    
                    # Store data
                    episode_prompts.append(prompt)
                    episode_completions.append(completion)
                    episode_rewards.append(reward)  # Store the raw environment reward
                    
                    # Update for next step
                    obs, infos = next_obs, next_infos
                    self.agent.update_state_after_action(obs, reward, done, next_infos)
                
                # Process episode rewards (compute returns)
                returns = self._compute_returns(episode_rewards, gamma=0.99)
                
                # Add to overall dataset
                all_prompts.extend(episode_prompts)
                all_completions.extend(episode_completions)
                all_rewards.extend(returns)
                all_format_checks.extend(episode_format_checks)
                
                # Add episode completion status for each step in this episode
                all_episode_completions.extend([episode_success] * len(episode_prompts))
        
        # Create dataset
        dataset_dict = {
            "prompt": all_prompts,
            "completion": all_completions,
            "reward": all_rewards,
            "episode_completion": all_episode_completions,
            "format_check": all_format_checks
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def _compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns
    
    def reward_function(self, completions, ground_truth=None, prompts=None, format_checks=None, **kwargs):
        """Custom reward function for GRPO
        
        Args:
            completions: List of model outputs
            ground_truth: List of episode completion statuses (True if episode was completed successfully)
            prompts: List of input prompts (optional)
            format_checks: List of format check results from data collection
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            # Check format of the current completion
            format_check_result = self.agent.check_format(completion)
            
            # Base reward for format adherence - use values from config
            if format_check_result["has_command_tags"] and format_check_result["has_room_tags"]:
                format_reward = self.config.format_success_reward
            else:
                format_reward = self.config.format_failure_penalty
                
            # Episode completion reward (if ground truth is provided)
            episode_reward = 0.0
            if ground_truth is not None and i < len(ground_truth):
                episode_reward = self.config.episode_completion_reward if ground_truth[i] else 0.0
                
            # Combine rewards
            total_reward = format_reward + episode_reward
            
            rewards.append(total_reward)
            
        return rewards
    
    def train(self):
        """Train the model using GRPO"""
        # Collect gameplay data
        train_dataset = self.collect_gameplay_data()
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.grpo_config,
            train_dataset=train_dataset,
            reward_funcs=self.reward_function,
            # Pass episode completion status as ground truth
            ground_truth_key="episode_completion",
            # Pass format check results
            format_checks_key="format_check"
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(self.config.checkpoint_dir + "/final_model")
        self.tokenizer.save_pretrained(self.config.checkpoint_dir + "/final_model")
        
    def evaluate(self, difficulties=None, episodes_per_difficulty=3):
        """Evaluate the trained model"""
        if difficulties is None:
            difficulties = [1, 5, 10, 15]
            
        results = {}
        
        # Load the latest model
        self.agent.model = self.model
        self.agent.tokenizer = self.tokenizer
        
        for difficulty in difficulties:
            success_count = 0
            total_reward = 0
            total_steps = 0
            
            for _ in range(episodes_per_difficulty):
                env = self.env_manager.get_or_create_env(difficulty)
                obs, infos = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                while not done and steps < self.config.max_steps:
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]
                    
                    action, _ = self.agent.get_action(env, obs, infos, valid_actions, steps)
                    next_obs, reward, done, next_infos = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    
                    obs, infos = next_obs, next_infos
                    self.agent.update_state_after_action(obs, reward, done, next_infos)
                
                total_reward += episode_reward
                total_steps += steps
                if episode_reward > 0:
                    success_count += 1
            
            results[difficulty] = {
                "success_rate": success_count / episodes_per_difficulty,
                "avg_reward": total_reward / episodes_per_difficulty,
                "avg_steps": total_steps / episodes_per_difficulty
            }
            
        return results