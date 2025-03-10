"""
Script to test the custom GRPO optimizer with pre-collected gameplay data and a fine-tuned FLAN-T5-BASE model.
"""

import os
import torch
import json
import traceback
from dataclasses import dataclass, field
from config.rl_config import RLConfig
from config.config import TextWorldConfig, ModelConfig, GameConfig, RewardType, GoalType, GameType
from training.trainer import TextWorldRLTrainer
from google.colab import drive

# Function to inspect data structure (helpful for debugging)
def inspect_data_structure(data, max_depth=3, current_depth=0, max_items=3):
    """
    Recursively inspect and print the structure of a data object
    """
    indent = "  " * current_depth
    
    if current_depth >= max_depth:
        print(f"{indent}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        print(f"{indent}Dict with {len(data)} keys:")
        for i, (key, value) in enumerate(list(data.items())[:max_items]):
            print(f"{indent}- {key}: ", end="")
            if isinstance(value, (dict, list, tuple)) and current_depth < max_depth:
                print()
                inspect_data_structure(value, max_depth, current_depth + 1, max_items)
            else:
                print(f"{type(value).__name__} {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
        if len(data) > max_items:
            print(f"{indent}... ({len(data) - max_items} more keys)")
    
    elif isinstance(data, (list, tuple)):
        print(f"{indent}{type(data).__name__} with {len(data)} items:")
        for i, item in enumerate(data[:max_items]):
            print(f"{indent}[{i}]: ", end="")
            if isinstance(item, (dict, list, tuple)) and current_depth < max_depth:
                print()
                inspect_data_structure(item, max_depth, current_depth + 1, max_items)
            else:
                print(f"{type(item).__name__} {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}")
        if len(data) > max_items:
            print(f"{indent}... ({len(data) - max_items} more items)")
    
    else:
        print(f"{indent}{type(data).__name__}: {str(data)[:100]}{'...' if len(str(data)) > 100 else ''}")

# Function to load and check gameplay data
def load_and_check_data(data_path):
    print(f"Loading gameplay data from {data_path}")
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Inspect the data structure
        print("\nInspecting data structure:")
        inspect_data_structure(data)
        print()
        
        # Check data structure and convert if necessary
        if isinstance(data, dict):
            # If data is a dictionary, it might be in a different format
            print("Data is in dictionary format, checking structure...")
            
            # Check if it's the format from collect_and_save_gameplay_data
            if "data" in data and "metadata" in data:
                print("Found 'data' and 'metadata' keys - this appears to be from collect_and_save_gameplay_data")
                print("The trainer will automatically convert this format during training")
                
                # Print some statistics about the data
                data_dict = data["data"]
                if "prompt" in data_dict and "completion" in data_dict and "reward" in data_dict:
                    prompts = data_dict.get("prompt", [])
                    completions = data_dict.get("completion", [])
                    rewards = data_dict.get("reward", [])
                    
                    print(f"Number of prompts: {len(prompts)}")
                    print(f"Number of completions: {len(completions)}")
                    print(f"Number of rewards: {len(rewards)}")
                    
                    if len(prompts) > 0:
                        print(f"Sample prompt: {prompts[0][:100]}...")
                    if len(completions) > 0:
                        print(f"Sample completion: {completions[0][:100]}...")
                    if len(rewards) > 0:
                        print(f"Sample reward: {rewards[0]}")
        
        # Now check if data is a list (expected format)
        elif isinstance(data, list):
            print(f"Loaded {len(data)} episodes")
            
            # Check if episodes have the expected structure
            if len(data) > 0 and "steps" in data[0] and len(data[0]["steps"]) > 0:
                step = data[0]["steps"][0]
                print(f"First episode has {len(data[0]['steps'])} steps")
                if "prompt" in step:
                    print(f"Sample prompt: {step['prompt'][:100]}...")
                if "completion" in step:
                    print(f"Sample completion: {step['completion'][:100]}...")
                if "reward" in step:
                    print(f"Sample reward: {step['reward']}")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None

def main():
    try:
        # Mount Google Drive
        drive.mount('/content/drive')
        
        # Set paths
        data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_20250307_223959.json'
        model_path = '/content/drive/MyDrive/textworld_rl_models/Format_Fine_Tuned_weights_only.pt'
        output_dir = '/content/drive/MyDrive/textworld_rl_models/grpo_trained'
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create a custom RLConfig class that includes all parameters needed for our GRPO optimizer
        @dataclass
        class CustomRLConfig(RLConfig):
            # GRPO specific parameters (ensure these match the parameters in MyGRPOOptimizer)
            num_samples: int = 4  # G in the writeup (number of completions per prompt)
            epsilon: float = 0.2  # PPO clipping parameter
            beta: float = 0.01  # KL penalty coefficient
            
            # Reward function parameters
            format_reward: float = 0.5
            format_penalty: float = -1.0
            room_reward: float = 0.5
            room_penalty: float = -0.5
            
            # Training parameters for GRPO
            num_iterations: int = 3
            num_episodes_per_iteration: int = 5
            
            # Optimizer type
            optimizer_type: str = 'custom'  # 'custom' or 'huggingface'
            
            # Maximum lengths for prompts and completions
            max_input_length: int = 512
            max_completion_length: int = 128
        
        # Initialize RL config with custom GRPO parameters
        rl_config = CustomRLConfig(
            # Environment parameters
            max_steps=50,
            
            # Model parameters
            learning_rate=5e-5,
            batch_size=8,
            gradient_accumulation_steps=2,
            max_output_length=128,
            max_input_length=512,
            num_epochs=3,
            
            # Logging and checkpointing
            log_steps=10,
            save_steps=100,
            checkpoint_dir=output_dir,
            
            # GRPO specific parameters
            num_samples=4,  # G=4 as mentioned in your data
            epsilon=0.2,
            beta=0.01,
            
            # Reward parameters
            gamma=0.99,
            format_reward=0.5,
            format_penalty=-1.0,
            room_reward=0.5,
            room_penalty=-0.5,
            
            # Training parameters
            num_iterations=3,
            num_episodes_per_iteration=5,
            
            # Specify optimizer type
            optimizer_type='custom',
            
            # Agent parameters
            temperature=0.7,
            use_map=True,
            
            # Maximum lengths for prompts and completions
            max_input_length=512,
            max_completion_length=128
        )
        
        # Print the config to verify all parameters are set
        print("RL Config Parameters:")
        for key, value in rl_config.__dict__.items():
            print(f"  {key}: {value}")
        
        # Initialize main config
        game_config = GameConfig(
            reward_type=RewardType.SPARSE,
            goal_type=GoalType.DETAILED,
            game_type=GameType.STANDARD,
            max_history_actions=3,
            seed=42,
            output_dir="./games"
        )
        
        model_config = ModelConfig(
            model_name="google/flan-t5-base",  # Base model name
            freeze_obs_base=True,
            unfreeze_last_n_obs_layers=2
        )
        
        main_config = TextWorldConfig(
            game_config=game_config,
            model_config=model_config
        )
        
        # Add max_input_length and max_completion_length to main_config
        # This is a workaround to make the optimizer work
        main_config.max_input_length = rl_config.max_input_length
        main_config.max_completion_length = rl_config.max_completion_length
        
        # Load and check the gameplay data
        print("\nLoading and checking gameplay data...")
        gameplay_data = load_and_check_data(data_path)
        
        if gameplay_data is None:
            print("Error: Failed to load gameplay data. Exiting.")
            return
        
        # Initialize the trainer with env_manager=None since we're using pre-collected data
        print("\nInitializing TextWorldRLTrainer...")
        trainer = TextWorldRLTrainer(
            rl_config=rl_config,
            main_config=main_config,
            model_path=model_path,
            use_map=rl_config.use_map
        )
        
        # Set the gameplay data
        trainer.gameplay_data = gameplay_data
        
        # Train the model using our custom GRPO optimizer
        print("\nStarting training with custom GRPO optimizer...")
        save_model_path = os.path.join(output_dir, "grpo_trained_model")
        
        try:
            metrics = trainer._train_with_custom_grpo(save_model_path)
            
            # Print training metrics
            print("\nTraining completed!")
            print("Final metrics:")
            for key, value in metrics.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"{key}: {value[-1]}")
                else:
                    print(f"{key}: {value}")
            
            print(f"\nTrained model saved to {save_model_path}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
        
        print("Script completed successfully!")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 