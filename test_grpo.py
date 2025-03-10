"""
Test script for running GRPO training with pre-collected gameplay data and a fine-tuned model.
"""

import os
import torch
import json
from config.rl_config import RLConfig
from config.config import TextWorldConfig, ModelConfig, GameConfig, RewardType, GoalType, GameType
from training.trainer import TextWorldRLTrainer

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set paths
data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_20250307_223959.json'
model_path = '/content/drive/MyDrive/textworld_rl_models/Format_Fine_Tuned_weights_only.pt'
output_dir = '/content/drive/MyDrive/textworld_rl_models/grpo_trained'
os.makedirs(output_dir, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize RL config
rl_config = RLConfig(
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
    
    # Optimizer selection
    optimizer_type='custom',
    
    # GRPO specific parameters
    num_samples=8,
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
    
    # Agent parameters
    temperature=0.7,
    use_map=True
)

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

# Utility function to inspect data structure
def inspect_data_structure(data, max_depth=3, current_depth=0, max_items=3):
    """
    Recursively inspect and print the structure of a data object
    
    Args:
        data: The data object to inspect
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth (used internally)
        max_items: Maximum number of items to show at each level
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
                data_dict = data["data"]
                
                # Check if data_dict contains parallel lists (prompt, completion, reward)
                if (isinstance(data_dict, dict) and 
                    "prompt" in data_dict and 
                    "completion" in data_dict and 
                    "reward" in data_dict):
                    
                    print("Converting parallel lists to episodes format...")
                    
                    # Extract lists
                    prompts = data_dict.get("prompt", [])
                    completions = data_dict.get("completion", [])
                    rewards = data_dict.get("reward", [])
                    completion_token_counts = data_dict.get("completion_token_count", [])
                    
                    # Ensure all lists have the same length
                    min_length = min(len(prompts), len(completions), len(rewards))
                    if min_length < len(prompts) or min_length < len(completions) or min_length < len(rewards):
                        print(f"Warning: Lists have different lengths. Truncating to {min_length} items.")
                    
                    # Create episodes (each step is its own episode for simplicity)
                    episodes = []
                    for i in range(min_length):
                        step = {
                            "prompt": prompts[i],
                            "completion": completions[i],
                            "reward": rewards[i]
                        }
                        
                        # Add completion token count if available
                        if i < len(completion_token_counts):
                            step["completion_token_count"] = completion_token_counts[i]
                        
                        # Each step becomes its own episode
                        episodes.append({"steps": [step]})
                    
                    data = episodes
                    print(f"Converted to {len(episodes)} episodes (one step per episode)")
                else:
                    print("Data format not recognized within 'data' key")
            elif "data" in data:
                print("Found 'data' key, extracting dataset...")
                data = data["data"]
            
            # If it's still a dictionary, try to convert to list of episodes
            if isinstance(data, dict):
                print("Converting dictionary to list of episodes...")
                episodes = []
                
                # Try to extract episodes from the dictionary
                for key, value in data.items():
                    if isinstance(value, dict) and "steps" in value:
                        episodes.append(value)
                    elif isinstance(value, list):
                        # If value is a list, check if it contains steps
                        for item in value:
                            if isinstance(item, dict) and "steps" in item:
                                episodes.append(item)
                
                if episodes:
                    data = episodes
                    print(f"Converted to {len(episodes)} episodes")
                else:
                    print("Could not convert dictionary to episodes format")
        
        # Now check if data is a list (expected format)
        if isinstance(data, list):
            print(f"Loaded {len(data)} episodes")
            
            # Check if episodes have the expected structure
            valid_episodes = []
            for episode in data:
                if isinstance(episode, dict) and "steps" in episode:
                    # Valid episode format
                    valid_episodes.append(episode)
                elif isinstance(episode, dict):
                    # Try to create a steps list from the episode data
                    steps = []
                    for key, value in episode.items():
                        if isinstance(value, dict) and "prompt" in value and "completion" in value:
                            steps.append(value)
                    
                    if steps:
                        valid_episodes.append({"steps": steps})
            
            if valid_episodes:
                data = valid_episodes
                print(f"Found {len(valid_episodes)} valid episodes")
                
                # Print sample data
                if len(data) > 0 and "steps" in data[0] and len(data[0]["steps"]) > 0:
                    step = data[0]["steps"][0]
                    print(f"First episode has {len(data[0]['steps'])} steps")
                    if "prompt" in step:
                        print(f"Sample prompt: {step['prompt'][:100]}...")
                    if "completion" in step:
                        print(f"Sample completion: {step['completion'][:100]}...")
                    if "reward" in step:
                        print(f"Sample reward: {step['reward']}")
            else:
                print("No valid episodes found in the data")
                return []
        else:
            print("Data format is not as expected (not a list or convertible dictionary)")
            return []
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return []

# Main execution
def main():
    try:
        # Initialize the trainer
        print("Initializing TextWorldRLTrainer...")
        trainer = TextWorldRLTrainer(
            rl_config=rl_config,
            main_config=main_config,
            model_path=model_path,
            use_map=rl_config.use_map
        )
        
        # Load and check the gameplay data
        print("\nLoading gameplay data...")
        gameplay_data = load_and_check_data(data_path)
        if not gameplay_data:
            print("No valid gameplay data found. Exiting.")
            return
        
        print(f"Successfully loaded {len(gameplay_data)} episodes")
        trainer.gameplay_data = gameplay_data
        
        # Train the model using our custom GRPO optimizer
        print("\nStarting training with custom GRPO optimizer...")
        save_model_path = os.path.join(output_dir, "grpo_trained_model")
        
        # Try to train the model
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
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            print("\nTraining failed. Please check the error message above.")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 