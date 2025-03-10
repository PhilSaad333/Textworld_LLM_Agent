import json
import os
from google.colab import drive
import argparse
from datetime import datetime

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

def convert_data_for_grpo(data):
    """
    Convert data to the format expected by the GRPO optimizer
    
    Args:
        data: Data to convert
        
    Returns:
        List of trajectories in the format expected by the GRPO optimizer
    """
    print("Converting data to GRPO format...")
    trajectories = []
    
    # Check if data is a dictionary with 'data' and 'metadata' keys
    if isinstance(data, dict) and "data" in data and "metadata" in data:
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
                
                # Each step becomes its own episode
                episodes.append({"steps": [step]})
            
            data = episodes
            print(f"Converted to {len(episodes)} episodes (one step per episode)")
        else:
            print("Data format not recognized within 'data' key")
            return []
    
    # Check if data is a list (expected format)
    if not isinstance(data, list):
        print(f"Warning: Data is not a list, it's a {type(data)}.")
        return []
    
    # Process each episode
    for episode_idx, episode in enumerate(data):
        # Check if episode is a dictionary with 'steps'
        if not isinstance(episode, dict) or 'steps' not in episode:
            print(f"Warning: Episode {episode_idx} does not have the expected format. Skipping...")
            continue
        
        # Check if steps is a list
        if not isinstance(episode['steps'], list):
            print(f"Warning: Steps in episode {episode_idx} is not a list. Skipping...")
            continue
        
        trajectory = {
            'episode': episode_idx,
            'steps': []
        }
        
        # Process each step
        for step_idx, step in enumerate(episode['steps']):
            # Check if step is a dictionary
            if not isinstance(step, dict):
                print(f"Warning: Step {step_idx} in episode {episode_idx} is not a dictionary. Skipping...")
                continue
            
            # Get the prompt and completion
            prompt = step.get('prompt', '')
            completion = step.get('completion', '')
            
            # Get the reward (default to 0 if not present)
            reward = step.get('reward', 0.0)
            
            # Create step data
            step_data = {
                'step': step_idx,
                'state': prompt,
                'states': [prompt],  # Same prompt for all samples
                'outputs': [completion],  # Just one completion for now
                'rewards': [reward],  # Just one reward for now
            }
            
            trajectory['steps'].append(step_data)
        
        # Only add trajectories with steps
        if len(trajectory['steps']) > 0:
            trajectories.append(trajectory)
    
    print(f"Converted {len(trajectories)} episodes with {sum(len(t['steps']) for t in trajectories)} total steps")
    
    # Verify the format
    if len(trajectories) > 0:
        print("Sample trajectory format:")
        print(f"  Episode: {trajectories[0]['episode']}")
        print(f"  Number of steps: {len(trajectories[0]['steps'])}")
        if len(trajectories[0]['steps']) > 0:
            step = trajectories[0]['steps'][0]
            print(f"  Sample step:")
            print(f"    Step index: {step['step']}")
            print(f"    State: {step['state'][:50]}..." if step['state'] else "    State: (empty)")
            print(f"    Outputs: {len(step['outputs'])} completions")
            print(f"    Rewards: {step['rewards']}")
    else:
        print("Warning: No valid trajectories were created!")
    
    return trajectories

def main():
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Get data path from user
    data_path = input("Enter the path to your data file (e.g., /content/drive/MyDrive/textworld_rl_data/gameplay_data_20250307_223959.json): ")
    
    # Load data
    print(f"Loading data from {data_path}")
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Inspect data structure
        print("\nOriginal data structure:")
        inspect_data_structure(data, max_depth=3, max_items=3)
        
        # Convert data to GRPO format
        trajectories = convert_data_for_grpo(data)
        
        # Ask if user wants to save the converted data
        save_converted = input("\nDo you want to save the converted data? (y/n): ").lower() == 'y'
        
        if save_converted and trajectories:
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.dirname(data_path)
            filename = os.path.basename(data_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base_name}_converted_{timestamp}{ext}")
            
            # Confirm output path
            output_path = input(f"Enter the path to save the converted data (default: {output_path}): ") or output_path
            
            # Save converted data
            print(f"Saving converted data to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(trajectories, f)
            
            print("Converted data saved successfully")
        
        print("\nDone!")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 