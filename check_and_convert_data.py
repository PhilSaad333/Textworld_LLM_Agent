import json
import os
import argparse
from datetime import datetime
from google.colab import drive
import numpy as np

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

def convert_parallel_lists_to_episodes(data_dict, group_by_episode=True):
    """
    Convert parallel lists format to episodes format
    
    Args:
        data_dict: Dictionary with parallel lists (prompt, completion, reward)
        group_by_episode: Whether to group steps into episodes (default: True)
        
    Returns:
        List of episode dictionaries
    """
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
    
    # Create steps
    steps = []
    for i in range(min_length):
        step = {
            "prompt": prompts[i],
            "completion": completions[i],
            "reward": rewards[i]
        }
        
        # Add completion token count if available
        if i < len(completion_token_counts):
            step["completion_token_count"] = completion_token_counts[i]
        
        steps.append(step)
    
    if not group_by_episode:
        # Each step is its own episode
        episodes = [{"steps": [step]} for step in steps]
        print(f"Created {len(episodes)} episodes (one step per episode)")
    else:
        # Try to group steps into episodes based on prompt content
        # This is a heuristic approach - it assumes that prompts from the same episode
        # have similar content or follow a pattern
        
        # For simplicity, let's create episodes with a fixed number of steps
        # You might want to implement a more sophisticated grouping logic
        steps_per_episode = 5  # Adjust as needed
        
        episodes = []
        for i in range(0, len(steps), steps_per_episode):
            episode_steps = steps[i:i+steps_per_episode]
            episodes.append({"steps": episode_steps})
        
        print(f"Created {len(episodes)} episodes with up to {steps_per_episode} steps each")
    
    return episodes

def check_and_convert_data(data_path, save_converted=False, output_path=None):
    """
    Check the format of the data and convert it if necessary
    
    Args:
        data_path: Path to the data file
        save_converted: Whether to save the converted data
        output_path: Path to save the converted data (if None, generates a path)
        
    Returns:
        Tuple of (original_data, converted_data)
    """
    print(f"Loading data from {data_path}")
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print("\nOriginal data structure:")
        inspect_data_structure(data, max_depth=3, max_items=3)
        
        # Check if data is in the format expected by GRPO optimizer
        is_episodes_format = False
        if isinstance(data, list):
            # Check if it's a list of episodes
            if len(data) > 0 and isinstance(data[0], dict) and "steps" in data[0]:
                is_episodes_format = True
                print("\nData is already in the expected format (list of episodes)")
        
        # If not in episodes format, try to convert
        converted_data = data
        if not is_episodes_format:
            print("\nData is not in the expected format. Attempting to convert...")
            
            # Check if it's a dictionary with 'data' key
            if isinstance(data, dict) and "data" in data:
                print("Found 'data' key, extracting dataset...")
                data_dict = data["data"]
                
                # Check if data_dict contains parallel lists
                if isinstance(data_dict, dict) and "prompt" in data_dict and "completion" in data_dict:
                    print("Found parallel lists format, converting to episodes...")
                    converted_data = convert_parallel_lists_to_episodes(data_dict)
                else:
                    print("Could not identify the format of the data")
            else:
                print("Could not identify the format of the data")
        
        # Print statistics about the converted data
        if isinstance(converted_data, list):
            print(f"\nConverted data has {len(converted_data)} episodes")
            
            total_steps = sum(len(episode.get("steps", [])) for episode in converted_data)
            print(f"Total steps: {total_steps}")
            
            if len(converted_data) > 0 and "steps" in converted_data[0] and len(converted_data[0]["steps"]) > 0:
                print(f"First episode has {len(converted_data[0]['steps'])} steps")
                
                # Print sample step
                step = converted_data[0]["steps"][0]
                print("\nSample step:")
                for key, value in step.items():
                    if isinstance(value, str):
                        print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
        
        # Save converted data if requested
        if save_converted and isinstance(converted_data, list):
            if output_path is None:
                # Generate output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.dirname(data_path)
                filename = os.path.basename(data_path)
                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base_name}_converted_{timestamp}{ext}")
            
            print(f"\nSaving converted data to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(converted_data, f)
            
            print(f"Converted data saved successfully")
        
        return data, converted_data
    
    except Exception as e:
        print(f"Error checking/converting data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Check and convert TextWorld gameplay data format")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--save_converted", action="store_true", help="Save the converted data")
    parser.add_argument("--output_path", type=str, help="Path to save the converted data")
    
    args = parser.parse_args()
    
    # Mount Google Drive if path is in Google Drive
    if args.data_path.startswith("/content/drive"):
        try:
            drive.mount('/content/drive')
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
    
    # Check and convert data
    original_data, converted_data = check_and_convert_data(
        args.data_path, 
        save_converted=args.save_converted,
        output_path=args.output_path
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main() 