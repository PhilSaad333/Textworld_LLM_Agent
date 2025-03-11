import json
import re
import os
from tqdm import tqdm

def check_format(text):
    """
    Check if the text follows the expected format with command tags.
    
    Args:
        text: Text to check
            
    Returns:
        bool: Whether the text has command tags
    """
    if text is None:
        return False
    
    # Check for command tags
    has_command_tags = '<command>' in text and '</command>' in text
    return has_command_tags

def filter_gameplay_data(input_file, output_file=None):
    """
    Filter gameplay data to keep only examples where at least one completion has correct command tags.
    
    Args:
        input_file: Path to the input gameplay data JSON file
        output_file: Path to save the filtered data (if None, will use input_file with '_filtered' suffix)
    
    Returns:
        dict: Statistics about the filtering process
    """
    # Set default output file if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filtered.json"
    
    print(f"Loading gameplay data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize statistics
    stats = {
        "total_episodes": len(data),
        "total_steps": 0,
        "steps_with_valid_format": 0,
        "steps_without_valid_format": 0,
        "filtered_episodes": 0
    }
    
    filtered_data = []
    
    # Process each episode
    for episode_idx, episode in enumerate(tqdm(data, desc="Filtering episodes")):
        filtered_episode = []
        episode_has_valid_steps = False
        
        # Process each step in the episode
        for step in episode:
            stats["total_steps"] += 1
            
            # Check if any completion has correct command tags
            has_valid_format = False
            if "completions" in step:
                for completion in step["completions"]:
                    if check_format(completion):
                        has_valid_format = True
                        break
            
            # If at least one completion has correct format, keep this step
            if has_valid_format:
                filtered_episode.append(step)
                stats["steps_with_valid_format"] += 1
                episode_has_valid_steps = True
            else:
                stats["steps_without_valid_format"] += 1
        
        # Only keep episodes that have at least one valid step
        if episode_has_valid_steps:
            filtered_data.append(filtered_episode)
            stats["filtered_episodes"] += 1
    
    # Save filtered data
    print(f"Saving filtered data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print statistics
    print("\nFiltering Statistics:")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Episodes with valid steps: {stats['filtered_episodes']} ({stats['filtered_episodes']/stats['total_episodes']*100:.2f}%)")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Steps with valid format: {stats['steps_with_valid_format']} ({stats['steps_with_valid_format']/stats['total_steps']*100:.2f}%)")
    print(f"Steps without valid format: {stats['steps_without_valid_format']} ({stats['steps_without_valid_format']/stats['total_steps']*100:.2f}%)")
    
    return stats

# Example usage in a Colab cell:
if __name__ == "__main__":
    # Set your gameplay data path
    gameplay_data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1.json'
    
    # Filter the data
    stats = filter_gameplay_data(gameplay_data_path)
    
    # The filtered data will be saved to '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1_filtered.json' 