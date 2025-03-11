import json
import re
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def check_format(text, require_command_tags=True, require_room_tags=False):
    """
    Check if the text follows the expected format with command and/or room tags.
    
    Args:
        text: Text to check
        require_command_tags: Whether to require command tags
        require_room_tags: Whether to require room tags
            
    Returns:
        dict: Format check results
    """
    if text is None:
        return {
            "format_correct": False,
            "has_command_tags": False,
            "has_room_tags": False
        }
    
    # Check for command and room tags
    has_command_tags = '<command>' in text and '</command>' in text
    has_room_tags = '<room>' in text and '</room>' in text
    
    # Determine if format is correct based on requirements
    format_correct = True
    if require_command_tags and not has_command_tags:
        format_correct = False
    if require_room_tags and not has_room_tags:
        format_correct = False
    
    return {
        "format_correct": format_correct,
        "has_command_tags": has_command_tags,
        "has_room_tags": has_room_tags
    }

def extract_command(text):
    """Extract command from text with command tags"""
    if text is None:
        return None
    
    command_match = re.search(r'<command>(.*?)</command>', text, re.DOTALL)
    if command_match:
        return command_match.group(1).strip()
    return None

def filter_gameplay_data(input_file, output_file=None, require_command_tags=True, 
                         require_room_tags=False, min_valid_completions=1,
                         analyze_only=False, visualize=False):
    """
    Filter gameplay data to keep only examples where at least N completions have correct format.
    
    Args:
        input_file: Path to the input gameplay data JSON file
        output_file: Path to save the filtered data (if None, will use input_file with '_filtered' suffix)
        require_command_tags: Whether to require command tags
        require_room_tags: Whether to require room tags
        min_valid_completions: Minimum number of valid completions required to keep a step
        analyze_only: If True, only analyze the data without saving filtered version
        visualize: If True, generate visualizations of the data
    
    Returns:
        dict: Statistics about the filtering process
    """
    # Set default output file if not provided
    if output_file is None and not analyze_only:
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
        "filtered_episodes": 0,
        "valid_completions_per_step": [],
        "valid_commands": Counter(),
        "completion_counts": []
    }
    
    filtered_data = []
    
    # Process each episode
    for episode_idx, episode in enumerate(tqdm(data, desc="Processing episodes")):
        filtered_episode = []
        episode_has_valid_steps = False
        
        # Process each step in the episode
        for step in episode:
            stats["total_steps"] += 1
            
            # Check if the step has completions
            if "completions" not in step:
                stats["steps_without_valid_format"] += 1
                continue
                
            # Count valid completions
            valid_completions = 0
            valid_commands = []
            
            for completion in step["completions"]:
                format_check = check_format(
                    completion, 
                    require_command_tags=require_command_tags,
                    require_room_tags=require_room_tags
                )
                
                if format_check["format_correct"]:
                    valid_completions += 1
                    if format_check["has_command_tags"]:
                        command = extract_command(completion)
                        if command:
                            valid_commands.append(command)
                            stats["valid_commands"][command] += 1
            
            # Track completion statistics
            stats["valid_completions_per_step"].append(valid_completions)
            stats["completion_counts"].append(len(step["completions"]))
            
            # If enough valid completions, keep this step
            if valid_completions >= min_valid_completions:
                if not analyze_only:
                    filtered_episode.append(step)
                stats["steps_with_valid_format"] += 1
                episode_has_valid_steps = True
            else:
                stats["steps_without_valid_format"] += 1
        
        # Only keep episodes that have at least one valid step
        if episode_has_valid_steps and not analyze_only:
            filtered_data.append(filtered_episode)
            stats["filtered_episodes"] += 1
    
    # Save filtered data if not in analyze-only mode
    if not analyze_only and output_file:
        print(f"Saving filtered data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
    
    # Print statistics
    print("\nFiltering Statistics:")
    print(f"Total episodes: {stats['total_episodes']}")
    if not analyze_only:
        print(f"Episodes with valid steps: {stats['filtered_episodes']} ({stats['filtered_episodes']/stats['total_episodes']*100:.2f}%)")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Steps with valid format: {stats['steps_with_valid_format']} ({stats['steps_with_valid_format']/stats['total_steps']*100:.2f}%)")
    print(f"Steps without valid format: {stats['steps_without_valid_format']} ({stats['steps_without_valid_format']/stats['total_steps']*100:.2f}%)")
    
    # Calculate average valid completions per step
    if stats["valid_completions_per_step"]:
        avg_valid = sum(stats["valid_completions_per_step"]) / len(stats["valid_completions_per_step"])
        print(f"Average valid completions per step: {avg_valid:.2f}")
    
    # Print top commands
    print("\nTop 10 valid commands:")
    for command, count in stats["valid_commands"].most_common(10):
        print(f"  {command}: {count}")
    
    # Generate visualizations if requested
    if visualize:
        # Create a directory for visualizations
        viz_dir = os.path.join(os.path.dirname(input_file), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot distribution of valid completions per step
        plt.figure(figsize=(10, 6))
        plt.hist(stats["valid_completions_per_step"], bins=range(max(stats["completion_counts"])+2), alpha=0.7)
        plt.title("Distribution of Valid Completions per Step")
        plt.xlabel("Number of Valid Completions")
        plt.ylabel("Number of Steps")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, "valid_completions_distribution.png"))
        
        # Plot top commands
        top_commands = stats["valid_commands"].most_common(15)
        if top_commands:
            plt.figure(figsize=(12, 8))
            commands, counts = zip(*top_commands)
            plt.barh(commands, counts)
            plt.title("Top 15 Valid Commands")
            plt.xlabel("Count")
            plt.ylabel("Command")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "top_commands.png"))
        
        print(f"\nVisualizations saved to {viz_dir}")
    
    return stats

# For use in a Colab cell
def filter_gameplay_data_colab(gameplay_data_path, 
                              require_command_tags=True,
                              require_room_tags=False,
                              min_valid_completions=1,
                              analyze_only=False,
                              visualize=True):
    """
    Wrapper function for use in Colab cells
    """
    return filter_gameplay_data(
        input_file=gameplay_data_path,
        require_command_tags=require_command_tags,
        require_room_tags=require_room_tags,
        min_valid_completions=min_valid_completions,
        analyze_only=analyze_only,
        visualize=visualize
    )

# Example usage in a Colab cell:
if __name__ == "__main__":
    # Set your gameplay data path
    gameplay_data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1.json'
    
    # Filter the data - only keep steps with at least 1 completion that has command tags
    stats = filter_gameplay_data_colab(
        gameplay_data_path,
        require_command_tags=True,
        require_room_tags=False,
        min_valid_completions=1,
        visualize=True
    )
    
    # The filtered data will be saved to '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1_filtered.json' 