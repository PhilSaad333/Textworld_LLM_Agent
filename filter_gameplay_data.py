import json
import os
import re
from agents.textworld_llm_agent import TextWorldLLMAgent
from config.config import get_game_config, RewardType, GoalType

def check_format_command_tags(text):
    """
    Check if text has valid command tags in the correct order with no nested tags.
    The text should contain exactly one pair of command tags with any text between them,
    but no additional command tags.
    """
    if text is None:
        return False
    
    # Find all occurrences of command tags and their content
    command_pattern = r'<command>(.*?)</command>'
    matches = re.findall(command_pattern, text, re.DOTALL)
    
    # Check if we have exactly one command tag pair
    if len(matches) != 1:
        return False
    
    # Check that the content between tags doesn't contain other command tags
    command_content = matches[0]
    if '<command>' in command_content or '</command>' in command_content:
        return False
    
    return True

def filter_gameplay_data(input_path, output_path):
    """
    Filter gameplay data to keep only steps where at least one completion has valid command tags.
    
    Args:
        input_path: Path to unfiltered gameplay data JSON
        output_path: Path to save filtered gameplay data JSON
    """
    print(f"Loading data from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'data' not in data:
        raise ValueError("Invalid data format: expected dict with 'data' key")
    
    # Get the original data lists
    prompts = data['data']['prompt']
    completions = data['data']['completion']
    rewards = data['data']['reward']
    completion_token_counts = data['data'].get('completion_token_counts', [])
    
    # Get number of completions per step (G)
    total_examples = len(prompts)
    if total_examples == 0:
        raise ValueError("No examples found in data")
    
    # Determine G (num_samples) by finding consecutive identical prompts
    G = 1
    first_prompt = prompts[0]
    while G < len(prompts) and prompts[G] == first_prompt:
        G += 1
    
    print(f"Detected {G} completions per step")
    print(f"Total examples: {total_examples}")
    num_steps = total_examples // G
    print(f"Number of steps: {num_steps}")
    
    # Lists to store filtered data
    filtered_prompts = []
    filtered_completions = []
    filtered_rewards = []
    filtered_token_counts = []
    
    # Process each step
    steps_kept = 0
    steps_filtered = 0
    
    for step_idx in range(num_steps):
        start_idx = step_idx * G
        end_idx = start_idx + G
        
        # Get completions for this step
        step_completions = completions[start_idx:end_idx]
        
        # Check if any completion has valid command tags
        has_valid_completion = any(check_format_command_tags(completion) for completion in step_completions)
        
        if has_valid_completion:
            # Keep this step's data
            steps_kept += 1
            # Add all G completions and corresponding data
            for i in range(G):
                idx = start_idx + i
                filtered_prompts.append(prompts[idx])
                filtered_completions.append(completions[idx])
                filtered_rewards.append(rewards[idx])
                if completion_token_counts:
                    filtered_token_counts.append(completion_token_counts[idx])
        else:
            steps_filtered += 1
    
    print(f"\nFiltering results:")
    print(f"Steps kept: {steps_kept}")
    print(f"Steps filtered: {steps_filtered}")
    print(f"Total steps processed: {steps_kept + steps_filtered}")
    
    # Create filtered data dictionary
    filtered_data = {
        'data': {
            'prompt': filtered_prompts,
            'completion': filtered_completions,
            'reward': filtered_rewards,
        },
        'metadata': {
            'original_file': input_path,
            'filtering_stats': {
                'steps_kept': steps_kept,
                'steps_filtered': steps_filtered,
                'completions_per_step': G,
                'total_completions_kept': len(filtered_completions)
            }
        }
    }
    
    # Add completion token counts if they existed in original data
    if completion_token_counts:
        filtered_data['data']['completion_token_counts'] = filtered_token_counts
    
    # Preserve any other metadata from original file
    if 'metadata' in data:
        filtered_data['metadata'].update({
            k: v for k, v in data['metadata'].items() 
            if k not in filtered_data['metadata']
        })
    
    # Save filtered data
    print(f"\nSaving filtered data to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    print("Done!")
    return filtered_data

if __name__ == "__main__":
    # Paths
    input_path = "/content/drive/MyDrive/textworld_rl_data/gameplay_data_1.json"
    output_path = "/content/drive/MyDrive/textworld_rl_data/gameplay_data_1_filtered_fixed.json"
    
    # Filter data
    filtered_data = filter_gameplay_data(input_path, output_path) 