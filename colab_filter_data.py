import json
import re
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

# Set your gameplay data path
gameplay_data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1.json'
output_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1_filtered.json'

def check_format(text):
    """Check if text has command tags"""
    if text is None:
        return False
    return '<command>' in text and '</command>' in text

def extract_command(text):
    """Extract command from text with command tags"""
    if text is None or not check_format(text):
        return None
    command_match = re.search(r'<command>(.*?)</command>', text, re.DOTALL)
    if command_match:
        return command_match.group(1).strip()
    return None

# Load the data
print(f"Loading gameplay data from {gameplay_data_path}...")
with open(gameplay_data_path, 'r') as f:
    data = json.load(f)

# Initialize statistics
stats = {
    "total_episodes": len(data),
    "total_steps": 0,
    "steps_with_valid_format": 0,
    "steps_without_valid_format": 0,
    "filtered_episodes": 0,
    "valid_completions_per_step": [],
    "valid_commands": Counter()
}

filtered_data = []

# Process each episode
for episode in tqdm(data, desc="Filtering episodes"):
    filtered_episode = []
    episode_has_valid_steps = False
    
    # Process each step in the episode
    for step in episode:
        stats["total_steps"] += 1
        
        # Check if any completion has correct command tags
        valid_completions = 0
        if "completions" in step:
            for completion in step["completions"]:
                if check_format(completion):
                    valid_completions += 1
                    command = extract_command(completion)
                    if command:
                        stats["valid_commands"][command] += 1
        
        stats["valid_completions_per_step"].append(valid_completions)
        
        # If at least one completion has correct format, keep this step
        if valid_completions >= 1:
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
print(f"Saving filtered data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

# Print statistics
print("\nFiltering Statistics:")
print(f"Total episodes: {stats['total_episodes']}")
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

# Plot distribution of valid completions per step
plt.figure(figsize=(10, 6))
plt.hist(stats["valid_completions_per_step"], bins=range(8), alpha=0.7)
plt.title("Distribution of Valid Completions per Step")
plt.xlabel("Number of Valid Completions")
plt.ylabel("Number of Steps")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nFiltered data saved to {output_path}")
print(f"Original data had {stats['total_steps']} steps, filtered data has {stats['steps_with_valid_format']} steps")
print(f"Retention rate: {stats['steps_with_valid_format']/stats['total_steps']*100:.2f}%") 