import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class DummyConfig:
    num_samples: int = 6  # G in the paper
    learning_rate: float = 1e-6
    batch_size: int = 3
    max_input_length: int = 512
    max_completion_length: int = 128

def load_gameplay_data(file_path: str) -> Dict[str, Any]:
    """Load gameplay data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_converted_data(converted_data: List[Dict], config: DummyConfig):
    """Validate the structure of converted data"""
    print("\nValidating converted data structure...")
    
    if not converted_data:
        print("❌ ERROR: No data was converted!")
        return False
    
    print(f"\nTotal examples after conversion: {len(converted_data)}")
    
    # Check structure of each example
    for i, example in enumerate(converted_data[:5]):  # Look at first 5 examples
        print(f"\nExample {i}:")
        
        # Check required keys
        required_keys = {"state", "outputs", "advantages", "old_logprobs"}
        actual_keys = set(example.keys())
        if not required_keys.issubset(actual_keys):
            print(f"❌ ERROR: Missing keys in example {i}")
            print(f"Expected: {required_keys}")
            print(f"Got: {actual_keys}")
            return False
        
        # Check number of completions
        num_completions = len(example["outputs"])
        print(f"Number of completions: {num_completions}")
        if num_completions != config.num_samples:
            print(f"❌ ERROR: Wrong number of completions in example {i}")
            print(f"Expected {config.num_samples}, got {num_completions}")
            return False
        
        # Check advantages match completions
        num_advantages = len(example["advantages"])
        print(f"Number of advantages: {num_advantages}")
        if num_advantages != num_completions:
            print(f"❌ ERROR: Mismatch between completions and advantages in example {i}")
            print(f"Completions: {num_completions}, Advantages: {num_advantages}")
            return False
        
        # Print first completion and its advantage
        print(f"First completion: {example['outputs'][0][:100]}...")
        print(f"First advantage: {example['advantages'][0]}")
        
        # Verify advantages are normalized
        advantages = np.array(example["advantages"])
        print(f"Advantages mean: {advantages.mean():.6f}")  # Should be close to 0
        print(f"Advantages std: {np.std(advantages):.6f}")  # Should be close to 1 if there's variance
    
    print("\n✅ All validation checks passed!")
    return True

def main():
    # Load the gameplay data
    file_path = "/content/drive/MyDrive/textworld_rl_data/gameplay_data_1.json"  # Update this path
    print(f"Loading data from {file_path}")
    data = load_gameplay_data(file_path)
    
    # Create config and optimizer
    config = DummyConfig()
    from training.optimizer import MyGRPOOptimizer
    optimizer = MyGRPOOptimizer(config)
    
    # Print original data structure
    print("\nOriginal data structure:")
    if isinstance(data, dict) and 'data' in data:
        data_content = data['data']
        print(f"Number of prompts: {len(data_content['prompt'])}")
        print(f"Number of completions: {len(data_content['completion'])}")
        print(f"Number of rewards: {len(data_content['reward'])}")
        
        # Print a few examples of the original data
        print("\nFirst few examples from original data:")
        for i in range(min(3, len(data_content['prompt']))):
            print(f"\nExample {i}:")
            print(f"Prompt: {data_content['prompt'][i][:100]}...")
            print(f"Completion: {data_content['completion'][i][:100]}...")
            print(f"Reward: {data_content['reward'][i]}")
    
    # Convert the data
    print("\nConverting data...")
    converted_data = optimizer._convert_data_for_custom_grpo(data)
    
    # Validate the converted data
    validate_converted_data(converted_data, config)

if __name__ == "__main__":
    main() 