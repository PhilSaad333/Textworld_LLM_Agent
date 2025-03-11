import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.textworld_llm_agent import TextWorldLLMAgent
from training.optimizer import MyGRPOOptimizer

@dataclass
class DummyConfig:
    num_samples: int = 6  # G in the paper
    learning_rate: float = 1e-6
    batch_size: int = 3
    max_input_length: int = 512
    max_completion_length: int = 128
    beta: float = 0.1  # KL penalty coefficient
    epsilon: float = 0.05  # PPO clipping parameter
    max_grad_norm: float = 0.5  # Gradient clipping
    model_name: str = "google/flan-t5-base"

def load_gameplay_data(file_path: str) -> Dict[str, Any]:
    """Load gameplay data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def debug_logprobs(model, tokenizer, device, prompt, completion):
    """Debug log probabilities calculation for a single prompt-completion pair"""
    print("\n=== Debug Log Probabilities ===")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Completion: {completion[:100]}...")
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Tokenize output
    output_tokens = tokenizer(
        completion,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # Get input for this example
    input_ids = inputs.input_ids[0:1]
    attention_mask = inputs.attention_mask[0:1] if hasattr(inputs, 'attention_mask') else None
    
    # Get output for this example
    output_ids = output_tokens.input_ids[0]
    
    # Print token IDs
    print(f"Input token IDs: {input_ids[0][:20]}...")
    print(f"Output token IDs: {output_ids[:20]}...")
    
    # Prepare model inputs
    model_inputs = {
        "input_ids": input_ids,
    }
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    
    # For encoder-decoder models, we need to provide decoder inputs
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(model.config, "pad_token_id", 0)
    
    # Create decoder_input_ids by shifting output_ids right and prepending decoder_start_token_id
    decoder_input_ids = torch.cat([
        torch.tensor([[decoder_start_token_id]], device=device),
        output_ids[:-1].unsqueeze(0)
    ], dim=1)
    
    model_inputs["decoder_input_ids"] = decoder_input_ids
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**model_inputs)
        
    # Get logits
    logits = outputs.logits
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Extract log probabilities for the actual output tokens
    token_log_probs = []
    
    # For encoder-decoder models, we compare each position in the output with the next token
    for i in range(len(output_ids) - 1):  # -1 because we don't need the last token's prediction
        if output_ids[i+1] == model.config.pad_token_id:
            continue  # Skip pad tokens
        token_log_prob = log_probs[0, i, output_ids[i+1]].item()
        token_log_probs.append(token_log_prob)
        print(f"Token {i+1}: {tokenizer.decode([output_ids[i+1]])} - Log prob: {token_log_prob:.4f}")
    
    # Combine token log probabilities by averaging over tokens
    if token_log_probs:
        avg_log_prob = sum(token_log_probs) / len(token_log_probs)
        print(f"Average log probability: {avg_log_prob:.4f}")
        print(f"Perplexity: {np.exp(-avg_log_prob):.4f}")
        return avg_log_prob
    else:
        print("No valid tokens found!")
        return 0.0

def debug_ppo_loss(old_logprobs, new_logprobs, advantages, epsilon=0.05):
    """Debug PPO loss calculation"""
    print("\n=== Debug PPO Loss ===")
    
    # Convert to tensors if they're not already
    if not isinstance(old_logprobs, torch.Tensor):
        old_logprobs = torch.tensor(old_logprobs)
    if not isinstance(new_logprobs, torch.Tensor):
        new_logprobs = torch.tensor(new_logprobs)
    if not isinstance(advantages, torch.Tensor):
        advantages = torch.tensor(advantages)
    
    # Compute ratio between new and old policies
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    
    # Compute losses
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    
    # Take the minimum (pessimistic bound)
    ppo_values = torch.min(surrogate1, surrogate2)
    
    # Average over completions
    ppo_loss = -ppo_values.mean()
    
    # Print detailed information
    print(f"Old log probs: {old_logprobs}")
    print(f"New log probs: {new_logprobs}")
    print(f"Advantages: {advantages}")
    print(f"Ratios: {ratio}")
    print(f"Clipped ratios: {clipped_ratio}")
    print(f"Surrogate1: {surrogate1}")
    print(f"Surrogate2: {surrogate2}")
    print(f"PPO values: {ppo_values}")
    print(f"PPO loss: {ppo_loss}")
    
    return ppo_loss.item()

def debug_kl_penalty(old_logprobs, new_logprobs):
    """Debug KL penalty calculation"""
    print("\n=== Debug KL Penalty ===")
    
    # Convert to tensors if they're not already
    if not isinstance(old_logprobs, torch.Tensor):
        old_logprobs = torch.tensor(old_logprobs)
    if not isinstance(new_logprobs, torch.Tensor):
        new_logprobs = torch.tensor(new_logprobs)
    
    # Compute delta of log probabilities
    delta_lp = old_logprobs - new_logprobs
    
    # Compute KL divergence: exp(delta_lp) - delta_lp - 1
    kl_div = torch.exp(delta_lp) - delta_lp - 1.0
    
    # Average over completions
    kl_penalty = kl_div.mean()
    
    # Print detailed information
    print(f"Old log probs: {old_logprobs}")
    print(f"New log probs: {new_logprobs}")
    print(f"Delta log probs: {delta_lp}")
    print(f"KL divergence: {kl_div}")
    print(f"KL penalty: {kl_penalty}")
    
    return kl_penalty.item()

def debug_batch_processing(model, tokenizer, device, batch, epsilon=0.05, beta=0.1):
    """Debug batch processing"""
    print("\n=== Debug Batch Processing ===")
    print(f"Batch size: {len(batch)} prompts")
    
    batch_loss = 0.0
    batch_ppo_loss = 0.0
    batch_kl_penalty = 0.0
    batch_ratios = []
    
    # Process each prompt in the batch
    for prompt_idx, prompt_data in enumerate(batch):
        print(f"\n--- Prompt {prompt_idx} ---")
        state = prompt_data["state"]
        outputs = prompt_data["outputs"]
        advantages = prompt_data["advantages"]
        
        print(f"State: {state[:100]}...")
        print(f"Number of completions: {len(outputs)}")
        print(f"Number of advantages: {len(advantages)}")
        
        # Compute old log probabilities
        old_logprobs_list = []
        for i, output in enumerate(outputs):
            print(f"\nCompletion {i}:")
            old_logprob = debug_logprobs(model, tokenizer, device, state, output)
            old_logprobs_list.append(old_logprob)
        
        # Simulate computing new log probabilities (slightly different)
        # In a real scenario, these would be computed after model updates
        new_logprobs_list = []
        for i, old_logprob in enumerate(old_logprobs_list):
            # Simulate a small change in log probabilities
            # This is just for testing - in reality, these would come from the updated model
            noise = np.random.normal(0, 0.1)  # Small random noise
            new_logprob = old_logprob + noise
            new_logprobs_list.append(new_logprob)
            print(f"Completion {i}: Old logprob={old_logprob:.4f}, New logprob={new_logprob:.4f}, Ratio={np.exp(new_logprob-old_logprob):.4f}")
        
        # Convert to tensors
        old_logprobs = torch.tensor(old_logprobs_list)
        new_logprobs = torch.tensor(new_logprobs_list)
        advantages_tensor = torch.tensor(advantages)
        
        # Compute ratios
        ratios = torch.exp(new_logprobs - old_logprobs)
        batch_ratios.extend(ratios.numpy().tolist())
        
        # Debug PPO loss
        prompt_ppo_loss = debug_ppo_loss(old_logprobs, new_logprobs, advantages_tensor, epsilon)
        
        # Debug KL penalty
        prompt_kl_penalty = debug_kl_penalty(old_logprobs, new_logprobs)
        
        # Compute total loss
        prompt_loss = prompt_ppo_loss + beta * prompt_kl_penalty
        
        # Accumulate losses
        batch_loss += prompt_loss
        batch_ppo_loss += prompt_ppo_loss
        batch_kl_penalty += prompt_kl_penalty
    
    # Compute batch statistics
    avg_batch_loss = batch_loss / len(batch)
    avg_batch_ppo_loss = batch_ppo_loss / len(batch)
    avg_batch_kl_penalty = batch_kl_penalty / len(batch)
    
    # Compute ratio statistics
    ratio_mean = sum(batch_ratios) / len(batch_ratios)
    ratio_min = min(batch_ratios)
    ratio_max = max(batch_ratios)
    
    print("\n=== Batch Summary ===")
    print(f"Average loss: {avg_batch_loss:.4f}")
    print(f"Average PPO loss: {avg_batch_ppo_loss:.4f}")
    print(f"Average KL penalty: {avg_batch_kl_penalty:.4f}")
    print(f"Ratios: mean={ratio_mean:.4f}, min={ratio_min:.4f}, max={ratio_max:.4f}")

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = DummyConfig()
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens for TextWorld - using the correct tokens from the agent
    special_tokens = {
        'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set model to evaluation mode for consistent results
    model.eval()
    
    # Use the specific gameplay data path
    gameplay_data_path = '/content/drive/MyDrive/textworld_rl_data/gameplay_data_1_filtered_fixed.json'
    print(f"Loading gameplay data from: {gameplay_data_path}")
    gameplay_data = load_gameplay_data(gameplay_data_path)
    
    # Convert data to the format expected by the optimizer
    print("Converting data to the format expected by the optimizer...")
    
    # Extract data
    if "data" in gameplay_data:
        data_dict = gameplay_data["data"]
    else:
        print("Unexpected data format!")
        return
    
    # Verify we have the required fields
    required_fields = ["prompt", "completion", "reward"]
    if not all(field in data_dict for field in required_fields):
        print(f"Missing required fields. Found: {list(data_dict.keys())}")
        return
    
    # Get the lists
    prompts = data_dict["prompt"]
    completions = data_dict["completion"]
    rewards = data_dict["reward"]
    
    # Get G from config
    G = config.num_samples
    print(f"Using {G} completions per step from config")
    
    # Calculate number of steps
    total_examples = len(prompts)
    if total_examples % G != 0:
        print(f"Warning: Total examples ({total_examples}) is not divisible by G ({G})")
        # Truncate to nearest multiple of G
        total_examples = (total_examples // G) * G
        prompts = prompts[:total_examples]
        completions = completions[:total_examples]
        rewards = rewards[:total_examples]
    
    num_steps = total_examples // G
    print(f"Total examples: {total_examples}")
    print(f"Number of steps: {num_steps}")
    
    # Create structured data for testing
    structured_data = []
    
    # Process each step (group of G completions)
    for step_idx in range(num_steps):
        start_idx = step_idx * G
        end_idx = start_idx + G
        
        # Get prompt (should be the same for all G completions)
        prompt = prompts[start_idx]
        
        # Verify all prompts in this group are the same
        group_prompts = prompts[start_idx:end_idx]
        if not all(p == prompt for p in group_prompts):
            print(f"Warning: Not all prompts in step {step_idx} are the same. Skipping...")
            continue
        
        # Get all G completions and rewards for this step
        step_completions = completions[start_idx:end_idx]
        step_rewards = rewards[start_idx:end_idx]
        
        # Verify we have exactly G completions and rewards
        if len(step_completions) != G or len(step_rewards) != G:
            print(f"Warning: Step {step_idx} has incorrect number of completions/rewards. Expected {G}, got {len(step_completions)}/{len(step_rewards)}")
            continue
        
        # Normalize rewards to get advantages
        rewards_array = np.array(step_rewards)
        advantages = (rewards_array - np.mean(rewards_array))
        if np.std(advantages) > 0:
            advantages = advantages / np.std(advantages)
        
        # Create a data point with the prompt and all its completions
        structured_data.append({
            "state": prompt,
            "outputs": step_completions,
            "advantages": advantages.tolist()
        })
    
    # Print first example
    if structured_data:
        print("\nFirst example:")
        print(f"  Number of completions: {len(structured_data[0]['outputs'])}")
        print(f"  Number of advantages: {len(structured_data[0]['advantages'])}")
        print(f"  First completion: {structured_data[0]['outputs'][0][:100]}...")
        print(f"  First advantage: {structured_data[0]['advantages'][0]}")
    
    # Create a small batch for testing
    batch_size = min(2, len(structured_data))
    test_batch = structured_data[:batch_size]
    
    # Debug batch processing
    debug_batch_processing(model, tokenizer, device, test_batch, 
                          epsilon=config.epsilon, beta=config.beta)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 