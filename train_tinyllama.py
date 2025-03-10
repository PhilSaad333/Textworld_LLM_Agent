"""
This script fine-tunes TinyLlama-1.1B-Chat on TextWorld data, optimized for A100 GPUs.
It includes tag checking between epochs to monitor the model's ability to use command and room tags.
It also implements layer freezing to reduce memory usage and improve training efficiency.
"""

import os
import json
import torch
import random
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Function to monitor GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        # Clear cache first to get accurate measurement
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        print(f"GPU Memory: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Max Allocated: {max_allocated:.2f} GB")
        
        # Reset max stats
        torch.cuda.reset_peak_memory_stats()
        return allocated, reserved, max_allocated
    else:
        print("No GPU available")
        return 0, 0, 0

# Mount Google Drive to access your data
from google.colab import drive
drive.mount('/content/drive')

# Set paths
data_path = '/content/drive/MyDrive/textworld_data/sft/combined_training_data_reformatted.json'
output_dir = '/content/drive/MyDrive/textworld_data/models/tinyllama_finetuned'
os.makedirs(output_dir, exist_ok=True)

# Print GPU info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    print(f"GPU Memory after clearing cache: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("No GPU available, using CPU")

# Load the dataset
print(f"Loading dataset from {data_path}")
with open(data_path, 'r') as f:
    dataset = json.load(f)

print(f"Dataset loaded with {len(dataset)} examples")

# Display a sample
print("\nSample input/output pair:")
sample_idx = random.randint(0, len(dataset) - 1)
print(f"Input: {dataset[sample_idx]['input'][:200]}...")
print(f"Output: {dataset[sample_idx]['output'][:200]}...")

# Split into train and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
print(f"Split dataset into {len(train_data)} training examples and {len(val_data)} validation examples")

# Import your existing CombinedSFTDataset and SFTTrainer
from training.fine_tuning import CombinedSFTDataset, SFTTrainer
from config.config import SFTConfig

# Create custom datasets from the split data
class TextWorldDataset(CombinedSFTDataset):
    def __init__(self, examples):
        # Override the initialization to use the provided examples
        self.examples = examples

# Create datasets
train_dataset = TextWorldDataset(train_data)
val_dataset = TextWorldDataset(val_data)

# Create SFT config
sft_config = SFTConfig()

# Update config settings to use more RAM
sft_config.checkpoint_dir = output_dir
sft_config.use_wandb = False  # Disable wandb

# Increase the number of unfrozen layers to use more RAM
sft_config.unfreeze_last_n_layers = 4  # Unfreeze 4 layers instead of 2

# Adjust batch size and gradient accumulation for better performance
sft_config.batch_size = 24  # Increase batch size
sft_config.gradient_accumulation_steps = 1  # Reduce gradient accumulation since we have larger batch size

# Enable mixed precision for better performance (optional)
# sft_config.mixed_precision = True  # Uncomment if you want to use mixed precision

print(f"Updated config: Unfreezing last {sft_config.unfreeze_last_n_layers} layers, batch size {sft_config.batch_size}")

# Function to analyze tag usage in model outputs
def analyze_tags(text):
    """Analyze the tag usage in the model's output"""
    results = {
        "has_command_tags": False,
        "has_room_tags": False,
        "correct_command_tags": False,
        "correct_room_tags": False,
        "command_content": None,
        "room_content": None,
        "wrong_tags": []
    }
    
    # Check for command tags
    command_match = re.search(r'<command>(.*?)</command>', text, re.DOTALL)
    if command_match:
        results["has_command_tags"] = True
        results["correct_command_tags"] = True
        results["command_content"] = command_match.group(1).strip()
    
    # Check for room tags
    room_match = re.search(r'<room>(.*?)</room>', text, re.DOTALL)
    if room_match:
        results["has_room_tags"] = True
        results["correct_room_tags"] = True
        results["room_content"] = room_match.group(1).strip()
    
    # Check for wrong command tag formats
    wrong_command_patterns = [
        r'<command>(.*?)<\/room>',
        r'<room>(.*?)<\/command>',
        r'<commands>(.*?)<\/commands>',
        r'<cmd>(.*?)<\/cmd>'
    ]
    
    for pattern in wrong_command_patterns:
        if re.search(pattern, text, re.DOTALL):
            results["wrong_tags"].append(pattern)
    
    # Check for wrong room tag formats
    wrong_room_patterns = [
        r'<rooms>(.*?)<\/rooms>',
        r'<location>(.*?)<\/location>'
    ]
    
    for pattern in wrong_room_patterns:
        if re.search(pattern, text, re.DOTALL):
            results["wrong_tags"].append(pattern)
    
    return results

# Function to check tag usage on a sample of validation data
def check_tag_usage(model, tokenizer, val_dataset, num_samples=20, device="cuda"):
    """Check how well the model is using command and room tags on validation data"""
    # Select random samples
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    samples = [val_dataset.examples[i] for i in sample_indices]
    
    results = []
    tag_stats = {
        "has_command_tags": 0,
        "has_room_tags": 0,
        "correct_command_tags": 0,
        "correct_room_tags": 0,
        "has_wrong_tags": 0
    }
    
    model.eval()
    
    for sample in tqdm(samples, desc="Checking tag usage"):
        # Format prompt using TinyLlama's chat format
        prompt = f"""<|user|>
You are playing a text adventure game. Analyze this game state and give a response formatted as requested:

{sample["input"]}

Generate a response in the following format:

A) One sentence reasoning about the game state, which actions seem relevant, and what those actions might achieve. 

B) Then, state your chosen action - Make sure it is in the available actions list:
Therefore, I choose: <command>[exact action]</command>

C) Then, state your prediction for the room you will be in after taking this action (say "New Room" if you think it will be a room you haven't been in yet):
I predict that I will be in room: <room>[room name]</room>
<|assistant|>"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=128,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the generated part (remove the input prompt)
        input_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))
        generated_text = full_response[input_length:].strip()
        
        # Analyze tag usage
        tag_analysis = analyze_tags(generated_text)
        
        # Update stats
        if tag_analysis["has_command_tags"]:
            tag_stats["has_command_tags"] += 1
        if tag_analysis["has_room_tags"]:
            tag_stats["has_room_tags"] += 1
        if tag_analysis["correct_command_tags"]:
            tag_stats["correct_command_tags"] += 1
        if tag_analysis["correct_room_tags"]:
            tag_stats["correct_room_tags"] += 1
        if tag_analysis["wrong_tags"]:
            tag_stats["has_wrong_tags"] += 1
        
        # Store results
        results.append({
            "input": sample["input"],
            "expected_output": sample["output"],
            "generated_text": generated_text,
            "tag_analysis": tag_analysis
        })
    
    # Calculate percentages - FIX: Create a copy of the keys to avoid modifying during iteration
    stats_with_percentages = tag_stats.copy()
    for key in list(tag_stats.keys()):
        stats_with_percentages[key + "_pct"] = tag_stats[key] / len(samples) * 100
    
    return results, stats_with_percentages

# Function to freeze layers in the model
def freeze_model_layers(model, unfreeze_last_n_layers=2):
    """Freeze all layers except the last n transformer layers and the LM head"""
    print(f"Freezing model layers, keeping last {unfreeze_last_n_layers} layers unfrozen...")
    
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Count total parameters and frozen parameters
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Initially frozen parameters: {frozen_params:,}/{total_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    # For TinyLlama (Llama architecture), the layers are in model.model.layers
    # Llama models have a different structure than GPT-2
    try:
        # First try the Llama architecture
        transformer_layers = list(model.model.layers)
        print(f"Detected Llama architecture with {len(transformer_layers)} layers")
    except AttributeError:
        try:
            # Try GPT-2 style architecture as fallback
            transformer_layers = list(model.transformer.h)
            print(f"Detected GPT-2 style architecture with {len(transformer_layers)} layers")
        except AttributeError:
            # If both fail, print model structure and raise error
            print("Model structure:")
            for name, _ in model.named_children():
                print(f"- {name}")
            raise ValueError("Could not determine model architecture. Please check model structure.")
    
    num_layers = len(transformer_layers)
    
    # Unfreeze the last n transformer layers
    for i in range(num_layers - unfreeze_last_n_layers, num_layers):
        print(f"Unfreezing layer {i}")
        for param in transformer_layers[i].parameters():
            param.requires_grad = True
    
    # Always unfreeze the LM head for generation
    try:
        for param in model.lm_head.parameters():
            param.requires_grad = True
        print("Unfrozen LM head")
    except AttributeError:
        print("Could not find lm_head, trying to unfreeze output layer...")
        # Try to find the output layer with a different name
        output_layer_found = False
        for name, module in model.named_children():
            if any(x in name for x in ['output', 'head', 'classifier']):
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfrozen output layer: {name}")
                output_layer_found = True
                break
        
        if not output_layer_found:
            print("Warning: Could not identify output layer to unfreeze")
    
    # Count unfrozen parameters
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen parameters: {unfrozen_params:,}/{total_params:,} ({unfrozen_params/total_params*100:.2f}%)")
    
    return model

# Initialize trainer
trainer = SFTTrainer(sft_config)

# Get the model and tokenizer from the trainer
model = trainer.model
tokenizer = trainer.tokenizer

# Add special tokens for command and room tags if not already added
special_tokens = {
    'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
}

# Check if tokens already exist
special_tokens_to_add = []
for token in special_tokens['additional_special_tokens']:
    if token not in tokenizer.get_vocab():
        special_tokens_to_add.append(token)

if special_tokens_to_add:
    special_tokens['additional_special_tokens'] = special_tokens_to_add
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens to the tokenizer and resized model embeddings")

# Freeze layers if configured
if sft_config.freeze_layers:
    model = freeze_model_layers(model, sft_config.unfreeze_last_n_layers)
    # Update the model in the trainer
    trainer.model = model

# Check memory usage before training
print("\nMemory usage before training:")
print_gpu_memory()

# Estimate memory usage for a single batch
def estimate_batch_memory(model, batch_size, seq_length=512):
    """Estimate memory usage for a single batch with more realistic overhead"""
    # Get model hidden size
    hidden_size = model.config.hidden_size
    
    # All model parameters (regardless of requires_grad)
    all_params_size = sum(p.numel() for p in model.parameters()) * 4  # bytes in float32
    
    # Trainable parameters only
    trainable_params_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4
    
    # Optimizer states (8 bytes per trainable parameter for Adam)
    optimizer_size = trainable_params_size * 2
    
    # Gradients (4 bytes per trainable parameter)
    gradient_size = trainable_params_size
    
    # Activations (more realistic estimate)
    # For each layer, we need to store:
    # - Input activations
    # - Output activations
    # - Attention key/query/value matrices
    # - Intermediate activations in MLP
    num_layers = len(list(model.model.layers))
    activation_size_per_token = hidden_size * 4  # bytes per token for basic activations
    
    # Each layer has multiple internal states
    layer_overhead_factor = 12  # More realistic factor accounting for attention states, etc.
    
    # Total activation memory
    activation_size = batch_size * seq_length * activation_size_per_token * num_layers * layer_overhead_factor
    
    # Batch data
    batch_size_bytes = batch_size * seq_length * 2 * 4  # 2 tensors (input_ids, attention_mask), 4 bytes per int
    
    # CUDA and PyTorch overhead (empirical estimate)
    cuda_overhead = 2 * 1024**3  # ~2 GB
    
    # Memory fragmentation (empirical estimate)
    fragmentation_factor = 1.2  # 20% fragmentation
    
    # Total
    raw_total = all_params_size + optimizer_size + gradient_size + activation_size + batch_size_bytes + cuda_overhead
    total_bytes = raw_total * fragmentation_factor
    
    # Convert to GB
    total_gb = total_bytes / (1024**3)
    
    print(f"\nDetailed Memory Estimate for Batch Size {batch_size}:")
    print(f"All Model Parameters: {all_params_size / (1024**3):.2f} GB")
    print(f"Trainable Parameters: {trainable_params_size / (1024**3):.2f} GB")
    print(f"Optimizer States: {optimizer_size / (1024**3):.2f} GB")
    print(f"Gradients: {gradient_size / (1024**3):.2f} GB")
    print(f"Activations: {activation_size / (1024**3):.2f} GB")
    print(f"Batch Data: {batch_size_bytes / (1024**3):.2f} GB")
    print(f"CUDA/PyTorch Overhead: {cuda_overhead / (1024**3):.2f} GB")
    print(f"Estimated Memory with Fragmentation: {total_gb:.2f} GB")
    
    return total_gb

# Estimate memory usage
estimate_batch_memory(model, sft_config.batch_size)

# Function to monitor memory usage during training
def monitor_memory_usage(model, batch, phase="forward"):
    """Monitor memory usage during different phases of training"""
    # Clear cache first to get accurate measurement
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get memory stats before operation
    before_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    before_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    
    # Perform operation based on phase
    if phase == "forward":
        # Forward pass
        with torch.no_grad():
            _ = model(**batch)
    elif phase == "backward":
        # Forward and backward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
    
    # Get memory stats after operation
    after_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    after_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    
    # Calculate difference
    diff_allocated = after_allocated - before_allocated
    diff_reserved = after_reserved - before_reserved
    
    print(f"\nMemory Usage - {phase.capitalize()} Phase:")
    print(f"Before: Allocated: {before_allocated:.2f} GB, Reserved: {before_reserved:.2f} GB")
    print(f"After: Allocated: {after_allocated:.2f} GB, Reserved: {after_reserved:.2f} GB")
    print(f"Difference: Allocated: {diff_allocated:.2f} GB, Reserved: {diff_reserved:.2f} GB")
    
    return after_allocated, after_reserved

# Function to log memory usage to a file
def log_memory_usage(log_file, phase, epoch=None):
    """Log memory usage to a file"""
    if torch.cuda.is_available():
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Create log entry
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        epoch_str = f"Epoch {epoch}" if epoch is not None else "N/A"
        log_entry = f"{timestamp} | {phase} | {epoch_str} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Max: {max_allocated:.2f} GB\n"
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        return allocated, reserved, max_allocated
    else:
        return 0, 0, 0

# Create memory log file
memory_log_file = os.path.join(output_dir, "memory_usage.log")
with open(memory_log_file, 'w') as f:
    f.write("Timestamp | Phase | Epoch | Allocated | Reserved | Max Allocated\n")
    f.write("-" * 80 + "\n")

# Log initial memory usage
log_memory_usage(memory_log_file, "Initial")

# Training loop with tag checking
total_epochs = 3  # Total number of epochs to train
tag_check_results = []

print("\nStarting training with tag checking between epochs...")

for epoch in range(total_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{total_epochs}")
    print(f"{'='*50}")
    
    # Log memory usage before training
    log_memory_usage(memory_log_file, "Before Training", epoch+1)
    
    # Train for one epoch
    print(f"Training epoch {epoch+1}...")
    
    # If this is the first epoch, monitor memory usage on the first batch
    if epoch == 0:
        print("\nMonitoring memory usage on first batch...")
        # Get a sample batch
        try:
            # Try to get a batch from the trainer's dataloader
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=sft_config.batch_size,
                shuffle=True,
                num_workers=sft_config.num_workers,
                pin_memory=sft_config.pin_memory
            )
            sample_batch = next(iter(train_dataloader))
            
            # Prepare the batch for the model
            inputs = tokenizer(
                [ex["input"] for ex in sample_batch], 
                padding="max_length",
                truncation=True,
                max_length=sft_config.max_input_length,
                return_tensors="pt"
            ).to(sft_config.device)
            
            # Monitor forward pass
            monitor_memory_usage(model, inputs, phase="forward")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Monitor backward pass
            monitor_memory_usage(model, inputs, phase="backward")
            
            # Clear memory again
            torch.cuda.empty_cache()
            gc.collect()
            
            print("\nContinuing with normal training...")
        except Exception as e:
            print(f"Error during memory monitoring: {str(e)}")
            print("Continuing with normal training...")
    
    train_results = trainer.train(train_dataset, val_dataset)
    
    # Log memory usage after training
    log_memory_usage(memory_log_file, "After Training", epoch+1)
    
    # Check memory usage after training
    print(f"\nMemory usage after epoch {epoch+1}:")
    print_gpu_memory()
    
    # Save checkpoint for this epoch
    epoch_checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}_model.pt")
    try:
        # Try with epoch and filename parameters
        trainer.save_checkpoint(epoch=epoch+1, filename=epoch_checkpoint_path)
    except TypeError:
        # Fallback if the function signature is different
        try:
            trainer.save_checkpoint(epoch+1, filename=epoch_checkpoint_path)
        except:
            # Last resort fallback
            trainer.save_checkpoint(epoch_checkpoint_path)
    print(f"Saved checkpoint to {epoch_checkpoint_path}")
    
    # Log memory usage after checkpoint
    log_memory_usage(memory_log_file, "After Checkpoint", epoch+1)
    
    # Check tag usage
    print(f"\nChecking tag usage after epoch {epoch+1}...")
    try:
        tag_results, tag_stats = check_tag_usage(
            model, 
            tokenizer, 
            val_dataset, 
            num_samples=sft_config.tag_check_samples, 
            device=sft_config.device
        )
        
        # Print tag usage statistics
        print("\nTag Usage Statistics:")
        print(f"Samples with command tags: {tag_stats['has_command_tags']} ({tag_stats['has_command_tags_pct']:.1f}%)")
        print(f"Samples with room tags: {tag_stats['has_room_tags']} ({tag_stats['has_room_tags_pct']:.1f}%)")
        print(f"Samples with correct command tags: {tag_stats['correct_command_tags']} ({tag_stats['correct_command_tags_pct']:.1f}%)")
        print(f"Samples with correct room tags: {tag_stats['correct_room_tags']} ({tag_stats['correct_room_tags_pct']:.1f}%)")
        print(f"Samples with wrong tag formats: {tag_stats['has_wrong_tags']} ({tag_stats['has_wrong_tags_pct']:.1f}%)")
        
        # Store tag check results
        tag_check_results.append({
            "epoch": epoch + 1,
            "tag_stats": tag_stats,
            "examples": tag_results[:5]  # Store first 5 examples
        })
        
        # Save tag check results
        tag_check_path = os.path.join(output_dir, f"epoch_{epoch+1}_tag_check.json")
        with open(tag_check_path, 'w') as f:
            json.dump(tag_check_results[-1], f, indent=2)
        print(f"Saved tag check results to {tag_check_path}")
    except Exception as e:
        print(f"Error during tag checking: {str(e)}")
        print("Continuing to next epoch...")
    
    # Log memory usage after tag checking
    log_memory_usage(memory_log_file, "After Tag Checking", epoch+1)
    
    # Clear memory between epochs
    torch.cuda.empty_cache()
    gc.collect()

# Final evaluation
print("\nFinal Evaluation:")
log_memory_usage(memory_log_file, "Before Final Evaluation")

val_dataloader = DataLoader(
    val_dataset,
    batch_size=sft_config.batch_size,
    shuffle=False,
    num_workers=sft_config.num_workers,
    pin_memory=sft_config.pin_memory
)
final_metrics = trainer.evaluate(val_dataloader)

print(f"Final Validation Loss: {final_metrics['loss']:.4f}")
print(f"Final Validation Perplexity: {final_metrics['perplexity']:.4f}")

log_memory_usage(memory_log_file, "After Final Evaluation")

# Save the final model and tokenizer
print("\nSaving final model...")
log_memory_usage(memory_log_file, "Before Saving Final Model")

final_model_path = os.path.join(output_dir, "final_model")
trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(os.path.join(final_model_path, "tokenizer"))
print(f"Final model saved to {final_model_path}")

log_memory_usage(memory_log_file, "After Saving Final Model")

# Save all tag check results
all_tag_checks_path = os.path.join(output_dir, "all_tag_checks.json")
with open(all_tag_checks_path, 'w') as f:
    json.dump(tag_check_results, f, indent=2)
print(f"All tag check results saved to {all_tag_checks_path}")

print("\nTraining and evaluation completed!")
print(f"Model saved to {output_dir}")
print(f"Memory usage log saved to {memory_log_file}")

# Final memory usage
log_memory_usage(memory_log_file, "Final")

# Print a summary of tag usage improvement over epochs
print("\nTag Usage Improvement Summary:")
print(f"{'Epoch':<10} {'Command Tags %':<15} {'Room Tags %':<15} {'Wrong Tags %':<15}")
print("-" * 55)
for result in tag_check_results:
    epoch = result["epoch"]
    cmd_pct = result["tag_stats"]["has_command_tags_pct"]
    room_pct = result["tag_stats"]["has_room_tags_pct"]
    wrong_pct = result["tag_stats"]["has_wrong_tags_pct"]
    print(f"{epoch:<10} {cmd_pct:<15.1f} {room_pct:<15.1f} {wrong_pct:<15.1f}") 