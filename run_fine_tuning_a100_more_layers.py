"""
This script fine-tunes a GPT-2 Medium model on TextWorld data, optimized for A100 GPUs.
This version unfreezes more layers to take advantage of the A100's memory capacity.
Paste this into a Colab cell after restarting your session.
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Mount Google Drive to access your data
from google.colab import drive
drive.mount('/content/drive')

# Set paths
data_path = '/content/drive/MyDrive/textword_data/sft/combined_training_data_reformatted.json'
output_dir = '/content/drive/MyDrive/textword_data/models/gpt2_medium_finetuned'
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

# Update config settings for A100 GPU
sft_config.model_name = "gpt2-medium"
sft_config.checkpoint_dir = output_dir
sft_config.use_wandb = False  # Disable wandb
sft_config.num_epochs = 3
sft_config.batch_size = 16  # Larger batch size for A100
sft_config.learning_rate = 2e-5
sft_config.mixed_precision = False  # Disable mixed precision for simplicity
sft_config.gradient_accumulation_steps = 2  # Fewer gradient accumulation steps

# Custom function to modify the trainer after initialization to unfreeze more layers
def unfreeze_more_layers(trainer, num_layers_to_keep_trainable=6):
    """Unfreeze more layers of the model to take advantage of A100's memory capacity"""
    # First, make sure all parameters are frozen
    for param in trainer.model.parameters():
        param.requires_grad = False
    
    # For GPT-2, unfreeze the specified number of layers from the end
    if hasattr(trainer.model, 'transformer') and hasattr(trainer.model.transformer, 'h'):
        num_layers = len(trainer.model.transformer.h)
        print(f"Model has {num_layers} layers total")
        
        # Unfreeze the last num_layers_to_keep_trainable layers
        for i in range(num_layers - num_layers_to_keep_trainable, num_layers):
            print(f"Unfreezing layer {i}")
            for param in trainer.model.transformer.h[i].parameters():
                param.requires_grad = True
    
    # Always unfreeze the LM head
    if hasattr(trainer.model, 'lm_head'):
        for param in trainer.model.lm_head.parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
    
    return trainer

# Initialize trainer
trainer = SFTTrainer(sft_config)

# Unfreeze more layers (6 instead of the default 2)
trainer = unfreeze_more_layers(trainer, num_layers_to_keep_trainable=6)

# Train the model
print("\nStarting training...")
results = trainer.train(train_dataset, val_dataset)

# Final evaluation
print("\nFinal Evaluation:")
val_dataloader = DataLoader(
    val_dataset,
    batch_size=sft_config.batch_size,
    shuffle=False,
    num_workers=4,  # More workers for faster data loading
    pin_memory=True
)
final_metrics = trainer.evaluate(val_dataloader)

print(f"Final Validation Loss: {final_metrics['loss']:.4f}")
print(f"Final Validation Perplexity: {final_metrics['perplexity']:.4f}")

# Save the tokenizer
tokenizer_path = os.path.join(output_dir, "tokenizer")
trainer.tokenizer.save_pretrained(tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path}")

print("\nTraining and evaluation completed!")
print(f"Model saved to {output_dir}")

# Test loading the best model
best_model_path = os.path.join(output_dir, "best_model.pt")
if os.path.exists(best_model_path):
    print(f"\nLoading best model from {best_model_path} to verify it works...")
    test_metrics = trainer.load_checkpoint(best_model_path)
    print(f"Best model metrics: {test_metrics}") 