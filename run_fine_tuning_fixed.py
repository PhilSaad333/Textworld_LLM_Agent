"""
This script fine-tunes a GPT-2 Medium model on TextWorld data.
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

# Update config settings
sft_config.model_name = "gpt2-medium"
sft_config.checkpoint_dir = output_dir
sft_config.use_wandb = False  # Disable wandb
sft_config.num_epochs = 3
sft_config.batch_size = 8
sft_config.learning_rate = 2e-5
sft_config.mixed_precision = True
sft_config.gradient_accumulation_steps = 4

# Adjust batch size based on GPU memory
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    if gpu_mem > 40:  # For A100 or similar high-memory GPUs
        sft_config.batch_size = 16
        sft_config.gradient_accumulation_steps = 2
    elif gpu_mem > 15:  # For V100 or similar
        sft_config.batch_size = 8
        sft_config.gradient_accumulation_steps = 4
    else:  # For smaller GPUs
        sft_config.batch_size = 4
        sft_config.gradient_accumulation_steps = 8

# Initialize trainer
trainer = SFTTrainer(sft_config)

# Train the model
print("\nStarting training...")
results = trainer.train(train_dataset, val_dataset)

# Final evaluation
print("\nFinal Evaluation:")
val_dataloader = DataLoader(
    val_dataset,
    batch_size=sft_config.batch_size,
    shuffle=False,
    num_workers=sft_config.num_workers if hasattr(sft_config, 'num_workers') else 2,
    pin_memory=sft_config.pin_memory if hasattr(sft_config, 'pin_memory') else True
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