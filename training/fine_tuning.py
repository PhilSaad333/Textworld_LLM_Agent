import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Any
import wandb
import numpy as np
from tqdm import tqdm
import os
import re
import random
import json
from torch.utils.data import Dataset

class SFTTrainer:
    def __init__(self, config):
        """
        Initialize SFT Trainer
        
        Args:
            config: Configuration object containing:
                - model_name (str): Base model name (e.g., "google/flan-t5-base")
                - learning_rate (float): Learning rate for training
                - batch_size (int): Batch size for training
                - num_epochs (int): Number of training epochs
                - warmup_steps (int): Number of warmup steps for scheduler
                - max_input_length (int): Maximum input sequence length
                - max_output_length (int): Maximum output sequence length
                - device (str): Device to use for training ("cuda" or "cpu")
                - checkpoint_dir (str): Directory to save model checkpoints
                - use_wandb (bool): Whether to use Weights & Biases logging
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens for command and room tags
        special_tokens = {
            'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Resize the model's token embeddings to account for the new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        # Freeze all layers except the last n_unfrozen_layers
        n_unfrozen_layers = 3  # Adjust this based on performance
        
        # Get all layer names
        layer_names = [name for name, _ in self.model.named_parameters()]
        encoder_layers = [name for name in layer_names if 'encoder.block' in name]
        decoder_layers = [name for name in layer_names if 'decoder.block' in name]
        
        # Calculate which layers to freeze
        n_encoder_layers = len(set([name.split('.')[2] for name in encoder_layers]))
        n_decoder_layers = len(set([name.split('.')[2] for name in decoder_layers]))
        
        # Freeze parameters
        for name, param in self.model.named_parameters():
            # Always train layer norm and bias terms
            if 'layer_norm' in name or 'bias' in name:
                param.requires_grad = True
            # Train only the last n_unfrozen_layers of encoder and decoder
            elif 'encoder.block' in name:
                layer_num = int(name.split('.')[2])
                param.requires_grad = layer_num >= (n_encoder_layers - n_unfrozen_layers)
            elif 'decoder.block' in name:
                layer_num = int(name.split('.')[2])
                param.requires_grad = layer_num >= (n_decoder_layers - n_unfrozen_layers)
            else:
                param.requires_grad = True  # Train all other parameters
            
        # Log number of trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
        
        # Initialize optimizer and scheduler (to be set in prepare_training)
        self.optimizer = None
        self.scheduler = None
        
    def prepare_training(self, dataset):
        """
        Prepare for training by setting up optimizer, scheduler, and dataloader
        
        Args:
            dataset: SFTData dataset instance
        """
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total training steps
        num_training_steps = len(dataset) * self.config.num_epochs // self.config.batch_size
        
        # Create learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def _prepare_batch(self, batch):
        """
        Prepare batch for training/evaluation
        
        Args:
            batch: Batch of examples from dataloader
            
        Returns:
            dict: Processed batch with input_ids, attention_mask, and labels
        """        
        # Extract inputs and outputs from batch
        if isinstance(batch, dict):
            inputs = batch["input"]
            outputs = batch["output"]
        else:
            # If batch is a list of dicts
            inputs = [example["input"] for example in batch]
            outputs = [example["output"] for example in batch]
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt"
        )
        
        # Tokenize outputs
        output_encodings = self.tokenizer(
            outputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_output_length,
            return_tensors="pt"
        )
        
        # Prepare model inputs
        model_inputs = {
            "input_ids": input_encodings.input_ids.to(self.device),
            "attention_mask": input_encodings.attention_mask.to(self.device),
            "labels": output_encodings.input_ids.to(self.device)
        }
        
        # Replace padding token id with -100 for labels
        model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100
        
        return model_inputs

    def train(self, train_dataset, val_dataset=None):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        """
        # Prepare training
        self.prepare_training(train_dataset)
        
        # Initialize best validation metric for model selection
        best_val_loss = float('inf')
        
        # Initialize Weights & Biases if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=vars(self.config)
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train for one epoch
            train_metrics = self._train_epoch(self.train_dataloader, epoch)
            
            # Log training metrics
            print(f"Training metrics:")
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
                if self.config.use_wandb:
                    wandb.log({f"train/{k}": v}, step=epoch)
            
            # Evaluate on validation set if provided
            if val_dataset is not None:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory
                )
                
                val_metrics = self.evaluate(val_dataloader)
                
                print(f"Validation metrics:")
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                    if self.config.use_wandb:
                        wandb.log({f"val/{k}": v}, step=epoch)
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(
                        epoch,
                        metrics=val_metrics,
                        filename=f"best_model.pt"
                    )
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(
                    epoch,
                    metrics=train_metrics,
                    filename=f"checkpoint_epoch_{epoch+1}.pt"
                )
        
        # Finish W&B run if enabled
        if self.config.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, train_dataloader, epoch):
        """
        Train for one epoch
        
        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        
        # Initialize progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Prepare batch
            inputs = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation if configured
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
            
            # Log to wandb if enabled
            if self.config.use_wandb and step % self.config.log_steps == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })
        
        # Calculate epoch metrics
        metrics = {
            'loss': total_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return metrics
    
    def evaluate(self, val_dataloader):
        """
        Evaluate model on validation set
        
        Args:
            val_dataloader: DataLoader for validation data
        
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(val_dataloader)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                # Prepare batch
                inputs = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Generate predictions
                predictions = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.config.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    remove_invalid_values=True,
                    # Add these parameters to improve formatting
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
                
                # Decode predictions and labels
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
                
                # Create a mask for non-padding tokens
                label_mask = inputs["labels"] != -100
                labels_to_decode = inputs["labels"].clone()
                labels_to_decode[~label_mask] = self.tokenizer.pad_token_id
                decoded_labels = self.tokenizer.batch_decode(labels_to_decode, skip_special_tokens=False)
                
                all_predictions.extend(decoded_preds)
                all_labels.extend(decoded_labels)
                total_loss += loss.item()
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / num_batches,
        }
        
        # Add any additional metrics
        additional_metrics = self._compute_metrics(all_predictions, all_labels)
        metrics.update(additional_metrics)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics=None, filename=None):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            metrics: Optional dictionary of metrics to save with checkpoint
            filename: Optional specific filename for the checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Clean up old checkpoints if needed
        if hasattr(self.config, 'keep_checkpoint_max') and self.config.keep_checkpoint_max > 0:
            checkpoints = sorted([
                f for f in os.listdir(self.config.checkpoint_dir)
                if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
            ])
            while len(checkpoints) > self.config.keep_checkpoint_max:
                os.remove(os.path.join(self.config.checkpoint_dir, checkpoints.pop(0)))

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if training
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if training
        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', None)
    
    def _compute_metrics(self, predictions, labels):
        """
        Compute training/evaluation metrics
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            dict: Computed metrics including:
                - format_adherence: % of responses following A/B/C format
                - action_format: % of responses with valid "Therefore, I choose:" format with <command> tags
                - room_format: % of responses with valid room prediction format with <room> tags
                - exact_match: % of responses matching labels exactly
        """
        metrics = {
            'format_adherence': 0,
            'action_format': 0,
            'room_format': 0,
            'exact_match': 0
        }
        
        total_samples = len(predictions)
        if total_samples == 0:
            return metrics
        
        for pred, label in zip(predictions, labels):
            # Check exact match
            if pred.strip() == label.strip():
                metrics['exact_match'] += 1
            
            # Check format adherence (A/B/C structure)
            has_section_a = bool(re.search(r'A\)', pred, re.IGNORECASE))
            has_section_b = bool(re.search(r'B\)', pred, re.IGNORECASE))
            has_section_c = bool(re.search(r'C\)', pred, re.IGNORECASE))
            
            if has_section_a and has_section_b and has_section_c:
                metrics['format_adherence'] += 1
            
            # Check action format with <command> tags
            action_match = re.search(r"Therefore,\s*I\s*choose:\s*<command>(.+?)</command>", 
                                   pred, re.IGNORECASE)
            if action_match:
                metrics['action_format'] += 1
            
            # Check room prediction format with <room> tags
            room_match = re.search(r"I\s*predict\s*that\s*I\s*will\s*be\s*in\s*room:\s*<room>(.+?)</room>", 
                                 pred, re.IGNORECASE)
            if room_match:
                metrics['room_format'] += 1
        
        # Convert counts to percentages
        for key in metrics:
            metrics[key] = (metrics[key] / total_samples) * 100
        
        # Add some additional analysis
        try:
            # Sample a few predictions for qualitative analysis
            sample_size = min(5, total_samples)
            sample_indices = random.sample(range(total_samples), sample_size)
            
            metrics['samples'] = [
                {
                    'prediction': predictions[i],
                    'label': labels[i],
                    'matches_format': bool(re.search(r'^A\)', predictions[i], re.MULTILINE | re.IGNORECASE) and
                                        re.search(r'^B\)', predictions[i], re.MULTILINE | re.IGNORECASE) and
                                        re.search(r'^C\)', predictions[i], re.MULTILINE | re.IGNORECASE)),
                    'has_command_tags': bool(re.search(r"<command>(.+?)</command>", predictions[i], re.IGNORECASE)),
                    'has_room_tags': bool(re.search(r"<room>(.+?)</room>", predictions[i], re.IGNORECASE))
                }
                for i in sample_indices
            ]
            
            # Log samples if using wandb
            if self.config.use_wandb:
                wandb.log({
                    "examples": wandb.Table(
                        columns=["Prediction", "Label", "Matches Format", "Has Command Tags", "Has Room Tags"],
                        data=[
                            [s['prediction'], s['label'], s['matches_format'], s['has_command_tags'], s['has_room_tags']]
                            for s in metrics['samples']
                        ]
                    )
                })
        
        except Exception as e:
            print(f"Warning: Error in sample analysis: {str(e)}")
        
        return metrics

class CombinedSFTDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.examples = json.load(f)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # Convert to tensor format if needed
        return {
            "input": example["input"],
            "output": example["output"]
        }



