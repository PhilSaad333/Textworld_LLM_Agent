import os
import json
import torch
import numpy as np
import random
import re
import logging
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Use PyTorch's AdamW
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
import torch.nn.functional as F

# Import wandb conditionally to avoid errors if it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class SFTTrainer:
    def __init__(self, config):
        """
        Initialize SFT Trainer
        
        Args:
            config: Configuration object containing:
                - model_name (str): Base model name (e.g., "google/flan-t5-base" or "gpt2-medium")
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
        
        print(f"Initializing SFT Trainer with device: {self.device}")
        print(f"Loading model: {config.model_name}")
        
        # Determine if model is autoregressive or seq2seq based on name
        self.is_autoregressive = self._is_autoregressive_model(config.model_name)
        print(f"Model type: {'Autoregressive' if self.is_autoregressive else 'Sequence-to-Sequence'}")
        
        # Initialize model and tokenizer based on model type
        if self.is_autoregressive:
            # For autoregressive models like GPT-2
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name
            )
        else:
            # For sequence-to-sequence models like T5
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
            
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens for command and room tags
        special_tokens = {
            'additional_special_tokens': ['<tag>']
        }
        
        # Add pad token if it doesn't exist (for some autoregressive models)
        if self.is_autoregressive and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Resize the model's token embeddings to account for the new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to device
        self.model.to(self.device)
        
        # Freeze most of the model parameters to reduce memory usage
        # Keep only the last few layers trainable
        num_layers_to_keep_trainable = config.unfreeze_last_n_layers 
        
        # Count total parameters before freezing
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Freeze parameters based on model type
        if self.is_autoregressive:
            # For GPT-2 and similar models
            if hasattr(self.model, 'transformer'):
                # Freeze embeddings
                for param in self.model.transformer.wte.parameters():
                    param.requires_grad = False
                for param in self.model.transformer.wpe.parameters():
                    param.requires_grad = False
                
                # Freeze most of the layers
                if hasattr(self.model.transformer, 'h'):
                    num_layers = len(self.model.transformer.h)
                    for i, layer in enumerate(self.model.transformer.h):
                        # Only keep the last few layers trainable
                        if i < num_layers - num_layers_to_keep_trainable:
                            for param in layer.parameters():
                                param.requires_grad = False
                
                # Keep LM head trainable
                if hasattr(self.model, 'lm_head'):
                    for param in self.model.lm_head.parameters():
                        param.requires_grad = True
        else:
            # For T5 and similar models
            if hasattr(self.model, 'encoder'):
                # Freeze encoder
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                
                # Freeze most of decoder layers
                if hasattr(self.model.decoder, 'block'):
                    num_layers = len(self.model.decoder.block)
                    for i, layer in enumerate(self.model.decoder.block):
                        # Only keep the last few layers trainable
                        if i < num_layers - num_layers_to_keep_trainable:
                            for param in layer.parameters():
                                param.requires_grad = False
        
        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Initialize mixed precision training if available
        if config.mixed_precision and torch.cuda.is_available():
            try:
                # Use the new recommended way to create GradScaler
                self.scaler = torch.amp.GradScaler('cuda')
            except TypeError:
                # Fall back to the old way if using an older PyTorch version
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb if enabled
        self.use_wandb = False
        if hasattr(config, 'use_wandb') and config.use_wandb:
            if WANDB_AVAILABLE:
                self.use_wandb = True
                print("Weights & Biases logging enabled")
            else:
                print("Warning: wandb not installed. Running without wandb logging.")
        else:
            print("Weights & Biases logging disabled")
            
        print("SFT Trainer initialized successfully")
        
    def _is_autoregressive_model(self, model_name):
        """
        Determine if a model is autoregressive based on its name
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model is autoregressive, False otherwise
        """
        # List of known autoregressive model families
        autoregressive_models = [
            "gpt", "opt", "bloom", "llama", "pythia", "falcon", "mistral", 
            "phi", "gemma", "qwen", "mpt", "cerebras", "stablelm"
        ]
        
        # Check if model name contains any autoregressive model family name
        model_name_lower = model_name.lower()
        return any(ar_model in model_name_lower for ar_model in autoregressive_models)
    
    def prepare_training(self, dataset):
        """
        Prepare for training by setting up optimizer and scheduler
        
        Args:
            dataset: Training dataset
            
        Returns:
            DataLoader: Training dataloader
        """
        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 2,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True
        )
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        # Initialize optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Use PyTorch's AdamW instead of transformers' AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # Create learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return train_dataloader
    
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
        
        if self.is_autoregressive:
            # For autoregressive models, we combine input and output and use shifted inputs as labels
            combined_texts = []
            for inp, out in zip(inputs, outputs):
                combined_texts.append(f"{inp}{out}")
            
            # Tokenize combined texts
            encodings = self.tokenizer(
                combined_texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_input_length + self.config.max_output_length,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for autoregressive models)
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # For each sequence, create labels where:
            # - Input tokens have label -100 (ignored in loss calculation)
            # - Output tokens have their token IDs as labels
            labels = input_ids.clone()
            
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                # Tokenize just the input to find its length
                input_len = len(self.tokenizer(inp, return_tensors="pt").input_ids[0])
                
                # Set labels for input tokens to -100 (ignored in loss)
                labels[i, :input_len] = -100
            
            return {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "labels": labels.to(self.device)
            }
        else:
            # For sequence-to-sequence models, we use the original approach
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

    def train(self, train_dataset, val_dataset=None, use_huggingface_trainer=False):
        """
        Train the model on the given dataset
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            use_huggingface_trainer: If True, use Hugging Face's SFTTrainer from TRL
            
        Returns:
            dict: Training results
        """
        if use_huggingface_trainer:
            return self.train_with_huggingface_sft(train_dataset, val_dataset)
            
        print(f"Starting training with {len(train_dataset)} examples")
        
        # Prepare training
        train_dataloader = self.prepare_training(train_dataset)
        
        # Prepare validation dataloader if provided
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 2,
                pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True
            )
            print(f"Validation set has {len(val_dataset)} examples")
        
        # Initialize Weights & Biases if enabled
        if self.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.config.wandb_project if hasattr(self.config, 'wandb_project') else "textworld-sft",
                    entity=self.config.wandb_entity if hasattr(self.config, 'wandb_entity') else None,
                    config=self.config.__dict__,
                    name=f"{self.config.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.use_wandb = False
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # Print training metrics
            print(f"\nTraining metrics for epoch {epoch+1}:")
            for k, v in train_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({f"train/{k}": v}, step=epoch)
            
            # Evaluate on validation set if provided
            if val_dataset is not None:
                print(f"\nEvaluating on validation set...")
                val_metrics = self.evaluate(val_dataloader)
                
                # Print validation metrics
                print(f"Validation metrics for epoch {epoch+1}:")
                for k, v in val_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({f"val/{k}": v}, step=epoch)
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(
                        epoch,
                        metrics=val_metrics,
                        filename="best_model.pt"
                    )
            
            # Save checkpoint for this epoch
            self.save_checkpoint(
                epoch,
                metrics=train_metrics,
                filename=f"checkpoint_epoch_{epoch+1}.pt"
            )
        
        # Save final model
        self.save_checkpoint(
            self.config.num_epochs - 1,
            metrics=train_metrics,
            filename="final_model.pt"
        )
        
        # Also save just the model state dict for compatibility with TextWorldRLTrainer
        self.save_model_only(os.path.join(self.config.checkpoint_dir, "final_model_state_dict.pt"))
        
        # Finish W&B run if enabled
        if self.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
        
        return train_metrics
    
    def _train_epoch(self, train_dataloader, epoch):
        """
        Train the model for one epoch
        
        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            dict: Training metrics for this epoch
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Create progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        # Training loop
        for step, batch in enumerate(progress_bar):
            # Prepare batch
            inputs = self._prepare_batch(batch)
            
            # Forward pass with mixed precision if enabled
            if self.config.mixed_precision and self.scaler is not None:
                try:
                    # Use the new recommended way for autocast
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                        loss = loss / self.config.gradient_accumulation_steps
                except TypeError:
                    # Fall back to the old way if using an older PyTorch version
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                        loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Check for FP16 gradients before unscaling
                    has_fp16_grads = False
                    for param in self.model.parameters():
                        if param.grad is not None and param.grad.dtype == torch.float16:
                            has_fp16_grads = True
                            break
                    
                    if not has_fp16_grads:
                        # Only unscale if we don't have FP16 gradients
                        self.scaler.unscale_(self.optimizer)
                        
                    # Clip gradients (only applied to non-FP16 gradients)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.grad is not None and p.grad.dtype != torch.float16],
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                # Standard forward and backward pass without mixed precision
                outputs = self.model(**inputs)
                loss = outputs.loss
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
                    self.global_step += 1
            
            # Update progress bar
            batch_size = inputs["input_ids"].size(0)
            total_loss += loss.item() * self.config.gradient_accumulation_steps * batch_size
            total_samples += batch_size
            
            # Update progress bar description
            progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
            
            # Log to wandb if enabled
            if self.use_wandb and WANDB_AVAILABLE and step % self.config.log_steps == 0:
                try:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # Log samples if using wandb
        if self.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.log({
                    "examples": wandb.Table(
                        columns=["input", "prediction", "target"],
                        data=self._get_prediction_samples(train_dataloader)
                    )
                })
            except Exception as e:
                print(f"Warning: Failed to log examples to wandb: {e}")
        
        return metrics
    
    def evaluate(self, val_dataloader):
        """
        Evaluate the model on the validation set
        
        Args:
            val_dataloader: DataLoader for validation data
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                # Prepare batch
                inputs = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Track loss
                batch_size = inputs["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics=None, filename=None):
        """
        Save a checkpoint of the model
        
        Args:
            epoch: Current epoch number
            metrics: Optional metrics to save with the checkpoint
            filename: Optional filename for the checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch+1}.pt"
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Create checkpoint path
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        # Save just the model state dict for compatibility with TextWorldRLTrainer
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model state dict saved to {checkpoint_path}")
        
        # Save tokenizer
        if "best_model" in filename or "final_model" in filename:
            tokenizer_path = os.path.join(self.config.checkpoint_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")
            
        # Also save a full checkpoint with optimizer state for resuming training if needed
        full_checkpoint_path = os.path.join(self.config.checkpoint_dir, f"full_{filename}")
        full_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__,
            'metrics': metrics,
            'global_step': self.global_step,
            'is_autoregressive': self.is_autoregressive
        }
        torch.save(full_checkpoint, full_checkpoint_path)
        print(f"Full checkpoint saved to {full_checkpoint_path}")
    
    def save_model_only(self, path):
        """
        Save only the model state dict to the specified path
        
        Args:
            path: Path to save the model state dict
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save just the model state dict
        torch.save(self.model.state_dict(), path)
        print(f"Model state dict saved to {path}")
        
        # Save tokenizer alongside the model
        tokenizer_path = os.path.join(os.path.dirname(path), "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            dict: Metrics from the checkpoint
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available and optimizer is initialized
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available and scheduler is initialized
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load global step if available
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        
        # Load is_autoregressive if available
        if 'is_autoregressive' in checkpoint:
            self.is_autoregressive = checkpoint['is_autoregressive']
        
        print(f"Checkpoint loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        
        return checkpoint.get('metrics', {})
    
    def _get_prediction_samples(self, dataloader, num_samples=5):
        """
        Get prediction samples for visualization
        
        Args:
            dataloader: DataLoader to get samples from
            num_samples: Number of samples to get
            
        Returns:
            list: List of [input, prediction, target] samples
        """
        self.model.eval()
        samples = []
        
        # Get a batch of data
        batch = next(iter(dataloader))
        inputs_batch = self._prepare_batch(batch)
        
        # Get raw inputs and targets for display
        if isinstance(batch, dict):
            raw_inputs = batch["input"][:num_samples]
            raw_targets = batch["output"][:num_samples]
        else:
            raw_inputs = [example["input"] for example in batch][:num_samples]
            raw_targets = [example["output"] for example in batch][:num_samples]
        
        # Generate predictions
        with torch.no_grad():
            if self.is_autoregressive:
                # For autoregressive models
                input_ids = inputs_batch["input_ids"][:num_samples]
                attention_mask = inputs_batch["attention_mask"][:num_samples]
                
                # Get input lengths for each sample
                input_lengths = []
                for i, inp in enumerate(raw_inputs[:num_samples]):
                    input_length = len(self.tokenizer(inp, return_tensors="pt").input_ids[0])
                    input_lengths.append(input_length)
                
                # Generate predictions
                predictions = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.size(1) + self.config.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode predictions, removing the input prefix
                decoded_preds = []
                for i, pred in enumerate(predictions):
                    # Get only the generated part (after the input)
                    generated_part = pred[input_lengths[i]:]
                    decoded = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    decoded_preds.append(decoded)
            else:
                # For sequence-to-sequence models
                input_ids = inputs_batch["input_ids"][:num_samples]
                attention_mask = inputs_batch["attention_mask"][:num_samples]
                
                # Generate predictions
                predictions = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
                
                # Decode predictions
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Create samples
        for i in range(min(num_samples, len(raw_inputs))):
            samples.append([raw_inputs[i], decoded_preds[i], raw_targets[i]])
        
        return samples

    def train_with_huggingface_sft(self, train_dataset, val_dataset=None):
        """
        Train the model using Hugging Face's SFTTrainer from the TRL library
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            dict: Training results
        """
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError(
                "To use Hugging Face's SFTTrainer, you need to install the TRL library. "
                "You can install it with `pip install trl`."
            )
        
        print(f"Starting training with Hugging Face's SFTTrainer with {len(train_dataset)} examples")
        
        # Convert config to SFTConfig
        sft_config = SFTConfig(
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1,
            max_grad_norm=self.config.max_grad_norm if hasattr(self.config, 'max_grad_norm') else 1.0,
            optim=self.config.optimizer if hasattr(self.config, 'optimizer') else "adamw_torch",
            output_dir=self.config.checkpoint_dir,
            max_seq_length=self.config.max_input_length if hasattr(self.config, 'max_input_length') else 512,
            save_strategy="epoch",
            logging_steps=100,
            remove_unused_columns=False,  # Important for custom datasets
        )
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text" if hasattr(train_dataset, "features") and "text" in train_dataset.features else None,
        )
        
        # Train the model
        train_results = trainer.train()
        
        # Save the final model
        trainer.save_model(os.path.join(self.config.checkpoint_dir, "final_model"))
        
        # Save the model state dict for compatibility with TextWorldRLTrainer
        self.save_model_only(os.path.join(self.config.checkpoint_dir, "final_model_state_dict.pt"))
        
        return train_results

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



