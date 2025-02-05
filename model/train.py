import torch
import torch.nn as nn
from transformers import (
    XLMRobertaForSequenceClassification, 
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import wandb
import argparse
from dataclasses import dataclass, asdict
import os
import warnings
from torch.cuda.amp import autocast, GradScaler
import time
from datetime import datetime, timedelta
import psutil
import GPUtil
import json
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from itertools import chain
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

from training_config import TrainingConfig, DynamicClassWeights, MetricsTracker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 32  # Reduced batch size for stability
    grad_accum_steps: int = 4  # Increased grad accumulation
    epochs: int = 10  # Increased epochs
    lr: float = 1e-5  # Slightly lower learning rate for stability
    warmup_ratio: float = 0.1  # Use ratio instead of steps
    device: str = None
    fp16: bool = True
    mixed_precision: str = 'bf16'
    num_workers: int = 4  # Reduced workers
    pin_memory: bool = True
    prefetch_factor: int = 2
    activation_checkpointing: bool = False  # Disabled for simplicity
    tensor_float_32: bool = True
    gc_frequency: int = 100  # More frequent GC
    
    def __post_init__(self):
        # Load language-specific weights
        with open('weights/language_class_weights.json', 'r') as f:
            weights_data = json.load(f)
            self.lang_weights = weights_data['weights']
            
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set TF32 if requested
        if torch.cuda.is_available() and self.tensor_float_32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # Define toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)

def init_model(config):
    """Initialize model"""
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        problem_type="multi_label_classification"
    )
    
    return model.to(config.device)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # No need to check for DDP since we're not using it
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': asdict(config)
    }
    
    # Save latest checkpoint
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = Path('weights') / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        # Log best model to wandb
        artifact = wandb.Artifact(
            name=f"best-model-auc{metrics['auc']:.4f}",
            type="model",
            description=f"Best model checkpoint with AUC {metrics['auc']:.4f}"
        )
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load training checkpoint"""
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0  # Start from epoch 0
        
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # Load model state
    if config.ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if it exists
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state if it exists
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint['metrics'].get('auc', 0)

def train(model, train_loader, val_loader, config):
    """Training loop with optimized configuration"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Use warmup ratio instead of fixed steps
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler(enabled=config.fp16)
    metrics_tracker = MetricsTracker()
    class_weights = config.class_weights
    
    # Initialize progress tracking
    best_auc = 0.0
    train_start_time = time.time()
    global_step = 0
    
    print("\n" + "="*80)
    print(f"Starting training for {config.epochs} epochs...")
    print(f"Total steps: {num_training_steps:,}, Warmup steps: {num_warmup_steps:,}")
    print("="*80 + "\n")
    
    try:
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            total_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            
            # Initialize per-class metrics
            class_losses = {label: 0.0 for label in config.toxicity_labels}
            class_counts = {label: 0 for label in config.toxicity_labels}
            
            # Calculate ETA (handle first epoch gracefully)
            if epoch == 0:
                eta = "Calculating..."
            else:
                eta = metrics_tracker.get_eta(epoch, config.epochs)
            
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print(f"{'='*40} Training {'='*40}")
            
            optimizer.zero_grad()
            
            # Track batch-level metrics for wandb
            batch_metrics = {
                'train/batch_loss': [],
                'train/learning_rate': [],
                'train/grad_norm': []
            }
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % config.gc_frequency == 0:
                    torch.cuda.empty_cache()
                
                try:
                    # Move batch to device
                    inputs = {
                        'input_ids': batch['input_ids'].to(config.device, non_blocking=True),
                        'attention_mask': batch['attention_mask'].to(config.device, non_blocking=True),
                        'labels': batch['labels'].to(config.device, non_blocking=True)
                    }
                    
                    # Forward pass
                    with autocast(enabled=config.fp16):
                        outputs = model(**inputs)
                        # Get weights for current batch and ensure proper shape
                        batch_weights = class_weights.get_weights_for_batch(batch['lang'], config.device)
                        
                        # Ensure proper shape for loss weighting
                        if outputs.loss.dim() == 2:  # If loss is per class
                            weighted_loss = outputs.loss * batch_weights.unsqueeze(0)  # [B, C]
                        else:  # If loss is already reduced
                            weighted_loss = outputs.loss * batch_weights  # [C]
                            
                        loss = weighted_loss.mean() / config.grad_accum_steps
                    
                    # Track per-class losses safely
                    with torch.no_grad():
                        if weighted_loss.dim() == 2:  # [B, C]
                            class_wise_loss = weighted_loss.mean(dim=0)  # Average over batch
                        else:  # [C]
                            class_wise_loss = weighted_loss
                            
                        for i, label in enumerate(config.toxicity_labels):
                            if i < class_wise_loss.size(0):  # Ensure index is valid
                                class_losses[label] += class_wise_loss[i].item()
                                class_counts[label] += 1
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    
                    running_loss += loss.item() * config.grad_accum_steps
                    total_loss += loss.item() * config.grad_accum_steps
                    num_batches += 1
                    global_step += 1
                    
                    if (batch_idx + 1) % config.grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        avg_loss = running_loss / config.grad_accum_steps
                        running_loss = 0.0
                        
                        # Track batch metrics
                        batch_metrics['train/batch_loss'].append(avg_loss)
                        batch_metrics['train/learning_rate'].append(scheduler.get_last_lr()[0])
                        batch_metrics['train/grad_norm'].append(grad_norm.item())
                    
                        if batch_idx % 50 == 0:
                            progress = batch_idx / len(train_loader)
                            batch_time = time.time() - epoch_start
                            batch_eta = str(timedelta(seconds=int(
                                (batch_time / (batch_idx + 1)) * (len(train_loader) - batch_idx)
                            )))
                            
                            # Get GPU stats
                            try:
                                gpu_stats = get_gpu_stats() if torch.cuda.is_available() else None
                            except Exception as e:
                                print(f"Warning: Could not get GPU stats: {str(e)}")
                                gpu_stats = None
                            
                            # Log detailed metrics to wandb
                            log_dict = {
                                'train/loss': avg_loss,
                                'train/lr': scheduler.get_last_lr()[0],
                                'train/progress': progress * 100,
                                'train/epoch': epoch + progress,
                                'train/grad_norm': grad_norm.item(),
                                'train/batch_size': config.batch_size,
                                'train/global_step': global_step,
                                'time/batch_eta': batch_eta,
                                'time/training_eta': eta,
                                'time/epoch_elapsed': str(timedelta(seconds=int(batch_time))),
                                'time/total_elapsed': str(timedelta(seconds=int(time.time() - train_start_time)))
                            }
                            
                            # Add GPU metrics if available
                            if gpu_stats:
                                log_dict.update({
                                    'system/gpu_util': gpu_stats['utilization'],
                                    'system/gpu_mem': gpu_stats['memory'],
                                    'system/gpu_temp': gpu_stats['temperature']
                                })
                            
                            # Add per-class losses
                            for label in config.toxicity_labels:
                                if class_counts[label] > 0:
                                    log_dict[f'train/loss_{label}'] = class_losses[label] / class_counts[label]
                            
                            try:
                                wandb.log(log_dict, step=global_step)
                            except Exception as e:
                                print(f"Warning: Failed to log to wandb: {str(e)}")
                            
                            # Print progress with better formatting
                            print(
                                f"\r[{batch_idx:>5}/{len(train_loader)}] "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                                f"GN: {grad_norm.item():.2f} | "
                                f"BT: {batch_eta} | "
                                f"ETA: {eta}",
                                end="", flush=True
                            )
                
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {str(e)}")
                    continue
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            epoch_avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            metrics_tracker.update_time(epoch_time)
            metrics_tracker.update_train(epoch_avg_loss)
            
            # Log epoch-level batch statistics
            try:
                wandb.log({
                    'train/epoch_avg_loss': np.mean(batch_metrics['train/batch_loss']),
                    'train/epoch_avg_lr': np.mean(batch_metrics['train/learning_rate']),
                    'train/epoch_avg_grad_norm': np.mean(batch_metrics['train/grad_norm']),
                    'train/epoch': epoch + 1
                }, step=global_step)
            except Exception as e:
                print(f"Warning: Failed to log epoch metrics to wandb: {str(e)}")
            
            # Validation
            print(f"\n\n{'='*40} Validation {'='*40}")
            model.eval()
            val_metrics = evaluate(model, val_loader, config)
            
            # Update metrics and save if best
            is_best = metrics_tracker.update_validation(val_metrics)
            if is_best:
                try:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metrics=val_metrics,
                        config=config,
                        is_best=True
                    )
                    print(f"\n🏆 New best model! AUC: {val_metrics['auc']:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {str(e)}")
            
            # Log epoch metrics
            epoch_metrics = {
                'val/auc': val_metrics['auc'],
                'val/loss': val_metrics['loss'],
                'val/f1': val_metrics['f1'],
                'train/epoch_loss': epoch_avg_loss,
                'time/epoch_minutes': epoch_time / 60,
                'epoch': epoch + 1
            }
            
            # Add per-class validation metrics
            for label, metrics in val_metrics['class_metrics'].items():
                for metric_name, value in metrics.items():
                    epoch_metrics[f'val/{label}/{metric_name}'] = value
            
            try:
                wandb.log(epoch_metrics, step=global_step)
            except Exception as e:
                print(f"Warning: Failed to log validation metrics to wandb: {str(e)}")
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"{'='*80}")
            print(f"Training Loss: {epoch_avg_loss:.4f}")
            print(f"Validation - AUC: {val_metrics['auc']:.4f}, Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"Time: {epoch_time/60:.2f}m ({str(timedelta(seconds=int(epoch_time)))})")
            print(f"{'='*80}\n")
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

def get_grad_norm(model):
    """Calculate gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_gpu_stats():
    """Get GPU statistics"""
    gpu = GPUtil.getGPUs()[0]
    return {
        'utilization': gpu.load * 100,
        'memory': gpu.memoryUtil * 100,
        'temperature': gpu.temperature
    }

def save_model(model, config, epoch, auc):
    """Save model checkpoint"""
    model_save_path = f"weights/toxic_classifier_{config.model_name}"
    model.save_pretrained(model_save_path)
    
    # Log model as wandb artifact
    artifact = wandb.Artifact(
        name=f"model-epoch{epoch+1}",
        type="model",
        description=f"Model checkpoint from epoch {epoch+1} with AUC {auc:.4f}"
    )
    artifact.add_dir(model_save_path)
    wandb.log_artifact(artifact)

def log_epoch_metrics(epoch, metrics, epoch_time, eta):
    """Log epoch-level metrics to wandb"""
    wandb.log({
        'val/auc': metrics['auc'],
        'val/loss': metrics['loss'],
        'val/precision': metrics['precision'],
        'val/recall': metrics['recall'],
        'val/f1': metrics['f1'],
        'time/epoch_minutes': epoch_time / 60,
        'time/epoch_eta': eta,
        'epoch': epoch + 1
    })
    
    # Log per-language metrics
    for lang, lang_metrics in metrics['lang_metrics'].items():
        wandb.log({
            f'val/{lang}/auc': lang_metrics['auc'],
            f'val/{lang}/precision': lang_metrics['precision'],
            f'val/{lang}/recall': lang_metrics['recall'],
            f'val/{lang}/f1': lang_metrics['f1']
        })

def parse_args():
    parser = argparse.ArgumentParser(description='Train toxic comment classifier')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                      help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large',
                      help='Model name')
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default='bf16',
                      help='Mixed precision mode')
    parser.add_argument('--num_workers', type=int, default=12,
                      help='Number of dataloader workers')
    parser.add_argument('--activation_checkpointing', type=lambda x: x.lower() == 'true',
                      default=True, help='Enable activation checkpointing')
    parser.add_argument('--tensor_float_32', type=lambda x: x.lower() == 'true',
                      default=True, help='Enable TF32 on Ampere GPUs')
    parser.add_argument('--gc_frequency', type=int, default=500,
                      help='Garbage collection frequency (in batches)')
    
    args = parser.parse_args()
    
    # Set fp16 based on mixed_precision argument
    args.fp16 = args.mixed_precision in ['fp16', 'bf16']
    
    return args

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
        required_columns = ['comment_text'] + config.toxicity_labels + ['lang']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Pre-tokenize all texts for faster training
        print("Pre-tokenizing texts...")
        # Process in batches to manage memory
        batch_size = 1000
        self.encodings = {'input_ids': [], 'attention_mask': []}
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['comment_text'].fillna('').iloc[i:i+batch_size].tolist()
            batch_encodings = self.tokenizer(
                batch_texts,
                max_length=config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings['input_ids'].append(batch_encodings['input_ids'])
            self.encodings['attention_mask'].append(batch_encodings['attention_mask'])
            
            if i % 10000 == 0:
                print(f"Processed {i}/{len(df)} texts")
        
        # Concatenate all batches
        self.encodings['input_ids'] = torch.cat(self.encodings['input_ids'])
        self.encodings['attention_mask'] = torch.cat(self.encodings['attention_mask'])
        
        # Convert labels to tensor
        self.labels = torch.FloatTensor(df[config.toxicity_labels].fillna(0).values)
        self.langs = df['lang'].fillna('en').values
        
        # Pin memory for faster transfer to GPU
        if torch.cuda.is_available():
            self.encodings['input_ids'] = self.encodings['input_ids'].pin_memory()
            self.encodings['attention_mask'] = self.encodings['attention_mask'].pin_memory()
            self.labels = self.labels.pin_memory()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
            'lang': self.langs[idx]
        }
        return item

# Weighted Focal Loss
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Load language-specific weights
        with open('weights/language_class_weights.json', 'r') as f:
            weights_data = json.load(f)
            self.lang_weights = weights_data['weights']
        
        # Get list of toxicity columns in order
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Default weights as fallback (use English weights)
        self.default_weights = torch.tensor([
            self.lang_weights['en'][col]['1'] for col in self.toxicity_columns
        ])
    
    def get_weights_for_batch(self, langs, device):
        """Get language-specific weights for each sample in the batch"""
        batch_weights = []
        
        for lang in langs:
            # Get weights for this language
            lang_weights = [
                self.lang_weights[lang][col]['1'] 
                for col in self.toxicity_columns
            ]
            batch_weights.append(lang_weights)
        
        # Convert to tensor and move to device
        weights = torch.tensor(batch_weights, dtype=torch.float32).to(device)
        
        # Take mean across batch if shape mismatch (shouldn't happen but just in case)
        if len(weights.shape) > 1:
            weights = weights.mean(dim=0)
            
        return weights

    def forward(self, preds, targets, langs=None):
        if langs is None:
            # Use default weights if no language information
            weights = self.default_weights.to(preds.device)
        else:
            # Get language-specific weights
            weights = self.get_weights_for_batch(langs, preds.device)
        
        # Calculate focal loss with dynamic weights
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss) * weights
        
        return focal_loss.mean()

# Evaluation
def evaluate(model, loader, config):
    """Evaluation function with per-class metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # Track per-class predictions and targets
    class_preds = {label: [] for label in config.toxicity_labels}
    class_targets = {label: [] for label in config.toxicity_labels}
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(config.device),
                'attention_mask': batch['attention_mask'].to(config.device),
                'labels': batch['labels'].to(config.device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            targets = batch['labels'].cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
            
            # Track per-class predictions
            for i, label in enumerate(config.toxicity_labels):
                class_preds[label].append(preds[:, i])
                class_targets[label].append(targets[:, i])
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate overall metrics
    auc = roc_auc_score(all_targets, all_preds, average='macro')
    binary_preds = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, binary_preds, average='macro'
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in config.toxicity_labels:
        label_preds = np.concatenate(class_preds[label])
        label_targets = np.concatenate(class_targets[label])
        label_binary_preds = (label_preds > 0.5).astype(int)
        
        label_auc = roc_auc_score(label_targets, label_preds)
        p, r, f, _ = precision_recall_fscore_support(
            label_targets, label_binary_preds, average='binary'
        )
        
        class_metrics[label] = {
            'auc': label_auc,
            'precision': p,
            'recall': r,
            'f1': f
        }
    
    return {
        'auc': auc,
        'loss': total_loss / len(loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }

def predict_toxicity(text, model, tokenizer, config):
    """
    Predict toxicity labels for a given text.
    Returns probabilities for each toxicity type.
    """
    # Prepare model
    model.eval()
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(outputs.logits)
    
    # Convert to probabilities
    probabilities = predictions[0].cpu().numpy()
    
    # Create results dictionary
    results = {}
    for label, prob in zip(config.toxicity_labels, probabilities):
        results[label] = float(prob)
    return results

def create_dataloaders(train_dataset, val_dataset, config):
    """Create optimized DataLoaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def main():
    """Main training function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize config
        config = TrainingConfig(
            model_name=args.model_name,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            mixed_precision=args.mixed_precision,
            num_workers=args.num_workers,
            activation_checkpointing=args.activation_checkpointing,
            tensor_float_32=args.tensor_float_32,
            gc_frequency=args.gc_frequency
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize wandb
        wandb.init(
            project="toxic-comment-classification",
            config=config.to_serializable_dict()
        )
        print("Initialized wandb")
        
        # Load data
        print("Loading datasets...")
        train_df = pd.read_csv("dataset/split/train.csv")
        val_df = pd.read_csv("dataset/split/val.csv")
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        
        # Initialize model and tokenizer
        print("Initializing model and tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        model = init_model(config)
        
        # Create datasets and dataloaders
        train_dataset = ToxicDataset(train_df, tokenizer, config)
        val_dataset = ToxicDataset(val_df, tokenizer, config)
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
        
        # Train
        train(model, train_loader, val_loader, config)
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if wandb.run is not None:
            wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
