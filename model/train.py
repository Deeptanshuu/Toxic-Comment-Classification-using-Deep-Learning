import torch
import torch.nn as nn
from transformers import (
    XLMRobertaForSequenceClassification, 
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, confusion_matrix
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

from training_config import TrainingConfig, DynamicClassWeights, MetricsTracker, EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 32
    grad_accum_steps: int = 4
    epochs: int = 10
    lr: float = 1e-5  # Lower learning rate for stability
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01  # Added weight decay
    max_grad_norm: float = 1.0  # Added gradient clipping
    label_smoothing: float = 0.01  # Added label smoothing
    device: str = None
    fp16: bool = True
    mixed_precision: str = 'bf16'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 100
    
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
    
    # Initialize threshold optimizer
    threshold_optimizer = ThresholdOptimizer(
        num_classes=len(config.toxicity_labels),
        languages=config.language_columns
    )
    
    # Create parameter groups with different learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
            'lr': config.lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.lr
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    # Use warmup ratio instead of fixed steps
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize scaler based on precision mode
    scaler = None
    if config.mixed_precision == 'fp16':
        scaler = GradScaler(enabled=True)
    elif config.mixed_precision == 'bf16':
        scaler = GradScaler(enabled=False)  # BF16 doesn't require scaling
    
    metrics_tracker = MetricsTracker()
    class_weights = config.class_weights
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3)
    
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
            
            # Validate distribution shift at the start of each epoch
            if epoch > 0:
                shifts = validate_distribution_shift(train_loader.dataset, val_loader.dataset)
                print("\nDistribution Shift Analysis:")
                for metric, value in shifts['token_usage'].items():
                    print(f"Token usage {metric}: {value:.4f}")
            
            running_loss = 0.0
            total_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            
            # Update weights based on validation performance every 2 epochs
            if epoch > 0 and epoch % 2 == 0:
                class_weights.update_weights_based_on_performance(val_metrics)
            
            # Initialize per-class metrics
            class_losses = {label: 0.0 for label in config.toxicity_labels}
            class_counts = {label: 0 for label in config.toxicity_labels}
            
            # Calculate ETA using metrics tracker
            eta = metrics_tracker.get_eta(epoch, config.epochs)
            
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print(f"{'='*40} Training {'='*40}")
            
            optimizer.zero_grad()
            
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
                        batch_weights = class_weights.get_weights_for_batch(batch['lang'], config.device)
                        
                        if outputs.loss.dim() == 2:
                            weighted_loss = outputs.loss * batch_weights
                        else:
                            weighted_loss = outputs.loss * batch_weights.mean(dim=0)
                            
                        loss = weighted_loss.mean() / config.grad_accum_steps
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    
                    # Track gradients with improved adaptive clipping
                    if (batch_idx + 1) % config.grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        
                        # Get adaptive max norm based on training progress
                        adaptive_max_norm = config.get_adaptive_max_norm(epoch, global_step)
                        
                        # Apply gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=adaptive_max_norm
                        )
                        
                        # Check for exploding gradients
                        if grad_norm > 1000:
                            print(f"\nExploding gradients detected: {grad_norm:.2f}")
                            print("Skipping batch and reducing learning rate")
                            optimizer.zero_grad()
                            # Reduce learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            continue
                        
                        param_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()]))
                        
                        # Log gradient metrics with improved tracking
                        wandb.log({
                            'grad/norm': grad_norm.item(),
                            'grad/param_norm': param_norm.item(),
                            'grad/ratio': grad_norm.item() / (param_norm.item() + 1e-6),
                            'grad/adaptive_max_norm': adaptive_max_norm,
                            'grad/norm_to_max_ratio': grad_norm.item() / adaptive_max_norm,
                            'grad/is_exploding': float(grad_norm > 1000)
                        }, step=global_step)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    running_loss += loss.item() * config.grad_accum_steps
                    total_loss += loss.item() * config.grad_accum_steps
                    num_batches += 1
                    global_step += 1
                    
                    avg_loss = running_loss / config.grad_accum_steps
                    running_loss = 0.0
                    
                    # Update metrics tracker
                    metrics_tracker.update_train(avg_loss)
                    
                    # Log step-level metrics
                    step_metrics = {
                        'train/step_loss': avg_loss,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/grad_norm': grad_norm.item(),
                        'train/step': global_step,
                    }
                    
                    try:
                        wandb.log(step_metrics, step=global_step)
                    except Exception as e:
                        print(f"Warning: Failed to log step metrics to wandb: {str(e)}")
                    
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
                        
                        # Prepare detailed metrics dict for less frequent logging
                        log_dict = {
                            'train/progress': progress * 100,
                            'train/epoch': epoch + progress,
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
                                class_loss = class_losses[label] / class_counts[label]
                                log_dict[f'train/loss_{label}'] = class_loss
                        
                        # Add validation metrics if available
                        if metrics_tracker.val_losses:
                            log_dict.update({
                                'val/loss': metrics_tracker.val_losses[-1],
                                'val/auc': metrics_tracker.val_aucs[-1],
                                'val/best_auc': metrics_tracker.best_auc
                            })
                        
                        # Log all metrics together
                        try:
                            wandb.log(log_dict, step=global_step)
                        except Exception as e:
                            print(f"Warning: Failed to log to wandb: {str(e)}")
                        
                        # Print progress
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
            
            # End of epoch updates
            epoch_time = time.time() - epoch_start
            metrics_tracker.update_time(epoch_time)
            epoch_avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Log epoch-level metrics
            epoch_metrics = {
                'train/epoch_avg_loss': epoch_avg_loss,
                'train/epoch': epoch + 1,
                'time/epoch_minutes': epoch_time / 60
            }
            
            try:
                wandb.log(epoch_metrics, step=global_step)
            except Exception as e:
                print(f"Warning: Failed to log epoch metrics to wandb: {str(e)}")
            
            # Validation
            print(f"\n\n{'='*40} Validation {'='*40}")
            model.eval()
            
            # Check gradient flow before validation
            grad_stats = check_class_wise_gradient_flow(model)
            if grad_stats:
                wandb.log({
                    'grad_flow/mean': grad_stats['mean'].tolist(),
                    'grad_flow/std': grad_stats['std'].tolist(),
                    'grad_flow/norm': grad_stats['norm'].tolist(),
                }, step=global_step)
            
            # Run validation with threshold optimization
            val_metrics = evaluate(model, val_loader, config)
            threshold_optimizer.optimize(
                val_preds=val_metrics['predictions'],
                val_labels=val_metrics['labels'],
                langs=val_metrics['langs']
            )
            
            # Log threshold values
            for lang in config.language_columns:
                if lang in threshold_optimizer.thresholds:
                    for i, label in enumerate(config.toxicity_labels):
                        wandb.log({
                            f'thresholds/{lang}/{label}': threshold_optimizer.thresholds[lang][i]
                        }, step=global_step)
            
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
            
            # Check early stopping
            if early_stopping(val_metrics['auc'], epoch):
                stop_reason = early_stopping.get_stop_reason()
                print(f"\nEarly stopping: {stop_reason}")
                # Log early stopping to wandb
                wandb.log({
                    'early_stopping/stopped_epoch': early_stopping.stopped_epoch,
                    'early_stopping/best_epoch': early_stopping.get_best_epoch(),
                    'early_stopping/best_auc': early_stopping.best_value
                }, step=global_step)
                break
            
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
        
        # Create cache directory
        Path('cache').mkdir(exist_ok=True)
        
        # Generate cache key based on data and config
        cache_key = f"tokenized_{len(df)}_{config.max_length}_{tokenizer.name_or_path.replace('/', '_')}"
        self.cache_file = Path('cache') / f"{cache_key}.pt"
        
        if self.cache_file.exists():
            print(f"Loading cached tokenized data from {self.cache_file}")
            cached_data = torch.load(self.cache_file, map_location='cpu')
            self.encodings = {
                k: cached_data[k].pin_memory() if torch.cuda.is_available() else cached_data[k]
                for k in ['input_ids', 'attention_mask']
            }
        else:
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
            self.encodings = {
                k: torch.cat(v).pin_memory() if torch.cuda.is_available() else torch.cat(v)
                for k, v in self.encodings.items()
            }
            
            # Save to cache
            torch.save(self.encodings, self.cache_file)
        
        # Convert labels to tensor
        self.labels = torch.FloatTensor(df[config.toxicity_labels].fillna(0).values)
        if torch.cuda.is_available():
            self.labels = self.labels.pin_memory()
        
        self.langs = df['lang'].fillna('en').values
        
        # Calculate and store feature statistics for distribution shift monitoring
        self._calculate_feature_stats()
    
    def _calculate_feature_stats(self):
        """Calculate feature statistics for monitoring distribution shift"""
        # Calculate mean token usage
        self.token_usage = (self.encodings['attention_mask'] == 1).float().mean(dim=1)
        
        # Calculate class distribution per language
        self.lang_class_dist = {}
        for lang in np.unique(self.langs):
            lang_mask = self.langs == lang
            self.lang_class_dist[lang] = {
                'mean': self.labels[lang_mask].mean(dim=0).numpy(),
                'std': self.labels[lang_mask].std(dim=0).numpy()
            }
    
    def check_distribution_shift(self, other_dataset):
        """Check for distribution shift between this dataset and another"""
        shifts = {}
        
        # Check token usage distribution
        shifts['token_usage'] = {
            'mean_diff': (self.token_usage.mean() - other_dataset.token_usage.mean()).item(),
            'std_diff': (self.token_usage.std() - other_dataset.token_usage.std()).item()
        }
        
        # Check class distribution shifts per language
        for lang in self.lang_class_dist:
            if lang in other_dataset.lang_class_dist:
                mean_diff = np.abs(
                    self.lang_class_dist[lang]['mean'] - 
                    other_dataset.lang_class_dist[lang]['mean']
                )
                std_diff = np.abs(
                    self.lang_class_dist[lang]['std'] - 
                    other_dataset.lang_class_dist[lang]['std']
                )
                shifts[f'lang_{lang}'] = {
                    'mean_diff': mean_diff.tolist(),
                    'std_diff': std_diff.tolist()
                }
        
        return shifts

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
    def __init__(self, alpha=0.25, gamma=2, label_smoothing=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = DynamicClassWeights()
    
    def forward(self, preds, targets, langs=None):
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Ensure all inputs have correct dimensions [B, C]
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Get weights with proper dimensions
        if langs is None:
            # Use default weights if no language information
            weights = self.class_weights.default_weights.to(preds.device)
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)  # [C] → [1, C]
        else:
            # Get language-specific weights
            weights = self.class_weights.get_weights_for_batch(langs, preds.device)  # [B, C]
        
        # Ensure weights have correct shape through proper broadcasting
        if weights.shape != preds.shape:
            weights = weights.expand_as(preds)  # Match dimensions exactly [B, C]
        
        # Calculate BCE loss with proper reduction
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)  # [B, C]
        
        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0)  # Numerical stability
        focal_weights = (1 - pt) ** self.gamma
        
        # Apply focal weighting and class weights
        loss = self.alpha * focal_weights * bce_loss  # [B, C]
        weighted_loss = loss * weights  # [B, C]
        
        # First sum over classes to preserve per-sample weighting, then average over batch
        return weighted_loss.sum(dim=-1).mean()

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
    
    # Calculate dynamic thresholds
    thresholds = {}
    for i, label in enumerate(config.toxicity_labels):
        fpr, tpr, thresh = roc_curve(all_targets[:, i], all_preds[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds[label] = thresh[optimal_idx]
    
    # Calculate overall metrics using dynamic thresholds
    threshold_array = np.array([thresholds[label] for label in config.toxicity_labels])
    binary_preds = (all_preds > threshold_array).astype(int)
    
    auc = roc_auc_score(all_targets, all_preds, average='macro')
    auc_weighted = roc_auc_score(all_targets, all_preds, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, binary_preds, average='macro'
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, label in enumerate(config.toxicity_labels):
        label_preds = np.concatenate(class_preds[label])
        label_targets = np.concatenate(class_targets[label])
        label_binary_preds = (label_preds > thresholds[label]).astype(int)
        
        label_auc = roc_auc_score(label_targets, label_preds)
        p, r, f, _ = precision_recall_fscore_support(
            label_targets, label_binary_preds, average='binary'
        )
        
        tn, fp, fn, tp = confusion_matrix(label_targets, label_binary_preds).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)
        
        class_metrics[label] = {
            'auc': label_auc,
            'precision': p,
            'recall': r,
            'f1': f,
            'specificity': specificity,
            'npv': npv,
            'threshold': thresholds[label],
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    return {
        'auc': auc,
        'auc_weighted': auc_weighted,
        'loss': total_loss / len(loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'thresholds': thresholds
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

class MultilabelStratifiedSampler(torch.utils.data.Sampler):
    """Samples elements maintaining balanced representation of classes per language"""
    def __init__(self, labels, groups, batch_size):
        self.labels = torch.tensor(labels)
        self.groups = np.array(groups)
        self.batch_size = batch_size
        self.unique_groups = np.unique(groups)
        self.num_classes = labels.shape[1]
        
        # Calculate weights for balanced sampling
        self.weights = self._calculate_weights()
        
        # Calculate samples per group ensuring minimum representation
        self.samples_per_group = max(2, self.batch_size // len(self.unique_groups))
        self.remainder = self.batch_size - (self.samples_per_group * len(self.unique_groups))
        
        # Validate minimum samples per group
        if self.samples_per_group < 2:
            print(f"Warning: Batch size {batch_size} too small for {len(self.unique_groups)} groups.")
            print(f"Consider increasing batch size to at least {2 * len(self.unique_groups)}")
        
    def _calculate_weights(self):
        weights = torch.ones(len(self.labels))
        for group in self.unique_groups:
            group_mask = self.groups == group
            group_labels = self.labels[group_mask]
            
            if len(group_labels) == 0:
                continue
            
            # Calculate inverse frequency for each class in this group
            class_weights = []
            for c in range(self.num_classes):
                pos_count = group_labels[:, c].sum()
                if pos_count > 0:
                    w = len(group_labels) / (2 * pos_count)
                    class_weights.append(w)
                else:
                    class_weights.append(1.0)
            
            # Apply weights to samples in this group
            group_weights = torch.tensor(class_weights).mean(dim=0)
            weights[group_mask] = group_weights
        
        return weights
    
    def __iter__(self):
        # Convert weights to probabilities
        probs = self.weights / self.weights.sum()
        
        # Generate indices ensuring each batch has samples from each group
        indices = []
        while len(indices) < len(self.labels):
            batch_indices = []
            
            # First, ensure minimum samples from each group
            for group in self.unique_groups:
                group_mask = self.groups == group
                group_probs = probs[group_mask].clone()
                
                if group_probs.sum() > 0:  # Only sample if group has samples
                    group_probs /= group_probs.sum()
                    
                    # Sample indices for this group
                    try:
                        group_indices = np.random.choice(
                            np.where(group_mask)[0],
                            size=self.samples_per_group,
                            p=group_probs.numpy(),
                            replace=False
                        )
                        batch_indices.extend(group_indices)
                    except ValueError as e:
                        print(f"Warning: Could not sample {self.samples_per_group} samples from group {group}")
                        # Fall back to sampling with replacement if necessary
                        group_indices = np.random.choice(
                            np.where(group_mask)[0],
                            size=self.samples_per_group,
                            p=group_probs.numpy(),
                            replace=True
                        )
                        batch_indices.extend(group_indices)
            
            # Add remaining samples randomly but weighted by class distribution
            if self.remainder > 0:
                remaining_probs = probs.clone()
                remaining_probs /= remaining_probs.sum()
                
                remaining_indices = np.random.choice(
                    len(self.labels),
                    size=self.remainder,
                    p=remaining_probs.numpy(),
                    replace=False
                )
                batch_indices.extend(remaining_indices)
            
            indices.extend(batch_indices)
        
        return iter(indices[:len(self.labels)])
    
    def __len__(self):
        return len(self.labels)

class ThresholdOptimizer:
    """Optimizes classification thresholds per language"""
    def __init__(self, num_classes, languages):
        self.num_classes = num_classes
        self.languages = languages
        self.thresholds = {lang: np.array([0.5] * num_classes) for lang in languages}
        
    def optimize(self, val_preds, val_labels, langs):
        """Optimize thresholds using validation data"""
        for lang in self.languages:
            lang_mask = langs == lang
            if not lang_mask.any():
                continue
                
            lang_preds = val_preds[lang_mask]
            lang_labels = val_labels[lang_mask]
            
            # Optimize threshold for each class
            for i in range(self.num_classes):
                # Skip classes with insufficient samples
                if np.sum(lang_labels[:, i]) < 100:
                    print(f"Skipping threshold optimization for {lang} class {i}: insufficient positive samples")
                    continue
                    
                try:
                    fpr, tpr, thresholds = roc_curve(lang_labels[:, i], lang_preds[:, i])
                    # Find threshold that maximizes f1 score
                    f1_scores = []
                    for t in thresholds:
                        binary_preds = (lang_preds[:, i] > t).astype(int)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            lang_labels[:, i], binary_preds, average='binary'
                        )
                        f1_scores.append(f1)
                    
                    optimal_idx = np.argmax(f1_scores)
                    self.thresholds[lang][i] = thresholds[optimal_idx]
                except:
                    continue
    
    def get_thresholds(self, langs):
        """Get thresholds for a batch of languages"""
        batch_thresholds = []
        for lang in langs:
            batch_thresholds.append(self.thresholds.get(lang, self.thresholds['en']))
        return np.array(batch_thresholds)

def create_dataloaders(train_dataset, val_dataset, config):
    """Create optimized DataLoaders with stratified sampling"""
    train_sampler = MultilabelStratifiedSampler(
        labels=train_dataset.labels,
        groups=train_dataset.langs,
        batch_size=config.batch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def check_class_wise_gradient_flow(model):
    """Monitor gradient flow for each class head"""
    grad_stats = {}
    
    # Get gradients for class-specific parameters
    classifier_params = list(model.classifier.parameters())
    if not classifier_params[-1].grad is None:  # Check if gradients exist
        class_grads = classifier_params[-1].grad  # Shape: [num_classes, hidden_dim]
        
        # Calculate statistics
        grad_stats['mean'] = class_grads.mean(dim=1).cpu().numpy()
        grad_stats['std'] = class_grads.std(dim=1).cpu().numpy()
        grad_stats['norm'] = torch.norm(class_grads, dim=1).cpu().numpy()
        
        # Calculate gradient diversity (correlation between class gradients)
        grad_corr = torch.corrcoef(class_grads)
        grad_stats['correlation'] = grad_corr.cpu().numpy()
    
    return grad_stats

def validate_distribution_shift(train_dataset, val_dataset):
    """Check for distribution shift between train and validation sets"""
    shifts = train_dataset.check_distribution_shift(val_dataset)
    
    # Log shifts to wandb
    wandb.log({
        'distribution_shift/token_usage_mean': shifts['token_usage']['mean_diff'],
        'distribution_shift/token_usage_std': shifts['token_usage']['std_diff']
    })
    
    # Log language-specific shifts
    for lang, lang_shifts in shifts.items():
        if lang.startswith('lang_'):
            for cls_idx, (mean_diff, std_diff) in enumerate(zip(
                lang_shifts['mean_diff'], lang_shifts['std_diff']
            )):
                wandb.log({
                    f'distribution_shift/{lang}_class{cls_idx}_mean': mean_diff,
                    f'distribution_shift/{lang}_class{cls_idx}_std': std_diff
                })
    
    return shifts

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
