# train.py
from collections import defaultdict
import torch
import torch.nn as nn
import logging
import os
import gc
import wandb
from datetime import datetime, timedelta
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

from transformers import (
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass, asdict
import os
import warnings
from torch.amp import autocast, GradScaler
import time
from datetime import datetime
import GPUtil
import json
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys
import signal
import atexit
from pathlib import Path
from model.training_config import TrainingConfig, DynamicClassWeights, EarlyStopping
from model.data.sampler import MultilabelStratifiedSampler
from model.evaluation.threshold_optimizer import ThresholdOptimizer
from model.language_aware_transformer import LanguageAwareTransformer

# Set environment variables if not already set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

# Initialize global variables with None
_model = None
_optimizer = None
_scheduler = None
_cleanup_handlers = []

def register_cleanup(handler):
    """Register cleanup handlers that will be called on exit"""
    _cleanup_handlers.append(handler)

def cleanup():
    """Cleanup function to be called on exit"""
    global _model, _optimizer, _scheduler
    
    print("\nPerforming cleanup...")
    
    # Call all registered cleanup handlers
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Warning: Cleanup handler failed: {str(e)}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {str(e)}")
    
    # Delete model and optimizer
    try:
        if _model is not None:
            del _model
        if _optimizer is not None:
            del _optimizer
        if _scheduler is not None:
            del _scheduler
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup function
atexit.register(cleanup)

# Handle termination signals
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 32
    grad_accum_steps: int = 4
    epochs: int = 10
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.01
    device: str = None
    fp16: bool = True
    mixed_precision: str = 'bf16'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 100
    freeze_layers: int = 0
    
    def __post_init__(self):
        try:
            # Load language-specific weights
            weights_path = Path('weights/language_class_weights.json')
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
                
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
                self.lang_weights = weights_data['weights']
        except Exception as e:
            print(f"Error loading language weights: {str(e)}")
            print("Using default weights...")
            self.lang_weights = self._get_default_weights()
            
        # Set device with error handling
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                self.device = torch.device('cuda')
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {str(e)}")
                self.device = torch.device('cpu')
        
        # Set TF32 if requested and available
        if torch.cuda.is_available() and self.tensor_float_32:
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                print("Warning: TF32 not supported on this GPU. Disabling.")
                self.tensor_float_32 = False
            
        # Define toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
    
    def _get_default_weights(self):
        """Provide default weights if loading fails"""
        return {
            'en': {label: {'0': 0.5, '1': 1.0} for label in self.toxicity_labels},
            'default': {label: {'0': 0.5, '1': 1.0} for label in self.toxicity_labels}
        }

def init_model(config):
    """Initialize model with error handling"""
    global _model
    
    try:
        # Initialize custom model
        _model = LanguageAwareTransformer(
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            model_name=config.model_name,
            dropout=config.model_dropout
        )
        
        # Enable gradient checkpointing if requested
        if config.activation_checkpointing:
            _model.gradient_checkpointing_enable()
        
        # Move model to device
        _model = _model.to(config.device)
        
        return _model
        
    except Exception as e:
        print(f"Fatal error initializing model: {str(e)}")
        raise

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, is_best=False):
    """Save training checkpoint with error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'config': asdict(config)
        }
        
        # Create weights directory if it doesn't exist
        Path('weights').mkdir(exist_ok=True)
        
        # Save latest checkpoint
        checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = Path('weights') / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
            # Log best model to wandb if initialized
            if wandb.run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"best-model-auc{metrics['auc']:.4f}",
                        type="model",
                        description=f"Best model checkpoint with AUC {metrics['auc']:.4f}"
                    )
                    artifact.add_file(str(best_path))
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Warning: Could not log best model to wandb: {str(e)}")
        
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {str(e)}")
        # Try to save with pickle protocol 4 as fallback
        try:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False, pickle_protocol=4)
        except Exception as e2:
            print(f"Error: Could not save checkpoint even with fallback method: {str(e2)}")

def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load training checkpoint with error handling"""
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0  # Start from epoch 0
        
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # Load model state with error handling
        try:
            if config.ddp:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load model state: {str(e)}")
            return 0, 0
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {str(e)}")
        
        # Load scheduler state if it exists
        if scheduler and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {str(e)}")
        
        # Load scaler state if it exists
        if scaler and checkpoint['scaler_state_dict']:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scaler state: {str(e)}")
        
        return checkpoint['epoch'] + 1, checkpoint.get('metrics', {}).get('auc', 0)
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return 0, 0

def calculate_lang_specific_loss(batch, weighted_loss, device):
    """Helper function to calculate language-specific losses with proper error handling"""
    lang_losses = {}
    try:
        # Get batch size and number of classes
        batch_size = weighted_loss.size(0)
        num_classes = weighted_loss.size(1) if weighted_loss.dim() > 1 else 1
        
        # Convert batch['lang'] to list if it's a tensor
        langs = batch['lang']
        if isinstance(langs, torch.Tensor):
            langs = langs.tolist()
        
        for lang in set(langs):  # Use set for unique languages
            if lang not in lang_losses:
                lang_losses[lang] = []
            
            try:
                # Create mask with correct shape
                lang_mask = torch.tensor([l == lang for l in langs], 
                                       device=device, dtype=torch.bool)
                
                # Skip if no samples for this language
                if not lang_mask.any():
                    continue
                
                # Ensure mask has correct shape for broadcasting
                if weighted_loss.dim() > 1:
                    # For 2D tensor [batch_size, num_classes]
                    lang_mask = lang_mask.view(-1, 1).expand(-1, num_classes)
                
                # Apply mask and calculate mean loss
                masked_loss = weighted_loss[lang_mask].view(-1, num_classes)
                if len(masked_loss) > 0:
                    lang_loss = masked_loss.mean().item()
                    lang_losses[lang].append(lang_loss)
            
            except Exception as e:
                print(f"Warning: Error processing language {lang}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Error in language-specific loss calculation: {str(e)}")
        return {}
    
    return lang_losses

def train(model, train_loader, val_loader, config):
    """Training loop with enhanced optimization and detailed logging"""
    model.train()
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"Total epochs: {config.epochs}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    print("="*50 + "\n")
    
    # Initialize monitoring metrics
    peak_memory = 0
    grad_norms_history = []
    
    def get_memory_stats():
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
        
        try:
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,  # Convert to GB
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'peak': torch.cuda.max_memory_allocated() / 1e9
            }
        except Exception as e:
            print(f"Warning: Could not get memory stats: {str(e)}")
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
    
    def get_class_grad_norm(model):
        """Calculate gradient norm for each class head"""
        try:
            classifier = model.module.output if hasattr(model, 'module') else model.output
            if not hasattr(classifier, 'weight') or classifier.weight.grad is None:
                return None
            
            # Shape: [num_classes, hidden_dim]
            class_grads = classifier.weight.grad
            
            # Calculate norm for each class
            class_norms = torch.norm(class_grads, dim=1)
            return class_norms.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Could not calculate class gradient norms: {str(e)}")
            return None
    
    def get_layer_grad_norms(model):
        """Calculate detailed gradient statistics for different layer groups"""
        try:
            grad_stats = {}
            
            # Get all parameter gradients
            grad_flow = [(name, param.grad.abs().mean().item()) 
                        for name, param in model.named_parameters() 
                        if param.grad is not None]
            
            # Check for problematic gradients
            for name, grad in grad_flow:
                if grad < 1e-7:
                    logger.warning(f"Vanishing gradient in {name}: {grad:.2e}")
                elif grad > 1e2:
                    logger.error(f"Exploding gradient in {name}: {grad:.2e}")
                
                # Store gradient stats by layer type
                layer_type = name.split('.')[0]
                if layer_type not in grad_stats:
                    grad_stats[layer_type] = []
                grad_stats[layer_type].append(grad)
            
            # Calculate aggregate statistics per layer type
            layer_norms = {}
            for layer_type, grads in grad_stats.items():
                layer_norms[layer_type] = {
                    'mean': np.mean(grads),
                    'std': np.std(grads),
                    'min': np.min(grads),
                    'max': np.max(grads)
                }
                
                # Log severe gradient issues
                if layer_norms[layer_type]['mean'] < 1e-7:
                    logger.error(f"Severe vanishing gradients in {layer_type} layer")
                elif layer_norms[layer_type]['mean'] > 1e2:
                    logger.error(f"Severe gradient explosion in {layer_type} layer")
            
            return layer_norms
            
        except Exception as e:
            logger.error(f"Error in gradient flow monitoring: {str(e)}")
            return {}
    
    # Freeze base model layers if specified
    if config.freeze_layers > 0:
        print(f"Freezing first {config.freeze_layers} layers of base model")
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(model.base_model.encoder.layer[:config.freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    # Calculate total training steps
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")
    
    # Initialize optimizer with automatic gradient accumulation
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Initialize cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=config.fp16)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
    best_auc = 0
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-"*20)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                inputs = {
                    k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass with automatic mixed precision
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type=device_type, enabled=config.fp16):
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['labels']
                    )
                    
                    # Apply label smoothing
                    if config.label_smoothing > 0:
                        smooth_labels = inputs['labels'] * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
                        targets = smooth_labels
                    else:
                        targets = inputs['labels']
                    
                    # Numerically stabilized BCEWithLogitsLoss
                    logits = outputs['logits']
                    
                    # Prevent overflow in exponentials
                    max_val = torch.clamp(-logits, min=0)
                    
                    # Compute loss with numerical stability
                    # loss = logits - logits * targets + max_val + log(exp(-max_val) + exp(-logits - max_val))
                    loss = (logits - logits * targets + max_val + 
                           torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val)))
                    
                    # Apply class weights if available
                    if hasattr(model, 'class_weights'):
                        class_weights = model.class_weights.to(logits.device)
                        loss = loss * class_weights
                    
                    # Take mean for final loss
                    loss = loss.mean()
                    
                    # Store for logging
                    outputs['loss'] = loss
                
                # Scale loss and backward pass
                scaled_loss = loss / config.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                
                # Step if we've accumulated enough gradients or at end of epoch
                if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    try:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Get language-specific clip values
                        lang_clip_values = {
                            'en': 1.0,   # English has more diverse samples
                            'ru': 0.9,   # Russian needs tighter clipping
                            'tr': 0.95,  # Turkish is intermediate
                            'fr': 0.95,  # French similar to Turkish
                            'es': 0.95,  # Spanish similar to Turkish
                            'it': 0.9,   # Italian needs tighter clipping like Russian
                            'pt': 0.9    # Portuguese similar to Italian
                        }
                        
                        # Get unique languages in batch
                        batch_langs = set(inputs['lang'])
                        
                        # Calculate effective clip value as minimum of all languages in batch
                        effective_clip = min(
                            lang_clip_values.get(lang, 1.0) * config.max_grad_norm 
                            for lang in batch_langs
                        )
                        
                        # Clip and check gradients with language-specific value
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=effective_clip,
                            error_if_nonfinite=True  # Catch NaN gradients
                        )
                        
                        # Log language-specific clipping
                        if wandb.run is not None:
                            wandb.log({
                                'gradients/effective_clip_value': effective_clip,
                                'gradients/pre_clip_norm': grad_norm,
                                'gradients/languages_in_batch': list(batch_langs)
                            })
                        
                        # Gradient sanity checks
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.error(f"Invalid gradient norm detected: {grad_norm}, skipping update")
                            optimizer.zero_grad()
                            continue
                            
                        if grad_norm > config.max_grad_norm * 10:
                            logger.warning(f"Unusually high gradient norm: {grad_norm:.2f}")
                        
                        # Get monitoring metrics before optimizer step
                        memory_stats = get_memory_stats()
                        class_grad_norms = get_class_grad_norm(model)
                        layer_grad_norms = get_layer_grad_norms(model)
                        
                        # Optimizer and scheduler steps
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Update peak memory tracking
                        peak_memory = max(peak_memory, memory_stats['peak'])
                        
                        # Store gradient norms for analysis
                        if class_grad_norms is not None:
                            grad_norms_history.append(class_grad_norms)
                            
                        # Log gradient statistics
                        if wandb.run is not None:
                            monitoring_metrics = {
                                # Training metrics
                                'train/loss': loss.item(),
                                'train/learning_rate': scheduler.get_last_lr()[0],
                                'train/epoch': epoch,
                                'train/grad_norm': grad_norm,
                                'train/gate_values': outputs['gate_values'].mean().item(),
                                
                                # Memory metrics
                                'memory/allocated_gb': memory_stats['allocated'],
                                'memory/reserved_gb': memory_stats['reserved'],
                                'memory/peak_gb': memory_stats['peak'],
                                
                                # Gradient metrics
                                'gradients/total_norm': grad_norm
                            }
                            
                            # Add class-specific gradient norms
                            if class_grad_norms is not None:
                                for i, norm in enumerate(class_grad_norms):
                                    monitoring_metrics[f'gradients/class_{i}_norm'] = norm
                            
                            # Add layer-specific gradient norms
                            for layer, norm in layer_grad_norms.items():
                                monitoring_metrics[f'gradients/{layer}_norm'] = norm
                            
                            # Add optimizer state
                            monitoring_metrics['optimizer/learning_rate'] = optimizer.param_groups[0]['lr']
                            monitoring_metrics['optimizer/weight_decay'] = optimizer.param_groups[0]['weight_decay']
                            
                            # Log all metrics
                            wandb.log(monitoring_metrics)
                    
                    except RuntimeError as e:
                        if "node is NaN" in str(e):
                            logger.error("NaN detected in gradients, skipping update")
                            optimizer.zero_grad()
                            continue
                        else:
                            raise
                
                # Update metrics
                total_loss += loss.item()
                
                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = total_loss / (batch_idx + 1)
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    
                    # Calculate ETA
                    elapsed_time = time.time() - epoch_start_time
                    batches_remaining = len(train_loader) - (batch_idx + 1)
                    eta = (elapsed_time / (batch_idx + 1)) * batches_remaining
                    
                    print(
                        f"\rProgress: [{batch_idx+1}/{len(train_loader)} ({progress:.1f}%)] "
                        f"Loss: {avg_loss:.4f} "
                        f"LR: {lr:.2e} "
                        f"Grad Norm: {grad_norm:.2f} "
                        f"Mem: {memory_stats['allocated']:.1f}GB "
                        f"ETA: {eta/60:.1f}min",
                        end=""
                    )
                    
                    # Log to wandb with enhanced monitoring
                    if wandb.run is not None:
                        monitoring_metrics = {
                            # Training metrics
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/epoch': epoch,
                            'train/grad_norm': grad_norm,
                            'train/gate_values': outputs['gate_values'].mean().item(),
                            
                            # Memory metrics
                            'memory/allocated_gb': memory_stats['allocated'],
                            'memory/reserved_gb': memory_stats['reserved'],
                            'memory/peak_gb': memory_stats['peak'],
                            
                            # Gradient metrics
                            'gradients/total_norm': grad_norm
                        }
                        
                        # Add class-specific gradient norms
                        if class_grad_norms is not None:
                            for i, norm in enumerate(class_grad_norms):
                                monitoring_metrics[f'gradients/class_{i}_norm'] = norm
                        
                        # Add layer-specific gradient norms
                        for layer, norm in layer_grad_norms.items():
                            monitoring_metrics[f'gradients/{layer}_norm'] = norm
                        
                        # Add optimizer state
                        monitoring_metrics['optimizer/learning_rate'] = optimizer.param_groups[0]['lr']
                        monitoring_metrics['optimizer/weight_decay'] = optimizer.param_groups[0]['weight_decay']
                        
                        # Log all metrics
                        wandb.log(monitoring_metrics)
                
            except Exception as e:
                print(f"\nError in training batch: {str(e)}")
                continue
        
        # End of epoch metrics
        epoch_time = time.time() - epoch_start_time
        print(f"\n\nEpoch {epoch+1} completed in {epoch_time/60:.1f} minutes")
        print(f"Average training loss: {total_loss/len(train_loader):.4f}")
        
        # Validation
        print("\nRunning validation...")
        val_metrics = evaluate(model, val_loader, config)
        print("\nValidation Metrics:")
        print(f"Loss: {val_metrics['loss']:.4f}")
        print(f"AUC: {val_metrics['auc']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        
        # Print per-class metrics
        print("\nPer-class Metrics:")
        for label, metrics in val_metrics['class_metrics'].items():
            print(f"\n{label}:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Threshold: {metrics['threshold']:.4f}")
        
        # Check early stopping
        if early_stopping(val_metrics['auc'], epoch):
            print(f"\nEarly stopping triggered. Best AUC: {best_auc:.4f}")
            break
            
        # Update best AUC and save model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            print(f"\nNew best AUC: {best_auc:.4f} - Saving model...")
            save_model(model, config, epoch, best_auc)
        
        # Log validation metrics
        if wandb.run is not None:
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/auc': val_metrics['auc'],
                'val/epoch': epoch,
                'val/best_auc': best_auc,
                'time/epoch_minutes': epoch_time/60
            })
            
            # Log per-class metrics
            for label, metrics in val_metrics['class_metrics'].items():
                wandb.log({
                    f'val/class/{label}/auc': metrics['auc'],
                    f'val/class/{label}/f1': metrics['f1'],
                    f'val/class/{label}/threshold': metrics['threshold']
                })
        
        print("\n" + "="*50)

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
        
        # Convert labels to tensor using recommended method
        labels_tensor = torch.tensor(df[config.toxicity_labels].fillna(0).values, dtype=torch.float32)
        self.labels = labels_tensor.clone().detach()
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

class LanguageAwareWeights:
    def __init__(self, lang_stats=None):
        # Base weights from global statistics
        self.base_weights = {
            'toxic': 1.0,
            'severe_toxic': 5.4,    # Global average
            'obscene': 2.1,
            'threat': 8.7,
            'insult': 2.8,
            'identity_hate': 6.2
        }
        
        # Language-specific boosts based on prevalence analysis
        self.lang_boost = {
            'en': {'threat': 1.8, 'identity_hate': 1.5},    # EN has 2.59% threats vs 2.04% avg
            'ru': {'identity_hate': 1.6},                   # RU: 5.34% vs 5.32% avg
            'tr': {'threat': 1.4}                           # TR: 2.05% vs 2.04% avg
        }
        
        # Update weights if statistics provided
        if lang_stats:
            self._update_weights_from_stats(lang_stats)
    
    def _update_weights_from_stats(self, lang_stats):
        """Update weights based on provided language statistics"""
        try:
            # Calculate global averages
            global_stats = defaultdict(list)
            for lang_data in lang_stats.values():
                for class_name, stats in lang_data.items():
                    if 'positive_ratio' in stats:
                        global_stats[class_name].append(stats['positive_ratio'])
            
            # Update base weights based on inverse frequency
            for class_name, ratios in global_stats.items():
                avg_ratio = np.mean(ratios)
                if avg_ratio > 0:
                    self.base_weights[class_name] = min(10.0, 1.0 / avg_ratio)
            
            # Update language boosts
            for lang, lang_data in lang_stats.items():
                for class_name, stats in lang_data.items():
                    if 'positive_ratio' in stats:
                        lang_ratio = stats['positive_ratio']
                        global_ratio = np.mean(global_stats[class_name])
                        if lang_ratio > global_ratio * 1.1:  # 10% higher than average
                            boost = min(2.0, lang_ratio / global_ratio)
                            if lang not in self.lang_boost:
                                self.lang_boost[lang] = {}
                            self.lang_boost[lang][class_name] = boost
                            
        except Exception as e:
            logger.warning(f"Could not update weights from statistics: {str(e)}")
    
    def get_weights(self, lang, class_name):
        """Get the weight for a specific language and class"""
        try:
            weight = self.base_weights[class_name]
            boost = self.lang_boost.get(lang, {}).get(class_name, 1.0)
            return weight * boost
        except Exception as e:
            logger.warning(f"Error getting weight for {lang}/{class_name}: {str(e)}")
            return 1.0
    
    def get_weights_tensor(self, langs, class_names, device):
        """Get weight tensor for a batch of samples"""
        try:
            weights = torch.zeros((len(langs), len(class_names)), device=device)
            for i, lang in enumerate(langs):
                for j, class_name in enumerate(class_names):
                    weights[i, j] = self.get_weights(lang, class_name)
            return weights
        except Exception as e:
            logger.warning(f"Error creating weight tensor: {str(e)}")
            return torch.ones((len(langs), len(class_names)), device=device)

# Update WeightedFocalLoss to use LanguageAwareWeights
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, label_smoothing=0.01):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        # Language-specific alpha values for each class
        self.alpha = {
            'en': [0.2, 0.8, 0.3, 0.9, 0.4, 0.85],  # [toxic, severe, obscene, threat, insult, hate]
            'ru': [0.18, 0.82, 0.28, 0.88, 0.35, 0.83],  # Higher alpha for threat/hate
            'tr': [0.19, 0.81, 0.29, 0.89, 0.36, 0.84],  # Similar pattern to RU
            'default': [0.25] * 6  # Default balanced alpha
        }
        
        # Low-resource language boost factors
        self.lang_boost = {
            'tr': 1.2,  # Turkish has fewer samples
            'it': 1.3,  # Italian has even fewer
            'pt': 1.3   # Portuguese similar to Italian
        }
    
    def calculate_alpha_weights(self, langs, labels):
        """Calculate language-specific alpha weights for each sample"""
        try:
            batch_size = labels.size(0)
            num_classes = labels.size(1)
            alpha_weights = torch.ones_like(labels, dtype=torch.float32)
            
            for i, (lang, label_vec) in enumerate(zip(langs, labels)):
                # Get alpha values for this language
                lang_alpha = torch.tensor(
                    self.alpha.get(lang, self.alpha['default']),
                    device=labels.device,
                    dtype=torch.float32
                )
                
                # Apply language-specific boost for low-resource languages
                if lang in self.lang_boost:
                    lang_alpha = lang_alpha * self.lang_boost[lang]
                
                alpha_weights[i] = lang_alpha
            
            return alpha_weights
            
        except Exception as e:
            logger.warning(f"Error calculating alpha weights: {str(e)}")
            return torch.ones_like(labels, dtype=torch.float32)
    
    def forward(self, preds, targets, langs=None):
        """Forward pass with language-specific focal loss"""
        try:
            # Apply label smoothing
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Ensure all inputs have correct dimensions [B, C]
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            
            # Calculate BCE loss with proper reduction
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)  # [B, C]
            
            # Calculate probabilities with numerical stability
            probs = torch.sigmoid(preds)
            pt = torch.where(targets == 1, probs, 1 - probs)
            pt = torch.clamp(pt, min=1e-7, max=1.0)  # Numerical stability
            
            # Calculate focal weights
            focal_weights = (1 - pt) ** self.gamma
            
            # Get language-specific alpha weights
            if langs is not None:
                alpha_weights = self.calculate_alpha_weights(langs, targets)
                alpha_weights = alpha_weights.to(preds.device)
            else:
                alpha_weights = torch.full_like(targets, 0.25)  # Default alpha
            
            # Apply focal weighting and alpha weights
            loss = alpha_weights * focal_weights * bce_loss  # [B, C]
            
            # First sum over classes to preserve per-sample weighting, then average over batch
            return loss.sum(dim=-1).mean()
            
        except Exception as e:
            logger.error(f"Error in focal loss calculation: {str(e)}")
            # Fallback to simple BCE loss
            return nn.BCEWithLogitsLoss()(preds, targets)

def calculate_metrics(labels, probs, thresholds, class_names):
    """Calculate classification metrics with robust handling of edge cases"""
    # Handle different threshold formats
    if isinstance(thresholds, dict):
        default_thresh = np.array([thresholds.get('en', {}).get(name, 0.5) for name in class_names])
    else:
        default_thresh = np.array(thresholds)
    
    # Ensure thresholds has correct shape
    if default_thresh.ndim == 1:
        default_thresh = default_thresh.reshape(1, -1)
    
    # Make predictions
    binary_preds = (probs > default_thresh).astype(int)
    
    # Calculate AUC with error handling
    try:
        auc = roc_auc_score(labels, probs, average='macro')
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {str(e)}")
        auc = 0.0
    
    # Manual calculation of metrics with proper handling of edge cases
    class_metrics = {}
    overall_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    for i, class_name in enumerate(class_names):
        try:
            # Calculate confusion matrix elements
            tp = np.sum((labels[:, i] == 1) & (binary_preds[:, i] == 1))
            fp = np.sum((labels[:, i] == 0) & (binary_preds[:, i] == 1))
            tn = np.sum((labels[:, i] == 0) & (binary_preds[:, i] == 0))
            fn = np.sum((labels[:, i] == 1) & (binary_preds[:, i] == 0))
            
            # Update overall metrics
            overall_metrics['tp'] += tp
            overall_metrics['fp'] += fp
            overall_metrics['tn'] += tn
            overall_metrics['fn'] += fn
            
            # Calculate metrics with explicit handling of edge cases
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = np.divide(tp, tp + fp) if (tp + fp) > 0 else 0.0
                recall = np.divide(tp, tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                specificity = np.divide(tn, tn + fp) if (tn + fp) > 0 else 0.0
                npv = np.divide(tn, tn + fn) if (tn + fn) > 0 else 0.0
            
            # Calculate class-specific AUC
            try:
                class_auc = roc_auc_score(labels[:, i], probs[:, i])
            except Exception:
                class_auc = 0.0
            
            class_metrics[class_name] = {
                'auc': float(class_auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity),
                'npv': float(npv),
                'threshold': float(default_thresh[0, i]) if default_thresh.ndim > 1 else float(default_thresh[i]),
                'support': {
                    'total': int(tp + fp + tn + fn),
                    'positive': int(tp + fn),
                    'negative': int(tn + fp),
                    'tp': int(tp),
                    'fp': int(fp),
                    'tn': int(tn),
                    'fn': int(fn)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {class_name}: {str(e)}")
            class_metrics[class_name] = {
                'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'specificity': 0.0, 'npv': 0.0, 'threshold': 0.5,
                'support': {'total': 0, 'positive': 0, 'negative': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
            }
    
    # Calculate overall metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        overall_precision = (
            np.divide(overall_metrics['tp'], overall_metrics['tp'] + overall_metrics['fp'])
            if (overall_metrics['tp'] + overall_metrics['fp']) > 0 else 0.0
        )
        overall_recall = (
            np.divide(overall_metrics['tp'], overall_metrics['tp'] + overall_metrics['fn'])
            if (overall_metrics['tp'] + overall_metrics['fn']) > 0 else 0.0
        )
        overall_f1 = (
            2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0.0
        )
    
    return {
        'auc': float(auc),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'class_metrics': class_metrics,
        'support': {
            'total': sum(m['support']['total'] for m in class_metrics.values()),
            'by_class': {name: m['support'] for name, m in class_metrics.items()}
        }
    }

def evaluate(model, loader, config):
    """Evaluation function for language-aware model"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_langs = []
    
    try:
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                inputs = {
                    k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
                
                # Accumulate results
                total_loss += outputs['loss'].item() if outputs['loss'] is not None else 0
                all_labels.append(inputs['labels'].cpu().numpy())
                all_probs.append(outputs['probabilities'].cpu().numpy())
                all_langs.extend(inputs['lang'])
        
        # Concatenate results
        try:
            labels = np.concatenate(all_labels)
            probs = np.concatenate(all_probs)
        except Exception as e:
            print(f"Error concatenating results: {str(e)}")
            return {'loss': float('inf'), 'auc': 0.0, 'class_metrics': {}}
        
        # Optimize thresholds per language
        try:
            threshold_optimizer = ThresholdOptimizer(
                min_samples=50,
                class_names=config.toxicity_labels
            )
            thresholds = threshold_optimizer.optimize(
                y_true=labels,
                y_pred=probs,
                languages=all_langs,
                class_names=config.toxicity_labels
            )
        except Exception as e:
            print(f"Error in threshold optimization: {str(e)}")
            # Fallback to default thresholds
            thresholds = {
                'en': {label: 0.5 for label in config.toxicity_labels}
            }
        
        # Calculate metrics
        try:
            metrics = calculate_metrics(
                labels=labels,
                probs=probs,
                thresholds=thresholds,
                class_names=config.toxicity_labels
            )
            metrics['loss'] = total_loss / len(loader)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            metrics = {
                'loss': total_loss / len(loader),
                'auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'class_metrics': {
                    label: {'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}
                    for label in config.toxicity_labels
                }
            }
        
        return metrics
        
    except Exception as e:
        print(f"Fatal error in evaluation: {str(e)}")
        return {
            'loss': float('inf'),
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'class_metrics': {
                label: {'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}
                for label in config.toxicity_labels
            }
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

def setup_distributed(config, rank):
    """Initialize distributed training with enhanced error handling"""
    if config.distributed:
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '23456'
            
            # Configure NCCL parameters for better error handling
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Network interface
            os.environ['NCCL_IB_TIMEOUT'] = '23'  # IB timeout in seconds
            os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
            
            # Initialize process group with timeout and error handling
            timeout = timedelta(seconds=30)  # 30 second timeout for initialization
            
            dist.init_process_group(
                backend=config.dist_backend,
                init_method=config.dist_url,
                world_size=config.world_size,
                rank=rank,
                timeout=timeout
            )
            
            # Set device with error handling
            try:
                torch.cuda.set_device(rank)
                config._device = torch.device(f'cuda:{rank}')
            except Exception as e:
                logger.error(f"Failed to set CUDA device for rank {rank}: {str(e)}")
                raise
            
            # Verify NCCL is working with barrier
            try:
                dist.barrier()  # Synchronize processes
                logger.info(f"Process group initialized successfully for rank {rank}")
            except Exception as e:
                logger.error(f"NCCL barrier failed for rank {rank}: {str(e)}")
                cleanup_distributed()
                raise
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training for rank {rank}: {str(e)}")
            cleanup_distributed()
            raise
            
    return False

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_dataloaders(train_dataset, val_dataset, config, rank=None):
    """Create optimized DataLoaders with distributed support"""
    # Create samplers
    if config.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = MultilabelStratifiedSampler(
            labels=train_dataset.labels,
            groups=train_dataset.langs,
            batch_size=config.batch_size
        )
        val_sampler = None
    
    # Create dataloaders
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
        sampler=val_sampler,
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
    
    try:
        # Get the base model from DDP if needed
        base_model = model.module if hasattr(model, 'module') else model
        
        # Get gradients for class-specific parameters
        classifier_params = list(base_model.classifier.parameters())
        if not classifier_params[-1].grad is None:  # Check if gradients exist
            class_grads = classifier_params[-1].grad  # Shape might be [num_classes, hidden_dim]
            
            # Ensure class_grads has the right shape
            if class_grads.dim() == 1:
                # If 1D, reshape to [1, hidden_dim]
                class_grads = class_grads.unsqueeze(0)
            
            # Calculate statistics along the correct dimension
            grad_stats['mean'] = class_grads.mean(dim=-1).cpu().numpy()  # Mean across hidden dimensions
            grad_stats['std'] = class_grads.std(dim=-1).cpu().numpy()    # Std across hidden dimensions
            grad_stats['norm'] = torch.norm(class_grads, dim=-1).cpu().numpy()  # Norm across hidden dimensions
            
            # Calculate gradient diversity only if we have multiple classes
            if class_grads.size(0) > 1:
                # Reshape for correlation calculation if needed
                flat_grads = class_grads.view(class_grads.size(0), -1)
                # Calculate correlation matrix
                grad_corr = torch.corrcoef(flat_grads)
                grad_stats['correlation'] = grad_corr.cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not calculate gradient stats: {str(e)}")
        return None
    
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

def train_worker(rank, config, train_dataset, val_dataset):
    """Worker function for distributed training with enhanced error handling"""
    # Setup distributed training
    is_distributed = setup_distributed(config, rank)
    
    try:
        # Create model
        model = init_model(config)
        if is_distributed:
            try:
                # Synchronize before DDP initialization
                dist.barrier()
                model = DDP(model, device_ids=[rank], find_unused_parameters=False)
                logger.info(f"DDP initialized successfully for rank {rank}")
            except Exception as e:
                logger.error(f"Failed to initialize DDP for rank {rank}: {str(e)}")
                cleanup_distributed()
                raise
        
        # Create dataloaders with proper error handling
        try:
            train_loader, val_loader = create_dataloaders(
                train_dataset, val_dataset, config, rank
            )
        except Exception as e:
            logger.error(f"Failed to create dataloaders for rank {rank}: {str(e)}")
            cleanup_distributed()
            raise
        
        # Initialize wandb only on master process
        if not is_distributed or rank == 0:
            try:
                wandb.init(
                    project="toxic-comment-classification",
                    config=config.to_serializable_dict(),
                    reinit=True
                )
            except Exception as e:
                logger.warning(f"Could not initialize wandb: {str(e)}")
        
        # Synchronize before training starts
        if is_distributed:
            try:
                dist.barrier()
                logger.info(f"All processes synchronized before training for rank {rank}")
            except Exception as e:
                logger.error(f"Synchronization failed before training for rank {rank}: {str(e)}")
                cleanup_distributed()
                raise
        
        # Train model with error handling
        try:
            train(model, train_loader, val_loader, config)
        except RuntimeError as e:
            if "NCCL" in str(e):
                logger.error(f"NCCL error during training for rank {rank}: {str(e)}")
                # Try to recover by reinitializing process group
                try:
                    cleanup_distributed()
                    is_distributed = setup_distributed(config, rank)
                    if not is_distributed:
                        raise RuntimeError("Failed to recover from NCCL error")
                except Exception as recovery_error:
                    logger.error(f"Failed to recover from NCCL error: {str(recovery_error)}")
                    raise
            else:
                raise
        
    except Exception as e:
        logger.error(f"Error in worker {rank}: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup
        if is_distributed:
            try:
                dist.barrier()  # Final sync before cleanup
                logger.info(f"Final synchronization successful for rank {rank}")
            except Exception as e:
                logger.warning(f"Final synchronization failed for rank {rank}: {str(e)}")
            
            cleanup_distributed()
        
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()

# Training Configuration
TRAINING_CONFIG = {
    "model_name": "xlm-roberta-large",
    "batch_size": 32,
    "grad_accum_steps": 2,
    "epochs": 4,
    "lr": 2e-5,
    "mixed_precision": "bf16",
    "num_workers": 12,
    "activation_checkpointing": True,
    "tensor_float_32": True,
    "gc_frequency": 500,
    "weight_decay": 0.01,
    "max_length": 128,
    "fp16": False,
    "distributed": False,
    "world_size": 1,
    # New architecture parameters
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "model_dropout": 0.1,
    "freeze_layers": 2,
    "warmup_ratio": 0.1,
    "label_smoothing": 0.01
}

def main():
    """Main training function with enhanced error handling"""
    try:
        # Initialize wandb
        try:
            wandb.init(
                project="toxic-comment-classification",
                name=f"toxic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=TRAINING_CONFIG
            )
            print("Initialized wandb logging")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {str(e)}")
        
        # Initialize config with parameters from TRAINING_CONFIG
        config = TrainingConfig(**TRAINING_CONFIG)
        
        # Load datasets with error handling
        print("Loading datasets...")
        try:
            train_df = pd.read_csv("dataset/split/train.csv")
            val_df = pd.read_csv("dataset/split/val.csv")
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise
        
        # Create datasets with error handling
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
            train_dataset = ToxicDataset(train_df, tokenizer, config)
            val_dataset = ToxicDataset(val_df, tokenizer, config)
        except Exception as e:
            print(f"Error creating datasets: {str(e)}")
            raise
        
        # Create dataloaders and train
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        model = init_model(config)
        
        # Train model
        train(model, train_loader, val_loader, config)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Could not finish wandb run: {str(e)}")
        cleanup()

if __name__ == "__main__":
    # Set numerical precision options
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    # Run main with error handling
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        cleanup()
        sys.exit(1)
