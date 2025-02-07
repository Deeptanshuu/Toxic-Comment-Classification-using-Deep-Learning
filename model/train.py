# train.py
from collections import defaultdict
import torch
import torch.nn as nn
import logging
import os
import gc
import tqdm
import wandb
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

from transformers import (
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass, asdict
import os
import warnings
from torch.amp import autocast, GradScaler
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
from model.training_config import MetricsTracker, TrainingConfig, EarlyStopping
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
        
        # Validate model configuration
        assert config.hidden_size == 1024, "XLM-R hidden size must be 1024"
        assert _model.base_model.config.num_attention_heads == 16, "Head count mismatch"
        
        # Freeze first 8 layers if specified
        if config.freeze_layers > 0:
            for param in list(_model.base_model.parameters())[:8]:
                param.requires_grad = False
        
        # Verify layers are frozen
        assert not any([p.requires_grad for p in _model.base_model.parameters()][:8]), "First 8 layers should be frozen"
        
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
    """Load checkpoint with proper error handling and security measures"""
    try:
        checkpoint_path = os.path.join('weights', 'latest_checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found. Starting from scratch.")
            return 0, {}  # Return epoch 0 and empty metrics
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Import required class for safe loading
        from transformers.tokenization_utils_base import BatchEncoding
        import torch.serialization
        
        # Add BatchEncoding to safe globals
        torch.serialization.add_safe_globals([BatchEncoding])
        
        # Load checkpoint with proper error handling
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
        except RuntimeError as e:
            if "WeightsUnpickler error" in str(e):
                logger.warning("Weights-only loading failed, falling back to full loading...")
                # Fall back to full loading if weights_only fails
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=config.device,
                    weights_only=False  # Explicitly disable weights_only
                )
            else:
                raise
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if it exists
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if it exists
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch, metrics
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        logger.warning("Starting from scratch due to checkpoint loading error")
        return 0, {}

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

def get_grad_stats(model):
    """Calculate gradient statistics for monitoring"""
    try:
        grad_norms = []
        grad_means = []
        grad_maxs = []
        grad_mins = []
        param_names = []
        
        for name, param in model.parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                grad_norms.append(grad_norm)
                grad_means.append(grad.mean().item())
                grad_maxs.append(grad.max().item())
                grad_mins.append(grad.min().item())
                param_names.append(name)
        
        if grad_norms:
            return {
                'grad/max_norm': max(grad_norms),
                'grad/min_norm': min(grad_norms),
                'grad/mean_norm': sum(grad_norms) / len(grad_norms),
                'grad/max_value': max(grad_maxs),
                'grad/min_value': min(grad_mins),
                'grad/mean_value': sum(grad_means) / len(grad_means),
                'grad/largest_layer': param_names[grad_norms.index(max(grad_norms))],
                'grad/smallest_layer': param_names[grad_norms.index(min(grad_norms))]
            }
        return {}
    except Exception as e:
        logger.warning(f"Error calculating gradient stats: {str(e)}")
        return {}

def train(model, train_loader, val_loader, config):
    """Training loop with gradient monitoring and class-aware clipping"""
    try:
        # Initialize metrics tracker
        metrics = MetricsTracker()
        
        # Get parameter groups with language-specific learning rates
        param_groups = config.get_param_groups(model)
        
        # Group parameters by component for class-aware clipping
        classifier_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Separate classifier parameters for class-aware clipping
            if 'classifier' in name or 'output' in name:
                classifier_params.append(param)
            else:
                base_params.append(param)
        
        # Initialize optimizer with parameter groups
        optimizer = torch.optim.AdamW(param_groups)
        
        # Get total training steps for scheduler
        total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision == "fp16")
        
        # Training loop
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            epoch_start = time.time()
            
            # Training steps
            for step, batch in enumerate(train_loader):
                try:
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs['loss']
                    
                    # Scale loss for gradient accumulation
                    if config.grad_accum_steps > 1:
                        loss = loss / config.grad_accum_steps
                    
                    # Backward pass with gradient monitoring
                    if config.use_mixed_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Monitor gradient flow after backward
                    if step % 10 == 0:  # Log every 10 steps to avoid overhead
                        grad_stats = get_grad_stats(model)
                        wandb.log(grad_stats)
                    
                    # Gradient accumulation and optimization step
                    if (step + 1) % config.grad_accum_steps == 0:
                        if config.use_mixed_precision:
                            scaler.unscale_(optimizer)
                        
                        # Class-aware gradient clipping
                        try:
                            # Validate classifier parameters
                            if classifier_params:
                                classifier_grad_norm = torch.nn.utils.clip_grad_norm_(
                                    classifier_params, 
                                    config.max_grad_norm * 0.5,  # More aggressive clipping for classifier
                                    error_if_nonfinite=True
                                )
                            else:
                                classifier_grad_norm = torch.tensor(0.0, device=config.device)
                                logger.info("No classifier parameters found for gradient clipping")
                            
                            # Validate base parameters
                            if base_params:
                                base_grad_norm = torch.nn.utils.clip_grad_norm_(
                                    base_params,
                                    config.max_grad_norm,
                                    error_if_nonfinite=True
                                )
                            else:
                                base_grad_norm = torch.tensor(0.0, device=config.device)
                                logger.info("No base parameters found for gradient clipping")
                            
                            # Log gradient norms
                            if step % 100 == 0:
                                wandb.log({
                                    'grad/classifier_norm': classifier_grad_norm.item(),
                                    'grad/base_norm': base_grad_norm.item()
                                })
                        except RuntimeError as e:
                            if "inf" in str(e).lower() or "nan" in str(e).lower():
                                logger.error(f"Non-finite gradients detected: {str(e)}")
                                # Skip this batch to prevent training instability
                                optimizer.zero_grad()
                                continue
                            else:
                                logger.warning(f"Gradient clipping failed: {str(e)}")
                                # Fallback to global clipping with non-finite check
                                try:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(),
                                        config.max_grad_norm,
                                        error_if_nonfinite=True
                                    )
                                except RuntimeError as clip_error:
                                    if "inf" in str(clip_error).lower() or "nan" in str(clip_error).lower():
                                        logger.error(f"Non-finite gradients in fallback clipping: {str(clip_error)}")
                                        optimizer.zero_grad()
                                        continue
                                    else:
                                        raise
                        
                        # Optimization step
                        if config.use_mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    # Update metrics
                    epoch_loss += loss.item() * config.grad_accum_steps
                    
                    # Log step metrics
                    if step % 100 == 0:
                        # Get learning rates for each parameter group
                        lrs = [group['lr'] for group in optimizer.param_groups]
                        
                        metrics_dict = {
                            'train/step_loss': loss.item(),
                            'train/learning_rate/base': lrs[0],
                            'system/gpu_memory': get_gpu_stats(),
                            'system/step': step
                        }
                        
                        # Log language-specific learning rates
                        for i, lr in enumerate(lrs[1:], 1):
                            metrics_dict[f'train/learning_rate/group_{i}'] = lr
                        
                        wandb.log(metrics_dict)
                        
                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    continue
            
            # Calculate epoch metrics
            epoch_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            
            # Log epoch-level metrics
            wandb.log({
                'train/epoch_loss': epoch_loss,
                'train/epoch': epoch,
                'train/epoch_time': epoch_time
            })
            
            # Validate and save checkpoint
            val_metrics = evaluate(model, val_loader, config)
            is_best = metrics.update_validation(val_metrics)
            
            if is_best:
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
        
        return {
            'loss': epoch_loss,
            'best_auc': metrics.best_auc
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

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
    def __init__(self, df, tokenizer, config, mode='train'):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.toxicity_labels = config.toxicity_labels
        
        # Language ID mapping with strict validation
        self.lang_to_id = {
            'en': 0,  # English
            'ru': 1,  # Russian
            'tr': 2,  # Turkish
            'es': 3,  # Spanish
            'fr': 4,  # French
            'it': 5,  # Italian
            'pt': 6   # Portuguese
        }
        
        # Validate and clean language codes with error handling
        try:
            # Convert with string key fallback
            self.df['lang'] = (
                self.df['lang']
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(lambda x: x if x in self.lang_to_id else 'en')  # Use string key 'en' for fallback
            )
            
            # Convert to numerical IDs with validation
            self.lang_ids = torch.tensor(
                [self.lang_to_id[lang] for lang in self.df['lang']],
                dtype=torch.long
            )
            
            # Numerical validation
            if not self.lang_ids.ge(0).all() or self.lang_ids.ge(7).any():
                raise ValueError("Invalid language IDs detected")
            
            # Log language distribution
            lang_dist = self.df['lang'].value_counts()
            logger.info(f"Language distribution in {mode} set:")
            for lang, count in lang_dist.items():
                logger.info(f"  {lang}: {count:,} samples")
                
        except Exception as e:
            logger.error(f"Error processing language IDs: {str(e)}")
            logger.warning("Falling back to English ('en') for all samples")
            self.df['lang'] = 'en'  # Use string key for fallback
            self.lang_ids = torch.tensor([self.lang_to_id['en']] * len(df), dtype=torch.long)
        
        # Process and cache the tokenized data
        self._process_and_cache_data()
        
        # Validate final data
        self._validate_dataset()
    
    def _process_and_cache_data(self):
        """Process and cache the tokenized data with error handling"""
        try:
            # Tokenize all texts in batches
            logger.info(f"Tokenizing {len(self.df):,} texts for {self.mode} set...")
            
            # Handle empty or invalid texts
            self.df['comment_text'] = self.df['comment_text'].fillna('').astype(str)
            
            # Tokenize with proper padding and truncation
            self.encodings = self.tokenizer(
                self.df['comment_text'].tolist(),
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Convert labels to tensor
            self.labels = torch.tensor(
                self.df[self.toxicity_labels].values,
                dtype=torch.float32
            )
            
            # Log tokenization stats
            total_tokens = self.encodings['attention_mask'].sum().item()
            avg_tokens = total_tokens / len(self.df)
            logger.info(f"Average tokens per text: {avg_tokens:.1f}")
            
            # Memory management
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise RuntimeError(f"Failed to process and cache data: {str(e)}")
    
    def _validate_dataset(self):
        """Validate the dataset integrity"""
        try:
            # Check lengths match
            if len(self.lang_ids) != len(self.encodings['input_ids']):
                raise ValueError(
                    f"Length mismatch: lang_ids ({len(self.lang_ids)}) != "
                    f"input_ids ({len(self.encodings['input_ids'])})"
                )
            
            # Validate language IDs are in correct range
            if not torch.all((self.lang_ids >= 0) & (self.lang_ids < len(self.lang_to_id))):
                invalid_ids = self.lang_ids[(self.lang_ids < 0) | (self.lang_ids >= len(self.lang_to_id))]
                raise ValueError(f"Invalid language IDs found: {invalid_ids.tolist()}")
            
            logger.info(f"Dataset validation passed for {self.mode} set")
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise
    
    def __getitem__(self, idx):
        """Get a single item with proper error handling"""
        try:
            # Get language ID with string fallback
            lang = str(self.df.iloc[idx]['lang']).strip().lower()
            lang_id = self.lang_to_id.get(lang, self.lang_to_id['en'])  # Use string key 'en' for fallback
            
            # Get base encodings
            item = {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'lang_ids': torch.tensor(lang_id, dtype=torch.long),
                'labels': torch.FloatTensor(self.df.iloc[idx][self.toxicity_labels].values)
            }
            
            # Validate item
            if not torch.is_tensor(item['input_ids']):
                item['input_ids'] = torch.tensor(item['input_ids'])
            if not torch.is_tensor(item['attention_mask']):
                item['attention_mask'] = torch.tensor(item['attention_mask'])
            
            return item
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            # Return a safe fallback item using string key
            return {
                'input_ids': torch.zeros(self.config.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.config.max_length, dtype=torch.long),
                'lang_ids': torch.tensor(self.lang_to_id['en'], dtype=torch.long),  # Use string key 'en'
                'labels': torch.zeros(len(self.toxicity_labels), dtype=torch.float)
            }

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
            except:
                class_auc = 0.0
            
            class_metrics[class_name] = {
                'auc': float(class_auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity),
                'npv': float(npv),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {class_name}: {str(e)}")
            class_metrics[class_name] = {
                'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'specificity': 0.0, 'npv': 0.0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
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
    """Evaluation function with enhanced error handling and metric calculation"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_langs = []
    
    try:
        with torch.no_grad():
            for batch in loader:
                try:
                    # Move batch to device with error handling
                    inputs = {
                        k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    
                    # Forward pass with error handling
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'],
                        lang_ids=inputs['lang_ids'],
                        mode='val'
                    )
                    
                    # Accumulate results with validation
                    if outputs['loss'] is not None:
                        if not torch.isnan(outputs['loss']) and not torch.isinf(outputs['loss']):
                            total_loss += outputs['loss'].item()
                    
                    # Validate predictions before accumulating
                    probs = outputs['probabilities'].cpu()
                    if not torch.any(torch.isnan(probs)) and not torch.any(torch.isinf(probs)):
                        all_probs.append(probs.numpy())
                        all_labels.append(inputs['labels'].cpu().numpy())
                        all_langs.extend(inputs['lang_ids'].cpu().tolist())
                    else:
                        logger.warning("Invalid probabilities detected in batch, skipping")
                        
                except Exception as batch_error:
                    logger.error(f"Error processing batch: {str(batch_error)}")
                    continue
        
        # Validate accumulated results
        if not all_labels or not all_probs:
            raise ValueError("No valid predictions accumulated during evaluation")
        
        # Concatenate results with error handling
        try:
            labels = np.concatenate(all_labels)
            probs = np.concatenate(all_probs)
            
            # Validate shapes
            if labels.shape != probs.shape:
                raise ValueError(f"Shape mismatch: labels {labels.shape} != probs {probs.shape}")
            
            # Validate values
            if np.any(np.isnan(labels)) or np.any(np.isnan(probs)):
                raise ValueError("NaN values detected in concatenated arrays")
            if np.any(np.isinf(labels)) or np.any(np.isinf(probs)):
                raise ValueError("Inf values detected in concatenated arrays")
                
        except Exception as concat_error:
            logger.error(f"Error concatenating results: {str(concat_error)}")
            return _get_default_metrics(config.toxicity_labels)
        
        # Optimize thresholds with error handling
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
        except Exception as thresh_error:
            logger.error(f"Error in threshold optimization: {str(thresh_error)}")
            # Fallback to default thresholds
            thresholds = {lang: {label: 0.5 for label in config.toxicity_labels} 
                         for lang in set(all_langs)}
        
        # Calculate metrics with comprehensive error handling
        try:
            metrics = calculate_metrics(
                labels=labels,
                probs=probs,
                thresholds=thresholds,
                class_names=config.toxicity_labels
            )
            
            # Validate metric values
            if not _validate_metrics(metrics):
                raise ValueError("Invalid metric values detected")
            
            # Add loss to metrics
            metrics['loss'] = total_loss / len(loader)
            
        except Exception as metric_error:
            logger.error(f"Error calculating metrics: {str(metric_error)}")
            metrics = _get_default_metrics(config.toxicity_labels)
            metrics['loss'] = total_loss / len(loader)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Fatal error in evaluation: {str(e)}")
        return _get_default_metrics(config.toxicity_labels)

def _get_default_metrics(class_names):
    """Return safe default metrics when calculation fails"""
    return {
        'loss': float('inf'),
        'auc': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'class_metrics': {
            label: {
                'auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'specificity': 0.0,
                'npv': 0.0,
                'threshold': 0.5,
                'support': {
                    'total': 0,
                    'positive': 0,
                    'negative': 0,
                    'tp': 0,
                    'fp': 0,
                    'tn': 0,
                    'fn': 0
                }
            }
            for label in class_names
        }
    }

def _validate_metrics(metrics):
    """Validate calculated metrics for sanity"""
    try:
        # Check for required keys
        required_keys = ['auc', 'precision', 'recall', 'f1', 'class_metrics']
        if not all(key in metrics for key in required_keys):
            return False
        
        # Validate main metrics
        main_metrics = [metrics['auc'], metrics['precision'], 
                       metrics['recall'], metrics['f1']]
        if not all(isinstance(m, (int, float)) for m in main_metrics):
            return False
        if not all(0 <= m <= 1 for m in main_metrics):
            return False
        
        # Validate class metrics
        for class_metrics in metrics['class_metrics'].values():
            class_values = [
                class_metrics['auc'], 
                class_metrics['precision'],
                class_metrics['recall'], 
                class_metrics['f1']
            ]
            if not all(isinstance(m, (int, float)) for m in class_values):
                return False
            if not all(0 <= m <= 1 for m in class_values):
                return False
        
        return True
        
    except Exception:
        return False

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
    """Initialize distributed training with enhanced error handling and device management"""
    if config.distributed:
        try:
            # Set environment variables with proper fallbacks
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
            
            # Configure NCCL parameters for better error handling
            os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'INFO')
            os.environ['NCCL_SOCKET_IFNAME'] = os.environ.get('NCCL_SOCKET_IFNAME', 'eth0')
            os.environ['NCCL_IB_TIMEOUT'] = os.environ.get('NCCL_IB_TIMEOUT', '23')
            os.environ['NCCL_DEBUG_SUBSYS'] = os.environ.get('NCCL_DEBUG_SUBSYS', 'ALL')
            
            # Proper device assignment based on available GPUs
            local_rank = rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            
            # Initialize process group with timeout and error handling
            timeout = timedelta(seconds=30)
            
            try:
                dist.init_process_group(
                    backend=config.dist_backend,
                    init_method=config.dist_url,
                    world_size=config.world_size,
                    rank=rank,
                    timeout=timeout
                )
            except Exception as e:
                logger.error(f"Failed to initialize process group: {str(e)}")
                raise
            
            # Set device and verify CUDA availability
            try:
                device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(device)
                config.device = device
                
                # Verify CUDA is working
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
            except Exception as e:
                logger.error(f"Failed to set up CUDA device for rank {rank}: {str(e)}")
                raise
            
            # Verify NCCL is working with barrier
            try:
                dist.barrier()
                logger.info(f"Process group initialized successfully for rank {rank} on device cuda:{local_rank}")
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
    """Cleanup distributed training with proper error handling"""
    try:
        if dist.is_initialized():
            # Synchronize before cleanup
            try:
                dist.barrier()
            except Exception as e:
                logger.warning(f"Final barrier failed during cleanup: {str(e)}")
            
            # Destroy process group
            dist.destroy_process_group()
            
            # Reset CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device('cuda:0')
                torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error during distributed cleanup: {str(e)}")

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
                
                # Configure DDP with proper error handling
                ddp_kwargs = {
                    'device_ids': [rank % torch.cuda.device_count()],
                    'output_device': rank % torch.cuda.device_count(),
                    'find_unused_parameters': config.find_unused_parameters,
                    'broadcast_buffers': True
                }
                
                model = DDP(model, **ddp_kwargs)
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
