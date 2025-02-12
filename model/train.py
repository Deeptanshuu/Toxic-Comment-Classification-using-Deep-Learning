# train.py
import pandas as pd
import torch
import logging
import os
import gc
import wandb
from datetime import datetime, timedelta
import time
import signal
import atexit
import sys
from pathlib import Path
import numpy as np
import warnings
import json

from transformers import (
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from model.evaluation.evaluate import ToxicDataset
from model.training_config import MetricsTracker, TrainingConfig
from model.data.sampler import MultilabelStratifiedSampler
from model.language_aware_transformer import LanguageAwareTransformer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set environment variables if not already set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="AVX2 detected")

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
    
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Warning: Cleanup handler failed: {str(e)}")
    
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {str(e)}")
    
    try:
        if _model is not None:
            del _model
        if _optimizer is not None:
            del _optimizer
        if _scheduler is not None:
            del _scheduler
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup handlers
atexit.register(cleanup)

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def init_model(config):
    """Initialize model with error handling"""
    global _model
    
    try:
        _model = LanguageAwareTransformer(
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            model_name=config.model_name,
            dropout=config.model_dropout
        )
        
        assert config.hidden_size == 1024, "XLM-R hidden size must be 1024"
        assert _model.base_model.config.num_attention_heads == 16, "Head count mismatch"
        
        if config.freeze_layers > 0:
            for param in list(_model.base_model.parameters())[:8]:
                param.requires_grad = False
        
        assert not any([p.requires_grad for p in _model.base_model.parameters()][:8]), "First 8 layers should be frozen"
        
        if config.activation_checkpointing:
            _model.gradient_checkpointing_enable()
        
        _model = _model.to(config.device)
        return _model
        
    except Exception as e:
        print(f"Fatal error initializing model: {str(e)}")
        raise

def get_grad_stats(model):
    """Calculate gradient statistics for monitoring"""
    try:
        grad_norms = []
        grad_means = []
        grad_maxs = []
        grad_mins = []
        param_names = []
        
        for name, param in model.named_parameters():
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

def train(model, train_loader, config):
    """Training loop with gradient monitoring and class-aware clipping"""
    try:
        # Initialize metrics tracker
        metrics = MetricsTracker()
        
        # Create save directory if it doesn't exist
        save_dir = Path('weights/toxic_classifier_xlm-roberta-large')
        save_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Calculate cycle lengths for cosine restarts
        total_steps = len(train_loader) * config.epochs
        first_cycle = int(total_steps / (2 ** (config.num_cycles - 1)))
        
        # Initialize cosine scheduler with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=first_cycle,  # Length of first cycle
            T_mult=2,  # Each cycle is twice as long
            eta_min=config.lr * config.min_lr_ratio  # Minimum learning rate
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
                    # Move all tensors in batch to the correct device
                    batch = {
                        k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    
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
                        # Get current learning rate
                        current_lr = scheduler.get_last_lr()[0]
                        loss_val = loss.item()
                        
                        # Calculate ETA
                        steps_done = epoch * len(train_loader) + step
                        total_steps = config.epochs * len(train_loader)
                        time_elapsed = time.time() - epoch_start
                        steps_per_sec = (step + 1) / time_elapsed if time_elapsed > 0 else 0
                        remaining_steps = total_steps - steps_done
                        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                        eta = str(timedelta(seconds=int(eta_seconds)))
                        
                        # Log to wandb
                        metrics_dict = {
                            'train/step_loss': loss_val,
                            'train/learning_rate': current_lr,
                            'train/eta': eta
                        }
                        wandb.log(metrics_dict)
                        
                        # Log to file
                        logger.info(
                            f"Epoch [{epoch+1}/{config.epochs}] "
                            f"Step [{step}/{len(train_loader)}] "
                            f"Loss: {loss_val:.4f} "
                            f"LR: {current_lr:.2e} "
                            f"ETA: {eta}"
                        )
                        
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
            
            # Save final model
            if epoch == config.epochs - 1:
                try:
                    model_save_path = save_dir / 'pytorch_model.bin'
                    torch.save(model.state_dict(), model_save_path)
                    
                    # Save config
                    config_save_path = save_dir / 'config.json'
                    with open(config_save_path, 'w') as f:
                        json.dump(config.to_serializable_dict(), f, indent=2)
                        
                    logger.info(f"Model saved successfully to {save_dir}")
                except Exception as save_error:
                    logger.error(f"Error saving model: {str(save_error)}")
        
        return {
            'loss': epoch_loss
        }
        
    except Exception as e:
        logger.error(f"Fatal error in training: {str(e)}")
        raise

def create_dataloaders(train_dataset, val_dataset, config):
    """Create optimized DataLoaders"""
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

def main():
    try:
        try:
            config = TrainingConfig(TrainingConfig.TRAINING_CONFIG)
            wandb.init(
                project="toxic-comment-classification",
                name=f"toxic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config.to_serializable_dict()
            )
            print("Initialized wandb logging")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {str(e)}")
        
        global _model, _optimizer, _scheduler
        _model = None
        _optimizer = None
        _scheduler = None
        
        print("Loading datasets...")
        try:
            train_df = pd.read_csv("dataset/split/train.csv")
            val_df = pd.read_csv("dataset/split/val.csv")
            print(f"Loaded train ({len(train_df)} samples) and val ({len(val_df)} samples) datasets")
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise
        
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
            train_dataset = ToxicDataset(train_df, tokenizer, config)
            val_dataset = ToxicDataset(val_df, tokenizer, config)
        except Exception as e:
            print(f"Error creating datasets: {str(e)}")
            raise
        
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
        model = init_model(config)
        train(model, train_loader, config)
        
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
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        cleanup()
        sys.exit(1)