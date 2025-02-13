# train.py
import pandas as pd
import torch
import logging
import os
import gc
import wandb
from datetime import datetime
import signal
import atexit
import sys
from pathlib import Path
import numpy as np
import warnings
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    XLMRobertaTokenizer
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
        
        # Enhanced gradient checkpointing setup
        if config.activation_checkpointing:
            logger.info("Enabling gradient checkpointing for memory efficiency")
            _model.gradient_checkpointing = True
            _model.base_model.gradient_checkpointing_enable()
            _model.base_model._set_gradient_checkpointing(enable=True)
            
            # Verify checkpointing is enabled
            assert _model.base_model.is_gradient_checkpointing, "Gradient checkpointing failed to enable"
        
        _model = _model.to(config.device)
        return _model
        
    except Exception as e:
        logger.error(f"Fatal error initializing model: {str(e)}")
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

class LanguageAwareFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, lang_weights=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if lang_weights is not None:
            # Expand lang_weights to match focal_loss dimensions
            if lang_weights.dim() == 1:
                lang_weights = lang_weights.unsqueeze(1).expand(-1, focal_loss.size(1))
            focal_loss = focal_loss * lang_weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def training_step(batch, model, optimizer, scheduler, config, scaler, batch_idx):
    """Execute a single training step with gradient accumulation"""
    # Move batch to device
    batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Calculate language weights and focal parameters
    lang_weights = None
    if hasattr(config, 'lang_weights') and config.lang_weights is not None:
        weight_dict = config.lang_weights.get_weights_for_batch(
            [lang.item() for lang in batch['lang']],
            batch['labels'],
            config.device
        )
        lang_weights = weight_dict['weights']
        gamma = weight_dict['gamma']
        alpha = weight_dict['alpha']
    else:
        gamma = torch.full_like(batch['labels'], 2.0, dtype=torch.float32)
        alpha = torch.full_like(batch['labels'], 0.25, dtype=torch.float32)
    
    # Forward pass
    with config.get_autocast_context():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            lang_ids=batch['lang']
        )
        
        # Calculate loss with focal loss
        loss_fct = LanguageAwareFocalLoss(
            alpha=alpha.mean(),
            gamma=gamma.mean()
        )
        loss = loss_fct(outputs['logits'], batch['labels'].float(), lang_weights)
        # Scale loss by gradient accumulation steps
        loss = loss / config.grad_accum_steps
        outputs['loss'] = loss
        
        # Check for numerical instability
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Numerical instability detected! Loss: {loss.item()}")
            logger.error(f"Batch stats - input_ids shape: {batch['input_ids'].shape}, labels shape: {batch['labels'].shape}")
            logger.error(f"Weights stats - min: {lang_weights.min():.3f}, max: {lang_weights.max():.3f}")
            logger.error(f"Focal params - gamma: {gamma.mean():.3f}, alpha: {alpha.mean():.3f}")
            optimizer.zero_grad()
            return None
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    
    # Only update weights after accumulating enough gradients
    if (batch_idx + 1) % config.grad_accum_steps == 0:
        # Gradient clipping
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    return loss.item() * config.grad_accum_steps  # Return unscaled loss for logging

def save_checkpoint(model, optimizer, scheduler, metrics, config, epoch):
    """Save model checkpoint"""
    save_dir = Path('weights/toxic_classifier_xlm-roberta-large')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_save_path = save_dir / 'pytorch_model.bin'
    torch.save(model.state_dict(), model_save_path)
    
    # Save config
    config_save_path = save_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

def train(model, train_loader, config):
    """Train the model"""
    global _model, _optimizer, _scheduler
    _model = model
    
    logger.info("Initializing training components...")
    logger.info(f"Using gradient accumulation with {config.grad_accum_steps} steps")
    logger.info(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    
    # Initialize gradient scaler for mixed precision
    logger.info("Setting up gradient scaler...")
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        config.get_param_groups(model),
        weight_decay=config.weight_decay
    )
    _optimizer = optimizer
    
    # Calculate total steps for cosine scheduler
    total_steps = (len(train_loader) // config.grad_accum_steps) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    logger.info(f"Training schedule: {total_steps} total steps, {warmup_steps} warmup steps")
    logger.info(f"Actual number of batches per epoch: {len(train_loader)}")
    
    # Initialize cosine scheduler with warm restarts
    logger.info("Creating learning rate scheduler...")
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // config.num_cycles,
        T_mult=1,
        eta_min=config.lr * config.min_lr_ratio
    )
    _scheduler = scheduler
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    
    logger.info("Starting training loop...")
    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        num_batches = 0
        
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        logger.info("Iterating through batches...")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                if batch_idx == 0:
                    logger.info("Successfully loaded first batch")
                    logger.info(f"Batch shapes - input_ids: {batch['input_ids'].shape}, "
                              f"attention_mask: {batch['attention_mask'].shape}, "
                              f"labels: {batch['labels'].shape}")
                
                loss = training_step(batch, model, optimizer, scheduler, config, scaler, batch_idx)
                if loss is not None:
                    epoch_loss += loss
                    num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}' if loss is not None else 'N/A',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Log to wandb
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    try:
                        wandb.log({
                            'batch_loss': loss if loss is not None else 0,
                            'learning_rate': scheduler.get_last_lr()[0]
                        })
                    except Exception as e:
                        logger.warning(f"Could not log to wandb: {str(e)}")
                
                if batch_idx % 100 == 0:
                    logger.info(f"Completed {batch_idx}/{len(train_loader)} batches")
                    # Explicit memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error("Batch contents:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        logger.error(f"{k}: type={type(v)}")
                continue
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        metrics.update_train(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        try:
            save_checkpoint(model, optimizer, scheduler, metrics, config, epoch)
            logger.info(f"Saved checkpoint for epoch {epoch + 1}")
        except Exception as e:
            logger.error(f"Could not save checkpoint: {str(e)}")
        
        # Log epoch metrics
        try:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_loss': avg_epoch_loss,
                'best_auc': metrics.best_auc
            })
        except Exception as e:
            logger.error(f"Could not log epoch metrics to wandb: {str(e)}")

def create_dataloaders(train_dataset, val_dataset, config):
    """Create optimized DataLoaders"""
    logger.info("Creating data loaders...")
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of workers: {config.num_workers}")
    
    logger.info("Initializing sampler...")
    try:
        # Create sampler with dataset size
        train_sampler = MultilabelStratifiedSampler(
            labels=train_dataset.labels,
            groups=train_dataset.langs,
            batch_size=config.batch_size,
            cached_size=len(train_dataset)  # Use dataset length directly
        )
        
        logger.info("Creating DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=2 if config.num_workers > 0 else None,
            drop_last=False
        )
        
        # Validate dataloader setup
        logger.info(f"DataLoader created with {len(train_loader)} batches")
        logger.info(f"Effective samples per epoch: {len(train_loader) * config.batch_size}")
        
        return train_loader
        
    except Exception as e:
        logger.error(f"Error creating data loader: {str(e)}")
        logger.error("DataLoader creation failed with traceback:", exc_info=True)
        raise

def main():
    try:
        logger.info("Initializing training configuration...")
        # Initialize config first
        config = TrainingConfig()
        
        # Initialize wandb
        try:
            wandb.init(
                project="toxic-comment-classification",
                name=f"toxic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config.to_serializable_dict()
            )
            logger.info("Initialized wandb logging")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {str(e)}")
        
        global _model, _optimizer, _scheduler
        _model = None
        _optimizer = None
        _scheduler = None
        
        logger.info("Loading datasets...")
        try:
            train_df = pd.read_csv("dataset/split/train.csv")
            logger.info(f"Loaded train dataset with {len(train_df)} samples")
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise
        
        try:
            logger.info("Creating tokenizer and dataset...")
            tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
            train_dataset = ToxicDataset(train_df, tokenizer, config)
            logger.info("Dataset creation successful")
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise
        
        logger.info("Creating data loaders...")
        train_loader = create_dataloaders(train_dataset, None, config)
        
        logger.info("Initializing model...")
        model = init_model(config)
        
        logger.info("Starting training...")
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