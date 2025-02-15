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
import time

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
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, lang_weights=None, alpha=None, gamma=None):
        """
        Compute focal loss with language-aware weighting and per-class parameters
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Target labels [batch_size, num_classes]
            lang_weights: Optional language weights [batch_size, num_classes]
            alpha: Optional class-wise weight factor [num_classes] or [batch_size, num_classes]
            gamma: Optional focusing parameter [num_classes] or [batch_size, num_classes]
        """
        if alpha is None:
            alpha = torch.full_like(inputs, 0.25)
        if gamma is None:
            gamma = torch.full_like(inputs, 2.0)
            
        # Ensure alpha and gamma have correct shape [batch_size, num_classes]
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0).expand(inputs.size(0), -1)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(0).expand(inputs.size(0), -1)
            
        # Compute binary cross entropy without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities for focusing
        pt = torch.exp(-bce_loss)  # [batch_size, num_classes]
        
        # Compute focal weights with per-class gamma
        focal_weights = (1 - pt) ** gamma  # [batch_size, num_classes]
        
        # Apply alpha weighting per-class
        weighted_focal_loss = alpha * focal_weights * bce_loss
        
        # Apply language-specific weights if provided
        if lang_weights is not None:
            weighted_focal_loss = weighted_focal_loss * lang_weights
            
        # Reduce if needed
        if self.reduction == 'mean':
            return weighted_focal_loss.mean()
        elif self.reduction == 'sum':
            return weighted_focal_loss.sum()
        return weighted_focal_loss

def training_step(batch, model, optimizer, scheduler, config, scaler, batch_idx):
    """Execute a single training step with gradient accumulation"""
    # Move batch to device
    batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Calculate language weights and focal parameters
    lang_weights = None
    alpha = None
    gamma = None
    
    if hasattr(config, 'lang_weights') and config.lang_weights is not None:
        weight_dict = config.lang_weights.get_weights_for_batch(
            [lang.item() for lang in batch['lang']],
            batch['labels'],
            config.device
        )
        lang_weights = weight_dict['weights']  # [batch_size, num_classes]
        alpha = weight_dict['alpha']           # [num_classes]
        gamma = weight_dict['gamma']           # [num_classes]
    else:
        # Default focal parameters if no language weights
        num_classes = batch['labels'].size(1)
        alpha = torch.full((num_classes,), 0.25, device=config.device)
        gamma = torch.full((num_classes,), 2.0, device=config.device)
    
    # Forward pass
    with config.get_autocast_context():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            lang_ids=batch['lang']
        )
        
        # Calculate loss with per-class focal parameters
        loss_fct = LanguageAwareFocalLoss()
        loss = loss_fct(
            outputs['logits'],
            batch['labels'].float(),
            lang_weights=lang_weights,
            alpha=alpha,
            gamma=gamma
        )
        outputs['loss'] = loss
        
        # Check for numerical instability
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Numerical instability detected! Loss: {loss.item()}")
            logger.error(f"Batch stats - input_ids shape: {batch['input_ids'].shape}, labels shape: {batch['labels'].shape}")
            if lang_weights is not None:
                logger.error(f"Weights stats - min: {lang_weights.min():.3f}, max: {lang_weights.max():.3f}")
            logger.error(f"Focal params - gamma range: [{gamma.min():.3f}, {gamma.max():.3f}], alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
            optimizer.zero_grad()
            return None
        
        # Scale loss for gradient accumulation
        if config.grad_accum_steps > 1:
            loss = loss / config.grad_accum_steps
    
    # Backward pass with scaled loss
    scaler.scale(loss).backward()
    
    # Only update weights after accumulating enough gradients
    if (batch_idx + 1) % config.grad_accum_steps == 0:
        # Log gradient stats before clipping
        if batch_idx % 100 == 0:
            grad_stats = get_grad_stats(model)
            if grad_stats:
                logger.debug("Gradient stats before clipping:")
                for key, value in grad_stats.items():
                    logger.debug(f"{key}: {value}")
        
        # Gradient clipping
        if config.max_grad_norm > 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )
            if grad_norm.isnan() or grad_norm.isinf():
                logger.warning(f"Gradient norm is {grad_norm}, skipping optimizer step")
                optimizer.zero_grad()
                return loss.item() * config.grad_accum_steps  # Return unscaled loss for logging
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Zero gradients after optimizer step
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Step scheduler after optimization
        scheduler.step()
        
        # Log gradient stats after update
        if batch_idx % 100 == 0:
            grad_stats = get_grad_stats(model)
            if grad_stats:
                logger.debug("Gradient stats after update:")
                for key, value in grad_stats.items():
                    logger.debug(f"{key}: {value}")
    
    # Return the original (unscaled) loss for logging
    return loss.item() * config.grad_accum_steps if config.grad_accum_steps > 1 else loss.item()

def save_checkpoint(model, optimizer, scheduler, metrics, config, epoch):
    """Save model checkpoint with versioning and timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base checkpoint directory
    base_dir = Path('weights/toxic_classifier_xlm-roberta-large')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create versioned checkpoint directory
    checkpoint_dir = base_dir / f"checkpoint_epoch{epoch:02d}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    try:
        # Save model state
        model_save_path = checkpoint_dir / 'pytorch_model.bin'
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Saved model state to {model_save_path}")
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'metrics': {
                'train_loss': metrics.train_losses[-1] if metrics.train_losses else None,
                'best_auc': metrics.best_auc,
                'timestamp': timestamp
            }
        }
        state_save_path = checkpoint_dir / 'training_state.pt'
        torch.save(training_state, state_save_path)
        logger.info(f"Saved training state to {state_save_path}")
        
        # Save config
        config_save_path = checkpoint_dir / 'config.json'
        with open(config_save_path, 'w') as f:
            json.dump(config.to_serializable_dict(), f, indent=2)
        logger.info(f"Saved config to {config_save_path}")
        
        # Create latest symlink
        latest_path = base_dir / 'latest'
        target_path = checkpoint_dir.relative_to(base_dir)
        
        # Remove existing symlink and target if they exist
        try:
            if latest_path.exists():
                if latest_path.is_symlink():
                    latest_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(latest_path)
            
            # Create new symlink
            latest_path.symlink_to(target_path, target_is_directory=True)
            logger.info(f"Updated 'latest' symlink to point to {checkpoint_dir.name}")
        except Exception as e:
            logger.warning(f"Failed to create symlink: {str(e)}")
            # Continue execution even if symlink creation fails
        
        # Save checkpoint metadata
        metadata = {
            'timestamp': timestamp,
            'epoch': epoch,
            'model_size': os.path.getsize(model_save_path) / (1024 * 1024),  # Size in MB
            'git_commit': os.environ.get('GIT_COMMIT', 'unknown'),
            'training_metrics': {
                'loss': metrics.train_losses[-1] if metrics.train_losses else None,
                'best_auc': metrics.best_auc
            }
        }
        meta_save_path = checkpoint_dir / 'metadata.json'
        with open(meta_save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved checkpoint metadata to {meta_save_path}")
        
        # Cleanup old checkpoints if needed
        keep_last_n = 3  # Keep last 3 checkpoints
        all_checkpoints = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint')])
        if len(all_checkpoints) > keep_last_n:
            for old_checkpoint in all_checkpoints[:-keep_last_n]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {str(e)}")
        
        logger.info(f"Successfully saved checkpoint for epoch {epoch + 1}")
        return checkpoint_dir
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        logger.error("Checkpoint save failed with traceback:", exc_info=True)
        raise

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
    
    # Verify data loader is properly initialized
    try:
        logger.info("Verifying data loader...")
        test_batch = next(iter(train_loader))
        logger.info(f"Data loader test successful. Batch keys: {list(test_batch.keys())}")
        logger.info(f"Input shape: {test_batch['input_ids'].shape}")
        logger.info(f"Label shape: {test_batch['labels'].shape}")
    except Exception as e:
        logger.error(f"Data loader verification failed: {str(e)}")
        raise
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        num_batches = 0
        
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        
        # Create progress bar with additional metrics
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            dynamic_ncols=True,  # Adapt to terminal width
            leave=True  # Keep progress bar after completion
        )
        
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
        
        logger.info("Iterating through batches...")
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Log first batch details
                if batch_idx == 0:
                    logger.info("Successfully loaded first batch")
                    logger.info(f"Batch shapes - input_ids: {batch['input_ids'].shape}, "
                              f"attention_mask: {batch['attention_mask'].shape}, "
                              f"labels: {batch['labels'].shape}")
                    logger.info(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                
                # Execute training step
                loss = training_step(batch, model, optimizer, scheduler, config, scaler, batch_idx)
                
                if loss is not None:
                    epoch_loss += loss
                    num_batches += 1
                
                # Calculate batch processing time
                batch_time = time.time() - batch_start_time
                
                # Format loss string outside of the postfix dict
                loss_str = "N/A" if loss is None else f"{loss:.4f}"
                
                # Update progress bar with detailed metrics
                progress_bar.set_postfix({
                    'loss': loss_str,
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'batch_time': f"{batch_time:.2f}s",
                    'processed': f"{(batch_idx + 1) * config.batch_size}"
                })
                
                # Log to wandb with more frequent updates
                if (batch_idx + 1) % max(1, config.grad_accum_steps // 2) == 0:
                    try:
                        wandb.log({
                            'batch_loss': loss if loss is not None else 0,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'batch_time': batch_time,
                            'gpu_memory': torch.cuda.memory_allocated() / 1024**2
                        })
                    except Exception as e:
                        logger.warning(f"Could not log to wandb: {str(e)}")
                
                # More frequent logging for debugging
                if batch_idx % 10 == 0:
                    loss_debug_str = "N/A" if loss is None else f"{loss:.4f}"
                    logger.debug(
                        f"Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss={loss_debug_str}, "
                        f"Time={batch_time:.2f}s"
                    )
                
                # Memory management
                if batch_idx % config.gc_frequency == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                batch_start_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error("Batch contents:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    else:
                        logger.error(f"{k}: type={type(v)}")
                if torch.cuda.is_available():
                    logger.error(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
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
                'best_auc': metrics.best_auc,
                'learning_rate': scheduler.get_last_lr()[0],
                'gpu_memory': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            })
        except Exception as e:
            logger.error(f"Could not log epoch metrics to wandb: {str(e)}")

def create_dataloaders(train_dataset, val_dataset, config):
    """Create DataLoader with simplified settings"""
    logger.info("Creating data loader...")
    
    # Create sampler
    train_sampler = MultilabelStratifiedSampler(
        labels=train_dataset.labels,
        groups=train_dataset.langs,
        batch_size=config.batch_size
    )
    
    # Create DataLoader with minimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Disable multiprocessing for now
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Verify DataLoader
    logger.info("Testing DataLoader...")
    try:
        test_batch = next(iter(train_loader))
        logger.info("DataLoader test successful")
        return train_loader
    except Exception as e:
        logger.error(f"DataLoader test failed: {str(e)}")
        raise

def main():
    try:
        # Set environment variables for CUDA and multiprocessing
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
        
        logger.info("Initializing training configuration...")
        # Initialize config first
        config = TrainingConfig()
        
        # Initialize CUDA settings
        if torch.cuda.is_available():
            # Disable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Set deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Set device to current CUDA device
            torch.cuda.set_device(torch.cuda.current_device())
            
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info("Configured CUDA settings for stability")
        
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
    # Set global PyTorch settings
    torch.set_num_threads(1)  # Limit CPU threads
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        cleanup()
        sys.exit(1)