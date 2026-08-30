# train.py
import pandas as pd
import torch
import logging
import os
import shutil
from datetime import datetime
import gc
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

from sklearn.metrics import roc_auc_score
from transformers import XLMRobertaTokenizerFast
# Not re-exported from the transformers top level in 4.48
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.evaluation.evaluate import ToxicDataset
from model.training_config import MetricsTracker, TrainingConfig
from model.data.sampler import MultilabelStratifiedSampler
from model.data.collate import DynamicPadCollator
from model.language_aware_transformer import LanguageAwareTransformer, SUPPORTED_LANGUAGES
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

os.makedirs('logs', exist_ok=True)

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
# Dataloader workers fork after the fast tokenizer has been used; its own Rust
# thread pool must be off or the fork warns and can deadlock.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="AVX2 detected")

# Language id -> code, taken from the single source of truth shared by the model
# and by ToxicDataset (model/evaluation/evaluate.py builds the same mapping).
# DynamicClassWeights keys everything by string code, so the integer ids that
# come out of the batch must be translated or every sample is silently skipped.
ID_TO_LANG = [lang for lang, _ in sorted(SUPPORTED_LANGUAGES.items(), key=lambda kv: kv[1])]

# Abort instead of grinding on if consecutive batches keep raising
MAX_CONSECUTIVE_BATCH_FAILURES = 5

# Initialize global variables with None
_model = None
_optimizer = None
_scheduler = None
_writer = None
_cleanup_handlers = []

def register_cleanup(handler):
    """Register cleanup handlers that will be called on exit"""
    _cleanup_handlers.append(handler)

def cleanup():
    """Cleanup function to be called on exit"""
    global _model, _optimizer, _scheduler, _writer

    print("\nPerforming cleanup...")

    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Warning: Cleanup handler failed: {str(e)}")

    if _writer is not None:
        try:
            _writer.close()
        except Exception as e:
            print(f"Warning: Could not close TensorBoard writer: {str(e)}")
        _writer = None

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {str(e)}")

    try:
        # Rebind rather than `del`: deleting the module global makes every later
        # cleanup() (atexit runs it again) fail with NameError
        _model = None
        _optimizer = None
        _scheduler = None
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


class SafeSummaryWriter:
    """TensorBoard writer that can never take the training run down with it.

    The previous run died at epoch 4 of 6 on a logging backend auth error
    (docs/KNOWN_ISSUES.md #11), which is why the headline checkpoint is a
    half-trained epoch-2 model. Every call here is wrapped; after a handful of
    consecutive failures the writer disables itself and training carries on
    with no logging at all.
    """

    MAX_FAILURES = 10

    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = None
        self.failures = 0
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except Exception as e:
            logger.warning(f"Could not start TensorBoard writer ({str(e)}); continuing without it")

    @property
    def enabled(self):
        return self.writer is not None

    def _guard(self, fn, what):
        if self.writer is None:
            return
        try:
            fn()
            self.failures = 0
        except Exception as e:
            self.failures += 1
            logger.warning(f"TensorBoard {what} failed ({str(e)}); training continues")
            if self.failures >= self.MAX_FAILURES:
                logger.warning(
                    f"Disabling TensorBoard after {self.failures} consecutive failures"
                )
                try:
                    self.writer.close()
                except Exception:
                    pass
                self.writer = None

    def add_scalar(self, tag, value, step):
        if value is None:
            return
        self._guard(lambda: self.writer.add_scalar(tag, float(value), step), f"add_scalar({tag})")

    def add_scalars(self, values, step):
        for tag, value in values.items():
            self.add_scalar(tag, value, step)

    def add_text(self, tag, text, step=0):
        self._guard(lambda: self.writer.add_text(tag, text, step), f"add_text({tag})")

    def flush(self):
        self._guard(lambda: self.writer.flush(), "flush")

    def close(self):
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception as e:
            logger.warning(f"TensorBoard close failed ({str(e)})")
        self.writer = None



# Fan out to MLflow as well as TensorBoard. TensorBoard keeps the dense
# per-step curves; MLflow records params, tags and artifacts so runs (in
# particular the lang-conditioning ablation) can actually be compared.
# The two backends fail independently; neither can take the run down.
from model.tracking import RunTracker as SafeSummaryWriter  # noqa: E402,F811

def unwrap_model(model):
    """Return the underlying module, whether or not it is DataParallel-wrapped"""
    return model.module if isinstance(model, nn.DataParallel) else model


def describe_trainable_parameters(model):
    """Count trainable vs frozen parameters, overall and for the base encoder"""
    base = model.base_model
    stats = {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'base_total': sum(p.numel() for p in base.parameters()),
        'base_trainable': sum(p.numel() for p in base.parameters() if p.requires_grad),
        'embeddings_total': sum(p.numel() for p in base.embeddings.parameters()),
        'embeddings_trainable': sum(
            p.numel() for p in base.embeddings.parameters() if p.requires_grad
        ),
    }
    stats['frozen'] = stats['total'] - stats['trainable']
    stats['base_frozen'] = stats['base_total'] - stats['base_trainable']
    return stats


def init_model(config):
    """Initialize model with error handling"""
    global _model

    try:
        model = LanguageAwareTransformer(
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            model_name=config.model_name,
            dropout=config.model_dropout,
            disable_lang_conditioning=getattr(config, 'disable_lang_conditioning', False)
        )
        if getattr(config, 'disable_lang_conditioning', False):
            logger.warning("ABLATION RUN: language conditioning is disabled in the model")

        assert config.hidden_size == 1024, "XLM-R hidden size must be 1024"
        assert model.base_model.config.num_attention_heads == 16, "Head count mismatch"

        # Freeze by module name, not by slicing the parameter list. The old code
        # froze list(base_model.parameters())[:8], which happened to be the
        # embeddings plus part of layer 0 (issue #4). Freezing the embeddings is
        # worth keeping - the word-embedding matrix is 256M of the 560M base
        # parameters and its AdamW update dominates the step - so it now has its
        # own flag, and freeze_layers means encoder layers.
        if config.freeze_embeddings:
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False

        if config.freeze_layers > 0:
            for layer in model.base_model.encoder.layer[:config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        stats = describe_trainable_parameters(model)
        logger.info(
            "Parameters: %.1fM total, %.1fM trainable (%.1f%%), %.1fM frozen",
            stats['total'] / 1e6, stats['trainable'] / 1e6,
            100.0 * stats['trainable'] / max(1, stats['total']), stats['frozen'] / 1e6
        )
        logger.info(
            "Base encoder: %.1fM total, %.1fM trainable, %.1fM frozen "
            "(embeddings %.1fM, frozen=%s; encoder layers frozen: %d)",
            stats['base_total'] / 1e6, stats['base_trainable'] / 1e6,
            stats['base_frozen'] / 1e6, stats['embeddings_total'] / 1e6,
            config.freeze_embeddings, config.freeze_layers
        )
        if stats['trainable'] == 0:
            raise ValueError("Every parameter is frozen, there is nothing to train")

        # Cross-checks the freezing against the real module tree and raises if
        # the intent and the model disagree
        config.validate_model_config(model)

        # Enhanced gradient checkpointing setup
        if config.activation_checkpointing:
            logger.info("Enabling gradient checkpointing for memory efficiency")
            model.gradient_checkpointing = True
            try:
                model.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.base_model.gradient_checkpointing_enable()

            assert model.base_model.is_gradient_checkpointing, "Gradient checkpointing failed to enable"

        model = model.to(config.device)

        # DataParallel must wrap an already-placed model, and only ever when
        # there is more than one visible device. train.sh has always exported
        # CUDA_VISIBLE_DEVICES="0,1" while nothing in the code used the second
        # GPU, so it sat idle for the whole run.
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if config.data_parallel and config.device.type == 'cuda' and device_count > 1:
            logger.info(f"Wrapping model in DataParallel across {device_count} GPUs")
            model = nn.DataParallel(model)
        elif config.data_parallel:
            logger.info(f"data_parallel requested but {device_count} CUDA device(s) visible; running single-GPU")

        _model = model
        return model

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
    def __init__(self, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

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

        # Confidence in the true (hard) label. Equals exp(-BCE) for hard targets,
        # but stays correct once the targets are smoothed below.
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Label smoothing pulls the regression targets off 0/1 so the logits are
        # not driven to +-inf. Only the BCE term is smoothed; the focal
        # modulation still keys off the hard label, so at label_smoothing=0 this
        # is numerically identical to the previous implementation.
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            bce_targets = targets * (1 - eps) + 0.5 * eps
        else:
            bce_targets = targets

        # Compute binary cross entropy without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, bce_targets, reduction='none'
        )

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


def batch_lang_codes(lang_ids):
    """Map integer language ids to the string codes DynamicClassWeights keys on"""
    ids = lang_ids.detach().cpu().tolist()
    last = len(ID_TO_LANG) - 1
    return [ID_TO_LANG[min(max(int(i), 0), last)] for i in ids]


def training_step(batch, model, optimizer, scheduler, config, scaler, batch_idx,
                  loss_fct, collect_stats=False):
    """Execute a single training step with gradient accumulation.

    Returns a dict with the unscaled loss, the pre-clip gradient norm on steps
    where the optimizer ran, and optional class-weight statistics.
    """
    result = {'loss': None, 'grad_norm': None, 'stepped': False, 'weight_stats': None}

    # Move batch to device
    batch = {k: v.to(config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}

    # Calculate language weights and focal parameters
    lang_weights = None
    alpha = None
    gamma = None

    if getattr(config, 'lang_weights', None) is not None:
        weight_dict = config.lang_weights.get_weights_for_batch(
            batch_lang_codes(batch['lang']),
            batch['labels'],
            config.device
        )
        lang_weights = weight_dict['weights']  # [batch_size, num_classes]
        alpha = weight_dict['alpha']           # [num_classes]
        gamma = weight_dict['gamma']           # [num_classes]
        if collect_stats:
            result['weight_stats'] = {
                'weight_min': lang_weights.min().item(),
                'weight_max': lang_weights.max().item(),
                'weight_mean': lang_weights.mean().item(),
                'alpha_min': alpha.min().item(),
                'alpha_max': alpha.max().item(),
                'gamma_min': gamma.min().item(),
                'gamma_max': gamma.max().item(),
            }
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
            lang_ids=batch['lang']
        )

        # Calculate loss with per-class focal parameters
        loss = loss_fct(
            outputs['logits'],
            batch['labels'].float(),
            lang_weights=lang_weights,
            alpha=alpha,
            gamma=gamma
        )

        # Check for numerical instability
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Numerical instability detected! Loss: {loss.item()}")
            logger.error(f"Batch stats - input_ids shape: {batch['input_ids'].shape}, labels shape: {batch['labels'].shape}")
            if lang_weights is not None:
                logger.error(f"Weights stats - min: {lang_weights.min():.3f}, max: {lang_weights.max():.3f}")
            logger.error(f"Focal params - gamma range: [{gamma.min():.3f}, {gamma.max():.3f}], alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
            optimizer.zero_grad(set_to_none=True)
            return result

        # Scale loss for gradient accumulation
        if config.grad_accum_steps > 1:
            loss = loss / config.grad_accum_steps

    # Backward pass with scaled loss
    scaler.scale(loss).backward()

    unscaled_loss = loss.item() * config.grad_accum_steps if config.grad_accum_steps > 1 else loss.item()
    result['loss'] = unscaled_loss

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
            result['grad_norm'] = grad_norm.item()

        # Optimizer step with scaler. scaler.step is already a no-op when the
        # unscale found non-finite gradients, so there is no separate "skip"
        # path: returning early here without scaler.update() leaves the scaler
        # stuck in the unscaled state and every later unscale_ raises
        # "unscale_() has already been called". fp16 overflows on the first few
        # steps by design, so that used to poison the whole run.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before

        # Zero gradients after optimizer step
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        if stepped:
            # Step scheduler after optimization - once per optimizer step, never per epoch
            scheduler.step()
            result['stepped'] = True
        else:
            logger.debug(
                "Non-finite gradients (norm=%s); AMP skipped the step and lowered "
                "the loss scale %g -> %g",
                result['grad_norm'], scale_before, scaler.get_scale()
            )

        # Log gradient stats after update
        if batch_idx % 100 == 0:
            grad_stats = get_grad_stats(model)
            if grad_stats:
                logger.debug("Gradient stats after update:")
                for key, value in grad_stats.items():
                    logger.debug(f"{key}: {value}")

    return result


@torch.no_grad()
def validate(model, val_loader, config):
    """Run one full validation pass and score it over the whole split.

    AUC is computed once over the concatenated split, not per batch: a batch
    frequently contains a single label value for a rare class, which makes
    roc_auc_score raise. Classes that are degenerate over the *whole* split are
    skipped with a warning and left out of the macro average.
    """
    was_training = model.training
    model.eval()

    # Validation loss uses the plain focal loss with fixed alpha/gamma. The
    # training weights are derived from running batch statistics, so scoring
    # with them would both move the target between epochs and pollute those
    # statistics with validation data.
    loss_fct = LanguageAwareFocalLoss(label_smoothing=config.label_smoothing)

    total_loss = 0.0
    total_samples = 0
    prob_chunks = []
    label_chunks = []

    try:
        for batch in tqdm(val_loader, desc="Validation", dynamic_ncols=True, leave=False):
            input_ids = batch['input_ids'].to(config.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(config.device, non_blocking=True)
            labels = batch['labels'].to(config.device, non_blocking=True).float()
            lang = batch['lang'].to(config.device, non_blocking=True)

            with config.get_autocast_context():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lang_ids=lang
                )

            logits = outputs['logits'].float()
            loss = loss_fct(logits, labels)

            n = labels.size(0)
            total_loss += loss.item() * n
            total_samples += n
            prob_chunks.append(torch.sigmoid(logits).cpu().numpy())
            label_chunks.append(labels.cpu().numpy())
    finally:
        if was_training:
            model.train()

    if total_samples == 0:
        logger.warning("Validation loader produced no samples")
        return {'loss': float('inf'), 'auc': 0.0, 'per_class_auc': {}, 'skipped_classes': [],
                'num_samples': 0}

    probs = np.concatenate(prob_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    per_class_auc = {}
    skipped = []
    for i, name in enumerate(config.label_columns):
        y_true = labels[:, i]
        if np.unique(y_true).size < 2:
            logger.warning(
                f"Validation AUC for '{name}' skipped: only one label value present "
                f"in the split ({int(y_true.sum())} positives of {len(y_true)})"
            )
            skipped.append(name)
            continue
        try:
            per_class_auc[name] = float(roc_auc_score(y_true, probs[:, i]))
        except Exception as e:
            logger.warning(f"Validation AUC for '{name}' failed ({str(e)}); skipping class")
            skipped.append(name)

    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else 0.0

    return {
        'loss': total_loss / total_samples,
        'auc': macro_auc,
        'per_class_auc': per_class_auc,
        'skipped_classes': skipped,
        'num_samples': total_samples
    }


def _write_checkpoint(checkpoint_dir, model, optimizer, scheduler, metrics, config,
                      epoch, val_metrics=None):
    """Write model, optimizer, scheduler, config and metadata into a directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Always unwrap DataParallel so the checkpoint loads into a plain model
    # (a DataParallel state_dict prefixes every key with "module.")
    state_dict = unwrap_model(model).state_dict()

    model_save_path = checkpoint_dir / 'pytorch_model.bin'
    torch.save(state_dict, model_save_path)
    logger.info(f"Saved model state to {model_save_path}")

    training_state = {
        'epoch': epoch,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'metrics': {
            'train_loss': metrics.train_losses[-1] if metrics.train_losses else None,
            'val_loss': metrics.val_losses[-1] if metrics.val_losses else None,
            'val_auc': metrics.val_aucs[-1] if metrics.val_aucs else None,
            'best_auc': metrics.best_auc,
            'timestamp': timestamp
        }
    }
    state_save_path = checkpoint_dir / 'training_state.pt'
    torch.save(training_state, state_save_path)
    logger.info(f"Saved training state to {state_save_path}")

    config_save_path = checkpoint_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

    metadata = {
        'timestamp': timestamp,
        'epoch': epoch,
        'model_size': os.path.getsize(model_save_path) / (1024 * 1024),  # Size in MB
        'git_commit': os.environ.get('GIT_COMMIT', 'unknown'),
        'training_metrics': {
            'loss': metrics.train_losses[-1] if metrics.train_losses else None,
            'best_auc': metrics.best_auc
        },
        'validation_metrics': val_metrics or {}
    }
    meta_save_path = checkpoint_dir / 'metadata.json'
    with open(meta_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return checkpoint_dir


def save_checkpoint(model, optimizer, scheduler, metrics, config, epoch, val_metrics=None):
    """Save model checkpoint with versioning and timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create base checkpoint directory
    base_dir = Path(config.checkpoint_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create versioned checkpoint directory
    checkpoint_dir = base_dir / f"checkpoint_epoch{epoch:02d}_{timestamp}"

    logger.info(f"Saving checkpoint to {checkpoint_dir}")

    try:
        _write_checkpoint(checkpoint_dir, model, optimizer, scheduler, metrics,
                          config, epoch, val_metrics)

        # Only create symlink after all files are saved successfully
        latest_path = base_dir / 'latest'
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()  # Remove existing symlink if it exists

        # Create relative symlink
        os.symlink(checkpoint_dir.name, latest_path)
        logger.info(f"Updated 'latest' symlink to point to {checkpoint_dir.name}")

        # Cleanup old checkpoints if needed
        keep_last_n = 3  # Keep last 3 checkpoints
        all_checkpoints = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint')])
        if len(all_checkpoints) > keep_last_n:
            for old_checkpoint in all_checkpoints[:-keep_last_n]:
                try:
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {str(e)}")

        logger.info(f"Successfully saved checkpoint for epoch {epoch + 1}")
        return checkpoint_dir

    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        logger.error("Checkpoint save failed with traceback:", exc_info=True)
        # If checkpoint save fails, ensure we don't leave a broken symlink
        latest_path = base_dir / 'latest'
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        raise


def _best_checkpoint_dir(config):
    return Path(config.checkpoint_dir) / 'best_model'


def save_best_checkpoint(model, optimizer, scheduler, metrics, config, epoch, val_metrics):
    """Save the best-so-far model to a fixed path, separate from epoch checkpoints.

    Lives outside the checkpoint_* rotation so it is never deleted by the
    keep-last-3 cleanup, and it is the checkpoint model selection should use.
    """
    best_dir = _best_checkpoint_dir(config)
    logger.info(f"New best macro AUC {val_metrics['auc']:.4f} at epoch {epoch + 1}, saving to {best_dir}")
    # Write to a sibling temp dir and swap. Writing 2.15 GB straight over the
    # previous best leaves a ~5 s window where a crash or Ctrl-C destroys the
    # only checkpoint that matters, with no recovery.
    staging = best_dir.parent / (best_dir.name + '.tmp')
    try:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        _write_checkpoint(staging, model, optimizer, scheduler, metrics, config,
                          epoch, val_metrics)
        with open(staging / 'best.json', 'w') as f:
            json.dump({
                'epoch': epoch + 1,
                'macro_auc': val_metrics['auc'],
                'val_loss': val_metrics['loss'],
                'per_class_auc': val_metrics['per_class_auc'],
                'skipped_classes': val_metrics['skipped_classes'],
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }, f, indent=2)
        # Swap into place only once the new copy is complete on disk.
        previous = best_dir.parent / (best_dir.name + '.prev')
        shutil.rmtree(previous, ignore_errors=True)
        if best_dir.exists():
            os.replace(best_dir, previous)
        os.replace(staging, best_dir)
        shutil.rmtree(previous, ignore_errors=True)
        return best_dir
    except Exception as e:
        logger.error(f"Could not save best checkpoint: {str(e)}")
        shutil.rmtree(staging, ignore_errors=True)
        return None


def build_scheduler(optimizer, config, total_steps):
    """Build the LR schedule and report exactly what it does"""
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    if config.use_warmup:
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
            min_lr_rate=config.min_lr_ratio
        )
        logger.info(
            "LR schedule: linear warmup 0 -> %.2e over %d steps, then a single "
            "half-cosine decay to %.2e over the remaining %d steps "
            "(num_cycles=0.5, min_lr_rate=%.3f)",
            config.lr, warmup_steps, config.lr * config.min_lr_ratio,
            total_steps - warmup_steps, config.min_lr_ratio
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps // config.num_cycles),
            T_mult=1,
            eta_min=config.lr * config.min_lr_ratio
        )
        logger.info(
            "LR schedule: CosineAnnealingWarmRestarts, no warmup, T_0=%d, eta_min=%.2e",
            max(1, total_steps // config.num_cycles), config.lr * config.min_lr_ratio
        )

    return scheduler, warmup_steps


def train(model, train_loader, val_loader, config, writer=None):
    """Train the model"""
    global _model, _optimizer, _scheduler
    _model = model

    if writer is None:
        writer = SafeSummaryWriter(Path('runs') / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logger.info("Initializing training components...")
    logger.info(f"Using gradient accumulation with {config.grad_accum_steps} steps")
    logger.info(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")

    # Initialize gradient scaler for mixed precision
    logger.info("Setting up gradient scaler...")
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)

    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        config.get_param_groups(unwrap_model(model)),
        weight_decay=config.weight_decay
    )
    _optimizer = optimizer

    # Total steps comes from the real loader length. The sampler used to
    # over-report its length, which made every step budget wrong.
    steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
    total_steps = steps_per_epoch * config.epochs

    logger.info("Creating learning rate scheduler...")
    scheduler, warmup_steps = build_scheduler(optimizer, config, total_steps)
    _scheduler = scheduler
    logger.info(f"Training schedule: {total_steps} total optimizer steps, {warmup_steps} warmup steps")
    logger.info(f"Actual number of batches per epoch: {len(train_loader)}")

    loss_fct = LanguageAwareFocalLoss(label_smoothing=config.label_smoothing)
    if config.label_smoothing > 0:
        logger.info(f"Label smoothing enabled: eps={config.label_smoothing}")

    # Initialize metrics tracker
    metrics = MetricsTracker()
    best_epoch = None

    writer.add_text('config', f"```json\n{json.dumps(config.to_serializable_dict(), indent=2)}\n```")

    # MLflow side: params and tags are what make runs comparable later. The
    # ablation tag in particular is how the real-vs-shuffled lang_ids
    # comparison gets read back off the run table.
    if hasattr(writer, 'log_params'):
        writer.log_params(config.to_serializable_dict())
        writer.set_tags({
            'ablation.disable_lang_conditioning': str(
                getattr(config, 'disable_lang_conditioning', False)).lower(),
            'run.kind': 'control' if getattr(config, 'disable_lang_conditioning', False) else 'treatment',
            'data.train_file': str(config.train_file),
            'data.val_file': str(config.val_file),
        })

    logger.info("Starting training loop...")
    # Training loop
    model.train()

    train_sampler = getattr(train_loader, 'sampler', None)
    # Padded width of the first batches actually delivered by the loader. The
    # startup profile predicts this from the sampler; comparing the two catches
    # a collator or lengths regression that silently reverts to max_length.
    observed_widths = []
    observed_report_after = 50

    for epoch in range(config.epochs):
        epoch_loss = 0
        num_batches = 0
        consecutive_failures = 0
        epoch_start_time = time.time()

        # Without this every epoch replays the exact same ordering
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

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
            global_step = epoch * len(train_loader) + batch_idx
            collect_stats = batch_idx < 5 or batch_idx % 50 == 0
            try:
                # Log first batch details
                if batch_idx == 0:
                    logger.info("Successfully loaded first batch")
                    logger.info(f"Batch shapes - input_ids: {batch['input_ids'].shape}, "
                              f"attention_mask: {batch['attention_mask'].shape}, "
                              f"labels: {batch['labels'].shape}")
                    if torch.cuda.is_available():
                        logger.info(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

                if epoch == 0 and len(observed_widths) < observed_report_after:
                    observed_widths.append(int(batch['input_ids'].shape[1]))
                    if len(observed_widths) == min(observed_report_after, len(train_loader)):
                        observed_mean = float(np.mean(observed_widths))
                        expected = getattr(config, 'expected_batch_length', None)
                        logger.info(
                            "Observed mean padded batch width over the first %d batches: "
                            "%.1f tokens (max %d) against max_length=%d%s",
                            len(observed_widths), observed_mean, max(observed_widths),
                            config.max_length,
                            f", predicted {expected:.1f}" if expected else ""
                        )
                        if config.dynamic_padding and observed_mean >= config.max_length:
                            logger.warning(
                                "Every batch is padding to max_length: dynamic padding is "
                                "not actually engaged, and this epoch will cost several "
                                "times what it should"
                            )

                # Execute training step
                step = training_step(batch, model, optimizer, scheduler, config, scaler,
                                     batch_idx, loss_fct, collect_stats=collect_stats)
                loss = step['loss']

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

                if step['weight_stats'] and (epoch == 0 and batch_idx < 5):
                    ws = step['weight_stats']
                    logger.info(
                        "Class weights batch %d: min=%.4f max=%.4f mean=%.4f | "
                        "alpha=[%.3f, %.3f] gamma=[%.3f, %.3f]",
                        batch_idx, ws['weight_min'], ws['weight_max'], ws['weight_mean'],
                        ws['alpha_min'], ws['alpha_max'], ws['gamma_min'], ws['gamma_max']
                    )

                writer.add_scalars({
                    'train/loss': loss,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/grad_norm': step['grad_norm'],
                    'train/batch_time': batch_time,
                    'train/gpu_memory_mb': (
                        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    ),
                }, global_step)
                if step['weight_stats']:
                    ws = step['weight_stats']
                    writer.add_scalars({
                        'train/class_weight_min': ws['weight_min'],
                        'train/class_weight_max': ws['weight_max'],
                        'train/class_weight_mean': ws['weight_mean'],
                    }, global_step)

                # More frequent logging for debugging
                if batch_idx % 10 == 0:
                    loss_debug_str = "N/A" if loss is None else f"{loss:.4f}"
                    logger.debug(
                        f"Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss={loss_debug_str}, "
                        f"Time={batch_time:.2f}s"
                    )

                # Memory management
                if config.gc_frequency > 0 and batch_idx % config.gc_frequency == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                batch_start_time = time.time()
                consecutive_failures = 0

            except torch.OutOfMemoryError as e:
                # Never swallow an OOM. Dropped batches are shuffled apart by the
                # length-bucketing sampler, so consecutive_failures would never
                # trip and the run would quietly finish having skipped the
                # longest sequences in the corpus. Fail loudly instead.
                width = batch['input_ids'].shape[1] if 'input_ids' in batch else -1
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM at batch {batch_idx} of epoch {epoch + 1} "
                    f"(batch shape {tuple(batch['input_ids'].shape)}, seq width {width}). "
                    f"Reduce batch_size or max_length rather than dropping batches."
                ) from e
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                logger.error("Batch contents:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    else:
                        logger.error(f"{k}: type={type(v)}")
                if torch.cuda.is_available():
                    logger.error(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                # A single bad batch is survivable; a broken run is not. Bail out
                # instead of silently burning an epoch on nothing but failures.
                if consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    raise RuntimeError(
                        f"Aborting: {consecutive_failures} consecutive batch failures "
                        f"at batch {batch_idx} of epoch {epoch + 1}"
                    ) from e
                optimizer.zero_grad(set_to_none=True)
                batch_start_time = time.time()
                continue

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        metrics.update_train(avg_epoch_loss)
        epoch_time = time.time() - epoch_start_time
        metrics.update_time(epoch_time)
        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_time:.1f}s. "
            f"Average loss: {avg_epoch_loss:.4f} over {num_batches} batches"
        )

        # Validation pass and model selection
        val_metrics = None
        if val_loader is not None and config.eval_every_epoch:
            val_start = time.time()
            try:
                val_metrics = validate(model, val_loader, config)
            except Exception as e:
                logger.error(f"Validation failed at epoch {epoch + 1}: {e}", exc_info=True)
                logger.error("Continuing training; this epoch has no validation metrics")
                model.train()
                val_metrics = None
            logger.info(
                f"Validation epoch {epoch + 1}: loss={val_metrics['loss']:.4f}, "
                f"macro AUC={val_metrics['auc']:.4f} over {val_metrics['num_samples']} samples "
                f"in {time.time() - val_start:.1f}s"
            )
            for name, auc in val_metrics['per_class_auc'].items():
                logger.info(f"  val AUC {name}: {auc:.4f}")
            if val_metrics['skipped_classes']:
                logger.warning(f"  val AUC skipped for degenerate classes: {val_metrics['skipped_classes']}")

            improved = metrics.update_validation(val_metrics)
            if improved:
                best_epoch = epoch + 1
                save_best_checkpoint(model, optimizer, scheduler, metrics, config,
                                     epoch, val_metrics)
            logger.info(
                f"Best macro AUC so far: {metrics.best_auc:.4f} "
                f"(epoch {best_epoch if best_epoch else 'n/a'})"
            )
        elif val_loader is None:
            logger.warning("No validation loader: skipping validation and model selection")

        # Save checkpoint
        try:
            save_checkpoint(model, optimizer, scheduler, metrics, config, epoch, val_metrics)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {str(e)}")

        # Log epoch metrics
        epoch_scalars = {
            'epoch/train_loss': avg_epoch_loss,
            'epoch/time_sec': epoch_time,
            'epoch/lr': scheduler.get_last_lr()[0],
            'epoch/best_auc': metrics.best_auc,
            'epoch/gpu_memory_mb': (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            ),
        }
        if val_metrics is not None:
            epoch_scalars['epoch/val_loss'] = val_metrics['loss']
            epoch_scalars['epoch/val_auc_macro'] = val_metrics['auc']
            for name, auc in val_metrics['per_class_auc'].items():
                epoch_scalars[f'epoch/val_auc/{name}'] = auc
        writer.add_scalars(epoch_scalars, epoch + 1)
        writer.flush()

    if best_epoch is not None:
        logger.info(
            f"Training finished. Best epoch: {best_epoch} with macro val AUC "
            f"{metrics.best_auc:.4f} ({_best_checkpoint_dir(config)})"
        )
    else:
        logger.info("Training finished. No validation metrics, so no best checkpoint was selected")

    writer.flush()
    return metrics


def log_batch_length_profile(sampler, lengths, config):
    """Report the padded length of the batches the sampler will actually emit.

    Length bucketing is the difference between ~9 and ~36 minutes an epoch, and
    when it silently falls back (lengths not wired through, or the collator not
    used) every batch quietly pads to max_length again. This makes that visible
    at startup instead of forty minutes in. The batches are derived the same way
    the DataLoader derives them: the flat index stream chopped every batch_size.
    """
    if lengths is None:
        logger.warning(
            "No token lengths: every batch pads to max_length=%d. Expect the full "
            "static-padding cost per epoch.", config.max_length
        )
        return None

    order = np.asarray(list(iter(sampler)))
    bs = config.batch_size
    multiple = max(1, config.pad_to_multiple_of)
    batch_max = np.array([
        lengths[order[i:i + bs]].max() for i in range(0, len(order), bs)
    ], dtype=np.float64)
    padded = np.ceil(batch_max / multiple) * multiple

    logger.info(
        "Batch length profile over %d batches: mean padded length %.1f tokens "
        "(p50=%d p90=%d p99=%d max=%d) against max_length=%d",
        len(padded), padded.mean(), int(np.percentile(padded, 50)),
        int(np.percentile(padded, 90)), int(np.percentile(padded, 99)),
        int(padded.max()), config.max_length
    )
    logger.info(
        "Length bucketing cuts padded tokens to %.1f%% of static padding "
        "(%.0f vs %d per sequence)",
        100.0 * padded.mean() / config.max_length, padded.mean(), config.max_length
    )
    return padded.mean()


def create_dataloaders(train_dataset, val_dataset, config):
    """Create the training and validation loaders"""
    logger.info("Creating data loaders...")

    collator = DynamicPadCollator(
        train_dataset.tokenizer,
        pad_to_multiple_of=config.pad_to_multiple_of
    )

    # Exact per-sample token lengths let the sampler put similar-length samples
    # in the same batch, so dynamic padding actually saves compute instead of
    # every batch being dragged up to its longest member.
    lengths = None
    if config.dynamic_padding:
        try:
            lengths = train_dataset.token_lengths
            logger.info(
                "Length-bucketed batching enabled (token lengths: min=%d, median=%d, max=%d)",
                int(np.min(lengths)), int(np.median(lengths)), int(np.max(lengths))
            )
        except Exception as e:
            logger.warning(f"Could not compute token lengths ({str(e)}); batching without length buckets")
            lengths = None

    train_sampler = MultilabelStratifiedSampler(
        labels=train_dataset.labels,
        groups=train_dataset.langs,
        batch_size=config.batch_size,
        lengths=lengths
    )

    config.expected_batch_length = log_batch_length_profile(train_sampler, lengths, config)

    num_workers = config.num_workers
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': collator,
        'drop_last': False,
    }
    if num_workers > 0:
        # Workers tokenize ahead of the GPU instead of the training process
        # stalling on it every batch
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        **loader_kwargs
    )
    logger.info(
        f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches, "
        f"batch_size={config.batch_size}, num_workers={num_workers}, "
        f"persistent_workers={num_workers > 0}, pin_memory={loader_kwargs['pin_memory']}"
    )

    val_loader = None
    if val_dataset is not None:
        # No stratified sampler and no shuffling: validation is a plain
        # deterministic pass so the metric is comparable across epochs.
        # Validation runs once per epoch and has no backward pass, so it needs
        # far less input bandwidth than training: 2 workers already feed it
        # several times over. They are also not persistent, so they are torn
        # down between epochs rather than holding a second full set of
        # DataFrame copy-on-write views alive for the whole run.
        val_workers = min(2, num_workers)
        val_kwargs = dict(loader_kwargs)
        val_kwargs.pop('persistent_workers', None)
        val_kwargs['num_workers'] = val_workers
        if val_workers == 0:
            val_kwargs.pop('prefetch_factor', None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            **val_kwargs
        )
        logger.info(
            f"Val loader: {len(val_dataset)} samples, {len(val_loader)} batches, "
            f"batch_size={config.eval_batch_size}, num_workers={val_workers} "
            f"(non-persistent)"
        )

    # Verify the dataset and collator without spawning workers
    logger.info("Testing dataset and collator...")
    try:
        probe = collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
        logger.info(
            f"Collator test successful. Keys: {list(probe.keys())}, "
            f"input_ids: {tuple(probe['input_ids'].shape)}, labels: {tuple(probe['labels'].shape)}"
        )
    except Exception as e:
        logger.error(f"Dataset/collator test failed: {str(e)}")
        raise

    return train_loader, val_loader


def configure_backend(config):
    """CUDA/cuDNN settings for throughput"""
    if not torch.cuda.is_available():
        logger.info("CUDA not available, running on CPU")
        return

    # CUDA_LAUNCH_BLOCKING used to be forced on here, which serializes every
    # kernel launch (issue #10). Deterministic cuDNN + benchmark off was the
    # other half of the same debugging setup; ordering is already reproducible
    # through the sampler seed, so trade it for autotuned kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # TF32 is decided by TrainingConfig, which checks compute capability. It
    # only exists on Ampere (sm_80) and later; on Turing the flags do nothing.
    capability = torch.cuda.get_device_capability()
    logger.info(
        f"Using CUDA device: {torch.cuda.get_device_name()} (compute capability "
        f"{capability[0]}.{capability[1]}), TF32={'on' if config.tensor_float_32 else 'unavailable/off'}"
    )
    logger.info(f"Visible CUDA devices: {torch.cuda.device_count()}")

    torch.cuda.empty_cache()
    torch.cuda.set_device(torch.cuda.current_device())


def main(config=None):
    global _writer
    writer = None
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        logger.info("Initializing training configuration...")
        if config is None:
            config = TrainingConfig()

        configure_backend(config)

        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SafeSummaryWriter(Path('runs') / run_name)
        _writer = writer

        global _model, _optimizer, _scheduler
        _model = None
        _optimizer = None
        _scheduler = None

        logger.info("Loading datasets...")
        try:
            train_df = pd.read_csv(config.train_file)
            logger.info(f"Loaded train dataset with {len(train_df)} samples from {config.train_file}")
            val_df = None
            if config.eval_every_epoch:
                val_df = pd.read_csv(config.val_file)
                logger.info(f"Loaded val dataset with {len(val_df)} samples from {config.val_file}")
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

        try:
            logger.info("Creating tokenizer and dataset...")
            # Fast (Rust) tokenizer. The slow sentencepiece one tokenized inside
            # the training process and stalled the GPU between batches.
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(config.model_name)
            train_dataset = ToxicDataset(train_df, tokenizer, config)
            val_dataset = ToxicDataset(val_df, tokenizer, config) if val_df is not None else None
            logger.info("Dataset creation successful")
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise

        # The class weighting maps integer language ids back to codes; a drift
        # between the two mappings would silently disable it
        if train_dataset.lang_to_id != SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language id mapping mismatch: dataset {train_dataset.lang_to_id} vs "
                f"model {SUPPORTED_LANGUAGES}"
            )

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

        logger.info("Initializing model...")
        model = init_model(config)

        logger.info("Starting training...")
        train(model, train_loader, val_loader, config, writer=writer)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        cleanup()

if __name__ == "__main__":
    # Set global PyTorch settings
    torch.set_num_threads(1)  # Limit CPU threads in the training process
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        cleanup()
        sys.exit(1)
