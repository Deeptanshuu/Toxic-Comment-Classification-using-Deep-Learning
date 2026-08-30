# training_config.py
from asyncio.log import logger
from dataclasses import dataclass
from typing import Dict, List
import json
import torch
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict
import os

@dataclass
class DynamicClassWeights:
    """Handles class weights per language using dynamic batch statistics"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __init__(self, weights_file: str = 'weights/language_class_weights.json'):
        self.weights_file = weights_file
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Initialize base scaling factors from file if available
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                self.lang_scaling = {}
                for lang in self.language_columns:
                    if lang in data['weights']:
                        # Calculate average scaling per language
                        scales = [float(data['weights'][lang][label]['1']) 
                                for label in self.toxicity_labels]
                        self.lang_scaling[lang] = sum(scales) / len(scales)
                    else:
                        self.lang_scaling[lang] = 1.0
        except Exception as e:
            logger.warning(f"Could not load weights from {self.weights_file}: {str(e)}")
            self._initialize_defaults()
        
        # Initialize running statistics for each language
        self.running_stats = {lang: {
            'pos_counts': torch.zeros(len(self.toxicity_labels)),
            'total_counts': torch.zeros(len(self.toxicity_labels)),
            'smoothing_factor': 0.95  # EMA smoothing factor
        } for lang in self.language_columns}
    
    def _initialize_defaults(self):
        """Initialize safe default scaling factors"""
        self.lang_scaling = {lang: 1.0 for lang in self.language_columns}
    
    def _update_running_stats(self, langs, labels):
        """Update running statistics for each language"""
        unique_langs = set(langs)
        for lang in unique_langs:
            if lang not in self.running_stats:
                continue

            lang_mask = torch.tensor([l == lang for l in langs], dtype=torch.bool,
                                     device=labels.device)
            lang_labels = labels[lang_mask]

            if len(lang_labels) == 0:
                continue

            # Calculate current batch statistics. Running stats are kept on CPU,
            # so bring the batch counts across before mixing them in - otherwise
            # every update raises a device mismatch and the caller silently falls
            # back to uniform weights.
            pos_count = lang_labels.sum(dim=0).float().detach().cpu()
            total_count = torch.full_like(pos_count, len(lang_labels))

            # Update running statistics with EMA
            alpha = self.running_stats[lang]['smoothing_factor']
            self.running_stats[lang]['pos_counts'] = (
                alpha * self.running_stats[lang]['pos_counts'] + 
                (1 - alpha) * pos_count
            )
            self.running_stats[lang]['total_counts'] = (
                alpha * self.running_stats[lang]['total_counts'] + 
                (1 - alpha) * total_count
            )
    
    def get_weights_for_batch(self, langs: List[str], labels: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Calculate dynamic weights and focal parameters based on batch and historical statistics
        Args:
            langs: List of language codes
            labels: Binary labels tensor [batch_size, num_labels]
            device: Target device for tensors
        Returns:
            Dict with weights, alpha, and gamma tensors
        """
        batch_size = len(langs)
        num_labels = labels.size(1)

        try:
            # Update running statistics
            self._update_running_stats(langs, labels)

            # Calculate positive ratio per language in current batch
            lang_pos_ratios = {}
            batch_pos_ratios = torch.zeros(num_labels, device=device)
            lang_counts = {}

            for lang in set(langs):
                lang_mask = torch.tensor([l == lang for l in langs], dtype=torch.bool, device=device)
                if not lang_mask.any():
                    continue

                # Calculate language-specific positive ratio
                lang_labels = labels[lang_mask]
                lang_pos_ratio = lang_labels.float().mean(dim=0)
                lang_pos_ratios[lang] = lang_pos_ratio

                # Weighted contribution to batch statistics
                lang_count = int(lang_mask.sum().item())
                lang_counts[lang] = lang_count
                batch_pos_ratios += lang_pos_ratio * (lang_count / batch_size)

            # A per-language class_adjustments table used to sit here, described
            # as "based on statistical analysis". It was not: five of its six
            # non-English rows were byte-identical and its comments contradicted
            # its values (the English row said "more obscene/threat" while
            # down-weighting both). It distorted class weights by ~3.5% on
            # average, 13% worst case. Removed after the language-conditioning
            # ablation, which needed it held constant across both arms to
            # cancel. See docs/KNOWN_ISSUES.md.

            # The per-sample weight row depends only on the sample's language, so
            # build one row per language present and index it out, rather than
            # looping over the batch (that cost ~1300 tiny CUDA launches/batch).
            weights = torch.ones(batch_size, num_labels, device=device)
            alpha = torch.zeros(num_labels, device=device)
            gamma = torch.zeros(num_labels, device=device)

            for lang, lang_count in lang_counts.items():
                if lang not in self.running_stats:
                    continue

                # Get historical statistics for this language
                hist_pos_ratio = (
                    self.running_stats[lang]['pos_counts'] /
                    (self.running_stats[lang]['total_counts'] + 1e-7)
                ).to(device)

                # Combine historical and current batch statistics
                current_pos_ratio = lang_pos_ratios.get(lang, batch_pos_ratios)
                combined_pos_ratio = 0.7 * hist_pos_ratio + 0.3 * current_pos_ratio

                # Calculate stable weights using log-space
                log_ratio = torch.log1p(1.0 / (combined_pos_ratio + 1e-7))
                class_weights = torch.exp(log_ratio.clamp(-2, 2))

                # Apply language-specific scaling
                row = class_weights * self.lang_scaling.get(lang, 1.0)

                lang_mask = torch.tensor([l == lang for l in langs], dtype=torch.bool, device=device)
                weights[lang_mask] = row

                # Focal parameters are a language-frequency-weighted average over
                # the languages in the batch, so the mixing weights sum to 1. The
                # previous version accumulated once per *sample*, which multiplied
                # every contribution by the language count and pinned alpha/gamma
                # at their clamp ceilings on every batch.
                mix = lang_count / batch_size
                alpha += (1.0 / (combined_pos_ratio + 1e-7).clamp(0.05, 0.95)) * mix
                gamma += log_ratio.clamp(1.0, 4.0) * mix

            # Normalize weights to prevent extreme values
            weights = weights / weights.mean()

            # Normalize alpha to mean 1.0. The relative per-class weighting is
            # what matters; the absolute scale only inflates the gradient norm.
            # Unnormalized, alpha averages ~4 and true grad norms run 53-61,
            # so max_grad_norm=1.0 clipped every single step by ~55x and stopped
            # being an outlier guard. Normalizing keeps the class weighting
            # intact and restores the clip's meaning (and makes train and val
            # losses comparable, since validation uses fixed alpha).
            alpha = alpha.clamp(0.1, 5.0)
            alpha = alpha / alpha.mean().clamp(min=1e-6)

            return {
                'weights': weights.clamp(0.1, 10.0),  # Prevent extreme values
                'alpha': alpha,                       # [num_labels], mean 1.0
                'gamma': gamma.clamp(1.0, 4.0)        # [num_labels]
            }

        except Exception as e:
            logger.error(f"Error computing batch weights: {str(e)}")
            # Fallback to safe default values
            return {
                'weights': torch.ones((batch_size, num_labels), device=device),
                'alpha': torch.full((num_labels,), 0.25, device=device),
                'gamma': torch.full((num_labels,), 2.0, device=device)
            }

@dataclass
class MetricsTracker:
    """Tracks training and validation metrics with error handling"""
    best_auc: float = 0.0
    train_losses: List[float] = None
    val_losses: List[float] = None
    val_aucs: List[float] = None
    epoch_times: List[float] = None
    
    def __post_init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.epoch_times = []
    
    def update_train(self, loss: float):
        """Update training metrics with validation"""
        try:
            if not isinstance(loss, (int, float)) or np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Invalid loss value: {loss}")
                return
            self.train_losses.append(float(loss))
        except Exception as e:
            print(f"Warning: Could not update training metrics: {str(e)}")
    
    def update_validation(self, metrics: Dict) -> bool:
        """Update validation metrics with error handling"""
        try:
            if not isinstance(metrics, dict):
                raise ValueError("Metrics must be a dictionary")
                
            loss = metrics.get('loss', float('inf'))
            auc = metrics.get('auc', 0.0)
            
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Invalid loss value: {loss}")
                loss = float('inf')
            
            if np.isnan(auc) or np.isinf(auc):
                print(f"Warning: Invalid AUC value: {auc}")
                auc = 0.0
            
            self.val_losses.append(float(loss))
            self.val_aucs.append(float(auc))
            
            # Update best AUC if needed
            if auc > self.best_auc:
                self.best_auc = auc
                return True
            return False
            
        except Exception as e:
            print(f"Warning: Could not update validation metrics: {str(e)}")
            return False
    
    def update_time(self, epoch_time: float):
        """Update timing metrics with validation"""
        try:
            if not isinstance(epoch_time, (int, float)) or epoch_time <= 0:
                print(f"Warning: Invalid epoch time: {epoch_time}")
                return
            self.epoch_times.append(float(epoch_time))
        except Exception as e:
            print(f"Warning: Could not update timing metrics: {str(e)}")
    
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """Calculate ETA based on average epoch time with error handling"""
        try:
            if not self.epoch_times:
                return "Calculating..."
                
            if current_epoch >= total_epochs:
                return "Complete"
                
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = total_epochs - current_epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            
            return f"{hours:02d}:{minutes:02d}:00"
            
        except Exception as e:
            print(f"Warning: Could not calculate ETA: {str(e)}")
            return "Unknown"

@dataclass
class TrainingConfig:
    """Basic training configuration with consolidated default values"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 512
    hidden_size: int = 1024
    num_attention_heads: int = 16
    model_dropout: float = 0.0
    # Number of *encoder layer blocks* frozen from the bottom up. This used to
    # default to 8 while the code actually froze the first 8 parameter tensors
    # (issue #4); the two are now separate and both mean what they say.
    freeze_layers: int = 0
    # Freeze the whole embedding module (word/position/token-type + LayerNorm).
    # This is what the old buggy slice was really doing, and it is worth keeping:
    # the word-embedding matrix is 256M of the 560M base parameters, so freezing
    # it removes the AdamW update that dominates the step at short sequence
    # lengths (measured 4.2x wall-clock, 3x peak memory at seq 96 / batch 128).
    freeze_embeddings: bool = True

    # Dataset parameters
    cache_dir: str = 'cached_dataset'
    # Separate directory from the archived 2025 run. The keep-last-3 rotation
    # sorts by name, so mixing two runs' checkpoints in one directory makes it
    # delete the older run's by epoch number -- including the one
    # docs/RESULTS.md cites. Keeping runs apart avoids that entirely and
    # leaves the previous model serving until the new one is evaluated.
    checkpoint_dir: str = 'weights/toxic_classifier_xlmr_v2'
    label_columns: List[str] = None  # Will be initialized in __post_init__
    
    # Training parameters
    # 64x2 keeps the effective batch at 128 while halving activation memory.
    # At batch 128 the 512-token batches peak at 18.30 GB against a 22.29 GB
    # cap and the real loop OOMs; at 64 they peak at 11.36 GB.
    batch_size: int = 64
    grad_accum_steps: int = 2
    epochs: int = 6
    lr: float = 2e-5
    num_cycles: int = 2
    weight_decay: float = 2e-7
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.01
    min_lr_ratio: float = 0.01
    
    # Memory optimization
    activation_checkpointing: bool = True
    mixed_precision: str = "fp16"
    _num_workers: int = None  # Private storage for num_workers
    gc_frequency: int = 500
    tensor_float_32: bool = True
    
    # Cosine scheduler parameters
    num_cycles: int = 2

    # --- Added by fix branch: see docs/KNOWN_ISSUES.md ---
    # Data paths
    train_file: str = 'dataset/split/train.csv'
    val_file: str = 'dataset/split/val.csv'
    test_file: str = 'dataset/split/test.csv'

    # Throughput: pad per-batch instead of to max_length (issue: pad-to-512)
    dynamic_padding: bool = True
    pad_to_multiple_of: int = 8

    # Correctness switches
    use_class_weights: bool = True      # activate DynamicClassWeights (issue #3)
    use_warmup: bool = True             # activate linear warmup (issue #5)
    data_parallel: bool = True          # use all visible GPUs (new issue)
    eval_batch_size: int = 256
    eval_every_epoch: bool = True       # validation loop (issue #2)

    # Populated in __post_init__ when use_class_weights is set
    lang_weights: object = None

    # Ablation: when True the model ignores lang_ids entirely, giving the
    # control run for "does language conditioning actually help". Settable via
    # the TOXIC_DISABLE_LANG_CONDITIONING env var so a control run needs no
    # code edit. See docs/KNOWN_ISSUES.md.
    disable_lang_conditioning: bool = False
    class_weights_file: str = 'weights/language_class_weights.json'

    def __post_init__(self):
        """Initialize and validate configuration"""
        # Initialize label columns
        self.label_columns = [
            'toxic', 'severe_toxic', 'obscene', 
            'threat', 'insult', 'identity_hate'
        ]
        
        # Set environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        
        # Rest of the initialization code...
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.lr < 1e-7:
            raise ValueError(f"Learning rate too small: {self.lr}")
        if self.lr > 1.0:
            raise ValueError(f"Learning rate too large: {self.lr}")
            
        # Validate weight decay and learning rate combination
        if self.weight_decay > 0:
            wd_to_lr_ratio = self.weight_decay / self.lr
            if wd_to_lr_ratio > 0.1:
                logger.warning(
                    "Weight decay too high: %.2e (%.2fx learning rate). "
                    "Should be 0.01-0.1x learning rate.", 
                    self.weight_decay, wd_to_lr_ratio
                )
            effective_lr = self.lr * (1 - self.weight_decay)
            if effective_lr < self.lr * 0.9:
                logger.warning(
                    "Weight decay %.2e reduces effective learning rate to %.2e (%.1f%% reduction)",
                    self.weight_decay, effective_lr, (1 - effective_lr/self.lr) * 100
                )
        
        # Set device with memory optimization
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                # Set memory allocation strategy
                torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some GPU memory free
                self.device = torch.device('cuda')
                
                if self.mixed_precision == "bf16":
                    if not torch.cuda.is_bf16_supported():
                        print("Warning: BF16 not supported on this GPU. Falling back to FP16")
                        self.mixed_precision = "fp16"
                
                if self.tensor_float_32:
                    if torch.cuda.get_device_capability()[0] >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    else:
                        print("Warning: TF32 not supported on this GPU. Disabling.")
                        self.tensor_float_32 = False
                
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {str(e)}")
                self.device = torch.device('cpu')
                self.mixed_precision = "no"
        else:
            self.device = torch.device('cpu')
            if self.mixed_precision != "no":
                print("Warning: Mixed precision not supported on CPU. Disabling.")
                self.mixed_precision = "no"
        
        # Create directories with error handling
        try:
            for directory in ["weights", "logs"]:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True)
                elif not dir_path.is_dir():
                    raise NotADirectoryError(f"{directory} exists but is not a directory")
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise
        
        # Ablation switch may be set from the environment so the control run
        # is launchable without editing code.
        _abl = os.environ.get('TOXIC_DISABLE_LANG_CONDITIONING', '').strip().lower()
        if _abl in ('1', 'true', 'yes'):
            self.disable_lang_conditioning = True
        if self.disable_lang_conditioning:
            logger.warning(
                "ABLATION: disable_lang_conditioning=True -- lang_ids will not "
                "affect the model. This is the control run, not the real one."
            )

        # Initialize toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
        
        # Set use_mixed_precision flag
        self.use_mixed_precision = self.mixed_precision != "no"

        # Instantiate the class weighting helper. Without this every training
        # step falls through to uniform focal parameters (issue #3).
        if self.use_class_weights:
            try:
                self.lang_weights = DynamicClassWeights(weights_file=self.class_weights_file)
                logger.info(f"Class weighting enabled from {self.class_weights_file}")
            except Exception as e:
                logger.error(f"Could not initialize DynamicClassWeights: {str(e)}")
                self.lang_weights = None
        else:
            self.lang_weights = None

    def validate_model_config(self, model):
        """Validate configuration against model architecture"""
        try:
            # Validate layer freezing
            total_layers = len(list(model.base_model.encoder.layer))
            if self.freeze_layers > total_layers:
                raise ValueError(f"Can't freeze {self.freeze_layers} layers in {total_layers}-layer model")
            if self.freeze_layers < 0:
                raise ValueError(f"freeze_layers must be >= 0, got {self.freeze_layers}")
            logger.info(f"Freezing {self.freeze_layers} out of {total_layers} encoder layers")

            # Validate embedding freezing actually took effect
            embed_trainable = sum(
                p.numel() for p in model.base_model.embeddings.parameters() if p.requires_grad
            )
            if self.freeze_embeddings and embed_trainable > 0:
                raise ValueError(
                    f"freeze_embeddings is set but {embed_trainable} embedding parameters "
                    "still require grad"
                )
            if not self.freeze_embeddings and embed_trainable == 0:
                logger.warning("freeze_embeddings is False but no embedding parameter requires grad")

            for idx, layer in enumerate(model.base_model.encoder.layer):
                trainable = any(p.requires_grad for p in layer.parameters())
                if idx < self.freeze_layers and trainable:
                    raise ValueError(f"encoder.layer[{idx}] should be frozen but is trainable")
                if idx >= self.freeze_layers and not trainable:
                    logger.warning(f"encoder.layer[{idx}] should be trainable but is fully frozen")

            # Report the parameter groups the optimizer will actually see. The
            # old check here warned whenever a group had lr < 0.01, which is
            # true of every transformer fine-tuning LR, so it fired on every
            # run and meant nothing.
            for i, group in enumerate(self.get_param_groups(model)):
                count = sum(p.numel() for p in group['params'])
                if count == 0:
                    logger.warning(f"Parameter group {i} is empty")
                logger.info(f"Optimizer group {i}: {count / 1e6:.1f}M parameters at lr={group['lr']:.2e}")

            return True
        except Exception as e:
            logger.error(f"Model configuration validation failed: {str(e)}")
            raise

    @property
    def dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on mixed precision settings"""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32
    
    def get_autocast_context(self):
        """Get the appropriate autocast context based on configuration."""
        if not self.use_mixed_precision:
            return nullcontext()
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)
    
    def to_serializable_dict(self):
        """Convert config to a dictionary for saving."""
        config_dict = asdict(self)
        # lang_weights holds a live helper object; record only what it was built from
        config_dict['lang_weights'] = self.class_weights_file if self.lang_weights else None
        config_dict['device'] = str(self.device)
        return config_dict

    def get_param_groups(self, model):
        """Get parameter groups with base learning rate.

        Frozen parameters are excluded so the optimizer never allocates state
        for them (the frozen embedding matrix alone is 256M parameters).
        """
        params = [p for p in model.parameters() if p.requires_grad]
        return [{'params': params, 'lr': self.lr}]


    @property
    def use_amp(self):
        """Check if AMP should be used based on device and mixed precision setting"""
        return self.device.type == 'cuda' and self.mixed_precision != "no"
    
    @property
    def grad_norm_clip(self):
        """Adaptive gradient clipping based on precision"""
        if self.mixed_precision == "bf16":
            return 1.5  # BF16 can handle slightly higher gradients than FP16
        if self.mixed_precision == "fp16":
            return 1.0  # Most conservative for FP16 due to lower precision
        return 5.0  # Full precision can handle larger gradients

    @property
    def num_workers(self):
        """Dynamically adjust workers based on system resources.

        The cap is 8 and deliberately not higher, even on a 36-core box. One
        worker tokenizes ~2700 samples/s unpadded; after length bucketing the
        GPU step is ~0.16 s per batch of 128, so the loader only has to sustain
        800-1000 samples/s. A single worker already covers that and workers past
        about 4 buy no throughput at all, while each one forks a
        copy-on-write view of a 285k-row DataFrame. The 8 is jitter headroom for
        the long-sequence batches in the tail, not throughput, and the rest of
        the CPU and RAM on this box is deliberately left for other processes.
        """
        if self._num_workers is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                self._num_workers = 0
            else:
                # Leave at least 2 CPUs free, max 8 workers
                self._num_workers = min(8, max(0, cpu_count - 2))
            logger.info(f"Dynamically set num_workers to {self._num_workers} (CPU count: {cpu_count})")
        return self._num_workers
    
    @num_workers.setter
    def num_workers(self, value):
        """Allow manual override of num_workers"""
        self._num_workers = value
        logger.info(f"Manually set num_workers to {value}") 