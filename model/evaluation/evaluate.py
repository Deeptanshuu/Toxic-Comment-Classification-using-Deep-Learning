import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    confusion_matrix, hamming_loss, 
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
import gc
import multiprocessing
from pathlib import Path
import hashlib
import logging
from sklearn.metrics import make_scorer

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

# Set memory optimization environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

logger = logging.getLogger(__name__)

# Disk cache for ToxicDataset.token_lengths (exact per-sample token counts),
# keyed by row count + max_length + tokenizer + a content hash so a stale
# cache is never reused across a data or config change.
TOKEN_LENGTH_CACHE_DIR = Path('cache')

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
        # Ensure label columns are defined
        if not hasattr(config, 'label_columns'):
            self.label_columns = [
                'toxic', 'severe_toxic', 'obscene', 
                'threat', 'insult', 'identity_hate'
            ]
            logger.warning("Label columns not provided in config, using defaults")
        else:
            self.label_columns = config.label_columns
        
        # Verify all label columns exist in DataFrame
        missing_columns = [col for col in self.label_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing label columns in dataset: {missing_columns}")
        
        # Convert labels to numpy array for efficiency
        self.labels = df[self.label_columns].values
        
        # Create language mapping
        self.lang_to_id = {
            'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
            'fr': 4, 'it': 5, 'pt': 6
        }
        
        # Convert language codes to numeric indices
        self.langs = np.array([self.lang_to_id.get(lang, 0) for lang in df['lang']])

        # Cache for the lazy `token_lengths` proxy (see property below).
        self._token_lengths = None

        print(f"Initialized dataset with {len(self)} samples")
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Label columns: {self.label_columns}")
        logger.info(f"Unique languages: {np.unique(df['lang'])}")
        logger.info(f"Language mapping: {self.lang_to_id}")

    def __len__(self):
        return len(self.df)

    def _fast_tokenizer_for_lengths(self):
        """Return a fast tokenizer to use for batch length computation.

        `self.tokenizer` is frequently a slow `XLMRobertaTokenizer` (both
        train.py and evaluate.py's `load_model` construct the slow class).
        Batch-tokenizing ~285k rows needs a Rust-backed fast tokenizer to be
        cheap (thousands of rows/sec vs. a slow per-call Python tokenizer).
        Falls back to `self.tokenizer` itself, with a warning, if a fast
        counterpart can't be loaded.
        """
        if getattr(self.tokenizer, 'is_fast', False):
            return self.tokenizer

        name_or_path = getattr(self.tokenizer, 'name_or_path', None)
        if not name_or_path:
            logger.warning(
                "token_lengths: tokenizer has no name_or_path, cannot load a fast "
                "counterpart; falling back to the slow tokenizer (this will be slow)."
            )
            return self.tokenizer

        try:
            try:
                fast_tokenizer = AutoTokenizer.from_pretrained(
                    name_or_path, use_fast=True, local_files_only=True
                )
            except Exception:
                fast_tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
            if not getattr(fast_tokenizer, 'is_fast', False):
                raise ValueError("AutoTokenizer did not return a fast tokenizer")
            logger.info(
                f"token_lengths: '{name_or_path}' was loaded as a slow tokenizer; "
                "using its fast counterpart for length computation only."
            )
            return fast_tokenizer
        except Exception as e:
            logger.warning(
                f"token_lengths: could not load a fast tokenizer for '{name_or_path}' "
                f"({e}); falling back to the slow tokenizer (this will be slow)."
            )
            return self.tokenizer

    @property
    def token_lengths(self):
        """Exact per-sample token length (post-truncation), for length-grouped batching.

        Computed once via a single batched call to a fast tokenizer over the
        whole text column (thousands of rows/sec, so ~285k rows takes tens of
        seconds, not minutes), then cached to disk under `cache/` so repeat
        runs and later epochs don't pay for it again. Only runs on first
        access -- nothing is computed if this is never read.
        """
        if self._token_lengths is not None:
            return self._token_lengths

        n = len(self.df)
        max_length = self.config.max_length
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', self.tokenizer.__class__.__name__)

        try:
            content_hash = hashlib.md5(
                pd.util.hash_pandas_object(self.df['comment_text'], index=False).values.tobytes()
            ).hexdigest()
        except Exception as e:
            logger.warning(f"token_lengths: pandas content hash failed ({e}); using a slower fallback hash")
            content_hash = hashlib.md5(
                '\n'.join(self.df['comment_text'].astype(str)).encode('utf-8', errors='ignore')
            ).hexdigest()

        cache_key = f"{n}_{max_length}_{tokenizer_name}_{content_hash}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_path = TOKEN_LENGTH_CACHE_DIR / f"token_lengths_{cache_key_hash}.npy"

        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                if len(cached) == n:
                    self._token_lengths = cached
                    logger.info(f"token_lengths: loaded {n} cached lengths from {cache_path}")
                    return self._token_lengths
                logger.warning(f"token_lengths: cache {cache_path} has wrong length, recomputing")
            except Exception as e:
                logger.warning(f"token_lengths: failed to read cache {cache_path} ({e}); recomputing")

        tokenizer = self._fast_tokenizer_for_lengths()
        texts = self.df['comment_text'].astype(str).tolist()
        logger.info(f"token_lengths: tokenizing {n} samples to compute exact lengths (max_length={max_length})...")
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            verbose=False
        )
        self._token_lengths = np.array([len(ids) for ids in encoded['input_ids']], dtype=np.int32)

        try:
            TOKEN_LENGTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self._token_lengths)
            logger.info(f"token_lengths: cached {n} lengths to {cache_path}")
        except Exception as e:
            logger.warning(f"token_lengths: failed to write cache {cache_path} ({e}); continuing without disk cache")

        return self._token_lengths

    def __getitem__(self, idx):
        if idx % 1000 == 0:
            logger.debug(f"Loading sample {idx}")

        # Get text and labels
        text = self.df.iloc[idx]['comment_text']
        labels = torch.FloatTensor(self.labels[idx])
        lang = torch.tensor(self.langs[idx], dtype=torch.long)  # Ensure long dtype

        # Dynamic padding: tokenize to the true (truncated) length and leave
        # per-batch padding to the collate function, instead of padding every
        # sample to max_length. Most comments are far shorter than max_length,
        # so fixed-length padding wastes the bulk of the compute (see
        # docs/KNOWN_ISSUES.md). Falls back to the old pad-to-max_length
        # behavior when the config doesn't opt in or explicitly disables it.
        dynamic_padding = getattr(self.config, 'dynamic_padding', True)

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_length,
            padding=False if dynamic_padding else 'max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'lang': lang
        }

class ThresholdOptimizer(ClassifierMixin, BaseEstimator):
    """Custom estimator for threshold optimization.

    ClassifierMixin MUST come first. On scikit-learn >= 1.6 is_classifier()
    resolves __sklearn_tags__ through the MRO; with BaseEstimator first it
    finds BaseEstimator's tags, estimator_type is None, is_classifier() is
    False, and GridSearchCV silently builds an unshuffled KFold instead of a
    StratifiedKFold. The rows are grouped by language with positives at the
    front, so contiguous folds then contain zero positives, and with
    zero_division=1 an empty fold scores a free 1.0 -- which dragged the
    argmax to a wrong threshold. See docs/KNOWN_ISSUES.md.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.probabilities_ = None
        
    def fit(self, X, y):
        # Store probabilities for prediction
        self.probabilities_ = X
        return self
        
    def predict(self, X):
        # Apply threshold to probabilities
        return (X > self.threshold).astype(int)
        
    def score(self, X, y):
        # Return F1 score with proper handling of edge cases
        predictions = self.predict(X)
        
        # Handle edge case where all samples are negative
        if y.sum() == 0:
            return 1.0 if predictions.sum() == 0 else 0.0
            
        # Calculate metrics with zero_division=1
        try:
            precision = precision_score(y, predictions, zero_division=1)
            recall = recall_score(y, predictions, zero_division=1)
            
            # Calculate F1 manually to avoid warnings
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        except Exception:
            return 0.0

def load_model(model_path):
    """Load model and tokenizer from versioned checkpoint directory"""
    try:
        # Check if model_path points to a specific checkpoint or base directory
        model_dir = Path(model_path)
        if model_dir.is_dir():
            # Check for 'latest' symlink first
            latest_link = model_dir / 'latest'
            if latest_link.exists() and latest_link.is_symlink():
                model_dir = latest_link.resolve()
                logger.info(f"Using latest checkpoint: {model_dir}")
            else:
                # Find most recent checkpoint
                checkpoints = sorted([
                    d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('checkpoint_epoch')
                ])
                if checkpoints:
                    model_dir = checkpoints[-1]
                    logger.info(f"Using most recent checkpoint: {model_dir}")
                else:
                    logger.info("No checkpoints found, using base directory")
        
        logger.info(f"Loading model from: {model_dir}")
        
        # Initialize the custom model architecture
        model = LanguageAwareTransformer(
            num_labels=6,
            hidden_size=1024,
            num_attention_heads=16,
            model_name='xlm-roberta-large'
        )
        
        # Load the trained weights
        weights_path = model_dir / 'pytorch_model.bin'
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
        
        # Load base XLM-RoBERTa tokenizer directly
        logger.info("Loading XLM-RoBERTa tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        # Load training metadata if available
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded checkpoint metadata: Epoch {metadata.get('epoch', 'unknown')}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

def optimize_threshold(y_true, y_pred_proba, n_steps=200):
    """Pick the probability threshold that maximises F1 on the data given.

    This is a direct sweep, not a cross-validated grid search, and the change is
    deliberate. The previous implementation wrapped a `ThresholdOptimizer`
    estimator in `GridSearchCV(cv=5)`, but that estimator's `fit()` only stores
    its input and learns nothing -- there is no model whose generalization a
    cross-validation could estimate. What CV did instead was split the data into
    folds, and because rows are grouped by language with positives at the front,
    entire folds could contain zero positives. With `zero_division=1` those folds
    scored a free 1.0, which inflated rare classes and dragged the argmax to a
    visibly wrong threshold: English `severe_toxic` was reported at F1 0.597 when
    the maximum achievable at any threshold is 0.442. See docs/KNOWN_ISSUES.md.

    Reported `f1_score` is now the F1 actually achieved at the returned threshold
    on this data, which is what the field name claims. It is fit on validation and
    applied unchanged to test, so the optimism of fitting on the scored split does
    not reach the headline numbers.

    The sweep covers [0.05, 0.95]; the old [0.3, 0.7] grid could not express the
    optimum for the rarest classes, whose best thresholds sit below 0.3. 200 steps
    rather than 50: without the cross-validation this is 200 f1_score calls, which
    is cheap, and 50 steps left measurable slack against the achievable optimum.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()

    if y_true.sum() == 0:
        # No positives: F1 is undefined. Report 0.0 rather than a free 1.0.
        return {
            'threshold': 0.5,
            'f1_score': 0.0,
            'support': 0,
            'total_samples': len(y_true),
            'degenerate': True
        }

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, n_steps):
        f1 = f1_score(y_true, (y_pred_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return {
        'threshold': float(best_t),
        'f1_score': float(best_f1),
        'support': int(y_true.sum()),
        'total_samples': len(y_true),
        'degenerate': False
    }

def calculate_optimal_thresholds(predictions, labels, langs):
    """Calculate optimal thresholds for each class and language combination using Bayesian optimization"""
    logger.info("Calculating optimal thresholds using Bayesian optimization...")
    
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    unique_langs = np.unique(langs)
    
    thresholds = {
        'global': {},
        'per_language': {}
    }
    
    # Calculate global thresholds
    logger.info("Computing global thresholds...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Global thresholds")):
        thresholds['global'][class_name] = optimize_threshold(
            labels[:, i],
            predictions[:, i],
            n_steps=50
        )
    
    # Calculate language-specific thresholds
    logger.info("Computing language-specific thresholds...")
    for lang in tqdm(unique_langs, desc="Language thresholds"):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        thresholds['per_language'][str(lang)] = {}
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        
        for i, class_name in enumerate(toxicity_types):
            # Only optimize if we have enough samples
            if lang_labels[:, i].sum() >= 100:  # Minimum samples threshold
                thresholds['per_language'][str(lang)][class_name] = optimize_threshold(
                    lang_labels[:, i],
                    lang_preds[:, i],
                    n_steps=30  # Fewer iterations for per-language optimization
                )
            else:
                # Use global threshold if not enough samples
                thresholds['per_language'][str(lang)][class_name] = thresholds['global'][class_name]
    
    return thresholds

def run_inference(model, data_loader, device, desc="Evaluating"):
    """Run the model over a DataLoader and return raw predictions, labels, and langs.

    Split out from `evaluate_model` so the same loop can be reused for both the
    validation pass (threshold tuning only) and the test pass (headline
    metrics), without duplicating the batch loop.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_langs = []

    total_samples = len(data_loader.dataset)
    total_batches = len(data_loader)

    logger.info(f"\nRunning inference on {total_samples:,} samples in {total_batches} batches")
    progress_bar = tqdm(
        data_loader,
        desc=desc,
        total=total_batches,
        unit="batch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.inference_mode():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            langs = batch['lang'].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=batch['lang'].to(device)
            )

            predictions = outputs['probabilities'].cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels)
            all_langs.append(langs)

            # Update progress bar description with batch size
            progress_bar.set_description(f"{desc} ({len(input_ids)} samples/batch)")

    # Concatenate all batches with progress bar
    logger.info("\nProcessing results...")
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    langs = np.concatenate(all_langs)

    return predictions, labels, langs

def evaluate_model(model, data_loader, device, output_dir, thresholds=None, desc="Evaluating"):
    """Run inference on one split, compute metrics, and save results/plots.

    `thresholds`, if given, must be the dict returned by
    `calculate_optimal_thresholds` (typically computed on a separate
    validation split) and is applied as-is instead of being re-tuned on this
    split. Leaving it as None reproduces the old behavior of tuning
    thresholds on the same data being reported on.
    """
    predictions, labels, langs = run_inference(model, data_loader, device, desc=desc)

    logger.info(f"Computing metrics for {len(predictions):,} samples...")

    # Calculate metrics with progress indication
    results = calculate_metrics(predictions, labels, langs, thresholds=thresholds)

    # Save results with progress indication
    logger.info("Saving evaluation results...")
    save_results(
        results=results,
        predictions=predictions,
        labels=labels,
        langs=langs,
        output_dir=output_dir
    )

    # Plot metrics
    logger.info("Generating metric plots...")
    plot_metrics(results, output_dir, predictions=predictions, labels=labels)

    logger.info("Evaluation complete!")
    return results, predictions

def calculate_metrics(predictions, labels, langs, thresholds=None):
    """Calculate detailed metrics.

    If `thresholds` is None, optimal thresholds are tuned on this same
    predictions/labels array -- the old single-split protocol, which is
    optimistically biased because the thresholds are fit and reported on the
    same data (see docs/KNOWN_ISSUES.md and docs/RESULTS.md). Pass thresholds
    computed by `calculate_optimal_thresholds` on a separate validation split
    to get an unbiased estimate on this split instead.
    """
    results = {
        'default_thresholds': {
            'overall': {},
            'per_language': {},
            'per_class': {}
        },
        'optimized_thresholds': {
            'overall': {},
            'per_language': {},
            'per_class': {}
        }
    }
    
    # Default threshold of 0.5
    DEFAULT_THRESHOLD = 0.5
    
    # Calculate metrics with default threshold
    logger.info("Computing metrics with default threshold (0.5)...")
    binary_predictions_default = (predictions > DEFAULT_THRESHOLD).astype(int)
    results['default_thresholds']['overall'] = calculate_overall_metrics(predictions, labels, binary_predictions_default)
    
    # Calculate per-language metrics with default threshold
    unique_langs = np.unique(langs)
    logger.info(f"Computing per-language metrics with default threshold...")
    for lang in tqdm(unique_langs, desc="Language metrics (default)", ncols=100):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        lang_binary_preds = binary_predictions_default[lang_mask]
        
        results['default_thresholds']['per_language'][str(lang)] = calculate_overall_metrics(
            lang_preds, lang_labels, lang_binary_preds
        )
        results['default_thresholds']['per_language'][str(lang)]['sample_count'] = int(lang_mask.sum())
    
    # Calculate per-class metrics with default threshold
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    logger.info("Computing per-class metrics with default threshold...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Class metrics (default)", ncols=100)):
        results['default_thresholds']['per_class'][class_name] = calculate_class_metrics(
            labels[:, i],
            predictions[:, i],
            binary_predictions_default[:, i],
            DEFAULT_THRESHOLD
        )
    
    # Calculate optimal thresholds and corresponding metrics
    if thresholds is None:
        logger.info("No pre-tuned thresholds given -- computing optimal thresholds on this same split "
                     "(optimistically biased; pass thresholds tuned on a separate split to avoid this).")
        thresholds = calculate_optimal_thresholds(predictions, labels, langs)
    else:
        logger.info("Applying pre-tuned thresholds (frozen from a separate split)...")

    # Apply optimal thresholds
    logger.info("Computing metrics with optimized thresholds...")
    binary_predictions_opt = np.zeros_like(predictions, dtype=int)
    for i, class_name in enumerate(toxicity_types):
        opt_threshold = thresholds['global'][class_name]['threshold']
        binary_predictions_opt[:, i] = (predictions[:, i] > opt_threshold).astype(int)
    
    # Calculate overall metrics with optimized thresholds
    results['optimized_thresholds']['overall'] = calculate_overall_metrics(predictions, labels, binary_predictions_opt)
    
    # Calculate per-language metrics with optimized thresholds
    logger.info(f"Computing per-language metrics with optimized thresholds...")
    for lang in tqdm(unique_langs, desc="Language metrics (optimized)", ncols=100):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        lang_binary_preds = binary_predictions_opt[lang_mask]
        
        results['optimized_thresholds']['per_language'][str(lang)] = calculate_overall_metrics(
            lang_preds, lang_labels, lang_binary_preds
        )
        results['optimized_thresholds']['per_language'][str(lang)]['sample_count'] = int(lang_mask.sum())
    
    # Calculate per-class metrics with optimized thresholds
    logger.info("Computing per-class metrics with optimized thresholds...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Class metrics (optimized)", ncols=100)):
        opt_threshold = thresholds['global'][class_name]['threshold']
        results['optimized_thresholds']['per_class'][class_name] = calculate_class_metrics(
            labels[:, i],
            predictions[:, i],
            binary_predictions_opt[:, i],
            opt_threshold
        )
    
    # Store the thresholds used
    results['thresholds'] = thresholds
    
    return results

def calculate_overall_metrics(predictions, labels, binary_predictions):
    """Calculate overall metrics for multi-label classification"""
    metrics = {}
    
    # AUC scores (threshold independent)
    # A class with only one label value present makes roc_auc_score return NaN
    # (with a warning) rather than raise, so an except ValueError never fires and
    # bare NaN -- which is not valid JSON -- reaches the results file. Score only
    # the classes that are actually well defined, and record which were skipped.
    usable = [i for i in range(labels.shape[1]) if 0 < labels[:, i].sum() < len(labels)]
    degenerate = [i for i in range(labels.shape[1]) if i not in usable]
    if degenerate:
        logger.warning(
            f"Skipping AUC for degenerate class indices {degenerate} "
            f"(only one label value present)"
        )
    metrics['auc_skipped_class_indices'] = degenerate
    if usable:
        try:
            metrics['auc_macro'] = float(
                roc_auc_score(labels[:, usable], predictions[:, usable], average='macro'))
            metrics['auc_weighted'] = float(
                roc_auc_score(labels[:, usable], predictions[:, usable], average='weighted'))
        except ValueError as e:
            logger.error(f"AUC computation failed: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
    else:
        logger.error("No class has both positive and negative samples; AUC undefined")
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    if not np.isfinite(metrics['auc_macro']):
        logger.error("AUC macro is not finite; coercing to 0.0")
        metrics['auc_macro'] = 0.0
    if not np.isfinite(metrics['auc_weighted']):
        metrics['auc_weighted'] = 0.0
    
    # Precision, recall, F1 (threshold dependent)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro', zero_division=1
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', zero_division=1
    )
    
    metrics.update({
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    })
    
    # Hamming loss
    metrics['hamming_loss'] = hamming_loss(labels, binary_predictions)
    
    # Exact match
    metrics['exact_match'] = accuracy_score(labels, binary_predictions)
    
    return metrics

def calculate_class_metrics(labels, predictions, binary_predictions, threshold):
    """Calculate metrics for a single class"""
    # Handle case where there are no positive samples
    if labels.sum() == 0:
        return {
            'auc': 0.0,
            'threshold': threshold,
            'precision': 1.0 if binary_predictions.sum() == 0 else 0.0,
            'recall': 1.0,  # All true negatives were correctly identified
            'f1': 1.0 if binary_predictions.sum() == 0 else 0.0,
            'support': 0,
            'brier': brier_score_loss(labels, predictions),
            'true_positives': 0,
            'false_positives': int(binary_predictions.sum()),
            'true_negatives': int((1 - binary_predictions).sum()),
            'false_negatives': 0
        }
    
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.0
    
    # Calculate metrics with zero_division=1
    precision = precision_score(labels, binary_predictions, zero_division=1)
    recall = recall_score(labels, binary_predictions, zero_division=1)
    f1 = f1_score(labels, binary_predictions, zero_division=1)
    
    metrics = {
        'auc': auc,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': int(labels.sum()),
        'brier': brier_score_loss(labels, predictions)
    }
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    })
    
    return metrics

def save_results(results, predictions, labels, langs, output_dir):
    """Save evaluation results and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for further analysis
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        labels=labels,
        langs=langs
    )
    
    # Log summary of results
    logger.info("\nResults Summary:")
    logger.info("\nDefault Threshold (0.5):")
    logger.info(f"Macro F1: {results['default_thresholds']['overall']['f1_macro']:.3f}")
    logger.info(f"Weighted F1: {results['default_thresholds']['overall']['f1_weighted']:.3f}")
    
    logger.info("\nOptimized Thresholds:")
    logger.info(f"Macro F1: {results['optimized_thresholds']['overall']['f1_macro']:.3f}")
    logger.info(f"Weighted F1: {results['optimized_thresholds']['overall']['f1_weighted']:.3f}")
    
    # Log threshold comparison
    if 'thresholds' in results:
        logger.info("\nOptimal Thresholds:")
        for class_name, data in results['thresholds']['global'].items():
            logger.info(f"{class_name:>12}: {data['threshold']:.3f} (F1: {data['f1_score']:.3f})")

def plot_metrics(results, output_dir, predictions=None, labels=None):
    """Generate visualization plots comparing default vs optimized thresholds"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot comparison of metrics between default and optimized thresholds
    if results.get('default_thresholds') and results.get('optimized_thresholds'):
        plt.figure(figsize=(15, 8))
        
        # Get metrics to compare
        metrics = ['precision_macro', 'recall_macro', 'f1_macro']
        default_values = [results['default_thresholds']['overall'][m] for m in metrics]
        optimized_values = [results['optimized_thresholds']['overall'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, default_values, width, label='Default Threshold (0.5)')
        plt.bar(x + width/2, optimized_values, width, label='Optimized Thresholds')
        
        plt.ylabel('Score')
        plt.title('Comparison of Default vs Optimized Thresholds')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'threshold_comparison.png'))
        plt.close()
        
        # Plot per-class comparison
        plt.figure(figsize=(15, 8))
        toxicity_types = list(results['default_thresholds']['per_class'].keys())
        
        default_f1 = [results['default_thresholds']['per_class'][c]['f1'] for c in toxicity_types]
        optimized_f1 = [results['optimized_thresholds']['per_class'][c]['f1'] for c in toxicity_types]
        
        x = np.arange(len(toxicity_types))
        width = 0.35
        
        plt.bar(x - width/2, default_f1, width, label='Default Threshold (0.5)')
        plt.bar(x + width/2, optimized_f1, width, label='Optimized Thresholds')
        
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Score Comparison')
        plt.xticks(x, toxicity_types, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_comparison.png'))
        plt.close()
        
        # Plot confusion matrices for each class with both default and optimized thresholds
        if predictions is not None and labels is not None:
            # Create a figure for each toxicity class
            for i, class_name in enumerate(toxicity_types):
                # Get default and optimized binary predictions
                default_threshold = 0.5
                opt_threshold = results['thresholds']['global'][class_name]['threshold']
                
                default_preds = (predictions[:, i] > default_threshold).astype(int)
                opt_preds = (predictions[:, i] > opt_threshold).astype(int)
                
                # Create confusion matrices
                cm_default = confusion_matrix(labels[:, i], default_preds)
                cm_opt = confusion_matrix(labels[:, i], opt_preds)
                
                # Save raw confusion matrices
                np.save(os.path.join(plots_dir, f'confusion_matrix_{class_name}_default.npy'), cm_default)
                np.save(os.path.join(plots_dir, f'confusion_matrix_{class_name}_optimized.npy'), cm_opt)
                
                # Normalize confusion matrices for visualization
                cm_default_norm = cm_default.astype('float') / (cm_default.sum(axis=1)[:, np.newaxis] + 1e-10)
                cm_opt_norm = cm_opt.astype('float') / (cm_opt.sum(axis=1)[:, np.newaxis] + 1e-10)
                
                # Plot default threshold confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cm_default_norm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Normalized Confusion Matrix - {class_name}\nDefault Threshold (0.5)')
                plt.colorbar()
                
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
                plt.yticks(tick_marks, ['Negative', 'Positive'])
                
                # Add text annotations
                thresh = cm_default_norm.max() / 2.
                for i, j in np.ndindex(cm_default_norm.shape):
                    plt.text(j, i, f'{cm_default[i, j]}\n({cm_default_norm[i, j]:.2f})',
                            ha="center", va="center",
                            color="white" if cm_default_norm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{class_name}_default.png'))
                plt.close()
                
                # Plot optimized threshold confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(cm_opt_norm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Normalized Confusion Matrix - {class_name}\nOptimized Threshold ({opt_threshold:.3f})')
                plt.colorbar()
                
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
                plt.yticks(tick_marks, ['Negative', 'Positive'])
                
                # Add text annotations
                thresh = cm_opt_norm.max() / 2.
                for i, j in np.ndindex(cm_opt_norm.shape):
                    plt.text(j, i, f'{cm_opt[i, j]}\n({cm_opt_norm[i, j]:.2f})',
                            ha="center", va="center",
                            color="white" if cm_opt_norm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{class_name}_optimized.png'))
                plt.close()
            
            # Create a combined confusion matrix report
            with open(os.path.join(plots_dir, 'confusion_matrix_report.md'), 'w') as f:
                f.write('# Confusion Matrix Report\n\n')
                f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
                
                for class_name in toxicity_types:
                    default_threshold = 0.5
                    opt_threshold = results['thresholds']['global'][class_name]['threshold']
                    
                    # Get metrics
                    default_metrics = results['default_thresholds']['per_class'][class_name]
                    opt_metrics = results['optimized_thresholds']['per_class'][class_name]
                    
                    f.write(f'## {class_name}\n\n')
                    f.write('### Default Threshold (0.5)\n\n')
                    f.write('| | Predicted Negative | Predicted Positive |\n')
                    f.write('|-|-|-|\n')
                    f.write(f'| True Negative | {default_metrics["true_negatives"]} | {default_metrics["false_positives"]} |\n')
                    f.write(f'| True Positive | {default_metrics["false_negatives"]} | {default_metrics["true_positives"]} |\n\n')
                    f.write(f'Precision: {default_metrics["precision"]:.4f}\n\n')
                    f.write(f'Recall: {default_metrics["recall"]:.4f}\n\n')
                    f.write(f'F1 Score: {default_metrics["f1"]:.4f}\n\n')
                    
                    f.write('### Optimized Threshold ({:.4f})\n\n'.format(opt_threshold))
                    f.write('| | Predicted Negative | Predicted Positive |\n')
                    f.write('|-|-|-|\n')
                    f.write(f'| True Negative | {opt_metrics["true_negatives"]} | {opt_metrics["false_positives"]} |\n')
                    f.write(f'| True Positive | {opt_metrics["false_negatives"]} | {opt_metrics["true_positives"]} |\n\n')
                    f.write(f'Precision: {opt_metrics["precision"]:.4f}\n\n')
                    f.write(f'Recall: {opt_metrics["recall"]:.4f}\n\n')
                    f.write(f'F1 Score: {opt_metrics["f1"]:.4f}\n\n')
                    f.write('---\n\n')

def main():
    parser = argparse.ArgumentParser(description='Evaluate toxic comment classifier')
    parser.add_argument('--model_path', type=str, 
                      default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to model directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str,
                      help='Specific checkpoint to evaluate (e.g., checkpoint_epoch05_20240213). If not specified, uses latest.')
    parser.add_argument('--val_file', type=str, default='dataset/split/val.csv',
                      help='Validation dataset. Per-class thresholds are tuned on this split. Ignored when --single_split_eval is set.')
    parser.add_argument('--test_file', type=str, default='dataset/split/test.csv',
                      help='Held-out test dataset. Headline metrics are reported on this split, using thresholds frozen from --val_file.')
    parser.add_argument('--single_split_eval', action='store_true',
                      help='Reproduce the old protocol: tune thresholds and report metrics on --test_file alone, ignoring --val_file. '
                           'Optimistically biased (thresholds are fit and scored on the same data) -- see docs/KNOWN_ISSUES.md and '
                           'docs/RESULTS.md. Kept only for comparison against old results; not the default.')
    parser.add_argument('--dynamic_padding', action='store_true',
                      help='Tokenize samples without fixed-length padding instead of padding every sample to --max_length. '
                           'Needs a length-aware collate_fn to batch variable-length samples, which this script does not '
                           'provide, so it defaults to off here to keep this DataLoader working standalone.')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Base directory to save results')
    parser.add_argument('--num_workers', type=int, default=16,
                      help='Number of workers for data loading')
    parser.add_argument('--cache_dir', type=str, default='cached_data',
                      help='Directory to store cached tokenized data')
    parser.add_argument('--force_retokenize', action='store_true',
                      help='Force retokenization even if cache exists')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch per worker')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length for tokenization (must match training; see docs/KNOWN_ISSUES.md issue #7)')
    parser.add_argument('--gc_frequency', type=int, default=500,
                      help='Frequency of garbage collection')
    parser.add_argument('--label_columns', nargs='+', 
                      default=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
                      help='List of label column names')
    
    args = parser.parse_args()
    
    # Create timestamped directory for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save evaluation parameters
    eval_params = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'checkpoint': args.checkpoint,
        'val_file': args.val_file,
        'test_file': args.test_file,
        'single_split_eval': args.single_split_eval,
        'dynamic_padding': args.dynamic_padding,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'cache_dir': args.cache_dir,
        'force_retokenize': args.force_retokenize,
        'prefetch_factor': args.prefetch_factor,
        'max_length': args.max_length,
        'gc_frequency': args.gc_frequency,
        'label_columns': args.label_columns
    }
    with open(os.path.join(eval_dir, 'eval_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    results = None
    
    try:
        # Load model
        print("Loading multi-language toxic comment classifier model...")
        model, tokenizer, device = load_model(args.model_path)
        
        if model is None:
            return
            
        def load_split(file_path, split_name):
            """Load one CSV split into a ToxicDataset + DataLoader."""
            print(f"\nLoading {split_name} dataset from {file_path}...")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df):,} {split_name} samples")

            missing_columns = [col for col in args.label_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing label columns in {split_name} dataset: {missing_columns}")

            dataset = ToxicDataset(df, tokenizer, args)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True if args.num_workers > 0 else False,
                drop_last=False
            )
            return loader

        thresholds_path = None
        headline_split = args.test_file

        if args.single_split_eval:
            # Legacy protocol, kept only for comparison: thresholds are tuned
            # and reported on the same file, which is optimistically biased --
            # see docs/KNOWN_ISSUES.md and docs/RESULTS.md. --val_file is
            # ignored in this mode.
            logger.warning(
                f"--single_split_eval: tuning and reporting thresholds on the same file "
                f"({args.test_file}). This reproduces the old, optimistically biased protocol."
            )
            test_loader = load_split(args.test_file, "test")
            results, predictions = evaluate_model(model, test_loader, device, eval_dir)
        else:
            # Correct protocol: tune thresholds on validation, freeze them,
            # and report headline metrics on the held-out test split.
            val_loader = load_split(args.val_file, "validation")
            print("\nTuning per-class thresholds on the validation split...")
            val_predictions, val_labels, val_langs = run_inference(
                model, val_loader, device, desc="Tuning (val)"
            )
            thresholds = calculate_optimal_thresholds(val_predictions, val_labels, val_langs)

            thresholds_path = os.path.join(eval_dir, 'tuned_thresholds.json')
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds, f, indent=2)
            print(f"Saved thresholds tuned on {args.val_file} to {thresholds_path}")

            test_loader = load_split(args.test_file, "test")
            print("\nEvaluating on the held-out test split with thresholds frozen from validation...")
            results, predictions = evaluate_model(
                model, test_loader, device, eval_dir, thresholds=thresholds, desc="Evaluating (test)"
            )

        # Print a detailed summary of results
        print(f"\nEvaluation Results Summary (headline split: {headline_split}):")
        print(f"Default Threshold (0.5):")
        print(f"  - Macro F1: {results['default_thresholds']['overall']['f1_macro']:.3f}")
        print(f"  - Weighted F1: {results['default_thresholds']['overall']['f1_weighted']:.3f}")

        print(f"\nOptimized Thresholds:")
        print(f"  - Macro F1: {results['optimized_thresholds']['overall']['f1_macro']:.3f}")
        print(f"  - Weighted F1: {results['optimized_thresholds']['overall']['f1_weighted']:.3f}")

        # Print optimal thresholds
        if 'thresholds' in results:
            print("\nOptimal Thresholds:")
            for class_name, data in results['thresholds']['global'].items():
                print(f"  - {class_name:>12}: {data['threshold']:.3f} (F1: {data['f1_score']:.3f})")

        if thresholds_path:
            print(f"\nTuned thresholds saved to {thresholds_path}")
        print(f"\nEvaluation complete! Results saved to {eval_dir}")
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()