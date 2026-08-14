# Model & training

## Architecture

```
input_ids, attention_mask, lang_ids
        │
        ▼
XLM-RoBERTa-large  (24 layers, hidden 1024, 16 heads)
        │  last_hidden_state [B, S, 1024]
        ▼
single extra self-attention block  (own q/k/v projections and scaling)
        │  + language bias  ← INERT, see KNOWN_ISSUES.md #1
        ▼
post-attention: Linear(1024→1024) → LayerNorm → GELU
        │
        ▼
[CLS] position → Linear(1024→512) → LayerNorm → GELU → Linear(512→6)
        │
        ▼
6 independent sigmoid outputs
```

Defined in `model/language_aware_transformer.py`. Because the language bias cancels under softmax
([issue #1](KNOWN_ISSUES.md#1-the-language-aware-attention-is-a-no-op)), the effective model is
XLM-R-large plus one extra self-attention block and an MLP head, with no language conditioning.

Loss is focal loss over BCE-with-logits (`LanguageAwareFocalLoss`, `model/train.py:173`) with
`alpha=0.25`, `gamma=2.0` — the per-language weighting path never activates
([issue #3](KNOWN_ISSUES.md#3-class-weighting-never-runs)). Sampling uses
`MultilabelStratifiedSampler` (`model/data/sampler.py`) to keep label and language mix stable
across batches.

## Training configuration

Values as they actually appear in `model/training_config.py`.

| Parameter | Value |
|---|---|
| base model | `xlm-roberta-large` |
| max_length | 512 |
| hidden_size / heads | 1024 / 16 |
| batch_size | 128 |
| grad_accum_steps | 1 |
| epochs | 6 configured / **3 completed** |
| optimizer | AdamW |
| lr | 2e-5 |
| scheduler | `CosineAnnealingWarmRestarts`, `num_cycles=2`, `min_lr_ratio=0.01` (no warmup — [issue #5](KNOWN_ISSUES.md#5-no-learning-rate-warmup)) |
| weight_decay | 2e-7 |
| max_grad_norm | 1.0 |
| model_dropout | 0.0 |
| freeze_layers | 8 (not implemented as described — [issue #4](KNOWN_ISSUES.md#4-freeze_layers-does-not-freeze-layers)) |
| label_smoothing | 0.01 (unused — [issue #8](KNOWN_ISSUES.md#8-label_smoothing-is-unused)) |
| mixed_precision | fp16 (`torch.cuda.amp.GradScaler`) |
| activation_checkpointing | enabled |
| gc_frequency | 500 |

Trained on 2× GPU; each checkpoint is ~2.2 GB. Logging goes to Weights & Biases when a valid API
key is present, otherwise it degrades to file logs under `logs/`.

## Files

```
model/
├── language_aware_transformer.py   # model definition
├── train.py                        # training loop, focal loss, checkpointing
├── training_config.py              # TrainingConfig, MetricsTracker, DynamicClassWeights (unused)
├── predict.py                      # single/batch prediction helpers
├── inference_optimized.py          # OptimizedToxicityClassifier used by the demo apps
├── hyperparameter_tuning.py        # Optuna sweep
├── data/sampler.py                 # MultilabelStratifiedSampler
└── evaluation/evaluate.py          # ToxicDataset, metrics, threshold search, plots
```
