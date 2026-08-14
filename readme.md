# Multilingual Toxic Comment Classification

Multi-label toxicity classification across 7 languages (en, ru, tr, es, fr, it, pt), built on
XLM-RoBERTa-large. Predicts 6 non-exclusive labels: `toxic`, `severe_toxic`, `obscene`, `threat`,
`insult`, `identity_hate`.

> **Status: archived / not maintained.** This was a learning project. It trains, evaluates, and
> serves predictions, but it has known correctness bugs that are documented in
> [Known issues](#known-issues) rather than fixed. Please read that section before reusing
> anything here or citing the numbers.

---

## Results

Measured with `model/evaluation/evaluate.py` on the checkpoint at
`weights/toxic_classifier_xlm-roberta-large/checkpoint_epoch02_20250401_141908`
(run `evaluation_results/eval_20250426_160137/`).

**Read these caveats first — they materially affect the numbers:**

- Metrics are computed on **`dataset/split/val.csv`**, not `test.csv`. The held-out test split
  exists but was never used.
- The "tuned threshold" column optimizes per-class thresholds on **the same split it reports on**.
  That gain is not an estimate of generalization.
- Evaluation truncates at `max_length=128` while training used `512` (see [Known issues](#known-issues)).
- There is **no baseline comparison and no ablation**, so none of this is evidence that the
  custom architecture beats plain XLM-R fine-tuning. See [Known issues](#known-issues) #1.

### Overall

| Metric | @ threshold 0.5 | @ tuned thresholds |
|---|---|---|
| AUC-ROC (macro) | **0.912** | 0.912 |
| AUC-ROC (weighted) | 0.931 | 0.931 |
| F1 (macro) | 0.523 | 0.604 |
| F1 (weighted) | 0.747 | — |
| Precision (macro) | 0.702 | 0.578 |
| Recall (macro) | 0.469 | 0.640 |
| Hamming loss | 0.085 | — |
| Exact match | 0.646 | 0.619 |

### Per class (threshold 0.5)

| Class | AUC | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| toxic | 0.962 | 0.907 | 0.889 | 0.898 | 17,697 |
| obscene | 0.925 | 0.764 | 0.686 | 0.723 | 8,626 |
| insult | 0.896 | 0.698 | 0.717 | 0.708 | 10,199 |
| severe_toxic | 0.902 | 0.562 | 0.156 | 0.244 | 1,655 |
| threat | 0.898 | 0.604 | 0.187 | 0.285 | 760 |
| identity_hate | 0.887 | 0.676 | 0.176 | 0.280 | 1,878 |

Ranking quality is decent everywhere (AUC 0.89–0.96), but recall on the three rare classes
collapses at a 0.5 threshold — the model is well-ordered but badly calibrated. Lowering thresholds
to ~0.37–0.39 raises rare-class F1 to 0.41–0.45. Note that the class-weighting code intended to
address this never actually runs (issue #3).

### Per language (macro AUC, threshold 0.5)

| Language | en | es | it | ru | fr | pt | tr |
|---|---|---|---|---|---|---|---|
| AUC | 0.945 | 0.914 | 0.912 | 0.906 | 0.905 | 0.905 | 0.895 |

English leads by ~4 points; Turkish trails. Since language conditioning is inert (issue #1), this
spread reflects XLM-R's own pretraining coverage and data quality, not any per-language modelling.

---

## Known issues

These are real bugs found by auditing the code after the fact. They are listed because the
results above cannot be interpreted without them.

1. **The language-aware attention is a no-op.** In
   `model/language_aware_transformer.py:296-298`, the language bias is shaped `[B, H, S, 1]` and
   added to attention scores of shape `[B, H, S, S]`. It therefore broadcasts as a constant across
   the key axis, and softmax is shift-invariant along that axis — so it cancels exactly.
   `lang_embed` and `lang_proj` receive zero gradient. `LanguageAwareClassifier` (line 24), which
   *does* concatenate language embeddings correctly, is never instantiated.

   **Consequence:** during training, `lang_ids` has no effect on the model output. What was
   actually trained is XLM-R-large + one extra self-attention block + an MLP head — a
   language-agnostic model. The project's central premise is unverified and, as implemented,
   inactive.

2. **No validation loop.** `create_dataloaders` in `model/train.py:589` is called with
   `val_dataset=None` and ignores it. `MetricsTracker.update_validation` is never called and
   `best_auc` stays `0.0`. There is no model selection, no early stopping, and no in-training
   metric despite those being wired up in `MetricsTracker`. Checkpoints are saved every epoch and
   one was picked by hand.

3. **Class weighting never runs.** `DynamicClassWeights` (`model/training_config.py:14`) is defined
   but never instantiated. `model/train.py:235` guards on `config.lang_weights`, which
   `TrainingConfig` never sets, so every step falls through to uniform focal-loss parameters
   (`alpha=0.25`, `gamma=2.0`). The per-language weights in `weights/language_class_weights.json`
   and all of `analysis/compute_class_weights.py` were unused at train time.

4. **`freeze_layers` does not freeze layers.** `model/train.py:116` freezes
   `list(base_model.parameters())[:8]` — the first 8 parameter *tensors* (embeddings plus part of
   layer 0), not 8 transformer layers. The assertion on line 119 checks the same wrong slice, so it
   passes vacuously. `TrainingConfig.validate_model_config`, which checks correctly against
   `encoder.layer`, is never called.

5. **No learning-rate warmup.** `warmup_steps` is computed at `model/train.py:443` and logged, then
   never used. The scheduler is a bare `CosineAnnealingWarmRestarts`, despite `warmup_ratio=0.1`
   in the config.

6. **Hardcoded "per-language" tables are copy-paste.** `LANG_THRESHOLD_ADJUSTMENTS`
   (`model/language_aware_transformer.py:322`) is described as derived from statistical patterns,
   but all six non-English rows are identical. The same applies to `class_adjustments` in
   `model/training_config.py:153`, where the comments contradict the values. Both blocks are dead
   in any case: the threshold block only fires under `mode='inference'`, which neither `train.py`
   nor `evaluate.py` passes.

7. **Train/eval sequence-length mismatch.** Training used `max_length=512`
   (`model/training_config.py:279`); `evaluate.py` defaults to `128` (line 760).

8. **`label_smoothing=0.01`** in the config is never referenced by any loss.

9. **Redundant loss computation.** The model computes a loss inside `forward()`
   (`language_aware_transformer.py:351`) that `train.py:268` immediately overwrites.

10. **`CUDA_LAUNCH_BLOCKING='1'`** is set unconditionally at `model/train.py:623`. This serializes
    every CUDA kernel and significantly slows training; it is a debugging flag left enabled.

11. **Training did not finish.** The config specifies 6 epochs; only 3 completed. The run died on a
    wandb auth error (`logs/train_20250401_143955.log`). The evaluated checkpoint is epoch 2.

### What would need to happen to make the core claim testable

Reshape the language bias to `[B, H, 1, S]` (or add it to keys rather than post-softmax scores),
add a validation loop reporting per-language AUC, tune thresholds on `val` and report on `test`,
then run the decisive ablation: real `lang_ids` vs. shuffled `lang_ids`. If the gap is zero,
language conditioning does not help on this data — which is itself a legitimate finding.

---

## Data

- **Corpus:** 356,580 comments across 7 languages, split 285,264 train / 35,658 val / 35,658 test.
- **Distribution:** roughly balanced by language (~14.5% each; English 13.0%).
- **Provenance:** built from the Jigsaw toxic comment data extended to 7 languages
  (`dataset/raw/MULTILINGUAL_TOXIC_DATASET_360K_7LANG.csv`). **This repo does not include the
  script that produced the multilingual corpus**, so that step is not reproducible from here.
- **Augmentation:** rare classes (notably `threat`) were topped up with synthetic samples generated
  by Mistral-7B-Instruct in 4-bit (`augmentation/toxic_augment.py`,
  `augmentation/threat_augment.py`), filtered through a lightweight sklearn validator and a
  language check. Labels on synthetic samples come from the generating prompt, so they are weakly
  labelled.

### Split hygiene

`utils/split_dataset.py` does multilabel-stratified splitting, exact-hash deduplication *before*
splitting, distribution verification, and a contamination check. Verified after the fact:

- Exact `comment_text` overlap between train/val/test: **0**.
- Near-duplicate leakage (char 3–4-gram TF-IDF, cosine ≥ 0.9 against train): **3.8%** of English
  val samples, **0.6%** of Russian. Non-zero — augmentation happens before the split, so
  LLM-generated variants of the same seed can land on both sides — but small.

---

## Architecture

```
input_ids, attention_mask, lang_ids
        │
        ▼
XLM-RoBERTa-large  (24 layers, hidden 1024, 16 heads)
        │  last_hidden_state [B, S, 1024]
        ▼
single extra self-attention block  (q/k/v projections, own scaling)
        │  + language bias  ← INERT, see Known issues #1
        ▼
post-attention: Linear(1024→1024) → LayerNorm → GELU
        │
        ▼
take [CLS] position → Linear(1024→512) → LayerNorm → GELU → Linear(512→6)
        │
        ▼
6 independent sigmoid outputs
```

Loss is focal loss over BCE-with-logits (`LanguageAwareFocalLoss`, `model/train.py:173`) with
`alpha=0.25`, `gamma=2.0`. Sampling uses `MultilabelStratifiedSampler`
(`model/data/sampler.py`) to keep label and language mix stable across batches.

## Training configuration

These are the values actually in `model/training_config.py`.

| Parameter | Value |
|---|---|
| base model | `xlm-roberta-large` |
| max_length | 512 |
| batch_size | 128 |
| grad_accum_steps | 1 |
| epochs | 6 configured / **3 completed** |
| optimizer | AdamW |
| lr | 2e-5 |
| scheduler | `CosineAnnealingWarmRestarts`, `num_cycles=2`, `min_lr_ratio=0.01` (no warmup — issue #5) |
| weight_decay | 2e-7 |
| max_grad_norm | 1.0 |
| model_dropout | 0.0 |
| freeze_layers | 8 (not implemented as described — issue #4) |
| mixed_precision | fp16 (`torch.cuda.amp.GradScaler`) |
| activation_checkpointing | enabled |

Trained on 2× GPU; each checkpoint is ~2.2 GB. Logging goes to Weights & Biases when a valid API
key is present, otherwise it degrades to file logs under `logs/`.

---

## Repository layout

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

augmentation/     # Mistral-7B synthetic generation for rare classes
analysis/         # class-weight computation, loss/ROC curve plots, language distribution
utils/            # dataset build, split, dedup, leakage check, attention/attribution viz
app.py            # Gradio demo
streamlit_app.py  # Streamlit demo
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights are not committed. You need a checkpoint at
`weights/toxic_classifier_xlm-roberta-large/` (or point the apps elsewhere) before inference works.

### Train

```bash
python -m model.train        # expects dataset/split/train.csv
```

### Evaluate

```bash
python -m model.evaluation.evaluate \
    --test_file dataset/split/test.csv \
    --max_length 512
```

Both flags are worth setting explicitly: the defaults are `val.csv` and `128`, which is how the
numbers above were produced.

### Demos

```bash
python app.py                    # Gradio
streamlit run streamlit_app.py   # Streamlit
```

A `Dockerfile`, `docker-compose.yml`, and `.devcontainer/` are included.

---

## Acknowledgements

Base model: [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large) (Conneau et al., 2020).
Label schema and English data: Jigsaw / Conversation AI Toxic Comment Classification.
Synthetic augmentation: Mistral-7B-Instruct-v0.3.
