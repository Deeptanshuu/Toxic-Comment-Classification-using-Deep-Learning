# Known issues

Bugs found by auditing the code after the fact. Listed because the
[results](RESULTS.md) cannot be interpreted without them. None are fixed — the project is
archived.

## 1. The language-aware attention is a no-op

In `model/language_aware_transformer.py:296-298`, the language bias is shaped `[B, H, S, 1]` and
added to attention scores of shape `[B, H, S, S]`. It therefore broadcasts as a constant across the
key axis, and softmax is shift-invariant along that axis — so it cancels exactly. `lang_embed` and
`lang_proj` receive zero gradient. `LanguageAwareClassifier` (line 24), which *does* concatenate
language embeddings correctly, is never instantiated.

**Consequence:** during training, `lang_ids` has no effect on the model output. What was actually
trained is XLM-R-large + one extra self-attention block + an MLP head — a language-agnostic model.
The project's central premise is unverified and, as implemented, inactive.

## 2. No validation loop

`create_dataloaders` (`model/train.py:589`) is called with `val_dataset=None` and ignores it.
`MetricsTracker.update_validation` is never called and `best_auc` stays `0.0`. There is no model
selection, no early stopping, and no in-training metric, despite those being wired up in
`MetricsTracker`. Checkpoints are saved every epoch and one was picked by hand.

## 3. Class weighting never runs

`DynamicClassWeights` (`model/training_config.py:14`) is defined but never instantiated.
`model/train.py:235` guards on `config.lang_weights`, which `TrainingConfig` never sets, so every
step falls through to uniform focal-loss parameters (`alpha=0.25`, `gamma=2.0`). The per-language
weights in `weights/language_class_weights.json` and all of `analysis/compute_class_weights.py`
were unused at train time. This is why rare-class recall is poor.

## 4. `freeze_layers` does not freeze layers

`model/train.py:116` freezes `list(base_model.parameters())[:8]` — the first 8 parameter *tensors*
(embeddings plus part of layer 0), not 8 transformer layers. The assertion on line 119 checks the
same wrong slice, so it passes vacuously. `TrainingConfig.validate_model_config`, which checks
correctly against `encoder.layer`, is never called.

## 5. No learning-rate warmup

`warmup_steps` is computed at `model/train.py:443` and logged, then never used. The scheduler is a
bare `CosineAnnealingWarmRestarts`, despite `warmup_ratio=0.1` in the config.

## 6. Hardcoded "per-language" tables are copy-paste

`LANG_THRESHOLD_ADJUSTMENTS` (`model/language_aware_transformer.py:322`) is described as derived
from statistical patterns, but all six non-English rows are identical. The same applies to
`class_adjustments` in `model/training_config.py:153`, where the comments contradict the values.
Both blocks are dead in any case: the threshold block only fires under `mode='inference'`, which
neither `train.py` nor `evaluate.py` passes.

## 7. Train/eval sequence-length mismatch

Training used `max_length=512` (`model/training_config.py:279`); `evaluate.py` defaults to `128`
(line 760).

## 8. `label_smoothing` is unused

`label_smoothing=0.01` in the config is never referenced by any loss.

## 9. Redundant loss computation

The model computes a loss inside `forward()` (`language_aware_transformer.py:351`) that
`train.py:268` immediately overwrites.

## 10. `CUDA_LAUNCH_BLOCKING` left enabled

Set unconditionally at `model/train.py:623`. This serializes every CUDA kernel and significantly
slows training; it is a debugging flag left on.

## 11. Training did not finish

The config specifies 6 epochs; only 3 completed. The run died on a wandb auth error
(`logs/train_20250401_143955.log`). The evaluated checkpoint is epoch 2.

---

## What would make the core claim testable

1. Reshape the language bias to `[B, H, 1, S]`, or add it to keys rather than to post-softmax
   scores.
2. Add a validation loop reporting per-language AUC, and select checkpoints on it.
3. Tune thresholds on `val`, report on `test`.
4. Run the decisive ablation: real `lang_ids` vs. shuffled `lang_ids`.

If the gap in (4) is zero, language conditioning does not help on this data — which is itself a
legitimate finding, and a more useful one than an unverified architecture.
