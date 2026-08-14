# Results

Produced by `model/evaluation/evaluate.py` on the checkpoint at
`weights/toxic_classifier_xlm-roberta-large/checkpoint_epoch02_20250401_141908`
(run: `evaluation_results/eval_20250426_160137/`).

## Caveats

These materially affect the numbers. Read them first.

- Metrics are computed on **`dataset/split/val.csv`**, not `test.csv`. The held-out test split
  exists but was never used.
- The tuned-threshold column optimizes per-class thresholds on **the same split it reports on**.
  That gain is not an estimate of generalization.
- Evaluation truncates at `max_length=128` while training used `512`
  ([issue #7](KNOWN_ISSUES.md#7-traineval-sequence-length-mismatch)).
- There is **no baseline and no ablation**, so none of this shows the custom architecture beats
  plain XLM-R fine-tuning — especially given
  [issue #1](KNOWN_ISSUES.md#1-the-language-aware-attention-is-a-no-op).
- The evaluated checkpoint is from an interrupted run (3 of 6 epochs,
  [issue #11](KNOWN_ISSUES.md#11-training-did-not-finish)).

## Overall

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

## Per class

At threshold 0.5:

| Class | AUC | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| toxic | 0.962 | 0.907 | 0.889 | 0.898 | 17,697 |
| obscene | 0.925 | 0.764 | 0.686 | 0.723 | 8,626 |
| insult | 0.896 | 0.698 | 0.717 | 0.708 | 10,199 |
| severe_toxic | 0.902 | 0.562 | 0.156 | 0.244 | 1,655 |
| threat | 0.898 | 0.604 | 0.187 | 0.285 | 760 |
| identity_hate | 0.887 | 0.676 | 0.176 | 0.280 | 1,878 |

At tuned thresholds:

| Class | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| toxic | 0.488 | 0.900 | 0.897 | 0.898 |
| obscene | 0.455 | 0.702 | 0.771 | 0.735 |
| insult | 0.463 | 0.657 | 0.785 | 0.715 |
| identity_hate | 0.373 | 0.424 | 0.477 | 0.449 |
| severe_toxic | 0.373 | 0.346 | 0.523 | 0.417 |
| threat | 0.390 | 0.437 | 0.387 | 0.410 |

Ranking quality is decent everywhere (AUC 0.89–0.96), but recall on the three rare classes
collapses at 0.5 — the model is well-ordered but badly calibrated. Dropping thresholds to
~0.37–0.39 roughly doubles rare-class F1. The class-weighting code intended to address this never
runs ([issue #3](KNOWN_ISSUES.md#3-class-weighting-never-runs)).

## Per language

Macro AUC at threshold 0.5:

| Language | AUC | n |
|---|---|---|
| en | 0.945 | 4,638 |
| es | 0.914 | 5,168 |
| it | 0.912 | 5,146 |
| ru | 0.906 | 5,193 |
| fr | 0.905 | 5,157 |
| pt | 0.905 | 5,192 |
| tr | 0.895 | 5,164 |

English leads by ~4 points; Turkish trails. Since language conditioning is inert
([issue #1](KNOWN_ISSUES.md#1-the-language-aware-attention-is-a-no-op)), this spread reflects
XLM-R's own pretraining coverage and data quality, not any per-language modelling in this repo.

## Reproducing

```bash
python -m model.evaluation.evaluate \
    --test_file dataset/split/val.csv \
    --max_length 128
```

To get an honest generalization estimate instead, point `--test_file` at `test.csv` and set
`--max_length 512` to match training.
