# Multilingual Toxic Comment Classification

Multi-label toxicity classification across 7 languages (en, ru, tr, es, fr, it, pt), built on
XLM-RoBERTa-large. Predicts 6 non-exclusive labels: `toxic`, `severe_toxic`, `obscene`, `threat`,
`insult`, `identity_hate`.

> **Status: archived / not maintained.** This was a learning project. It trains, evaluates, and
> serves predictions, but it has known correctness bugs — most importantly, the language-aware
> attention that the design is built around is a no-op, so `lang_ids` has no effect on the trained
> model. Read **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** before reusing anything here or
> citing the numbers.

## Results

| Metric | @ 0.5 | @ tuned |
|---|---|---|
| AUC-ROC (macro) | **0.912** | 0.912 |
| F1 (macro) | 0.523 | 0.604 |
| F1 (weighted) | 0.747 | — |
| Exact match | 0.646 | 0.619 |

Measured on `dataset/split/val.csv` — not the held-out test split — with thresholds tuned on that
same split, and no baseline or ablation to compare against. Per-class and per-language breakdowns
plus the full caveats are in **[docs/RESULTS.md](docs/RESULTS.md)**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights are not committed. Inference needs a checkpoint at
`weights/toxic_classifier_xlm-roberta-large/`.

```bash
python -m model.train                        # train (expects dataset/split/train.csv)

python -m model.evaluation.evaluate \        # evaluate — set both flags explicitly;
    --test_file dataset/split/test.csv \     # defaults are val.csv and 128
    --max_length 512

python app.py                                # Gradio demo
streamlit run streamlit_app.py               # Streamlit demo
```

A `Dockerfile`, `docker-compose.yml`, and `.devcontainer/` are included.

## Layout

```
model/         model definition, training loop, inference, evaluation
augmentation/  Mistral-7B synthetic generation for rare classes
analysis/      class weights, loss/ROC curves, language distribution
utils/         dataset build, split, dedup, leakage check, attention/attribution viz
app.py         Gradio demo   ·   streamlit_app.py   Streamlit demo
```

## Docs

| | |
|---|---|
| [Known issues](docs/KNOWN_ISSUES.md) | 11 audited bugs, and what would make the core claim testable |
| [Results](docs/RESULTS.md) | Full metrics per class and per language, with caveats |
| [Model & training](docs/MODEL.md) | Architecture and the training config as actually configured |
| [Data](docs/DATA.md) | Provenance, augmentation, split hygiene and measured leakage |

## Acknowledgements

Base model: [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large) (Conneau et al., 2020).
Label schema and English data: Jigsaw / Conversation AI Toxic Comment Classification.
Synthetic augmentation: Mistral-7B-Instruct-v0.3.
