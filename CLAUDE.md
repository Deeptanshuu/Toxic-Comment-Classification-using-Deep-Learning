# CLAUDE.md

Instructions for an AI assistant working in this repository.

## What this is

Multi-label toxic-comment classification: 6 non-exclusive labels, 7 languages, fine-tuning
XLM-RoBERTa-large with a custom attention block that conditions on language ID. The branch
`fix/training-correctness` is an audit-and-fix pass over a 2025 run that shipped with eleven
bugs; `main` still has the original, unfixed code — verified: `main`'s `model/train.py` still
freezes `list(_model.base_model.parameters())[:8]` and has no `model/tracking.py` at all.

## Critical context

**The April 2025 published model never fine-tuned its backbone.** Two bugs stacked: an
over-broad freeze (slicing the raw parameter list instead of naming modules caught the
embedding matrix) plus `gradient_checkpointing_enable()` defaulting to `use_reentrant=True`
on a frozen input, which silently drops the whole encoder's backward graph. Result: 4.8M of
565M parameters (0.8%) received any gradient. Every historical number in this repo describes
a frozen XLM-R feature extractor with a small trainable head on top — not a fine-tuned model.
Do not cite old numbers as if they measure the current architecture; `docs/RESULTS.md` and
`docs/MODEL.md` say what does and does not carry over.

## Landmines

| Rule | Why | Where |
|---|---|---|
| Never call `gradient_checkpointing_enable()` without `use_reentrant=False` while embeddings are frozen | `use_reentrant=True` (the default) then builds no backward graph through the encoder — no error, no warning, silently zero encoder gradient | `model/train.py:282-292` |
| The per-language attention bias must vary along the KEY axis | A bias constant along the key axis cancels exactly under softmax's shift-invariance — this was the original bug. Adding it to keys instead of queries is ALSO wrong, for the same reason | `model/language_aware_transformer.py:247-255` |
| Do not raise training's `num_workers` above 8 | Past ~4 workers buys no throughput (the GPU step is the bottleneck); each extra worker forks a copy-on-write view of a 285k-row DataFrame for no benefit | `model/training_config.py:573-593` |
| Never point a new run's `checkpoint_dir` at a directory holding another run's checkpoints | Rotation keeps only the last 3 by filename sort and will delete the OTHER run's checkpoints once yours sorts past them | `model/train.py:721-726`, `model/training_config.py:316` |

## Commands

| Task | Command |
|---|---|
| Train (tmux, survives SSH drop) | `GPUS=1 scripts/train_tmux.sh` |
| Train (foreground) | `uv run python -m model.train` |
| Evaluate (defaults are already correct) | `uv run python -m model.evaluation.evaluate` |
| Ablation control run | `TOXIC_DISABLE_LANG_CONDITIONING=1 uv run python -m model.train` |
| Training dashboard | `PY=.venv-uv/bin/python scripts/monitor.sh` — `.venv` lacks streamlit |
| MLflow UI | `scripts/mlflow_ui.sh` |
| Lint | `uv run ruff check .` |

Full explanations of all of these: `docs/DEVELOPMENT.md`.

## Conventions

- No emoji, anywhere — code, docs, commit messages.
- Tables over prose for anything enumerable: comparisons, flag lists, status.
- Cite `file:line` in docs, not just a filename, so a claim can be checked against the code.

## Where to read, not guess

| Topic | File |
|---|---|
| Architecture, freezing, the language-bias fix, loss | `docs/MODEL.md` |
| Every audited bug, fixed or still open | `docs/KNOWN_ISSUES.md` |
| Metrics, per-class/per-language, with caveats | `docs/RESULTS.md` |
| Dataset provenance, splits, leakage checks | `docs/DATA.md` / `datacard.md` |
| Environment setup, tmux layout, monitoring, evaluation flags | `docs/DEVELOPMENT.md` |

A training run is very likely live right now (`tmux ls` → session `toxic`, window `train`).
Check before assuming otherwise: do not start a second run against the same `checkpoint_dir`,
and do not kill tmux sessions you did not start.
