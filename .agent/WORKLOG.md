# Autonomous worklog

Durable state for the overnight autonomous session. Context gets auto-compacted;
this file does not. **Re-read this file first after any compaction.**

Owner went to sleep ~05:00 on 2026-08-30 and granted broad permission to improve
the model, including downloading data and making large architectural changes.

## Hard rules (do NOT violate even with broad permission)

1. NEVER publish to Hugging Face. `scripts/publish_hf.sh` stays unrun. The owner
   must choose a license first (the repo has none) and must review. Preparing is
   fine; publishing is not.
2. NEVER force-push, rewrite history, or delete the `weights/*.2025run` backup.
3. NEVER kill a training run to start another. One trainer at a time on GPU 1.
4. GPU 0 has ~6.3 GB held by ANOTHER USER's process. Do not OOM them. Only use
   GPU 0 for short, low-memory jobs, and check free memory first.
5. Keep `.venv` intact while training runs. Never run `uv run`/`uv sync` against
   it (implicit sync deletes and rebuilds it). Dashboard uses `.venv-uv`.
6. Commit freely on `fix/training-correctness`; push is fine. No PR merges.

## Baseline to beat (old epoch-2 checkpoint, thresholds tuned on val, TEST split)

AUC macro 0.9147 | F1 macro 0.6036 tuned / 0.5284 @0.5 | exact match 0.6194
per-class AUC: toxic .9666 obscene .9278 threat .9051 insult .9035
               severe_toxic .8988 identity_hate .8866

## Run log

| Run | What | Status |
|---|---|---|
| train_20260830_030414 | main retrain, 6 epochs, lang conditioning ON | RUNNING, ~06:46 |

Validation macro AUC by epoch: 1: 0.9578 | 2: 0.9697

## Plan (update as it changes)

1. [ ] Wait for main run to finish (~06:46). Do not disturb it.
2. [ ] Full test-split eval of best_model, thresholds tuned on val -> REAL headline number
3. [ ] Launch ABLATION control run (TOXIC_DISABLE_LANG_CONDITIONING=1) on GPU 1.
       This is the decisive experiment the whole project exists to answer.
4. [ ] Compare treatment vs control -> does language conditioning actually help?
5. [ ] Improvements to try, in rough value order (see IDEAS below)

## While waiting (safe, no GPU contention)

- [ ] Commit the rewritten docs
- [ ] Remove the fabricated `class_adjustments` table (training_config.py:122).
      5 of 6 non-English rows identical, comments contradict values, ~3.5% mean
      weight distortion. Applied identically to treatment and control so the
      ablation stays valid, but remove it for runs after that.
- [ ] Fix open issues: per-language threshold MRO bug (ThresholdOptimizer has
      BaseEstimator before ClassifierMixin -> KFold not StratifiedKFold),
      roc_auc_score NaN not raising, evaluate.py --dynamic_padding crash,
      token_lengths cache key, optuna undeclared
- [ ] `oversample_rare_classes()` in utils/split_dataset.py is dead code

## IDEAS to improve the score (test, do not assume)

- Ablation first; it may show lang conditioning is worthless, which changes design
- Rare-class F1 is the weak spot: severe_toxic .398, threat .419, identity_hate .437
  at baseline. AUC is high (.89-.91) so RANKING is fine and CALIBRATION is the gap.
  -> per-class threshold tuning already helps; try better calibration
- Try unfreezing embeddings now that use_reentrant=False is correct (was never a
  fair test before, since the encoder was not training at all)
- Longer training / different LR schedule; loss was still falling at epoch 2
- severe_toxic is a SUBSET of toxic in Jigsaw's scheme; independent sigmoids
  ignore that structure. Try modelling the hierarchy.
- Near-duplicate leakage (3.8% en val) inflates val; test split is cleaner
- The corpus is ~50% toxic vs a few % in real traffic. Consider reporting
  precision at a realistic base rate.

## Notes for the owner (write anything they must know here)

- (nothing yet)
