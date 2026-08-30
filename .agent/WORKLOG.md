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

Validation macro AUC by epoch: 1: 0.9578 | 2: 0.9697 | 3: 0.9799 | 4: 0.9849 | 5: 0.9868
Gains decelerate cleanly: +0.0119, +0.0102, +0.0051, +0.0019 -> converged.
Val loss ticked UP at e5 (0.0070 -> 0.0071) while train loss kept falling
(0.0140 -> 0.0089): overfitting onset. Model selection on val AUC handles it, and
this is exactly why the validation loop needed to exist. 6 epochs was the right
budget; more would likely have cost accuracy.
Baseline is 0.9147, so epoch 3 is already +0.0652. Rare classes gained most:
identity_hate +0.0908, threat +0.0755, insult +0.0725, severe_toxic +0.0713,
obscene +0.0563, toxic +0.0244. Best/worst class spread collapsed 0.080 -> 0.021.
Both losses still falling at epoch 3, so the remaining epochs should still buy something.

## Plan (update as it changes)

1. [~] Main run in progress, ~06:52. Do not disturb it.
2. [~] AUTOMATED: scripts/after_training.sh is armed in tmux window `after`.
       It waits for the trainer to exit, checks 6 epochs actually completed and a
       best checkpoint exists, then runs the test-split eval, then starts the
       ablation. It REFUSES to start the ablation if the run ended early.
3. [~] AUTOMATED by the same script. Control run uses scripts/run_ablation.py ->
       weights/toxic_classifier_xlmr_v2_ablation, MLflow tag run.kind=control.
4. [~] READY: scripts/compare_ablation.py. Paired bootstrap over test rows
       (both arms score the SAME rows, so unpaired would overstate uncertainty),
       2000 resamples, per-class and per-language breakdown. Reports effect size
       WITH significance because n=35,658 makes trivial diffs "significant".
       Smoke-tested by comparing an eval to itself: diff exactly 0.0000, p=1.000.
       Run: python scripts/compare_ablation.py <treat_eval_dir> <control_eval_dir>
       KEY TEST: if lang conditioning is real, the gain concentrates in
       NON-ENGLISH languages. A uniform lift across all 7 means a better run,
       not a working language signal. The script says this in words.
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

## Experiments run (results, not plans)

| Experiment | Result |
|---|---|
| Per-language vs global thresholds | **NEGATIVE, -0.0047 macro F1, 1/5 splits improved.** Delete the per_language block. Damage concentrates in rare classes (threat -0.0144) because 7 thresholds fit on 1/7 the data each is variance, not signal. Writeup: experiments/per_language_thresholds.md |
| Enforce label hierarchy P(child)<=P(toxic) | **NEGATIVE, -0.0001 macro F1.** The hierarchy IS real (severe_toxic is a perfect subset of toxic, 1648/1648) but the model already respects it: only 2.35% of rows violate it and the mean excess is 0.0127, too small to cross a threshold. Six sigmoids over a shared encoder are not independent -- the structure was learned from the labels. Writeup: experiments/label_hierarchy.md |

## IDEAS to improve the score (test, do not assume)

- Ablation first; it may show lang conditioning is worthless, which changes design
- Rare-class F1 is the weak spot: severe_toxic .398, threat .419, identity_hate .437
  at baseline. AUC is high (.89-.91) so RANKING is fine and CALIBRATION is the gap.
  -> per-class threshold tuning already helps; try better calibration
- Try unfreezing embeddings now that use_reentrant=False is correct (was never a
  fair test before, since the encoder was not training at all)
- Longer training / different LR schedule; loss was still falling at epoch 2
- severe_toxic is a SUBSET of toxic. POST-HOC clamping tested and does nothing
  (see experiments table). Hierarchy-aware TRAINING is still untested, but the
  low violation rates suggest small headroom -- not a priority.
- Near-duplicate leakage (3.8% en val) inflates val; test split is cleaner
- The corpus is ~50% toxic vs a few % in real traffic. Consider reporting
  precision at a realistic base rate.

## Iteration log

### 05:0x - first autonomous iteration
- Committed + pushed the docs rewrite (974c117) and the eval fixes (b8fd59a).
- FIXED the threshold search properly. It was not just the MRO bug: the whole
  GridSearchCV wrapper was meaningless because ThresholdOptimizer.fit() learns
  nothing, so there was no model to cross-validate. Replaced with a direct sweep
  over [0.05, 0.95] in 200 steps. Evidence it was broken: English severe_toxic
  was REPORTED at F1 0.6907 when the achievable max is 0.4439 (impossible);
  English toxic reported 0.6341 against an achievable 0.8903. Mean gap to the
  optimum is now 0.0008.
  NOTE: the old grid was [0.3, 0.7] and the rare classes' optima sit below 0.36,
  so the grid literally could not express the right answer.
- Fixed roc_auc_score NaN handling (it returns NaN with a warning, it does not
  raise, so the except ValueError never fired and invalid NaN reached the JSON).
- My first attempt at the MRO fix BROKE the estimator (sklearn then demanded a
  classes_ attribute). Caught it by testing against cached predictions rather
  than assuming. Worth remembering: is_classifier()=True changes what sklearn
  requires of the estimator.

## Notes for the owner (write anything they must know here)

- Threshold numbers will differ from the earlier baseline because the old ones
  were partly artifacts. Global F1 moves both ways: toxic .8966 -> .9045,
  threat .4003 -> .4245, but severe_toxic .4160 -> .3978 and
  identity_hate .4452 -> .4389 (those two were inflated by the free-1.0 folds).
  The new numbers are honest; some are lower.
- Nothing published to Hugging Face. Still blocked on YOUR license choice
  (the repo declares none) - see hf_release/README.md placeholders.
