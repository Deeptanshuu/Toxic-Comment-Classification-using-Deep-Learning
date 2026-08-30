# Do per-language thresholds beat a single global threshold?

**Answer: no. They are measurably worse. Delete the feature rather than fix it.**

## Why this was worth asking

`evaluate.py` computed a `per_language` threshold block for every class and
language, serialized it to `tuned_thresholds.json`, and never read it back --
`calculate_metrics` uses `thresholds['global'][class]` only. It was dead code
carrying an obvious-looking idea: languages calibrate differently, so surely a
threshold per language beats one threshold for all of them.

The block was also computed by a broken search (see docs/KNOWN_ISSUES.md, the
`KFold` / `zero_division=1` bug), which produced impossible numbers -- English
`severe_toxic` at F1 0.597 when the achievable maximum is 0.442. With the search
now fixed, the question is whether the idea itself was ever any good.

## Protocol

Split the test set in half at random, fit thresholds on half A, score half B.
Repeat over 5 random splits.

Fitting and scoring on the same rows would flatter per-language thresholds
badly, because they have seven times more free parameters to overfit with. That
is precisely the mistake the original code made, and it is why its numbers
looked good.

Run with `experiments/per_language_thresholds.py`. Uses the cached predictions
in `evaluation_results/eval_20260830_011818/predictions.npz`, so it is CPU-only
and takes about a minute.

## Result

Run `experiments/per_language_thresholds.py`. Measured on both models, since the
question is whether the conclusion depends on which one you ask.

**Current model** (5 splits): macro F1 0.8822 global vs 0.8806 per-language,
**-0.0016**, and **0 of 5 splits improved**.

| Class | Global F1 | Per-language F1 | Delta |
|---|---|---|---|
| toxic | 0.9639 | 0.9633 | -0.0006 |
| severe_toxic | 0.7591 | 0.7527 | -0.0064 |
| obscene | 0.9360 | 0.9368 | +0.0008 |
| threat | 0.8446 | 0.8439 | -0.0007 |
| insult | 0.9234 | 0.9219 | -0.0015 |
| identity_hate | 0.8662 | 0.8648 | -0.0014 |
| **Macro** | **0.8822** | **0.8806** | **-0.0016** |

**April model** (5 splits): macro F1 0.6047 vs 0.6000, **-0.0047**, 1 of 5
splits improved and that one by exactly 0.0000.

| Class | Global F1 | Per-language F1 | Delta |
|---|---|---|---|
| toxic | 0.9036 | 0.9032 | -0.0004 |
| severe_toxic | 0.3944 | 0.3871 | -0.0073 |
| obscene | 0.7406 | 0.7394 | -0.0013 |
| threat | 0.4267 | 0.4123 | **-0.0144** |
| insult | 0.7225 | 0.7207 | -0.0017 |
| identity_hate | 0.4402 | 0.4370 | -0.0032 |
| **Macro** | **0.6047** | **0.6000** | **-0.0047** |

The conclusion holds on both, and is cleaner on the current model: not a single
split improved. The damage is smaller in absolute terms (-0.0016 against -0.0047)
for a reason worth noting -- the current model's probabilities are well separated,
so where you put the threshold matters less to begin with. On this model,
threshold tuning is worth only 0.0007 macro F1 over a flat 0.5; on the April
model it was worth +0.0752. Less to gain from tuning also means less to lose from
tuning badly.

## Why it loses, and why the pattern is the giveaway

The damage is concentrated in the rarest classes: `threat` -0.0144 and
`severe_toxic` -0.0073, against `toxic` -0.0004. That ordering is the
explanation.

A threshold is fit by finding the value that maximises F1 on the data in front
of it. The global threshold for `threat` is fit on all 766 test positives. Split
in half, that is ~383. A per-language threshold for `threat` in English is fit
on roughly 118/7 of that -- a few dozen positives, and after the split, a few
dozen halved. At that sample size the argmax is chasing noise, and the threshold
it lands on does not transfer to held-out rows of the same language.

`toxic` is barely affected because it has ~17,700 positives; even split seven
ways there is enough data for the argmax to be stable.

So the effect is not "languages do not differ in calibration." It is that
whatever real per-language calibration difference exists is smaller than the
variance introduced by estimating seven thresholds instead of one. More
parameters, less data each, worse generalisation -- the ordinary bias/variance
trade, showing up exactly where the theory says it should.

## What to do

Delete the `per_language` block from the threshold computation. It costs about
40% of the tuning time, produces a misleading artifact, and the idea it
implements makes the model worse.

If someone wants per-language calibration later, the way to get it without the
variance problem is to shrink towards the global threshold rather than fit
independently -- for example a partial-pooling estimate where a language's
threshold is pulled towards the global one in proportion to how little data it
has. That is a real technique with a real justification, unlike fitting seven
independent argmaxes on a seventh of the data each.
