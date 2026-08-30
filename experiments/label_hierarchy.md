# Does enforcing the label hierarchy help?

**No. Macro F1 moves +0.0000 on the current model and -0.0001 on the previous
one. But the reason is not what it first appeared to be.**

## The idea

Jigsaw's annotation scheme is hierarchical: `severe_toxic` is a stricter grade of
`toxic`, not an independent category. The model predicts six independent sigmoids
and structurally cannot know that. So an obvious free win: clamp
`P(child) <= P(toxic)` at prediction time and let the constraint repair cases
where the heads disagree with each other.

## The hierarchy is real in the labels

Measured on the 35,658-row test split -- given a row carries the child label, how
often does it also carry `toxic`:

| Child label | Count | Also toxic | Rate |
|---|---|---|---|
| severe_toxic | 1,648 | 1,648 | **1.000** |
| obscene | 8,626 | 8,525 | 0.988 |
| insult | 10,199 | 10,049 | 0.985 |
| identity_hate | 1,891 | 1,791 | 0.947 |
| threat | 766 | 718 | 0.937 |

`severe_toxic` is a perfect subset. The clamp is applied to the three labels
above 0.98 containment.

## Result

Clamped, re-tuned thresholds on half the test split, scored the other half, 3
random splits. Run `experiments/label_hierarchy.py`.

| Class | Base F1 | Clamped | Delta |
|---|---|---|---|
| toxic | 0.9640 | 0.9640 | +0.0000 |
| severe_toxic | 0.7611 | 0.7617 | +0.0006 |
| obscene | 0.9367 | 0.9367 | -0.0001 |
| threat | 0.8424 | 0.8424 | +0.0000 |
| insult | 0.9226 | 0.9223 | -0.0003 |
| identity_hate | 0.8672 | 0.8672 | +0.0000 |
| **Macro** | **0.8823** | **0.8824** | **+0.0000** |

## Why it does nothing -- and a correction

An earlier version of this document, written against the April model, explained
the null by saying the model already respected the constraint: only 2.35% of
rows violated it. **That explanation does not survive contact with the current
model, which violates it far more often.**

Violation rates, `P(child) > P(toxic)`:

| Child | April model | Current model | Mean excess (current) |
|---|---|---|---|
| severe_toxic | 2.35% | **35.47%** | 0.0084 |
| obscene | 5.99% | **48.83%** | 0.0272 |
| threat | 6.43% | **44.66%** | 0.1241 |
| identity_hate | 6.98% | **44.18%** | 0.1063 |
| insult | 21.05% | **50.15%** | 0.0424 |

So the current model violates the hierarchy on roughly half its rows, by a larger
margin than the old one, and clamping still changes nothing. The original
explanation was wrong.

The actual reason is that **violations happen where the decision is not close**.
Of the rows that violate the constraint, the number where clamping would flip a
prediction:

| Child | Violations | Would flip a decision | Share of all rows |
|---|---|---|---|
| severe_toxic | 12,647 | 2 | 0.006% |
| obscene | 17,412 | 39 | 0.109% |
| threat | 15,924 | 8 | 0.022% |
| identity_hate | 15,754 | 22 | 0.062% |
| insult | 17,881 | 87 | 0.244% |

A flip requires `P(child)` to be above its threshold while `P(toxic)` is below
it. Almost never happens. Taking `threat` as the clearest case, with a tuned
threshold of 0.528, here is where its violations actually live:

| P(threat) among violating rows | Count | Share |
|---|---|---|
| 0.0 - 0.1 | 4,435 | 27.9% |
| 0.1 - 0.3 | 11,326 | 71.1% |
| 0.3 - 0.5 | 108 | 0.7% |
| 0.5 - 0.7 | 18 | 0.1% |
| 0.7 - 1.0 | 37 | 0.2% |

**99% of violations sit below 0.3**, on rows the model is confidently calling
negative for both labels. `P(threat) = 0.18` against `P(toxic) = 0.04` violates
the constraint, and clamping it to 0.04 is a change no threshold will ever
notice. The constraint is broken exactly where breaking it is harmless.

**What this says about the model.** It has not learned the hierarchy as a
hard rule -- the violation rates rule that out. What it has learned is to be
confident and correct in the region that determines predictions, and to be
loosely ordered in the region that does not. Independent sigmoids over a shared
encoder do not need to respect the label structure everywhere; they only need to
respect it where the decision boundary is, and there they do.

## What would still be worth trying

This tests a post-hoc constraint, the weakest version of the idea. It does not
rule out hierarchy-aware *training* -- a loss penalising `P(severe_toxic) >
P(toxic)` during optimisation would shape the representation rather than patch
its output, and might tighten the low-probability region this experiment shows is
loose. Whether that buys anything is untested. Given the constraint is already
respected where it matters, the headroom looks small.
