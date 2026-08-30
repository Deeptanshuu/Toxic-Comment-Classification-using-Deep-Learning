# Does enforcing the label hierarchy help?

**Answer: no, because the model already learned it. Macro F1 -0.0001.**

## The idea

Jigsaw's annotation scheme is hierarchical: `severe_toxic` is a stricter grade of
`toxic`, not an independent category. The model predicts six independent
sigmoids, which structurally cannot know that. So an obvious free win: clamp
`P(child) <= P(toxic)` at prediction time and let the constraint fix cases where
the heads disagree with each other.

## The hierarchy is real in the labels

Measured on the 35,658-row test split -- given a row carries the child label,
how often does it also carry `toxic`:

| Child label | Count | Also toxic | Rate |
|---|---|---|---|
| severe_toxic | 1,648 | 1,648 | **1.000** |
| obscene | 8,626 | 8,525 | 0.988 |
| insult | 10,199 | 10,049 | 0.985 |
| identity_hate | 1,891 | 1,791 | 0.947 |
| threat | 766 | 718 | 0.937 |

`severe_toxic` is a perfect subset. `obscene` and `insult` are near-perfect. So
the constraint is a genuine property of the data, not a guess.

## Result of enforcing it

Clamped `P(child) = min(P(child), P(toxic))` for the labels with containment
above 0.98, re-tuned thresholds on half the test split, scored the other half,
3 random splits.

| Class | Base F1 | Clamped | Delta |
|---|---|---|---|
| toxic | 0.9029 | 0.9029 | +0.0000 |
| severe_toxic | 0.3912 | 0.3912 | +0.0000 |
| obscene | 0.7421 | 0.7421 | -0.0000 |
| threat | 0.4220 | 0.4220 | +0.0000 |
| insult | 0.7209 | 0.7203 | -0.0007 |
| identity_hate | 0.4419 | 0.4419 | +0.0000 |
| **Macro** | **0.6035** | **0.6034** | **-0.0001** |

## Why it does nothing

Because the model already respects the constraint almost everywhere. Violation
rates in the raw predictions:

| Child | Rows violating P(child) > P(toxic) | Share | Mean excess |
|---|---|---|---|
| severe_toxic | 839 | 2.35% | 0.0127 |
| obscene | 2,135 | 5.99% | 0.0137 |
| threat | 2,292 | 6.43% | 0.0345 |
| identity_hate | 2,489 | 6.98% | 0.0352 |
| insult | 7,507 | 21.05% | 0.0386 |

Where the constraint is violated at all, it is violated by ~0.01-0.04 in
probability -- far too little to move a row across a threshold. Clamping shaves
those tiny excesses and changes essentially no decisions.

**The interesting part is what this says about the model.** Six independent
sigmoids over a shared encoder are not really independent: they read the same
representation, and that representation encodes the hierarchy because the
training labels do. The structure was learned rather than imposed. An explicit
constraint is redundant with something the network already figured out.

## What would still be worth trying

This tested a post-hoc constraint, which is the weakest version of the idea. It
does not rule out that hierarchy-aware *training* helps -- for example a loss
that penalises `P(severe_toxic) > P(toxic)` during optimisation, which shapes the
representation rather than patching its output. That is a real experiment and
this one does not answer it. But the low violation rates suggest the headroom is
small, so it should not be near the top of anyone's list.
