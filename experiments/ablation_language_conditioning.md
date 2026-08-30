# Does language-aware attention help?

**No. Measured on held-out test data, the effect is +0.0003 macro AUC with a 95%
confidence interval of [-0.0007, +0.0012] and p = 0.588. The interval includes
zero.**

This is the experiment the project was built to answer, and until now it could
not be run: the language bias was mathematically inert (it cancelled under
softmax), so there was nothing to ablate. Fixing that made the question
answerable for the first time.

## Design

Two runs identical in every respect except one flag:

| | Treatment | Control |
|---|---|---|
| `disable_lang_conditioning` | False | True |
| Everything else | identical | identical |
| Epochs | 6 | 6 |
| Best epoch | 5 | 5 |
| Seed, data order, config | same | same |

Both were evaluated on the same 35,658-row test split with per-class thresholds
tuned on validation and applied unchanged. Comparison is a **paired** bootstrap
over test rows, 1,000 resamples: both arms score the same resampled rows, because
their errors are correlated and an unpaired test would overstate the uncertainty.

The fabricated `class_adjustments` table was deliberately left active in both
arms. It distorts class weights by ~3.5%, but identically on both sides, so it
cancels in the difference. Removing it from one arm only would have broken the
comparison.

## Result

```
MACRO AUC     treat 0.9852   control 0.9849   diff +0.0003
              95% CI [-0.0007, +0.0012]   p = 0.588
```

Per class:

| Class | Treatment | Control | Diff | 95% CI |
|---|---|---|---|---|
| toxic | 0.9921 | 0.9916 | +0.0006 | [+0.0001, +0.0011] |
| severe_toxic | 0.9863 | 0.9854 | +0.0009 | [-0.0003, +0.0021] |
| obscene | 0.9899 | 0.9897 | +0.0001 | [-0.0006, +0.0008] |
| insult | 0.9855 | 0.9850 | +0.0004 | [-0.0002, +0.0013] |
| identity_hate | 0.9818 | 0.9801 | +0.0018 | [-0.0004, +0.0042] |
| **threat** | 0.9755 | 0.9776 | **-0.0021** | [-0.0063, +0.0021] |

Only `toxic` has an interval excluding zero, at +0.0006 on the easiest and most
abundant class. `threat` is worse with conditioning on.

## The per-language result is the decisive one

If conditioning on language id does anything real, the benefit must appear in the
languages the model handles worst. It does not.

| Language | Treatment | Control | Diff | n |
|---|---|---|---|---|
| it | 0.9893 | 0.9877 | +0.0016 | 5,146 |
| tr | 0.9726 | 0.9713 | +0.0013 | 5,163 |
| pt | 0.9832 | 0.9829 | +0.0003 | 5,192 |
| es | 0.9882 | 0.9881 | +0.0001 | 5,168 |
| en | 0.9902 | 0.9908 | -0.0006 | 4,638 |
| fr | 0.9877 | 0.9885 | -0.0008 | 5,158 |
| ru | 0.9790 | 0.9800 | -0.0010 | 5,193 |

**Three of seven languages are worse with language conditioning on.** English
moves -0.0006 while the non-English mean is +0.0003 — so the effect does not
concentrate where a language signal should help. Mixed signs across languages,
with the largest gain (Italian, +0.0016) smaller than the noise floor implied by
the macro confidence interval, is what an absent effect looks like.

## Why this is the expected outcome

XLM-RoBERTa is pretrained on 100 languages. To do multilingual transfer at all it
must already represent which language it is reading — that information is
necessarily present in its hidden states. Adding an explicit per-language bias
supplies the model with something it already has. The architecture was solving a
problem the backbone had solved during pretraining.

## What this does not say

It does not say language-aware attention is a bad idea in general. It says that
on **this** corpus, with **this** backbone, at **this** scale, a learned
per-language attention bias adds nothing measurable. A weaker or monolingual
backbone, or languages absent from pretraining, could give a different answer.

## Watch out for the validation numbers

Per-epoch **validation** macro AUC favoured the treatment in all six epochs
(+0.000055, +0.000553, +0.001244, +0.000301, +0.000976, +0.001069; mean +0.0007,
sign-test p about 0.016). That looked like a small consistent real effect.

It did not survive on test. The paired interval on held-out data includes zero.

This is a clean demonstration of why the val/test split protocol matters. Six
consistent same-sign validation results were still not evidence of a
generalising effect, because the checkpoint was selected on validation and the
comparison inherited that optimism. Had the project reported validation numbers,
as the original version did, it would have concluded that language conditioning
works.
