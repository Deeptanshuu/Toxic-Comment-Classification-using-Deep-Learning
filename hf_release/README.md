---
language:
  - en
  - ru
  - tr
  - es
  - fr
  - it
  - pt
license: other
license_name: "<<PENDING_LICENSE>>"
library_name: transformers
pipeline_tag: text-classification
base_model: xlm-roberta-large
tags:
  - toxicity
  - content-moderation
  - multilingual
  - multi-label
  - xlm-roberta
  - custom_code
metrics:
  - roc_auc
  - f1
model-index:
  - name: toxic-comment-multilingual-xlmr
    results:
      - task:
          type: text-classification
          name: Multi-label toxicity classification
        dataset:
          type: custom
          name: Multilingual Jigsaw-derived toxicity corpus (7 languages)
          split: test
        metrics:
          - type: roc_auc
            name: Macro AUC-ROC
            value: "<<PENDING_FINAL_METRICS>>"
          - type: f1
            name: Macro F1 (tuned thresholds)
            value: "<<PENDING_FINAL_METRICS>>"
          - type: f1
            name: Macro F1 (threshold 0.5)
            value: "<<PENDING_FINAL_METRICS>>"
---

# toxic-comment-multilingual-xlmr

Multi-label toxicity classification for online comments in seven languages:
English, Russian, Turkish, Spanish, French, Italian, Portuguese.

The model is XLM-RoBERTa-large with one extra attention block on top whose
attention scores carry a per-language bias, followed by a small classification
head. It emits six independent probabilities per comment.

**Read the [Known limitations](#known-limitations) and
[Out-of-scope use](#out-of-scope-use) sections before you deploy this.** They are
not boilerplate. This model has a documented history of a published version whose
headline number meant something quite different from what it appeared to mean,
and the card explains that in full.

---

## What the six labels mean

| Label | Index | Meaning in the source annotation scheme |
|---|---|---|
| `toxic` | 0 | Rude, disrespectful, or likely to make someone leave a discussion |
| `severe_toxic` | 1 | An extreme case of the above. In the source scheme this is a *subset* of `toxic` |
| `obscene` | 2 | Obscene or vulgar language |
| `threat` | 3 | A threat of violence against a person or group |
| `insult` | 4 | Insulting, inflammatory, or negative toward a person |
| `identity_hate` | 5 | Hatred toward an identity group: race, religion, gender, sexuality, nationality |

The index column is the position in the output vector. It is worth memorising
that index 1 is `severe_toxic` and index 2 is `obscene`, because that is neither
alphabetical nor the order most people assume.

### Why six sigmoids and not a softmax

The labels are **non-exclusive**. A single comment can be toxic *and* obscene
*and* an insult at the same time, and most toxic comments in the training data
carry two or three labels at once. That is a different problem from picking one
class out of six.

A softmax forces the six scores to sum to 1, so raising one necessarily lowers
the others. That is exactly wrong here: the fact that a comment is obscene
should not reduce the model's belief that it is also an insult.

So the head produces six raw scores (logits) and each one goes through its own
sigmoid independently:

```
p_k = 1 / (1 + exp(-logit_k))    for k = 0..5
```

Each `p_k` is that class's own answer to its own yes/no question. The six values
do not sum to anything in particular. You compare each one to its own threshold
and take every class that clears it.

---

## Quick start

This is a **custom architecture**, not a stock Hugging Face model. `AutoModel`
alone will not build it. The model definition ships in this repo as
`modeling_toxic_xlmr.py`, and you need `trust_remote_code=True` so that
`transformers` will execute it.

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

REPO = "Deeptanshuu/toxic-comment-multilingual-xlmr"
LANGUAGE_IDS = {"en": 0, "ru": 1, "tr": 2, "es": 3, "fr": 4, "it": 5, "pt": 6}
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO)

with open(hf_hub_download(REPO, "thresholds.json")) as f:
    thresholds = json.load(f)["thresholds"]
cut = torch.tensor([thresholds[name] for name in LABELS])

texts = ["You are an absolute idiot.", "Sei un cretino, vattene via."]
langs = ["en", "it"]

enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
lang_ids = torch.tensor([LANGUAGE_IDS[l] for l in langs])

with torch.no_grad():
    probs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lang_ids=lang_ids,
    )["probabilities"]

for text, row in zip(texts, probs):
    fired = [name for name, p, t in zip(LABELS, row, cut) if p >= t]
    print(text, "->", fired or ["clean"])
```

`inference_example.py` in this repo is the same thing as a runnable script with
batching, and it is the version that has actually been executed end to end.

### If you would rather not run remote code

`modeling_toxic_xlmr.py` also contains a plain `load_model()` helper. Download
the repo, then:

```python
from huggingface_hub import snapshot_download
import sys

path = snapshot_download("Deeptanshuu/toxic-comment-multilingual-xlmr")
sys.path.insert(0, path)

from modeling_toxic_xlmr import load_model
model, tokenizer = load_model(path, device="cpu")
```

This constructs the module and calls `load_state_dict` directly. No auto
classes, no dynamic module machinery. You are still executing the same Python
file, but you can read it first, which is the point.

---

## The `lang_ids` input, which will trip you up

The forward pass takes **three** inputs, not the usual two:

```python
model(input_ids=..., attention_mask=..., lang_ids=...)
```

`lang_ids` is a `LongTensor` of shape `[batch_size]`, one integer per sequence:

| Language | `lang_id` |
|---|---|
| English | 0 |
| Russian | 1 |
| Turkish | 2 |
| Spanish | 3 |
| French | 4 |
| Italian | 5 |
| Portuguese | 6 |

The model looks the id up in a learned embedding table and uses it to bias the
attention scores in the final block. It is the one place language identity
enters the network.

**If you omit `lang_ids`, the model does not fail.** It fills the batch with
zeros, which means every comment is scored as if it were English. You get a
`UserWarning` once per process and correct-looking output that is quietly worse
on the other six languages. This is the single most likely way to misuse this
model.

The same applies to the `text-classification` pipeline, which works but cannot
pass `lang_ids`:

```python
# Works, but treats every input as English and applies no thresholds.
from transformers import pipeline
pipe = pipeline("text-classification", model=REPO, trust_remote_code=True, top_k=None)
```

Ids outside 0-6 are clamped into range with a warning rather than raising. If
your text is in a language not in the table, there is no good id to pass: this
model has never seen that language's toxicity labels, and picking the
typologically nearest option is a guess, not a fallback.

---

## Thresholds: do not use 0.5

`thresholds.json` holds one decision threshold per class. Use it.

Across every version of this model the pattern has been the same: **ranking
quality is good on all six classes, calibration is poor on the three rare
ones.** AUC sits high, which means the model orders comments correctly, but the
absolute probabilities for `severe_toxic`, `threat` and `identity_hate` cluster
well below 0.5 even for true positives, because those classes are only 2-5% of
the training data. Cutting at 0.5 therefore throws away most of their recall for
very little precision in return.

The thresholds in this repo were chosen to maximise per-class F1 on the
validation split, and every one of them sits below 0.5.

There is deliberately **no per-language threshold block** in this repo. The
evaluation script can emit one, but that code path builds an unshuffled
cross-validation split instead of a stratified one, so folds containing zero
positive examples combine with a `zero_division=1` setting to inject free F1
scores of 1.0. It reported English `severe_toxic` at F1 0.597 when the maximum
achievable at any threshold is 0.442. Nothing ever read those numbers, and they
are not published here.

---

## Intended use

- Flagging comments for **human** review in a moderation queue.
- Prioritising a review backlog by likely severity.
- Research and coursework on multilingual toxicity detection, multi-label
  classification, or threshold calibration on imbalanced data.
- A baseline to beat.

## Out-of-scope use

**Do not use this as the sole automated decision-maker for moderation.** No
comment should be deleted, no account banned, and no user penalised on this
model's output alone. It is a triage aid. Keep a human in the loop for anything
with a consequence attached.

Beyond that:

- **Performance is uneven across languages, measurably so.** In the previous
  version's evaluation English led the worst language by about 5 points of macro
  AUC (English 0.946, Turkish 0.894). Expect the same shape here. A single
  global threshold will therefore be tighter or looser in practice depending on
  the language, even though the number itself is the same.
- **Reclaimed slurs and in-group language are a known failure mode of this class
  of model.** Communities that use slurs about themselves affectionately, drag
  and roast culture, AAVE, and queer in-group speech all reliably draw false
  positives from toxicity classifiers trained on annotator judgements of the
  kind used here. This model has not been evaluated for that, which means the
  problem is unmeasured, not absent. Deploying it against a community that talks
  that way will disproportionately flag that community.
- **Quoted and counter-speech toxicity is scored the same as first-person
  toxicity.** A comment reporting abuse, or quoting a slur to object to it,
  looks much like the abuse itself to this model.
- **Not a substitute for a threat-assessment process.** The `threat` class is
  the rarest and among the weakest. Do not route safety-critical escalation
  through it.
- **Not calibrated probabilities.** A score of 0.8 does not mean 80% of
  annotators would call the comment toxic. Treat the outputs as ranks, and use
  the thresholds.
- **The base rate in the evaluation data is nothing like a real queue.** Roughly
  half the comments in this corpus are labelled `toxic`. A live moderation
  stream is typically a few percent. Precision on real traffic will be
  substantially worse than any precision figure in this card, because precision
  depends on the base rate and the base rate here is inflated by roughly an
  order of magnitude. This is the single most common way people are disappointed
  by a toxicity classifier in production.
- Not intended for languages outside the seven listed, for long-form documents,
  or for detecting misinformation, spam, or self-harm content.

---

## Training data

Built from Jigsaw toxic comment data, extended to seven languages. 356,580
comments, split 285,264 train / 35,658 validation / 35,658 test, roughly balanced
across languages (about 14.5% each; English 13.0%).

Class balance is heavily skewed: roughly 48-50% `toxic`, but only 2-5% for
`threat`, `severe_toxic` and `identity_hate`.

### Provenance limits you should know about

These are real caveats, not disclaimers.

- **The script that produced the multilingual corpus is not in the source
  repository.** The pipeline is reproducible from the multilingual CSV onward,
  but the step that got from Jigsaw's English data to seven languages is not.
  Nobody outside the project can audit or rerun it.
- **The label definitions are English Wikipedia talk-page definitions.** The
  original Jigsaw annotation guidelines were written for one community, in one
  language, at one point in time. Whatever the missing step did to carry those
  labels into Russian, Turkish, Spanish, French, Italian and Portuguese,
  the label semantics for those six languages are inherited from an unauditable
  process. Non-English performance numbers should be read with that in mind.
- **Rare classes were topped up with synthetic data.** `threat` in particular
  was augmented with samples generated by Mistral-7B-Instruct-v0.3 in 4-bit.
  Their labels come from the generating prompt, not from annotation, so they are
  weakly labelled. The lightweight classifier used to filter the generated
  samples was trained on the same distribution it filters, which is circular:
  it keeps the synthetic samples that look like what the model already believes
  toxicity looks like, and discards the ones that would have taught it something
  new.
- **Measured near-duplicate leakage.** Exact `comment_text` overlap between
  splits is zero. But near-duplicates (character 3-4-gram TF-IDF, cosine at or
  above 0.9 against train) account for **3.8% of English validation rows and
  0.6% of Russian**. The cause is that augmentation runs before the split, so
  LLM-generated variants of the same seed text can land on both sides. Exact-hash
  deduplication does not catch these. Held-out numbers are therefore mildly
  optimistic on top of everything else.

---

## Evaluation protocol

Thresholds are tuned on the **validation** split. All reported metrics are
computed on the **held-out test** split, which was not used for tuning or model
selection.

**Why that separation matters.** Per-class thresholds are free parameters, six of
them. If you pick the thresholds that maximise F1 on a split and then report F1
on that same split, you have fitted those six parameters to that split's noise
and are reporting the fit as if it were a prediction. The number goes up and
means less. AUC is threshold-free and so is unaffected, which is exactly why AUC
and F1 can disagree about whether a change helped.

An earlier version of this project's results reported tuned-threshold metrics on
the same split it tuned on. Correcting the protocol moved the numbers by less
than 0.003 in the end, so nobody was materially misled in practice. It was still
wrong in principle, and it is fixed.

Evaluation uses `max_length=512`, matching training. An earlier evaluation
defaulted to 128, which truncated about 16% of real comments in the test split
and quietly changed what was being measured.

---

## Results

Final metrics for this version are not filled in yet.

| Metric (test split) | Value |
|---|---|
| Macro AUC-ROC | `<<PENDING_FINAL_METRICS>>` |
| Weighted AUC-ROC | `<<PENDING_FINAL_METRICS>>` |
| Macro F1 at threshold 0.5 | `<<PENDING_FINAL_METRICS>>` |
| Macro F1 at tuned thresholds | `<<PENDING_FINAL_METRICS>>` |
| Weighted F1 at tuned thresholds | `<<PENDING_FINAL_METRICS>>` |
| Exact match | `<<PENDING_FINAL_METRICS>>` |

Per class, on the test split at tuned thresholds:

| Class | AUC | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `toxic` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |
| `severe_toxic` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |
| `obscene` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |
| `threat` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |
| `insult` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |
| `identity_hate` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` | `<<PENDING_FINAL_METRICS>>` |

Per language, macro AUC on the test split:

| Language | Macro AUC |
|---|---|
| English | `<<PENDING_FINAL_METRICS>>` |
| Russian | `<<PENDING_FINAL_METRICS>>` |
| Turkish | `<<PENDING_FINAL_METRICS>>` |
| Spanish | `<<PENDING_FINAL_METRICS>>` |
| French | `<<PENDING_FINAL_METRICS>>` |
| Italian | `<<PENDING_FINAL_METRICS>>` |
| Portuguese | `<<PENDING_FINAL_METRICS>>` |

`metrics.json` in this repo carries the same values in machine-readable form.

---

## The previous version, and why its 0.9147 does not mean what it looks like

This section is the most useful thing in this card for anyone comparing versions.
It is here because the honest version of this project's history is more
instructive than a clean one would be.

### What was wrong

An earlier version of this model was published in April 2025 with a macro AUC of
0.9147. **That model never fine-tuned its XLM-RoBERTa backbone.** Measured
directly: 4.8M of 564.7M parameters, **0.8%**, actually received a gradient. All
381 encoder tensors had `grad = None` after a backward pass.

Two bugs multiplied together to cause it:

1. A layer-freezing option intended to freeze the bottom encoder layers instead
   froze the first eight *parameter tensors* of the base model. That is 258.6M
   parameters, almost all of it the 256M-row word-embedding matrix. The
   assertion written to catch exactly this checked the same wrong slice, so it
   passed.
2. With the embeddings frozen, the input to the gradient-checkpointed segment no
   longer required grad. PyTorch's `gradient_checkpointing_enable()` defaults to
   `use_reentrant=True`, and a reentrant checkpoint whose input does not require
   grad builds **no backward graph at all** through the segment. Every encoder
   layer above the frozen embeddings silently stopped receiving gradient.

Neither bug throws. Training runs, loss goes down, AUC comes out at 0.9147. What
was actually being trained was the small head on top of a frozen feature
extractor: a linear probe on stock XLM-R representations.

That is a legitimate thing to build. It is not what the card said it was.

### The other thing that was inert

The architecture's central claim is that biasing attention by language helps.
In the April version that bias had a shape that was constant along the axis the
softmax normalises over. Softmax is shift-invariant along that axis, so the bias
cancelled **exactly**. `lang_ids` had literally no effect on the output.

Measured: swapping `lang_ids` between two languages on identical text changed
the logits by 3.6e-07 in the old code, which is float32 rounding noise. In this
version the same test gives 1.9e-01, five orders of magnitude larger. The fix
adds the language vector to the attention *queries* rather than to the scores or
the keys, which is the only one of the three placements that survives the
softmax.

Worth knowing if you are testing something similar: the broken version leaked
about 4e-08 of float noise into the language embedding's gradient, so a naive
`assert grad is not None and grad.sum() != 0` **passes** on the broken model. A
test for "does this parameter actually learn" needs a magnitude threshold, not a
comparison against exact zero.

### What changed in this version

| | April 2025 version | This version |
|---|---|---|
| Parameters receiving gradient | 4.8M of 564.7M (0.8%) | 307.1M of 564.7M (54.4%) |
| Encoder fine-tuned | No | Yes |
| `lang_ids` affects output | No (cancels under softmax) | Yes (1.9e-01 logit delta) |
| Sampler | Drew with replacement; 36.9% of the training set never seen in an epoch | Exact one-pass, 285,264 unique |
| Class weighting | Never activated | Active, rare classes get about 2.6x the weight of `toxic` |
| LR warmup | Computed, then never applied | Linear warmup over 10% of steps, then cosine decay |
| Validation during training | None; best checkpoint picked by hand | Per-epoch, with per-class AUC and automatic model selection |
| Serving sequence length | 128 (truncated 15.7% of test rows) | 512, matching training |
| Run completion | Crashed at epoch 4 of 6 on a logging auth error; the published checkpoint is epoch 2 | `<<PENDING_FINAL_METRICS>>` |

The word/position embedding module is **still frozen in this version, on
purpose**. It is 256M of the 564.7M parameters, and freezing it removes the
optimizer update that dominates step time without measurably costing quality.
That is a deliberate choice, unlike last time.

For reference, the April model's real numbers, measured on the test split with
tuned thresholds under the corrected protocol:

| Metric | April 2025 version |
|---|---|
| Macro AUC | 0.9147 |
| Macro F1 at 0.5 | 0.5284 |
| Macro F1 at tuned thresholds | 0.6036 |
| Weighted F1 | 0.7732 |
| Exact match | 0.6194 |

Per class: `toxic` 0.9666 AUC / 0.9038 F1, `obscene` 0.9278 / 0.7392, `insult`
0.9035 / 0.7248, `threat` 0.9051 / 0.4189, `severe_toxic` 0.8988 / 0.3980,
`identity_hate` 0.8866 / 0.4370. Per language macro AUC: English 0.9463, French
0.9157, Spanish 0.9152, Italian 0.9139, Portuguese 0.9088, Russian 0.9065,
Turkish 0.8944.

**These are the previous version's numbers.** They are recorded so the two can
be compared. They are not this model's results, and this model's results are the
`<<PENDING_FINAL_METRICS>>` fields above.

---

## Known limitations

- **The architecture's central claim is unverified.** The language-conditioning
  ablation has not been run. The bias is now demonstrably live, and it
  demonstrably changes the output, but nobody has yet trained the control model
  with the language pathway switched off and compared. Until that run exists,
  "language-aware attention helps" is a hypothesis, not a result. The switch to
  run it exists (`disable_lang_conditioning` in the config).
- Non-English label semantics are inherited from a corpus-building step that is
  not in the repository and cannot be audited. See
  [Training data](#training-data).
- Rare-class performance is limited by rare-class data, and part of that data is
  weakly labelled synthetic text.
- The measured near-duplicate leakage (3.8% English validation) makes held-out
  metrics slightly optimistic.
- Fairness has not been evaluated. There is no subgroup analysis, no test for
  identity-term bias, and no evaluation on reclaimed-slur or in-group language.
  Absence of a measured problem is not evidence of absence.
- Weights ship as a pickled `pytorch_model.bin` state dict rather than
  safetensors, because that is what the training loop writes.
- `severe_toxic` is a subset of `toxic` in the annotation scheme, so the two are
  strongly correlated by construction. Independent sigmoids do not know that.

---

## Files in this repo

| File | What it is |
|---|---|
| `modeling_toxic_xlmr.py` | The model definition. Loaded by `trust_remote_code=True`, or importable directly for `load_model()` |
| `config.json` | Architecture config, including the nested XLM-RoBERTa encoder config and the `auto_map` that points the auto classes at the file above |
| `pytorch_model.bin` | Trained weights, a raw state dict (about 2.2 GB) |
| `thresholds.json` | Per-class decision thresholds tuned on validation, plus their provenance |
| `metrics.json` | Evaluation results, machine-readable, including the previous version's for comparison |
| `training_config.json` | The exact configuration the training run used |
| `inference_example.py` | Runnable batch-prediction example with thresholds applied |
| tokenizer files | Stock `xlm-roberta-large` tokenizer, unmodified, shipped so the repo is self-contained |

## Training configuration

Full detail in `training_config.json`. The parts that matter:

| Setting | Value |
|---|---|
| Base model | `xlm-roberta-large` (24 layers, hidden 1024, 16 heads) |
| Max sequence length | 512 |
| Batch size | 64 with gradient accumulation 2 (effective 128) |
| Epochs | 6 |
| Optimizer | AdamW, lr 2e-5, weight decay 2e-7 |
| Schedule | Linear warmup over 10% of steps, then cosine to 1% of peak |
| Mixed precision | fp16 |
| Gradient checkpointing | Enabled, `use_reentrant=False` |
| Frozen | Embedding module only (256M parameters), deliberately |
| Loss | Focal loss over BCE-with-logits, with per-language dynamic class weights |
| Hardware | Single Quadro RTX 6000 (24 GB) |

`use_reentrant=False` is not a detail. It costs about 4x the step time and twice
the memory of the reentrant variant, and it is the difference between the
encoder receiving gradients and receiving nothing at all.

---

## License

`<<PENDING_LICENSE>>`

There is no license declared in the source repository, so none is asserted here.
The constraints that will apply to whatever is chosen:

- `xlm-roberta-large`, the base model, is MIT licensed.
- The underlying Jigsaw toxic comment data derives from Wikipedia talk pages and
  carries its own terms.
- Synthetic augmentation was generated with Mistral-7B-Instruct-v0.3.

Resolve this before making the repository public.

## Citation

```bibtex
@misc{toxic_comment_multilingual_xlmr,
  author       = {Deeptanshu Lal},
  title        = {toxic-comment-multilingual-xlmr: multilingual multi-label toxicity classification},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Deeptanshuu/toxic-comment-multilingual-xlmr}}
}
```
