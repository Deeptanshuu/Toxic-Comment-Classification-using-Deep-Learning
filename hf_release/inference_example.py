#!/usr/bin/env python3
"""Runnable example: batch toxicity prediction with tuned per-class thresholds.

    python inference_example.py                       # pull from the Hub
    python inference_example.py /path/to/local/repo   # use a local checkout
    python inference_example.py --device cuda

Only needs `transformers`, `torch` and `huggingface_hub`. Nothing from the
training repo is imported; `modeling_toxic_xlmr.py` ships alongside this file
and is loaded for you by `trust_remote_code=True`.

The three things people get wrong with this model:

1. It needs `lang_ids`. Omit it and every row is scored as English (id 0),
   silently and with no error.
2. The six labels are independent, not one choice out of six. Any number of
   them can fire on the same comment.
3. 0.5 is the wrong cut. Use `thresholds.json`; the rare classes need cuts
   near 0.37.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ID = "Deeptanshuu/toxic-comment-multilingual-xlmr"

# Baked into the trained language embedding table. Do not renumber.
LANGUAGE_IDS = {"en": 0, "ru": 1, "tr": 2, "es": 3, "fr": 4, "it": 5, "pt": 6}

# Classifier output order. Index 1 is severe_toxic and index 2 is obscene,
# which is neither alphabetical nor the order most people would guess.
LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_thresholds(source: str) -> dict:
    """Read thresholds.json from a local directory or from the Hub."""
    if os.path.isdir(source):
        path = os.path.join(source, "thresholds.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=source, filename="thresholds.json")
    with open(path) as f:
        return json.load(f)["thresholds"]


@torch.no_grad()
def predict(model, tokenizer, texts, langs, thresholds, batch_size=8, max_length=512):
    """Score a list of texts and apply the per-class thresholds.

    Args:
        texts: list of strings.
        langs: list of language codes, one per text, from LANGUAGE_IDS.
        thresholds: {label_name: float}.

    Returns:
        list of dicts, one per input text.
    """
    if len(texts) != len(langs):
        raise ValueError(f"{len(texts)} texts but {len(langs)} language codes")

    unknown = sorted({l for l in langs if l not in LANGUAGE_IDS})
    if unknown:
        raise ValueError(
            f"Unsupported language codes {unknown}. This model knows "
            f"{sorted(LANGUAGE_IDS)}. Pick the closest supported language rather "
            "than letting it default silently."
        )

    device = next(model.parameters()).device
    cut = torch.tensor([thresholds[name] for name in LABEL_NAMES], device=device)

    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_langs = langs[start : start + batch_size]

        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,  # must be 512; training used 512
            return_tensors="pt",
        ).to(device)
        lang_ids = torch.tensor(
            [LANGUAGE_IDS[l] for l in chunk_langs], dtype=torch.long, device=device
        )

        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            lang_ids=lang_ids,
        )
        probs = out["probabilities"]
        fired = probs >= cut

        for text, lang, prob_row, fired_row in zip(chunk, chunk_langs, probs, fired):
            labels = [n for n, f in zip(LABEL_NAMES, fired_row.tolist()) if f]
            results.append(
                {
                    "text": text,
                    "language": lang,
                    "probabilities": {
                        n: round(float(p), 4) for n, p in zip(LABEL_NAMES, prob_row.tolist())
                    },
                    "labels": labels,
                    "any_toxic": bool(labels),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "source",
        nargs="?",
        default=REPO_ID,
        help="Hub repo id or a local directory holding config.json and pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print(f"loading {args.source} on {args.device}", file=sys.stderr)
    model = AutoModel.from_pretrained(args.source, trust_remote_code=True)
    model.to(args.device).eval()
    # The tokenizer is stock xlm-roberta-large, unmodified. It ships in this
    # repo so the whole thing is self-contained, but the line below works
    # equally well against "xlm-roberta-large".
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    thresholds = load_thresholds(args.source)

    samples = [
        ("You are an absolute idiot and everyone here hates you.", "en"),
        ("Thanks for catching that typo, much appreciated.", "en"),
        ("Sei un cretino, vattene via da qui.", "it"),
        ("Merci beaucoup pour ton aide, c'etait tres clair.", "fr"),
        ("Defol git buradan, seni asagilik herif.", "tr"),
    ]
    texts = [t for t, _ in samples]
    langs = [l for _, l in samples]

    for row in predict(model, tokenizer, texts, langs, thresholds, batch_size=args.batch_size):
        verdict = ", ".join(row["labels"]) if row["labels"] else "clean"
        print(f"\n[{row['language']}] {row['text']}")
        print(f"  fires: {verdict}")
        print(
            "  probs: "
            + "  ".join(f"{n}={row['probabilities'][n]:.3f}" for n in LABEL_NAMES)
        )
    print("\nthresholds used: " + json.dumps(thresholds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
