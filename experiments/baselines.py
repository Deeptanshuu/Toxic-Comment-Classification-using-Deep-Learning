"""Compare Mill against public toxicity classifiers on Mill's test split. See the .md beside this.

  run KEY   -- score one public model on dataset/split/test.csv and save
               experiments/baselines_out/KEY.npz (row indices + scores).
               Needs a GPU for the XLM-R-large model; the rest run on CPU, slowly.
  score     -- compare every saved baseline against Mill's saved test predictions
               (evaluation_results/eval_20260830_072515/predictions.npz, same row
               order as test.csv) on exactly the same rows. Prints markdown tables.

Usage (from the repo root):
    uv run python experiments/baselines.py run detoxify_multi
    uv run python experiments/baselines.py run textdetox_xlmr_large
    uv run python experiments/baselines.py run citizenlab_mdistilbert
    uv run python experiments/baselines.py run toxic_bert
    uv run python experiments/baselines.py score
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

TEST = Path("dataset/split/test.csv")
MILL = Path("evaluation_results/eval_20260830_072515/predictions.npz")
OUT = Path("experiments/baselines_out")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LANGS = ["en", "ru", "tr", "es", "fr", "it", "pt"]

# key: (hf id, how logits become scores, languages it is scored on)
# Label order checked by hand on obvious toxic/benign inputs:
#   textdetox: index 1 = toxic; citizenlab: index 0 = toxic (id2label says so too).
MODELS = {
    "detoxify_multi": ("unitary/multilingual-toxic-xlm-roberta", "sigmoid1", LANGS),
    "textdetox_xlmr_large": ("textdetox/xlmr-large-toxicity-classifier-v2", "softmax_idx1", LANGS),
    "citizenlab_mdistilbert": ("citizenlab/distilbert-base-multilingual-cased-toxicity", "softmax_idx0", LANGS),
    "toxic_bert": ("unitary/toxic-bert", "sigmoid6", ["en"]),
}


def load_test():
    df = pd.read_csv(TEST)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    return df


def run(key, batch_size=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_id, head, langs = MODELS[key]
    df = load_test()
    idx = np.flatnonzero(df.lang.isin(langs).values)
    texts = df.comment_text.values[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(device).eval()
    if device == "cuda":
        model.half()

    # Length-sorted batches: same outputs, far less padding.
    order = np.argsort([len(t) for t in texts])
    scores = np.zeros((len(idx), 6 if head == "sigmoid6" else 1), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            enc = tok([texts[i] for i in b], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            if head == "sigmoid1":
                p = torch.sigmoid(logits[:, :1])
            elif head == "sigmoid6":
                p = torch.sigmoid(logits)
            elif head == "softmax_idx1":
                p = torch.softmax(logits, -1)[:, 1:2]
            else:
                p = torch.softmax(logits, -1)[:, 0:1]
            scores[b] = p.cpu().numpy()
            if (s // batch_size) % 50 == 0:
                print(f"{key}: {s}/{len(order)} rows, {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{key}.npz", idx=idx, scores=scores, hf_id=hf_id)
    print(f"{key}: {len(idx)} rows in {time.time() - t0:.0f}s on {device} -> {OUT / f'{key}.npz'}")


def paired_auc_diff(y, a, b, n_boot=1000, seed=0):
    """AUC(a) - AUC(b) on the same rows, with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    diff = roc_auc_score(y, a) - roc_auc_score(y, b)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff, lo, hi


def score():
    df = load_test()
    mill = np.load(MILL)
    y_all = mill["labels"]
    assert (df[LABELS].values.astype(np.float32) == y_all).all(), "Mill predictions are not aligned with test.csv"
    p_mill = mill["predictions"]
    langs = df.lang.values

    files = sorted(OUT.glob("*.npz"))
    if not files:
        sys.exit(f"no baseline outputs in {OUT}; run `baselines.py run KEY` first")

    # 1. `toxic` label, every multilingual baseline, all 7 languages.
    print("## `toxic` label, all languages (AUC; Mill scored on the same rows)\n")
    print("| Model | Rows | AUC | Mill AUC | Mill - model [95% CI] | F1 @0.5 | Mill F1 @0.5 |")
    print("|---|---|---|---|---|---|---|")
    per_lang = {}
    for f in files:
        d = np.load(f)
        key, idx, s = f.stem, d["idx"], d["scores"][:, 0]
        if set(langs[idx]) != set(LANGS):
            continue
        y = y_all[idx, 0].astype(int)
        m = p_mill[idx, 0]
        diff, lo, hi = paired_auc_diff(y, m, s)
        print(f"| {key} | {len(idx)} | {roc_auc_score(y, s):.4f} | {roc_auc_score(y, m):.4f} "
              f"| {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] | {f1_score(y, s >= 0.5):.4f} | {f1_score(y, m >= 0.5):.4f} |")
        per_lang[key] = {lg: roc_auc_score(y[langs[idx] == lg], s[langs[idx] == lg]) for lg in LANGS}
        per_lang["Mill"] = {lg: roc_auc_score(y[langs[idx] == lg], m[langs[idx] == lg]) for lg in LANGS}

    if per_lang:
        print("\n## `toxic` AUC by language\n")
        print("| Model | " + " | ".join(LANGS) + " |")
        print("|---|" + "---|" * len(LANGS))
        for key, row in per_lang.items():
            print(f"| {key} | " + " | ".join(f"{row[lg]:.4f}" for lg in LANGS) + " |")

    # 2. All six labels, English, for the one baseline that predicts all six.
    for f in files:
        d = np.load(f)
        if d["scores"].shape[1] != 6:
            continue
        idx, s = d["idx"], d["scores"]
        print(f"\n## All six labels, English only ({f.stem}, {len(idx)} rows)\n")
        print("| Label | Model AUC | Mill AUC | Mill - model [95% CI] |")
        print("|---|---|---|---|")
        for k, name in enumerate(LABELS):
            y = y_all[idx, k].astype(int)
            diff, lo, hi = paired_auc_diff(y, p_mill[idx, k], s[:, k])
            print(f"| `{name}` | {roc_auc_score(y, s[:, k]):.4f} | {roc_auc_score(y, p_mill[idx, k]):.4f} "
                  f"| {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        macro_b = np.mean([roc_auc_score(y_all[idx, k], s[:, k]) for k in range(6)])
        macro_m = np.mean([roc_auc_score(y_all[idx, k], p_mill[idx, k]) for k in range(6)])
        print(f"| **macro** | {macro_b:.4f} | {macro_m:.4f} | {macro_m - macro_b:+.4f} |")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
