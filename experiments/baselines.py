"""Compare Mill against public toxicity classifiers on Mill's test split. See the .md beside this.

  run KEY   -- score one public model on dataset/split/test.csv and save
               experiments/baselines_out/KEY.npz (row indices + scores).
               Needs a GPU for the XLM-R-large model; the rest run on CPU, slowly.
  overlap DIR -- mark test rows whose text also appears in public Jigsaw data, saved
               as experiments/baselines_out/overlap.npz. DIR must hold
               jigsaw2018_train.csv and jigsaw2018_test.csv (Kaggle 2018 challenge) and
               civil_comments*.parquet (google/civil_comments, the Jigsaw 2019 base that
               the 2020 multilingual competition trained on).
  speed     -- time Mill and every baseline on one GPU in fp16: throughput on 2,048
               test rows (batch 64) and single-comment latency (batch 1, 300 rows).
               Saves experiments/baselines_out/speed.json.
  plot      -- draw the comparison and speed charts into docs/images/ and
               hf_release/images/.
  score     -- compare every saved baseline against Mill's saved test predictions
               (evaluation_results/eval_20260830_072515/predictions.npz, same row
               order as test.csv) on exactly the same rows. Prints markdown tables,
               then the same tables with overlapping rows removed if overlap.npz exists.

Usage (from the repo root):
    uv run python experiments/baselines.py run detoxify_multi
    uv run python experiments/baselines.py run textdetox_xlmr_large
    uv run python experiments/baselines.py run citizenlab_mdistilbert
    uv run python experiments/baselines.py run toxic_bert
    uv run python experiments/baselines.py overlap PATH/TO/DOWNLOADS
    uv run python experiments/baselines.py score
    uv run python experiments/baselines.py speed
    uv run python experiments/baselines.py plot
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

TEST = Path("dataset/split/test.csv")
MILL = Path("evaluation_results/eval_20260830_072515/predictions.npz")
MILL_WEIGHTS = Path("weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin")
MILL_CONFIG = Path("hf_release/config.json")  # the shipped config for those weights
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


_NON_WORD = re.compile(r"[\W_]+")


def _norm(text):
    # Mill's English text was cleaned (apostrophes and punctuation stripped, whitespace
    # collapsed), so an exact match misses most copies. Compare lowercase letters/digits only.
    return _NON_WORD.sub(" ", str(text).lower()).strip()


def overlap(src_dir):
    src_dir = Path(src_dir)
    df = load_test()
    sources = {
        "jigsaw2018_train": pd.read_csv(src_dir / "jigsaw2018_train.csv").comment_text,
        "jigsaw2018_test": pd.read_csv(src_dir / "jigsaw2018_test.csv").comment_text,
        "civil_comments": pd.concat([pd.read_parquet(f, columns=["text"]).text
                                     for f in sorted(src_dir.glob("civil_comments*.parquet"))]),
    }
    norm = df.comment_text.map(_norm)
    masks = {}
    print("| Source | " + " | ".join(LANGS) + " | Total |")
    print("|---|" + "---|" * (len(LANGS) + 1))
    for name, texts in sources.items():
        exact = df.comment_text.isin(set(texts)).values
        seen = {_norm(t) for t in texts}
        masks[name] = (norm.isin(seen) & (norm.str.len() > 0)).values
        for kind, m in (("exact", exact), ("normalised", masks[name])):
            counts = [int(m[df.lang.values == lg].sum()) for lg in LANGS]
            print(f"| {name} ({kind}) | " + " | ".join(map(str, counts)) + f" | {int(m.sum())} |")
    any_ = np.logical_or.reduce(list(masks.values()))
    counts = [int(any_[df.lang.values == lg].sum()) for lg in LANGS]
    print("| any (normalised) | " + " | ".join(map(str, counts)) + f" | {int(any_.sum())} |")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "overlap.npz", any=any_, **masks)

def _load(key, device):
    """(tokenizer, model, forward(enc, langs) -> toxic score) for Mill or a baseline."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if key == "mill":
        sys.path.insert(0, "hf_release")
        from modeling_toxic_xlmr import LANGUAGE_IDS, ToxicCommentConfig, ToxicCommentModel

        model = ToxicCommentModel(ToxicCommentConfig.from_dict(json.loads(MILL_CONFIG.read_text())))
        model.load_state_dict(torch.load(MILL_WEIGHTS, map_location="cpu", weights_only=True), strict=True)
        tok = AutoTokenizer.from_pretrained("xlm-roberta-large")

        def fwd(enc, langs):
            lang_ids = torch.tensor([LANGUAGE_IDS[lg] for lg in langs], device=device)
            return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         lang_ids=lang_ids).probabilities[:, 0]
    else:
        hf_id, head, _ = MODELS[key]
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_id)

        def fwd(enc, langs):
            return model(**enc).logits[:, 0]  # timing only; which column does not matter
    model.to(device).eval().half()
    return tok, model, fwd


def speed(n_rows=2048, batch_size=64, n_latency=300):
    import torch

    device = "cuda"
    df = load_test()
    rows = df.sample(n_rows, random_state=0)
    lat_rows = df.sample(n_latency, random_state=1)
    texts, langs = rows.comment_text.tolist(), rows.lang.tolist()
    order = np.argsort([len(t) for t in texts])

    def batches():
        for s in range(0, n_rows, batch_size):
            b = order[s:s + batch_size]
            yield [texts[i] for i in b], [langs[i] for i in b]

    results = {"gpu": torch.cuda.get_device_name(0), "dtype": "fp16", "n_rows": n_rows,
               "batch_size": batch_size, "n_latency": n_latency, "max_length": 512,
               "timing": "wall clock, tokenisation included", "models": {}}
    for key in ["mill", *MODELS]:
        tok, model, fwd = _load(key, device)
        n_params = sum(p.numel() for p in model.parameters())
        enc_kw = dict(padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            if key == "mill":  # the timed model must be the one that produced the saved predictions
                ref = np.load(MILL)["predictions"][:, 0]
                chk = df.iloc[:256]
                got = fwd(tok(chk.comment_text.tolist(), **enc_kw).to(device), chk.lang.tolist())
                gap = float(np.abs(got.float().cpu().numpy() - ref[:256]).max())
                assert gap < 0.02, f"Mill fp16 scores differ from saved predictions by {gap}"
                results["mill_fp16_max_abs_gap"] = gap
            for i, (t, lg) in enumerate(batches()):  # warm-up
                fwd(tok(t, **enc_kw).to(device), lg)
                if i == 4:
                    break
            runs = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for t, lg in batches():
                    fwd(tok(t, **enc_kw).to(device), lg)
                torch.cuda.synchronize()
                runs.append(n_rows / (time.perf_counter() - t0))
            lat = []
            for i, (t, lg) in enumerate(zip(lat_rows.comment_text, lat_rows.lang, strict=True)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fwd(tok([t], **enc_kw).to(device), [lg])
                torch.cuda.synchronize()
                if i >= 20:  # first 20 are warm-up
                    lat.append((time.perf_counter() - t0) * 1000)
        results["models"][key] = {
            "params_m": round(n_params / 1e6, 1),
            "throughput_per_s": round(float(np.median(runs)), 1),
            "latency_ms_median": round(float(np.median(lat)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        }
        print(key, results["models"][key], flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed.json").write_text(json.dumps(results, indent=2) + "\n")


# Display names and a fixed colour + marker per model, the same in every chart.
# Colours are slots 1-5 of the dataviz reference palette; markers are the secondary encoding.
STYLE = {
    "mill": ("Mill", "#2a78d6", "o"),
    "detoxify_multi": ("Detoxify multilingual", "#eb6834", "s"),
    "textdetox_xlmr_large": ("textdetox XLM-R large", "#1baf7a", "D"),
    "citizenlab_mdistilbert": ("citizenlab mDistilBERT", "#eda100", "^"),
    "toxic_bert": ("toxic-bert", "#e87ba4", "v"),
}
LANG_NAMES = {"en": "English", "ru": "Russian", "tr": "Turkish", "es": "Spanish",
              "fr": "French", "it": "Italian", "pt": "Portuguese"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    for d in (Path("docs/images"), Path("hf_release/images")):
        fig.savefig(d / name, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote docs/images/{name} and hf_release/images/{name}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_test()
    mill = np.load(MILL)
    y_all, p_mill, langs = mill["labels"], mill["predictions"], df.lang.values

    # --- Chart 1: toxic AUC by language, and all six labels vs toxic-bert in English.
    rows = ["all"] + LANGS
    auc = {"mill": {}}
    for r in rows:
        m = np.ones(len(df), bool) if r == "all" else langs == r
        auc["mill"][r] = roc_auc_score(y_all[m, 0], p_mill[m, 0])
    for key in ("detoxify_multi", "textdetox_xlmr_large", "citizenlab_mdistilbert"):
        d = np.load(OUT / f"{key}.npz")
        full = np.full(len(df), np.nan)
        full[d["idx"]] = d["scores"][:, 0]
        auc[key] = {}
        for r in rows:
            m = np.ones(len(df), bool) if r == "all" else langs == r
            auc[key][r] = roc_auc_score(y_all[m, 0], full[m])
    tb = np.load(OUT / "toxic_bert.npz")
    tb_auc = [roc_auc_score(y_all[tb["idx"], k], tb["scores"][:, k]) for k in range(6)]
    mill_en = [roc_auc_score(y_all[tb["idx"], k], p_mill[tb["idx"], k]) for k in range(6)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.45})
    fig.patch.set_facecolor(SURFACE)
    ypos = np.arange(len(rows))[::-1]
    # Each model sits at its own small vertical offset so near-ties do not hide one another.
    for j, key in enumerate(auc):
        name, color, marker = STYLE[key]
        a1.scatter([auc[key][r] for r in rows], ypos + 0.24 - 0.16 * j, s=46, color=color,
                   marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
    a1.set_yticks(ypos, ["All languages"] + [LANG_NAMES[lg] for lg in LANGS], color=INK)
    a1.set_xlim(0.74, 1.0)
    a1.set_xlabel("AUC, `toxic` label", color=INK2, fontsize=9)
    a1.set_title("toxic, by language (35,658 test rows)", loc="left", color=INK, fontsize=11)
    _axes_style(a1)

    ypos2 = np.arange(6)[::-1]
    for vals, key, dy in ((mill_en, "mill", 0.12), (tb_auc, "toxic_bert", -0.12)):
        name, color, marker = STYLE[key]
        a2.scatter(vals, ypos2 + dy, s=46, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
    a2.set_yticks(ypos2, LABELS, color=INK)
    a2.set_xlim(0.86, 1.0)
    a2.set_xlabel("AUC", color=INK2, fontsize=9)
    a2.set_title("All six labels, English only (4,638 rows)", loc="left", color=INK, fontsize=11)
    _axes_style(a2)

    handles = [plt.Line2D([], [], linestyle="", marker=STYLE[k][2], color=STYLE[k][1],
                          markersize=7, label=STYLE[k][0]) for k in STYLE]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.0, -0.06, "Higher is better. Mill's test split; baselines used as published. "
             "Mill trained on this distribution and the baselines did not.",
             color=INK2, fontsize=8)
    _save(fig, "baseline_auc.png")
    plt.close(fig)

    # --- Chart 2: speed. Two panels, never one dual axis.
    sp = json.loads((OUT / "speed.json").read_text())
    keys = list(STYLE)
    names = [f"{STYLE[k][0]} ({sp['models'][k]['params_m']:.0f}M)" for k in keys]
    colors = [STYLE[k][1] for k in keys]
    thr = [sp["models"][k]["throughput_per_s"] for k in keys]
    med = [sp["models"][k]["latency_ms_median"] for k in keys]
    p95 = [sp["models"][k]["latency_ms_p95"] for k in keys]

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    yb = np.arange(len(keys))[::-1]
    b1.barh(yb, thr, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for yv, v in zip(yb, thr, strict=True):
        b1.text(v, yv, f"  {v:,.0f}", va="center", color=INK, fontsize=9)
    b1.set_yticks(yb, names, color=INK)
    b1.set_xlim(0, max(thr) * 1.18)
    b1.set_xlabel("comments per second, batch 64 (higher is faster)", color=INK2, fontsize=9)
    b1.set_title("Throughput", loc="left", color=INK, fontsize=11)
    _axes_style(b1)

    b2.barh(yb, med, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    b2.hlines(yb, med, p95, color=INK2, linewidth=1)
    b2.scatter(p95, yb, marker="|", s=60, color=INK2)
    for yv, v, q in zip(yb, med, p95, strict=True):
        b2.text(q, yv, f"  {v:.1f} ms", va="center", color=INK, fontsize=9)
    b2.set_yticks(yb, [""] * len(keys))
    b2.set_xlim(0, max(p95) * 1.3)
    b2.set_xlabel("ms per comment, batch 1: median, whisker to p95 (lower is faster)",
                  color=INK2, fontsize=9)
    b2.set_title("Latency", loc="left", color=INK, fontsize=11)
    _axes_style(b2)
    fig.text(0.0, -0.1, f"One {sp['gpu']}, fp16, max length 512, tokenisation included. "
             f"{sp['n_rows']:,} random test rows for throughput, {sp['n_latency'] - 20} for latency.",
             color=INK2, fontsize=8)
    _save(fig, "baseline_speed.png")
    plt.close(fig)


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


def tables(df, y_all, p_mill, files, keep):
    langs = df.lang.values

    # 1. `toxic` label, every multilingual baseline, all 7 languages.
    print("## `toxic` label, all languages (AUC; Mill scored on the same rows)\n")
    print("| Model | Rows | AUC | Mill AUC | Mill - model [95% CI] | F1 @0.5 | Mill F1 @0.5 |")
    print("|---|---|---|---|---|---|---|")
    per_lang = {}
    for f in files:
        d = np.load(f)
        idx, s = d["idx"], d["scores"]
        if set(langs[idx]) != set(LANGS):
            continue
        k = keep[idx]
        key, idx, s = f.stem, idx[k], s[k, 0]
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
        k = keep[d["idx"]]
        idx, s = d["idx"][k], d["scores"][k]
        print(f"\n## All six labels, English only ({f.stem}, {len(idx)} rows)\n")
        print("| Label | Positives | Model AUC | Mill AUC | Mill - model [95% CI] |")
        print("|---|---|---|---|---|")
        for j, name in enumerate(LABELS):
            y = y_all[idx, j].astype(int)
            diff, lo, hi = paired_auc_diff(y, p_mill[idx, j], s[:, j])
            print(f"| `{name}` | {y.sum()} | {roc_auc_score(y, s[:, j]):.4f} | {roc_auc_score(y, p_mill[idx, j]):.4f} "
                  f"| {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        macro_b = np.mean([roc_auc_score(y_all[idx, j], s[:, j]) for j in range(6)])
        macro_m = np.mean([roc_auc_score(y_all[idx, j], p_mill[idx, j]) for j in range(6)])
        print(f"| **macro** | | {macro_b:.4f} | {macro_m:.4f} | {macro_m - macro_b:+.4f} |")


def score():
    df = load_test()
    mill = np.load(MILL)
    y_all = mill["labels"]
    assert (df[LABELS].values.astype(np.float32) == y_all).all(), "Mill predictions are not aligned with test.csv"
    p_mill = mill["predictions"]

    files = sorted(f for f in OUT.glob("*.npz") if f.stem != "overlap")
    if not files:
        sys.exit(f"no baseline outputs in {OUT}; run `baselines.py run KEY` first")

    print("# All test rows\n")
    tables(df, y_all, p_mill, files, np.ones(len(df), dtype=bool))

    ov = OUT / "overlap.npz"
    if ov.exists():
        keep = ~np.load(ov)["any"]
        print(f"\n# Excluding {int((~keep).sum())} rows whose text appears in public Jigsaw data\n")
        tables(df, y_all, p_mill, files, keep)
    else:
        print("\n# Not checked for training-data overlap (run `baselines.py overlap DIR`)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "overlap":
        overlap(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        score()
    elif len(sys.argv) == 2 and sys.argv[1] == "speed":
        speed()
    elif len(sys.argv) == 2 and sys.argv[1] == "plot":
        plot()
    else:
        sys.exit(__doc__)
