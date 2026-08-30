#!/usr/bin/env python
"""Regenerate the README / model-card figure set for the retrained toxicity model.

Reads evaluation artifacts that are already on disk -- saved predictions, computed
metrics, and TensorBoard training scalars -- and writes PNGs to docs/images/. No
model loading and no GPU: everything here is post-hoc analysis of saved outputs.

Run from the repo root:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES="" ./.venv/bin/python scripts/make_figures.py

If the evaluation or training run directories change, update the constants below --
nothing else in this file should need to change.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --------------------------------------------------------------------------------
# Inputs. Edit these if the run directories change; nothing else should need to.
# --------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

FINAL_DIR = REPO_ROOT / "evaluation_results" / "eval_20260830_072515"  # retrained model, test split
OLD_DIR = REPO_ROOT / "evaluation_results" / "eval_20260830_011818"  # April 2025 baseline, test split
# Language-conditioning ablation control, test split. Not plotted by any figure
# below (the ablation showed no measurable effect) -- kept here for reference in
# case a future figure needs it.
ABLATION_DIR = REPO_ROOT / "evaluation_results" / "eval_20260830_123615"

FINAL_PREDICTIONS = FINAL_DIR / "predictions.npz"
OLD_PREDICTIONS = OLD_DIR / "predictions.npz"
FINAL_METRICS = FINAL_DIR / "evaluation_results.json"
OLD_METRICS = OLD_DIR / "evaluation_results.json"
FINAL_THRESHOLDS = FINAL_DIR / "tuned_thresholds.json"  # per-class operating thresholds; not plotted below, kept for reference

TRAIN_RUN_DIR = REPO_ROOT / "runs" / "train_20260830_030414"  # main training run, TensorBoard events
SELECTED_EPOCH = 5  # checkpoint actually shipped: best val macro AUC, right before val loss turns up

OUTPUT_DIR = REPO_ROOT / "docs" / "images"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LANG_NAMES = {0: "English", 1: "Russian", 2: "Turkish", 3: "Spanish", 4: "French", 5: "Italian", 6: "Portuguese"}

# --------------------------------------------------------------------------------
# Palette. Colourblind-safe (Okabe-Ito), one fixed colour per class, reused across
# every figure that shows per-class series -- so e.g. `threat` is always the same
# colour. For old-vs-new comparisons: grey for "old", and either the class colour
# (when the comparison axis is the class itself) or NEW_ACCENT (when it's not,
# e.g. per-language) for "new".
# --------------------------------------------------------------------------------
CLASS_COLORS = {
    "toxic": "#0072B2",
    "severe_toxic": "#D55E00",
    "obscene": "#009E73",
    "threat": "#CC79A7",
    "insult": "#E69F00",
    "identity_hate": "#56B4E9",
}
OLD_GREY = "#999999"
NEW_ACCENT = "#0072B2"
CHANCE_GREY = "#BBBBBB"
INK = "#111111"

FIGSIZE_WIDE = (10.67, 6.0)  # ~1600x900 @ dpi 150, single-panel figures
FIGSIZE_BAR = (10.67, 6.5)  # a bit taller, room for 6 grouped bar rows + labels
DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.grid": False,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 9.5,
    }
)


def load_npz(path: Path):
    d = np.load(path)
    return d["predictions"], d["labels"], d["langs"]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def savefig(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return out


# --------------------------------------------------------------------------------
# Figure 1: ROC curves, final model, all six classes
# --------------------------------------------------------------------------------
def fig_roc(preds: np.ndarray, labels: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    aucs = {}
    for cls in CLASSES:
        i = CLASSES.index(cls)
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        auc = roc_auc_score(labels[:, i], preds[:, i])
        aucs[cls] = auc
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], linewidth=2.2, label=f"{cls}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=CHANCE_GREY, linewidth=1.5, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, alpha=0.6)
    lo, hi = min(aucs.values()), max(aucs.values())
    ax.set_title(f"All six classes rank well above chance (ROC-AUC {lo:.3f}-{hi:.3f})")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    return savefig(fig, "roc_curves.png")


# --------------------------------------------------------------------------------
# Figure 2: Precision-recall curves, final model, all six classes
# --------------------------------------------------------------------------------
def fig_pr(preds: np.ndarray, labels: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    base_rates = labels.mean(axis=0)

    for cls in CLASSES:
        i = CLASSES.index(cls)
        precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
        ap = average_precision_score(labels[:, i], preds[:, i])
        color = CLASS_COLORS[cls]
        base = base_rates[i]
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.2,
            label=f"{cls}  (AP={ap:.3f}, base rate={base * 100:.1f}%)",
        )
        # Dotted floor from recall 0.8 to 1.0 marking this class's positive base
        # rate -- the precision a "flag everything" classifier gets, and exactly
        # where each curve mathematically lands at recall=1. Confined to the
        # right side of the panel so it never competes with the legend.
        ax.plot([0.8, 1.0], [base, base], color=color, linewidth=1.6, linestyle=":", alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.6)
    ax.set_title("Precision-recall exposes what ROC hides: rare classes pay for every point of recall")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    return savefig(fig, "pr_curves.png")


# --------------------------------------------------------------------------------
# Figure 3: Before vs after F1, per class, sorted by gain
# --------------------------------------------------------------------------------
def fig_f1_gains(final_metrics: dict, old_metrics: dict) -> Path:
    final_pc = final_metrics["optimized_thresholds"]["per_class"]
    old_pc = old_metrics["optimized_thresholds"]["per_class"]

    rows = []
    for cls in CLASSES:
        old_f1 = old_pc[cls]["f1"]
        new_f1 = final_pc[cls]["f1"]
        rows.append((cls, old_f1, new_f1, new_f1 - old_f1))
    rows.sort(key=lambda r: r[3], reverse=True)  # biggest gain first

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bar_h = 0.34
    y = np.arange(len(rows))
    for j, (_cls, old_f1, new_f1, delta) in enumerate(rows):
        ax.barh(j + bar_h / 2, old_f1, height=bar_h, color=OLD_GREY)
        ax.barh(j - bar_h / 2, new_f1, height=bar_h, color=NEW_ACCENT)
        ax.text(new_f1 + 0.015, j - bar_h / 2, f"+{delta:.3f}", va="center", ha="left", fontsize=9.5, color=INK, fontweight="bold")

    # Label the two series directly on the first (top, biggest-gain) row instead
    # of a floating legend. Every row here has a bar or a delta label running
    # most of its width, so there is no rectangle a legend box reliably avoids;
    # every prior placement collided with some row's "+x.xxx" text.
    ax.text(0.015, 0 + bar_h / 2, "Old model (April 2025)", va="center", ha="left", fontsize=8.8, color=INK, fontweight="bold")
    ax.text(0.015, 0 - bar_h / 2, "New model (retrained)", va="center", ha="left", fontsize=8.8, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()  # biggest gain at the top
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("F1 score (test split, per-class tuned threshold)")
    ax.grid(True, axis="x", alpha=0.6)
    ax.set_title("Rare classes gained the most from retraining; common classes had less room to move")
    return savefig(fig, "f1_gains_by_class.png")


# --------------------------------------------------------------------------------
# Figure 4: Training curves -- loss (train vs val) and val macro AUC, by epoch
# --------------------------------------------------------------------------------
def fig_training_curves() -> Path:
    ea = EventAccumulator(str(TRAIN_RUN_DIR))
    ea.Reload()

    def scalars(tag: str):
        events = ea.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])

    ep_tl, train_loss = scalars("epoch/train_loss")
    ep_vl, val_loss = scalars("epoch/val_loss")
    ep_auc, val_auc = scalars("epoch/val_auc_macro")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)

    ax1.plot(ep_tl, train_loss, color="#333333", marker="o", markersize=5, linewidth=2, label="Train loss")
    ax1.plot(ep_vl, val_loss, color="#D55E00", marker="o", markersize=5, linewidth=2, label="Validation loss")
    ax1.axvline(SELECTED_EPOCH, color=OLD_GREY, linestyle="--", linewidth=1.5)

    idx_sel = int(np.where(ep_vl == SELECTED_EPOCH)[0][0])
    ax1.annotate(
        "val loss turns up here\n(train loss keeps falling)",
        xy=(ep_vl[idx_sel], val_loss[idx_sel]),
        xytext=(ep_vl[idx_sel] - 2.55, val_loss.max() * 1.01),
        fontsize=9,
        color="#D55E00",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.3),
    )
    ax1.set_ylabel("Focal loss")
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.legend(loc="center right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    ax1.set_title("Validation loss turns up at epoch 5 while training loss keeps falling")

    ax2.plot(ep_auc, val_auc, color=NEW_ACCENT, marker="o", markersize=5, linewidth=2)
    ax2.axvline(
        SELECTED_EPOCH,
        color=OLD_GREY,
        linestyle="--",
        linewidth=1.5,
        label=f"Epoch {SELECTED_EPOCH}: selected checkpoint",
    )
    ax2.set_ylabel("Validation macro AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(ep_auc.astype(int))
    ax2.grid(True, axis="y", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    fig.suptitle("Six epochs was the right budget: gains flatten just as validation loss starts to diverge", y=0.995)
    fig.tight_layout()
    return savefig(fig, "training_curves.png")


# --------------------------------------------------------------------------------
# Figure 5: Predicted-probability distributions for `threat`, old model vs new
# model, one panel per model so the shape change is visible without tracing a
# four-way legend. Reference point is p=0.5 (not either model's own tuned
# threshold) so the same vertical line and the same "share below it" statistic
# are directly comparable across panels -- this matches the framing already
# used for this comparison in docs/RESULTS.md.
# --------------------------------------------------------------------------------
def fig_threat_distributions(
    old_preds: np.ndarray,
    old_labels: np.ndarray,
    new_preds: np.ndarray,
    new_labels: np.ndarray,
    final_metrics: dict,
    old_metrics: dict,
) -> Path:
    idx = CLASSES.index("threat")
    threat_color = CLASS_COLORS["threat"]
    neg_color = OLD_GREY
    bins = np.linspace(0, 1, 31)
    REF_P = 0.5

    old_pc = old_metrics["optimized_thresholds"]["per_class"]["threat"]
    new_pc = final_metrics["optimized_thresholds"]["per_class"]["threat"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True)
    panels = [
        (axL, old_preds, old_labels, "Old model", old_pc["auc"], old_pc["f1"]),
        (axR, new_preds, new_labels, "New model", new_pc["auc"], new_pc["f1"]),
    ]

    shares_below = []
    for ax, preds, labels, model_label, auc, f1 in panels:
        pos = preds[labels[:, idx] == 1, idx]
        neg = preds[labels[:, idx] == 0, idx]
        ax.hist(neg, bins=bins, density=True, color=neg_color, alpha=0.55, edgecolor="none", label=f"non-threat (n={len(neg):,})")
        ax.hist(pos, bins=bins, density=True, color=threat_color, alpha=0.8, edgecolor="none", label=f"threat (n={len(pos):,})")
        ax.axvline(REF_P, color=INK, linestyle=":", linewidth=1.6, zorder=5)
        ax.set_title(f"{model_label}  (AUC={auc:.3f}, F1={f1:.3f})", fontsize=12)
        ax.set_xlabel("Predicted probability of 'threat'")
        ax.grid(True, axis="y", alpha=0.5)
        shares_below.append(float((pos < REF_P).mean()))

    # One legend for the pair (the colour coding is identical in both panels):
    # placed lower-right in the old-model panel, the one spot in either panel
    # that the histograms don't reach, leaving the top of both panels free for
    # the "share below 0.5" annotations below.
    axL.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=9)

    axL.set_ylabel("Density")
    axL.set_xlim(0, 1)
    # Shared y-limit (sharey already enforces this) plus headroom for the two
    # annotations, which sit at a fixed height in axes-fraction coordinates so
    # they clear the bars regardless of where each panel's peak lands.
    ymax = axL.get_ylim()[1]
    axL.set_ylim(0, ymax * 1.28)

    for (ax, *_rest), share in zip(panels, shares_below, strict=True):
        ax.annotate(
            f"{share * 100:.0f}% of real threats\nscore below p=0.5",
            xy=(REF_P, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(10, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )

    auc_delta = final_metrics["optimized_thresholds"]["overall"]["auc_macro"] - old_metrics["optimized_thresholds"]["overall"]["auc_macro"]
    f1_delta = final_metrics["optimized_thresholds"]["overall"]["f1_macro"] - old_metrics["optimized_thresholds"]["overall"]["f1_macro"]
    ratio = f1_delta / auc_delta
    fig.suptitle(f"Same ranking, different separation: why macro F1 rose {ratio:.1f}x more than macro AUC", y=1.02)
    fig.tight_layout()
    return savefig(fig, "threat_probability_shift.png")


# --------------------------------------------------------------------------------
# Figure 6: Per-language AUC and F1, new vs old
# --------------------------------------------------------------------------------
def fig_per_language(final_metrics: dict, old_metrics: dict) -> Path:
    final_pl = final_metrics["optimized_thresholds"]["per_language"]
    old_pl = old_metrics["optimized_thresholds"]["per_language"]

    lang_ids = sorted(LANG_NAMES.keys(), key=lambda k: -final_pl[str(k)]["auc_macro"])
    names = [LANG_NAMES[k] for k in lang_ids]
    y = np.arange(len(lang_ids))

    auc_old = np.array([old_pl[str(k)]["auc_macro"] for k in lang_ids])
    auc_new = np.array([final_pl[str(k)]["auc_macro"] for k in lang_ids])
    f1_old = np.array([old_pl[str(k)]["f1_macro"] for k in lang_ids])
    f1_new = np.array([final_pl[str(k)]["f1_macro"] for k in lang_ids])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 6.0), sharey=True)

    for ax, old_vals, new_vals, title in [
        (axL, auc_old, auc_new, "Macro AUC"),
        (axR, f1_old, f1_new, "Macro F1"),
    ]:
        for i in range(len(lang_ids)):
            ax.plot([old_vals[i], new_vals[i]], [y[i], y[i]], color="#CCCCCC", linewidth=1.8, zorder=1)
        ax.scatter(old_vals, y, color=OLD_GREY, s=70, zorder=2, label="Old model (April 2025)")
        ax.scatter(new_vals, y, color=NEW_ACCENT, s=70, zorder=3, label="New model (retrained)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.6)
        ax.set_xlabel("Score")
        span = new_vals.max() - old_vals.min()
        ax.set_xlim(old_vals.min() - 0.12 * span, new_vals.max() + 0.12 * span)

    axL.set_yticks(y)
    axL.set_yticklabels(names)
    axL.invert_yaxis()  # best-performing language (English) on top
    # The F1 panel's old-model cluster (~0.56-0.65) and new-model cluster
    # (~0.81-0.90) leave a wide gap with no dots in it at all -- "center" lands
    # there, crossing at most a couple of thin connector lines. The AUC panel's
    # two clusters sit much closer together, so every corner and its own center
    # is close to some language's dot; the F1 panel is the one clean option.
    axR.legend(loc="center", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=9)

    auc_spread_old = auc_old.max() - auc_old.min()
    auc_spread_new = auc_new.max() - auc_new.min()
    fig.suptitle(
        f"Retraining narrows the language gap: AUC spread {auc_spread_old:.3f} -> {auc_spread_new:.3f}",
        y=1.01,
    )
    fig.tight_layout()
    return savefig(fig, "per_language_performance.png")


# --------------------------------------------------------------------------------
def main() -> None:
    final_preds, final_labels, final_langs = load_npz(FINAL_PREDICTIONS)
    old_preds, old_labels, old_langs = load_npz(OLD_PREDICTIONS)
    final_metrics = load_json(FINAL_METRICS)
    old_metrics = load_json(OLD_METRICS)

    assert np.array_equal(final_labels, old_labels), "old and new eval runs must share the same test-set labels"
    assert np.array_equal(final_langs, old_langs), "old and new eval runs must share the same test-set language ids"

    fig_roc(final_preds, final_labels)
    fig_pr(final_preds, final_labels)
    fig_f1_gains(final_metrics, old_metrics)
    fig_training_curves()
    fig_threat_distributions(old_preds, old_labels, final_preds, final_labels, final_metrics, old_metrics)
    fig_per_language(final_metrics, old_metrics)

    print("\n=== sanity check: per-class ROC-AUC recomputed from predictions.npz vs evaluation_results.json ===")
    for cls in CLASSES:
        i = CLASSES.index(cls)
        computed = roc_auc_score(final_labels[:, i], final_preds[:, i])
        reported = final_metrics["optimized_thresholds"]["per_class"][cls]["auc"]
        status = "OK" if abs(computed - reported) < 1e-4 else "MISMATCH"
        print(f"  {cls:15s} computed={computed:.4f}  reported={reported:.4f}  [{status}]")

    print("\n=== sanity check: per-language macro AUC recomputed from predictions.npz + langs vs evaluation_results.json ===")
    for lang_id in (0, 2):  # English, Turkish -- the two endpoints of the figure 6 spread
        mask = final_langs == lang_id
        computed = float(np.mean([roc_auc_score(final_labels[mask, i], final_preds[mask, i]) for i in range(len(CLASSES))]))
        reported = final_metrics["optimized_thresholds"]["per_language"][str(lang_id)]["auc_macro"]
        status = "OK" if abs(computed - reported) < 1e-4 else "MISMATCH"
        print(f"  {LANG_NAMES[lang_id]:10s} computed={computed:.4f}  reported={reported:.4f}  [{status}]")

    print(f"\nAll figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
