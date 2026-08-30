"""Compare the language-conditioning treatment against its control.

The project's central claim is that conditioning attention on the language id
helps. Two runs differ in exactly one thing -- disable_lang_conditioning -- so
the difference between them is what that claim is worth.

Reported honestly:

  * Paired bootstrap over test rows, not an unpaired comparison. Both arms score
    the SAME rows, so their errors are correlated and an unpaired test would
    overstate the uncertainty.
  * Effect size alongside significance. n = 35,658, so a difference far too
    small to matter can still clear p < 0.05. A confidence interval that
    excludes zero but sits at +0.001 AUC is a null result in every sense the
    owner cares about, and this script says so in words.
  * A per-language breakdown, because that is where the effect must appear if it
    is real. If conditioning on language helps, it should help the languages the
    model is worst at, not English. A uniform lift across all seven languages is
    evidence of a better run, not of working language conditioning.

Usage:
  python scripts/compare_ablation.py <treatment_eval_dir> <control_eval_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def fast_auc(y, score):
    """AUC via the rank identity, much faster than roc_auc_score in a bootstrap.

    AUC = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg).
    sklearn rebuilds an ROC curve on every call, which dominates when you call it
    tens of thousands of times. scipy's rankdata does the tie-averaging in C.
    """
    n = len(y)
    pos = int(y.sum())
    if pos == 0 or pos == n:
        return np.nan
    r = rankdata(score)
    return (r[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * (n - pos))


CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LANGS = {0: 'en', 1: 'ru', 2: 'tr', 3: 'es', 4: 'fr', 5: 'it', 6: 'pt'}
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load(d):
    z = np.load(Path(d) / 'predictions.npz')
    return z['predictions'], z['labels'], z['langs']


def macro_auc(y, p):
    vals = [fast_auc(y[:, i], p[:, i]) for i in range(y.shape[1])
            if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(vals)) if vals else np.nan


def paired_bootstrap(y, pa, pb, fn, n=N_BOOT):
    """Resample ROWS once and score both arms on the same resample."""
    obs = fn(y, pa) - fn(y, pb)
    diffs = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        diffs[i] = fn(y[s], pa[s]) - fn(y[s], pb[s])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: how often the difference crosses zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, min(p, 1.0)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pa, ya, la = load(sys.argv[1])
    pb, yb, lb = load(sys.argv[2])
    if not (np.array_equal(ya, yb) and np.array_equal(la, lb)):
        sys.exit("the two evaluations do not cover the same rows; cannot pair")

    print(f"treatment (lang conditioning ON) : {sys.argv[1]}")
    print(f"control   (lang conditioning OFF): {sys.argv[2]}")
    print(f"paired over {len(ya):,} test rows, {N_BOOT} bootstrap resamples\n")

    obs, lo, hi, p = paired_bootstrap(ya, pa, pb, macro_auc)
    print(f"{'':<16}{'treat':>9}{'control':>9}{'diff':>9}{'95% CI':>20}{'p':>9}")
    print('-' * 74)
    print(f"{'MACRO AUC':<16}{macro_auc(ya,pa):>9.4f}{macro_auc(yb,pb):>9.4f}"
          f"{obs:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>20}{p:>9.3f}")

    print(f"\n{'per class':<16}{'treat':>9}{'control':>9}{'diff':>9}{'95% CI':>20}")
    print('-' * 65)
    for i, c in enumerate(CLASSES):
        f = lambda y, q, i=i: fast_auc(y[:, i], q[:, i])
        o, l, h, _ = paired_bootstrap(ya, pa, pb, f, n=300)
        print(f"  {c:<14}{roc_auc_score(ya[:,i],pa[:,i]):>9.4f}"
              f"{roc_auc_score(yb[:,i],pb[:,i]):>9.4f}{o:>+9.4f}{f'[{l:+.4f}, {h:+.4f}]':>20}")

    print(f"\n{'per language':<16}{'treat':>9}{'control':>9}{'diff':>9}{'n':>8}")
    print('-' * 53)
    rows = []
    for k, name in LANGS.items():
        m = la == k
        if m.sum() == 0:
            continue
        a, b = macro_auc(ya[m], pa[m]), macro_auc(yb[m], pb[m])
        rows.append((name, a, b, a - b, int(m.sum())))
    for name, a, b, d, n in sorted(rows, key=lambda r: -r[3]):
        print(f"  {name:<14}{a:>9.4f}{b:>9.4f}{d:>+9.4f}{n:>8,}")

    en = next((r for r in rows if r[0] == 'en'), None)
    non_en = [r for r in rows if r[0] != 'en']
    print("\nverdict")
    print('-' * 74)
    if p >= 0.05:
        print(f"  No detectable effect. The paired 95% CI [{lo:+.4f}, {hi:+.4f}] includes zero.")
        print("  Language conditioning does not measurably help on this data. That is a")
        print("  real finding, not a failure: XLM-R already encodes language identity")
        print("  implicitly, so an explicit per-language bias is plausibly redundant.")
    elif abs(obs) < 0.005:
        print(f"  Statistically detectable ({p:.3f}) but negligible: {obs:+.4f} AUC.")
        print(f"  At n={len(ya):,} a trivial difference clears significance. This is a")
        print("  null result in every sense that matters for the design.")
    else:
        print(f"  Real effect: {obs:+.4f} macro AUC, 95% CI [{lo:+.4f}, {hi:+.4f}], p={p:.3f}.")
    if en and non_en:
        mean_non_en = float(np.mean([r[3] for r in non_en]))
        print(f"\n  English delta {en[3]:+.4f} vs non-English mean {mean_non_en:+.4f}.")
        if mean_non_en > en[3] + 0.002:
            print("  The gain concentrates in non-English languages, which is what a working")
            print("  language signal should look like.")
        else:
            print("  The gain does NOT concentrate in non-English languages. A uniform shift")
            print("  across all seven is evidence of a better run, not of language")
            print("  conditioning doing the thing it was designed to do.")


if __name__ == '__main__':
    main()
