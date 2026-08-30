"""Per-language vs global thresholds, measured honestly. See the .md beside this.

Split test in half, fit thresholds on half A, score half B, repeat over 5 splits.
Fitting and scoring on the same rows flatters per-language thresholds because
they carry 7x the free parameters -- that is the mistake this experiment exists
to avoid.
"""
import numpy as np
from sklearn.metrics import f1_score
from model.evaluation.evaluate import optimize_threshold

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LANGS = {0: 'en', 1: 'ru', 2: 'tr', 3: 'es', 4: 'fr', 5: 'it', 6: 'pt'}
# Default is the CURRENT model. The original run of this experiment used the
# April model's predictions (eval_20260830_011818); both results are in the .md.
PRED = 'evaluation_results/eval_20260830_072515/predictions.npz'
MIN_FIT = 50   # below this many rows for a language, fall back to the global threshold


def main(path=PRED, n_splits=5):
    d = np.load(path)
    P, Y, L = d['predictions'], d['labels'], d['langs']
    rng = np.random.default_rng(0)
    glob, perlang = [], []

    for _ in range(n_splits):
        idx = rng.permutation(len(Y))
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]
        gf, pf = [], []
        for i in range(len(CLASSES)):
            t = optimize_threshold(Y[a][:, i], P[a][:, i])['threshold']
            gf.append(f1_score(Y[b][:, i], (P[b][:, i] >= t).astype(int), zero_division=0))

            pred = np.zeros(len(b), dtype=int)
            for k in LANGS:
                ma, mb = L[a] == k, L[b] == k
                if mb.sum() == 0:
                    continue
                tk = t if ma.sum() < MIN_FIT else optimize_threshold(
                    Y[a][ma][:, i], P[a][ma][:, i])['threshold']
                pred[mb] = (P[b][mb][:, i] >= tk).astype(int)
            pf.append(f1_score(Y[b][:, i], pred, zero_division=0))
        glob.append(gf)
        perlang.append(pf)

    glob, perlang = np.array(glob), np.array(perlang)
    print(f"{'class':<15}{'global F1':>11}{'per-lang F1':>13}{'delta':>9}")
    print('-' * 48)
    for i, c in enumerate(CLASSES):
        print(f"{c:<15}{glob[:, i].mean():>11.4f}{perlang[:, i].mean():>13.4f}"
              f"{perlang[:, i].mean() - glob[:, i].mean():>+9.4f}")
    print(f"{'MACRO':<15}{glob.mean():>11.4f}{perlang.mean():>13.4f}"
          f"{perlang.mean() - glob.mean():>+9.4f}")
    dm = perlang.mean(1) - glob.mean(1)
    print(f"\nper-split macro deltas: {np.round(dm, 4)}")
    print(f"mean {dm.mean():+.4f}  sd {dm.std():.4f}  improved: {(dm > 0).sum()}/{n_splits}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else PRED)
