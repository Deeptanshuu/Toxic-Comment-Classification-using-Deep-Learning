"""Does enforcing the label hierarchy help? See the .md beside this.

severe_toxic is a strict subset of toxic in Jigsaw's scheme, but the model
predicts six independent sigmoids and structurally cannot know that. Clamping
P(child) <= P(toxic) looks like free accuracy. It is not, because the model has
already learned the constraint from the labels.

Protocol: clamp, re-tune thresholds on half the test split, score the other half,
3 random splits. Fitting and scoring on the same rows would flatter the clamp.
"""
import sys

import numpy as np
from sklearn.metrics import f1_score

from model.evaluation.evaluate import optimize_threshold

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TOXIC = 0
CONTAINMENT_MIN = 0.98      # only clamp where the hierarchy actually holds
PRED = 'evaluation_results/eval_20260830_072515/predictions.npz'


def main(path=PRED, n_splits=3):
    d = np.load(path)
    P, Y = d['predictions'], d['labels']

    print("containment in the labels: P(row is also toxic | row has label X)")
    keep = []
    for i, c in enumerate(CLASSES):
        if i == TOXIC:
            continue
        m = Y[:, i] == 1
        rate = Y[m][:, TOXIC].mean()
        if rate > CONTAINMENT_MIN:
            keep.append(i)
        print(f"  {c:<15}{int(m.sum()):>7}{rate:>9.3f}{'  clamped' if rate > CONTAINMENT_MIN else ''}")

    print("\nviolation rate in the raw predictions: P(child) > P(toxic)")
    for i, c in enumerate(CLASSES):
        if i == TOXIC:
            continue
        v = P[:, i] > P[:, TOXIC]
        exc = (P[v, i] - P[v, TOXIC]).mean() if v.sum() else 0.0
        print(f"  {c:<15}{int(v.sum()):>8}{v.mean() * 100:>8.2f}%   mean excess {exc:.4f}")

    rng = np.random.default_rng(0)
    base, clamp = [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(Y))
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]
        Pca, Pcb = P[a].copy(), P[b].copy()
        for i in keep:
            Pca[:, i] = np.minimum(Pca[:, i], Pca[:, TOXIC])
            Pcb[:, i] = np.minimum(Pcb[:, i], Pcb[:, TOXIC])
        bf, cf = [], []
        for i in range(len(CLASSES)):
            t = optimize_threshold(Y[a][:, i], P[a][:, i])['threshold']
            bf.append(f1_score(Y[b][:, i], (P[b][:, i] >= t).astype(int), zero_division=0))
            tc = optimize_threshold(Y[a][:, i], Pca[:, i])['threshold']
            cf.append(f1_score(Y[b][:, i], (Pcb[:, i] >= tc).astype(int), zero_division=0))
        base.append(bf)
        clamp.append(cf)

    base, clamp = np.array(base), np.array(clamp)
    print(f"\n{'class':<15}{'base F1':>10}{'clamped':>10}{'delta':>9}")
    print('-' * 44)
    for i, c in enumerate(CLASSES):
        print(f"{c:<15}{base[:, i].mean():>10.4f}{clamp[:, i].mean():>10.4f}"
              f"{clamp[:, i].mean() - base[:, i].mean():>+9.4f}")
    print(f"{'MACRO':<15}{base.mean():>10.4f}{clamp.mean():>10.4f}"
          f"{clamp.mean() - base.mean():>+9.4f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else PRED)
