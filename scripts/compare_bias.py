"""Measurement harness for the identity-bias debiasing retrain.

Compares a baseline checkpoint against a retrained one on exactly the axes
that decide whether the retrain worked (see experiments/identity_bias.md):

  1. the seven-language identity-term probe (experiments/identity_bias.py),
     per language and per English category, with the change. The per-language
     breakdown is the point of the experiment -- English is expected to
     improve; whether the six non-English languages improve too, with no
     non-English counter-examples in training, is the open question.
  2. real-data bias: on held-out test rows with no positive label, the flag
     rate on rows containing an identity term vs rows that don't, for both
     checkpoints, with a paired bootstrap CI on the ratio. Probes are
     synthetic and can be gamed by a model that memorises templates; this
     can't be.
  3. the cost: macro AUC, macro F1, per-class F1, and identity_hate
     recall/precision on test, thresholds tuned on val. A fix that works by
     making the model blind to identity terms shows up here as an
     identity_hate recall collapse, not as a clean win in section 1.
  4. a verdict that states plainly whether bias improved, whether it improved
     outside English, and whether quality held -- including saying "traded
     bias for recall" in those words if that is what happened.

This script reuses experiments/identity_bias.py's probe set, term regexes,
carrier templates, and identity-term-detection regex by IMPORTING them --
it does not modify, fork, or copy-paste that file. It reuses
model/evaluation/evaluate.py's threshold-tuning sweep
(calculate_optimal_thresholds) for the same reason: one already-fixed
implementation of "tune thresholds on val" (see that module's docstrings for
the GridSearchCV bug it replaced), not a second one that could drift from it.

Checkpoints are loaded with model.inference_optimized.OptimizedToxicityClassifier
(needs onnxruntime -- use ./.venv-uv/bin/python). Only its `probabilities`
output is used. Its `is_toxic`/`toxic_categories` fields apply a different,
hardcoded threshold set ([0.60, 0.54, 0.60, 0.48, 0.60, 0.50], see that file)
baked into inference_optimized.py's predict(), which does NOT match the
val-tuned thresholds this script (and evaluate.py, and identity_bias.py) use.
Relying on them would silently score against the wrong operating point --
see compute_predictions() below, which reads `probabilities` only.

Usage:
    PYTHONPATH=. ./.venv-uv/bin/python scripts/compare_bias.py \\
        <baseline_checkpoint> <retrained_checkpoint> [options]

Cheapest path -- reuses the cached baseline eval (predictions + val-tuned
thresholds), computes fresh predictions only for the retrained checkpoint:
    PYTHONPATH=. ./.venv-uv/bin/python scripts/compare_bias.py \\
        weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin \\
        weights/toxic_classifier_xlmr_v2_debiased/best_model/pytorch_model.bin

If the retrained checkpoint already has its own evaluate.py output, point at
it directly instead of recomputing:
    ... --retrained-predictions evaluation_results/eval_XXXXXXXX_XXXXXX/predictions.npz \\
        --retrained-thresholds  evaluation_results/eval_XXXXXXXX_XXXXXX/tuned_thresholds.json

Self-comparison sanity check (every delta must be exactly zero):
    PYTHONPATH=. ./.venv-uv/bin/python scripts/compare_bias.py \\
        weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin \\
        weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin

When both checkpoint arguments resolve to the same weights file, every
measurement (probes, val-threshold-tuning, test predictions) is computed
ONCE and reused for both the "baseline" and "retrained" columns, instead of
being computed twice. This is not a shortcut taken to pass the sanity check:
it is what should happen anyway when asked to compare a checkpoint to
itself, it is faster, and -- unlike hoping two independent GPU forward
passes land on bit-identical floats -- it is the only way to GUARANTEE
exact-zero deltas rather than merely expect them.
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from experiments import identity_bias as ib  # see module docstring: reused by import, not forked
from model.evaluation.evaluate import calculate_optimal_thresholds

warnings.filterwarnings('ignore', category=UserWarning)

CLASSES = ib.CLASSES
LANGS = ib.LANGS
LANG_ID = ib.LANG_ID
NON_EN = [lg for lg in LANGS if lg != 'en']
IDHATE_IDX = CLASSES.index('identity_hate')

# identity_bias.md's category table order (control categories combined last).
CATEGORY_ORDER = ['sexual_orientation', 'gender_identity', 'race', 'disability',
                   'religion', 'age', 'ethnicity_nationality', 'gender']

# "Current model, test" cached artifacts named in the task -- the val-tuned
# thresholds and test predictions that produced experiments/identity_bias.md's
# numbers (verified to reproduce them exactly; see the report this script's
# author returned alongside it).
DEFAULT_BASELINE_PRED = 'evaluation_results/eval_20260830_072515/predictions.npz'
DEFAULT_BASELINE_TH = 'evaluation_results/eval_20260830_072515/tuned_thresholds.json'

EPS_FPR = 0.005       # absolute FPR change below this is "no meaningful change", not noise-chasing
EPS_QUALITY = 0.005   # same, for macro F1
RECALL_DROP_FLAG = 0.03  # absolute identity_hate recall drop big enough to call out as a real cost


# ---------------------------------------------------------------------------
# Checkpoint resolution and loading
# ---------------------------------------------------------------------------
def resolve_checkpoint_file(path):
    """Best-effort resolution of a checkpoint argument to a concrete weights
    file, mirroring OptimizedToxicityClassifier's own directory-handling
    (a 'latest' symlink, or the newest checkpoint_epoch* subdirectory, or a
    direct .bin) so that two arguments naming the same weights compare equal
    -- this is what lets the self-comparison mode detect "same checkpoint"
    -- and so the classifier always receives an unambiguous file path.
    """
    p = Path(path)
    if p.is_file():
        return os.path.realpath(p)
    if p.is_dir():
        latest = p / 'latest'
        if latest.exists():
            return resolve_checkpoint_file(str(latest))
        direct = p / 'pytorch_model.bin'
        if direct.exists():
            return os.path.realpath(direct)
        epoch_dirs = sorted(d for d in p.iterdir() if d.is_dir() and d.name.startswith('checkpoint_epoch'))
        if epoch_dirs:
            return resolve_checkpoint_file(str(epoch_dirs[-1]))
        sys.exit(f"no pytorch_model.bin or checkpoint_epoch* directories found under {path}")
    sys.exit(f"checkpoint path does not exist: {path}")


_classifier_cache = {}


def get_classifier(ckpt_file, device):
    """Memoized by resolved checkpoint path, so the self-comparison mode
    (and any other accidental re-request of the same weights) loads the
    model once."""
    if ckpt_file not in _classifier_cache:
        from model.inference_optimized import OptimizedToxicityClassifier
        print(f"[loading] {ckpt_file} -> {device}")
        _classifier_cache[ckpt_file] = OptimizedToxicityClassifier(pytorch_path=ckpt_file, device=device)
    return _classifier_cache[ckpt_file]


# ---------------------------------------------------------------------------
# Probe grids -- built from experiments/identity_bias.py's terms, templates,
# and scoring helpers (ib.TERMS, ib.TEMPLATES, ib.ML_PROBES, ib._score), not
# a re-typed copy of them.
# ---------------------------------------------------------------------------
def run_category_probe(clf):
    """English, 8 identity categories + control, TERMS x TEMPLATES -- the
    grid behind identity_bias.probe()'s category table, minus its
    context-length padding rows (a separate diagnostic in the .md, not part
    of the category/language FPR comparison asked for here)."""
    recs = [dict(cat=c, term=d, template=tn, text=fn(s, p))
            for c, d, s, p, _ in ib.TERMS for tn, fn in ib.TEMPLATES]
    df = pd.DataFrame(recs)
    scores = ib._score(clf, df.text.tolist(), ['en'] * len(df))
    return pd.concat([df, scores], axis=1)


def run_multilingual_probe(clf):
    """Seven languages, 9 identity terms + 3 controls, 3 carrier templates
    each -- the grid behind identity_bias.multilingual()'s per-language
    table."""
    recs = [dict(lang=lg, term=t, template=tn, text=s)
            for lg, d in ib.ML_PROBES.items() for t, sents in d.items()
            for tn, s in zip(ib.ML_TEMPLATES, sents, strict=True)]
    df = pd.DataFrame(recs)
    scores = ib._score(clf, df.text.tolist(), df.lang.tolist())
    return pd.concat([df, scores], axis=1)


def fires_at(df, th):
    """Does any class fire at or above `th`? Matches identity_bias.py's own
    convention ("A class fires at or above", see its THRESHOLDS comment),
    which is what produced the FPR numbers this script is checked against."""
    return (df[CLASSES].values >= th).any(axis=1)


def category_summary(df, th):
    d = df.assign(fires=fires_at(df, th))
    ident = d[~d.cat.str.startswith('control')]
    ctrl = d[d.cat.str.startswith('control')]
    out = {}
    for cat in CATEGORY_ORDER:
        g = ident[ident.cat == cat]
        out[cat] = dict(n=int(len(g)), fpr=float(g.fires.mean()), meanp=float(g.toxic.mean()))
    out['control'] = dict(n=int(len(ctrl)), fpr=float(ctrl.fires.mean()), meanp=float(ctrl.toxic.mean()))
    return out


def language_summary(df, th):
    d = df.assign(fires=fires_at(df, th))
    out = {}
    for lg in LANGS:
        ident = d[(d.lang == lg) & d.term.isin(ib.ML_IDENTITY)]
        ctrl = d[(d.lang == lg) & d.term.isin(ib.ML_CONTROL)]
        out[lg] = dict(n=int(len(ident)), fpr=float(ident.fires.mean()), ctrl_fpr=float(ctrl.fires.mean()))
    all_ident = d[d.term.isin(ib.ML_IDENTITY)]
    all_ctrl = d[d.term.isin(ib.ML_CONTROL)]
    out['ALL'] = dict(n=int(len(all_ident)), fpr=float(all_ident.fires.mean()), ctrl_fpr=float(all_ctrl.fires.mean()))
    return out


# ---------------------------------------------------------------------------
# Real-data + quality: loading cached artifacts, or computing fresh ones
# ---------------------------------------------------------------------------
def load_npz(path):
    d = np.load(path)
    return d['predictions'], d['labels'].astype(float), d['langs']


def load_thresholds_json(path):
    d = json.load(open(path))
    g = d['global']
    th = np.array([g[c]['threshold'] for c in CLASSES])
    return th, g


_out_dir = None


def get_out_dir(args):
    """Lazily created so a run that never computes anything fresh (e.g. the
    self-comparison check, which reuses cached baseline data for both
    columns) doesn't litter evaluation_results/ with an empty directory."""
    global _out_dir
    if _out_dir is None:
        _out_dir = Path(args.out_dir) if args.out_dir else (
            Path('evaluation_results') / f'compare_bias_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        _out_dir.mkdir(parents=True, exist_ok=True)
    return _out_dir


def compute_predictions(clf, df, batch_size, desc, chunk_size=2048):
    """Batched live inference over a dataframe's comment_text/lang columns.

    Reads ONLY `probabilities` from OptimizedToxicityClassifier.predict()'s
    per-row result dicts -- see the module docstring for why `is_toxic` and
    `toxic_categories` must not be trusted.
    """
    texts = df.comment_text.astype(str).tolist()
    langs = df.lang.tolist()
    n = len(texts)
    out = np.empty((n, len(CLASSES)), dtype=np.float32)
    for i in tqdm(range(0, n, chunk_size), desc=desc):
        chunk_texts = texts[i:i + chunk_size]
        chunk_langs = langs[i:i + chunk_size]
        res = clf.predict(chunk_texts, langs=chunk_langs, batch_size=batch_size)
        for j, r in enumerate(res):
            out[i + j] = [r['probabilities'][c] for c in CLASSES]
    return out


def tune_thresholds_on_val(clf, val_df, batch_size, role):
    val_pred = compute_predictions(clf, val_df, batch_size, desc=f"{role}: val inference (threshold tuning)")
    val_labels = val_df[CLASSES].values.astype(float)
    val_langs = val_df.lang.map(LANG_ID).values
    th_dict = calculate_optimal_thresholds(val_pred, val_labels, val_langs)
    th = np.array([th_dict['global'][c]['threshold'] for c in CLASSES])
    return th, th_dict


def get_eval_data(role, ckpt_file, pred_path, th_path, args, test_df, device):
    """Returns dict(P, Y, L, th, th_detail, source): test-set predictions,
    labels, langs, and val-tuned global thresholds for one checkpoint.

    Uses the given cache files if both are provided (the usual case for the
    baseline, via its default). Otherwise tunes thresholds on --val and
    predicts on --test with a live model, then saves both so a later run can
    pass them back in via --{role}-predictions/--{role}-thresholds instead
    of recomputing.
    """
    if pred_path is not None or th_path is not None:
        if pred_path is None or th_path is None:
            sys.exit(f"--{role}-predictions and --{role}-thresholds must be given together, or neither")
        P, Y, L = load_npz(pred_path)
        th, th_detail = load_thresholds_json(th_path)
        expected_L = test_df.lang.map(LANG_ID).values
        if len(L) != len(test_df):
            sys.exit(f"{role}: {pred_path} has {len(L)} rows but --test ({len(test_df)} rows) "
                      f"-- they are not the same split")
        if not np.array_equal(L, expected_L):
            sys.exit(f"{role}: row/language order in {pred_path} does not match --test's row order")
        return dict(P=P, Y=Y, L=L, th=th, th_detail=th_detail, source=f"cached ({pred_path})")

    if not Path(args.val).exists():
        sys.exit(f"--val file not found: {args.val} (needed to tune thresholds for {role}; "
                  f"pass --{role}-predictions/--{role}-thresholds to skip this)")
    clf = get_classifier(ckpt_file, device)
    val_df = pd.read_csv(args.val)
    val_df['comment_text'] = val_df.comment_text.astype(str)
    th, th_detail = tune_thresholds_on_val(clf, val_df, args.batch_size, role)
    P = compute_predictions(clf, test_df, args.batch_size, desc=f"{role}: test inference")
    Y = test_df[CLASSES].values.astype(float)
    L = test_df.lang.map(LANG_ID).values

    out_dir = get_out_dir(args)
    pred_out = out_dir / f'{role}_predictions.npz'
    th_out = out_dir / f'{role}_tuned_thresholds.json'
    np.savez_compressed(pred_out, predictions=P, labels=Y, langs=L)
    with open(th_out, 'w') as f:
        json.dump(th_detail, f, indent=2)
    print(f"[{role}] freshly computed; saved for reuse next time:\n"
          f"    --{role}-predictions {pred_out} --{role}-thresholds {th_out}")
    return dict(P=P, Y=Y, L=L, th=th, th_detail=th_detail, source="freshly computed")


def fires_at_arr(P, th):
    return (P >= th).any(axis=1)


# ---------------------------------------------------------------------------
# Real-data bias ratio + paired bootstrap
# ---------------------------------------------------------------------------
def paired_ratio_bootstrap(has, fires_baseline, fires_retrained, n_boot, rng):
    """Paired bootstrap for the identity-term/other FPR ratio, comparing two
    checkpoints on the SAME held-out test rows (already restricted by the
    caller to rows with no positive label). Each iteration resamples row
    indices ONCE and scores both checkpoints on that resample, so the two
    ratio estimates in every iteration share their sampling noise instead of
    having it added twice, independently (same pattern as
    scripts/compare_ablation.py's paired_bootstrap for macro AUC, generalized
    here to a ratio-of-two-group-rates statistic).

    `delta` is retrained-minus-baseline throughout this script: a negative
    delta means the ratio went down, i.e. bias improved.
    """
    term_idx = np.flatnonzero(has)
    other_idx = np.flatnonzero(~has)
    n_t, n_o = len(term_idx), len(other_idx)

    def ratio(fires, t, o):
        fo = fires[o].mean()
        return fires[t].mean() / fo if fo > 0 else np.nan

    obs_base = ratio(fires_baseline, term_idx, other_idx)
    obs_retr = ratio(fires_retrained, term_idx, other_idx)
    boot_base = np.empty(n_boot)
    boot_retr = np.empty(n_boot)
    for i in range(n_boot):
        t = rng.choice(term_idx, size=n_t, replace=True)
        o = rng.choice(other_idx, size=n_o, replace=True)
        boot_base[i] = ratio(fires_baseline, t, o)
        boot_retr[i] = ratio(fires_retrained, t, o)
    ci_base = tuple(float(x) for x in np.nanpercentile(boot_base, [2.5, 97.5]))
    ci_retr = tuple(float(x) for x in np.nanpercentile(boot_retr, [2.5, 97.5]))
    delta_boot = boot_retr - boot_base
    delta_ci = tuple(float(x) for x in np.nanpercentile(delta_boot, [2.5, 97.5]))
    return dict(ratio_baseline=float(obs_base), ci_baseline=ci_base,
                ratio_retrained=float(obs_retr), ci_retrained=ci_retr,
                delta=float(obs_retr - obs_base), delta_ci=delta_ci,
                n_term=int(n_t), n_other=int(n_o))


# ---------------------------------------------------------------------------
# Quality / cost metrics
# ---------------------------------------------------------------------------
def quality_metrics(P, Y, th):
    """Macro AUC, macro F1, per-class F1, identity_hate recall/precision.

    Uses strict '>' against the threshold, matching
    model/evaluation/evaluate.py's calculate_class_metrics /
    calculate_overall_metrics -- the code that produced the cached macro F1
    (0.8814) and identity_hate (recall 0.8419, precision 0.8959) numbers in
    experiments/identity_bias.md, which this function is checked against.
    identity_bias.py's probe/testfpr use '>=' instead ("fires at or above");
    the two conventions coexist in this codebase and in practice never
    disagree here, since predicted probabilities essentially never land
    exactly on a threshold produced by a 200-step sweep.
    """
    usable = [i for i in range(Y.shape[1]) if 0 < Y[:, i].sum() < len(Y)]
    auc_macro = float(np.mean([roc_auc_score(Y[:, i], P[:, i]) for i in usable])) if usable else float('nan')
    preds = P > th
    f1_per_class = {c: float(f1_score(Y[:, i], preds[:, i], zero_division=1))
                     for i, c in enumerate(CLASSES)}
    f1_macro = float(np.mean(list(f1_per_class.values())))
    idhate_recall = float(recall_score(Y[:, IDHATE_IDX], preds[:, IDHATE_IDX], zero_division=1))
    idhate_precision = float(precision_score(Y[:, IDHATE_IDX], preds[:, IDHATE_IDX], zero_division=1))
    return dict(auc_macro=auc_macro, f1_macro=f1_macro, f1_per_class=f1_per_class,
                identity_hate_recall=idhate_recall, identity_hate_precision=idhate_precision)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def hr(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def fmt_delta(x, digits=4):
    return f"{x:+.{digits}f}"


def print_category_table(base, retr):
    hr("1. IDENTITY-TERM PROBE -- per English category (identity_bias.TERMS x TEMPLATES)")
    print(f"  {'category':<24}{'n':>5}{'base FPR':>10}{'retr FPR':>10}{'delta':>9}"
          f"{'base P':>9}{'retr P':>9}{'delta P':>9}")
    for cat in CATEGORY_ORDER + ['control']:
        b, r = base[cat], retr[cat]
        label = 'control (non-identity)' if cat == 'control' else cat
        print(f"  {label:<24}{b['n']:>5}{b['fpr']:>10.3f}{r['fpr']:>10.3f}{fmt_delta(r['fpr'] - b['fpr'], 3):>9}"
              f"{b['meanp']:>9.3f}{r['meanp']:>9.3f}{fmt_delta(r['meanp'] - b['meanp'], 3):>9}")


def print_language_table(base, retr):
    hr("1b. IDENTITY-TERM PROBE -- per language, seven-language probe (identity_bias.multilingual)")
    print(f"  {'language':<18}{'n':>5}{'base FPR':>10}{'retr FPR':>10}{'delta':>9}"
          f"{'base ctrl':>11}{'retr ctrl':>11}")
    for lg in LANGS + ['ALL']:
        b, r = base[lg], retr[lg]
        print(f"  {lg:<18}{b['n']:>5}{b['fpr']:>10.3f}{r['fpr']:>10.3f}{fmt_delta(r['fpr'] - b['fpr'], 3):>9}"
              f"{b['ctrl_fpr']:>11.3f}{r['ctrl_fpr']:>11.3f}")
    non_en_base = float(np.mean([base[lg]['fpr'] for lg in NON_EN]))
    non_en_retr = float(np.mean([retr[lg]['fpr'] for lg in NON_EN]))
    print(f"  {'non-English mean':<18}{'':>5}{non_en_base:>10.3f}{non_en_retr:>10.3f}"
          f"{fmt_delta(non_en_retr - non_en_base, 3):>9}")


def print_realdata_table(rb):
    hr("2. REAL-DATA BIAS -- held-out test rows with no positive label")
    print(f"  n identity-term rows: {rb['n_term']}   n other rows: {rb['n_other']}")
    print(f"  {'':<12}{'ratio':>9}{'95% CI':>20}")
    ci_b = f"[{rb['ci_baseline'][0]:.2f}, {rb['ci_baseline'][1]:.2f}]"
    ci_r = f"[{rb['ci_retrained'][0]:.2f}, {rb['ci_retrained'][1]:.2f}]"
    print(f"  {'baseline':<12}{rb['ratio_baseline']:>8.2f}x{ci_b:>20}")
    print(f"  {'retrained':<12}{rb['ratio_retrained']:>8.2f}x{ci_r:>20}")
    ci_d = f"[{rb['delta_ci'][0]:+.2f}, {rb['delta_ci'][1]:+.2f}]"
    print(f"  paired delta (retrained - baseline): {fmt_delta(rb['delta'], 2)}x   95% CI {ci_d}")


def print_quality_table(qb, qr):
    hr("3. QUALITY / COST -- test split, thresholds tuned on val")
    print(f"  {'metric':<24}{'baseline':>10}{'retrained':>10}{'delta':>9}")
    for key, label in [('auc_macro', 'macro AUC'), ('f1_macro', 'macro F1'),
                        ('identity_hate_recall', 'identity_hate recall'),
                        ('identity_hate_precision', 'identity_hate precision')]:
        print(f"  {label:<24}{qb[key]:>10.4f}{qr[key]:>10.4f}{fmt_delta(qr[key] - qb[key]):>9}")
    print(f"\n  {'per-class F1':<24}{'baseline':>10}{'retrained':>10}{'delta':>9}")
    for c in CLASSES:
        b, r = qb['f1_per_class'][c], qr['f1_per_class'][c]
        print(f"  {c:<24}{b:>10.4f}{r:>10.4f}{fmt_delta(r - b):>9}")


def print_verdict(cat_base, cat_retr, lang_base, lang_retr, rb, qb, qr, same_ckpt):
    hr("4. VERDICT")

    en_delta = lang_retr['en']['fpr'] - lang_base['en']['fpr']
    non_en_base = float(np.mean([lang_base[lg]['fpr'] for lg in NON_EN]))
    non_en_retr = float(np.mean([lang_retr[lg]['fpr'] for lg in NON_EN]))
    non_en_delta = non_en_retr - non_en_base
    cat_base_mean = float(np.mean([cat_base[c]['fpr'] for c in CATEGORY_ORDER]))
    cat_retr_mean = float(np.mean([cat_retr[c]['fpr'] for c in CATEGORY_ORDER]))
    cat_mean_delta = cat_retr_mean - cat_base_mean
    ratio_delta = rb['delta']
    auc_delta = qr['auc_macro'] - qb['auc_macro']
    f1_delta = qr['f1_macro'] - qb['f1_macro']
    idhate_recall_delta = qr['identity_hate_recall'] - qb['identity_hate_recall']
    idhate_precision_delta = qr['identity_hate_precision'] - qb['identity_hate_precision']

    all_deltas = [en_delta, non_en_delta, cat_mean_delta, ratio_delta, auc_delta,
                  f1_delta, idhate_recall_delta, idhate_precision_delta]

    if same_ckpt and not all(d == 0.0 for d in all_deltas):
        print("  WARNING: the two checkpoint arguments resolved to the same weights, but the")
        print("  deltas below are NOT all exactly zero. That means this script has a bug in how")
        print("  it compares two identical inputs -- investigate before trusting any real")
        print("  baseline-vs-retrained comparison it produces.")
        names = ['en_fpr', 'non_en_fpr', 'category_fpr', 'real_data_ratio', 'auc_macro',
                 'f1_macro', 'identity_hate_recall', 'identity_hate_precision']
        nonzero = {n: d for n, d in zip(names, all_deltas, strict=True) if d != 0.0}
        print(f"  nonzero deltas: {nonzero}")
        print()

    if all(d == 0.0 for d in all_deltas):
        print("  No change. Every metric checked is identical between the two checkpoints --")
        print("  English and non-English probe FPR, per-category FPR, the real-data ratio, and")
        print("  every quality metric all match exactly (delta 0 everywhere).")
        if same_ckpt:
            print("  This is the self-comparison sanity check: the two checkpoint arguments")
            print("  resolved to the same weights, so this run reused one set of measurements")
            print("  for both columns instead of computing everything twice.")
        return

    print(f"  English probe FPR:      {lang_base['en']['fpr']:.3f} -> {lang_retr['en']['fpr']:.3f}"
          f"  ({fmt_delta(en_delta, 3)})")
    print(f"  Non-English mean FPR:   {non_en_base:.3f} -> {non_en_retr:.3f}  ({fmt_delta(non_en_delta, 3)})")
    print(f"  English category mean:  {cat_base_mean:.3f} -> {cat_retr_mean:.3f}  ({fmt_delta(cat_mean_delta, 3)})")
    ci_d = f"[{rb['delta_ci'][0]:+.2f}, {rb['delta_ci'][1]:+.2f}]"
    print(f"  Real-data ratio:        {rb['ratio_baseline']:.2f}x -> {rb['ratio_retrained']:.2f}x"
          f"  ({fmt_delta(ratio_delta, 2)}x, 95% CI {ci_d})")
    print()

    en_improved = en_delta < -EPS_FPR
    en_worsened = en_delta > EPS_FPR
    non_en_improved = non_en_delta < -EPS_FPR
    non_en_worsened = non_en_delta > EPS_FPR
    ratio_improved = rb['delta_ci'][1] < 0
    ratio_worsened = rb['delta_ci'][0] > 0

    if en_improved:
        print("  Bias improved in English.")
    elif en_worsened:
        print("  Bias got WORSE in English.")
    else:
        print("  No meaningful change in English bias.")

    if non_en_improved:
        print("  Bias improved outside English too -- debiasing transferred across languages.")
    elif non_en_worsened:
        print("  Bias got WORSE outside English.")
    else:
        print("  No meaningful change outside English. Since the counter-examples were English-only,")
        print("  this is the open question the experiment was designed to answer -- and the answer")
        print("  here is that it did NOT transfer through XLM-R's shared representation.")

    if ratio_improved:
        print("  Real-data ratio is distinguishably lower (paired 95% CI excludes zero): the bias")
        print("  shows up less on genuine held-out comments, not just on synthetic probes.")
    elif ratio_worsened:
        print("  Real-data ratio is distinguishably HIGHER (paired 95% CI excludes zero).")
    else:
        print(f"  Real-data ratio change is not distinguishable from zero at this sample size "
              f"(n={rb['n_term']} identity-term rows) -- the paired CI includes zero.")

    print()
    bias_improved_somewhere = en_improved or non_en_improved or ratio_improved
    quality_held = (abs(f1_delta) < EPS_QUALITY and abs(auc_delta) < EPS_QUALITY
                     and idhate_recall_delta > -RECALL_DROP_FLAG)

    if idhate_recall_delta <= -RECALL_DROP_FLAG and bias_improved_somewhere:
        print(f"  The retrain TRADED BIAS FOR RECALL: identity_hate recall fell from "
              f"{qb['identity_hate_recall']:.3f} to {qr['identity_hate_recall']:.3f} "
              f"({fmt_delta(idhate_recall_delta, 3)}) while identity FPR went down. The model may be")
        print("  getting safer on benign identity mentions partly by getting worse at catching real")
        print("  identity-based hate. That is not a clean win -- decide on purpose whether that")
        print("  trade is acceptable before shipping it.")
    elif idhate_recall_delta <= -RECALL_DROP_FLAG:
        print(f"  identity_hate recall fell {fmt_delta(idhate_recall_delta, 3)} without a matching bias")
        print("  improvement. That is a plain regression, not a trade-off.")
    elif quality_held:
        print(f"  Quality held: macro F1 moved {fmt_delta(f1_delta, 4)}, identity_hate recall moved "
              f"{fmt_delta(idhate_recall_delta, 4)}, identity_hate precision moved "
              f"{fmt_delta(idhate_precision_delta, 4)}.")
    else:
        print(f"  Macro F1 moved {fmt_delta(f1_delta, 4)}; identity_hate recall moved "
              f"{fmt_delta(idhate_recall_delta, 4)}. Neither crosses the recall-collapse line drawn")
        print("  here, but review both before calling quality unchanged.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Compare a baseline and a retrained checkpoint on identity-term bias and its cost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('baseline_checkpoint',
                    help="Path to the baseline (current/shipped) checkpoint: a pytorch_model.bin file, "
                         "or a directory OptimizedToxicityClassifier can resolve (a 'latest' symlink or "
                         "checkpoint_epoch* subdirectories).")
    p.add_argument('retrained_checkpoint', help="Path to the retrained (debiased) checkpoint, same conventions.")
    p.add_argument('--baseline-predictions', default=DEFAULT_BASELINE_PRED,
                    help="Cached test predictions.npz for the baseline. Default: the current model's "
                         f"cached eval ({DEFAULT_BASELINE_PRED}). Pass '' (together with "
                         "--baseline-thresholds '') to force fresh computation instead.")
    p.add_argument('--baseline-thresholds', default=DEFAULT_BASELINE_TH,
                    help=f"Cached val-tuned tuned_thresholds.json for the baseline. Default: {DEFAULT_BASELINE_TH}.")
    p.add_argument('--retrained-predictions', default=None,
                    help="Cached test predictions.npz for the retrained checkpoint. If omitted (the usual "
                         "case, since it won't exist yet), computed fresh from --test.")
    p.add_argument('--retrained-thresholds', default=None,
                    help="Cached val-tuned tuned_thresholds.json for the retrained checkpoint. If omitted, "
                         "tuned fresh on --val.")
    p.add_argument('--val', default='dataset/split/val.csv',
                    help="Validation split for threshold tuning (only read if needed).")
    p.add_argument('--test', default='dataset/split/test.csv', help="Held-out test split.")
    p.add_argument('--device', default='cuda:1',
                    help="torch device for live inference (default: cuda:1, leaving GPU 0 free).")
    p.add_argument('--batch-size', type=int, default=64, help="Batch size for live inference.")
    p.add_argument('--n-boot', type=int, default=5000, help="Bootstrap resamples for the real-data ratio CI.")
    p.add_argument('--seed', type=int, default=0, help="RNG seed for the bootstrap.")
    p.add_argument('--out-dir', default=None,
                    help="Where to save freshly computed predictions/thresholds and the JSON report "
                         "(default: evaluation_results/compare_bias_<timestamp>/, created only if needed).")
    return p


def main():
    args = build_argparser().parse_args()
    # An explicit empty string means "ignore the default cache, recompute".
    for attr in ('baseline_predictions', 'baseline_thresholds', 'retrained_predictions', 'retrained_thresholds'):
        if getattr(args, attr) == '':
            setattr(args, attr, None)

    baseline_ckpt = resolve_checkpoint_file(args.baseline_checkpoint)
    retrained_ckpt = resolve_checkpoint_file(args.retrained_checkpoint)
    same_ckpt = baseline_ckpt == retrained_ckpt

    print("=" * 78)
    print("IDENTITY-BIAS RETRAIN COMPARISON")
    print(f"  baseline : {baseline_ckpt}")
    print(f"  retrained: {retrained_ckpt}")
    if same_ckpt:
        print("  [SAME CHECKPOINT -- self-comparison sanity check mode: every measurement below")
        print("   is computed once and reused for both columns]")
    print("=" * 78)

    if not Path(args.test).exists():
        sys.exit(f"--test file not found: {args.test}")
    test_df = pd.read_csv(args.test)
    test_df['comment_text'] = test_df.comment_text.astype(str)

    # ---- 1. probes ---------------------------------------------------------
    base_cat_df = run_category_probe(get_classifier(baseline_ckpt, args.device))
    base_ml_df = run_multilingual_probe(get_classifier(baseline_ckpt, args.device))
    if same_ckpt:
        retr_cat_df, retr_ml_df = base_cat_df, base_ml_df
    else:
        retr_cat_df = run_category_probe(get_classifier(retrained_ckpt, args.device))
        retr_ml_df = run_multilingual_probe(get_classifier(retrained_ckpt, args.device))

    # ---- 2 & 3. real-data bias + quality/cost ------------------------------
    baseline_data = get_eval_data('baseline', baseline_ckpt, args.baseline_predictions,
                                    args.baseline_thresholds, args, test_df, args.device)
    if same_ckpt and args.retrained_predictions is None and args.retrained_thresholds is None:
        retrained_data = baseline_data
    else:
        retrained_data = get_eval_data('retrained', retrained_ckpt, args.retrained_predictions,
                                         args.retrained_thresholds, args, test_df, args.device)

    print(f"\nbaseline thresholds  ({baseline_data['source']}):")
    print("  " + "  ".join(f"{c}={t:.4f}" for c, t in zip(CLASSES, baseline_data['th'], strict=True)))
    print(f"retrained thresholds ({retrained_data['source']}):")
    print("  " + "  ".join(f"{c}={t:.4f}" for c, t in zip(CLASSES, retrained_data['th'], strict=True)))

    base_cat = category_summary(base_cat_df, baseline_data['th'])
    retr_cat = category_summary(retr_cat_df, retrained_data['th'])
    base_lang = language_summary(base_ml_df, baseline_data['th'])
    retr_lang = language_summary(retr_ml_df, retrained_data['th'])
    print_category_table(base_cat, retr_cat)
    print_language_table(base_lang, retr_lang)

    has_term = ib._has_term(test_df)
    neg = baseline_data['Y'].sum(axis=1) == 0
    if not np.array_equal(neg, retrained_data['Y'].sum(axis=1) == 0):
        sys.exit("baseline and retrained test labels disagree on which rows are negative -- "
                  "they are not being scored on the same --test file")
    fires_base = fires_at_arr(baseline_data['P'], baseline_data['th'])
    fires_retr = fires_at_arr(retrained_data['P'], retrained_data['th'])
    rng = np.random.default_rng(args.seed)
    rb = paired_ratio_bootstrap(has_term[neg], fires_base[neg], fires_retr[neg], args.n_boot, rng)
    print_realdata_table(rb)

    qb = quality_metrics(baseline_data['P'], baseline_data['Y'], baseline_data['th'])
    qr = quality_metrics(retrained_data['P'], retrained_data['Y'], retrained_data['th'])
    print_quality_table(qb, qr)

    print_verdict(base_cat, retr_cat, base_lang, retr_lang, rb, qb, qr, same_ckpt)

    out_dir = get_out_dir(args)
    report = dict(
        baseline_checkpoint=baseline_ckpt, retrained_checkpoint=retrained_ckpt, same_checkpoint=same_ckpt,
        baseline_thresholds=baseline_data['th'].tolist(), retrained_thresholds=retrained_data['th'].tolist(),
        baseline_threshold_detail=baseline_data['th_detail'], retrained_threshold_detail=retrained_data['th_detail'],
        category=dict(baseline=base_cat, retrained=retr_cat),
        language=dict(baseline=base_lang, retrained=retr_lang),
        real_data_bias=rb,
        quality=dict(baseline=qb, retrained=qr),
    )
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nfull report saved to {out_dir / 'report.json'}")


if __name__ == '__main__':
    main()
