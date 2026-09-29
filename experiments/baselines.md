# Baseline comparison: Mill versus public toxicity classifiers

Status: **not yet run.** Harness written and tested on synthetic scores; no baseline has been
scored on real data. Nothing here is a result yet.

## Question

How does Mill compare with public toxicity classifiers on the same held-out comments?

## Design

Every model is scored on Mill's own test split (`dataset/split/test.csv`, 35,658 rows). Mill is
not re-run: its saved predictions (`evaluation_results/eval_20260830_072515/predictions.npz`) are
in the same row order as `test.csv`, checked by an assert in `score`, and are compared on exactly
the rows each baseline was scored on. Differences in AUC come with a paired-bootstrap 95% CI
(1,000 resamples).

| Key | Model | Labels | Scored on |
|---|---|---|---|
| `detoxify_multi` | `unitary/multilingual-toxic-xlm-roberta` (Detoxify multilingual, XLM-R base) | `toxic` only | all 7 languages |
| `textdetox_xlmr_large` | `textdetox/xlmr-large-toxicity-classifier-v2` | binary toxic | all 7 languages |
| `citizenlab_mdistilbert` | `citizenlab/distilbert-base-multilingual-cased-toxicity` | binary toxic | all 7 languages |
| `toxic_bert` | `unitary/toxic-bert` (Detoxify original) | all 6 Jigsaw labels | English only |

No public multilingual model predicts all six labels, so the multilingual comparison is on `toxic`
alone. The six-label comparison is English only.

Label order for the two softmax models was checked by hand on obvious toxic and benign inputs:
textdetox index 1 is toxic, citizenlab index 0 is toxic.

**AUC is the headline**, because it needs no threshold. F1 at 0.5 is reported for reference only:
the baselines were not threshold-tuned on `val`, and Mill's tuned thresholds are not used for this
table either.

## Caveats to settle before publishing numbers

| Caveat | Why it matters | How to check |
|---|---|---|
| Training-data overlap | `toxic-bert` and Detoxify multilingual were trained on Jigsaw data, and Mill's English data derives from Jigsaw (`datacard.md:16`). They may have seen Mill's test comments, which would inflate their scores | Exact-text match of `test.csv` against the Jigsaw 2018 `train.csv` and the 2020 multilingual release; report results with overlapping rows removed |
| Label definitions differ | textdetox and citizenlab were trained on other toxicity definitions, so a lower AUC partly measures definition mismatch, not model quality | State it next to the numbers |
| Not calibrated to this data | F1 at 0.5 penalises models whose scores sit on a different scale | Keep AUC as the headline |

## How to run

```bash
uv run python experiments/baselines.py run detoxify_multi
uv run python experiments/baselines.py run textdetox_xlmr_large
uv run python experiments/baselines.py run citizenlab_mdistilbert
uv run python experiments/baselines.py run toxic_bert
uv run python experiments/baselines.py score > experiments/baselines_result.md
```

Each `run` writes `experiments/baselines_out/KEY.npz`. Uses the GPU in fp16 when one is available.
