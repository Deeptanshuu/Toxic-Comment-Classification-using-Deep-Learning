# Baseline comparison: Mill versus public toxicity classifiers

Status: **done** (2026-09-29). Full output: [`baselines_result.md`](baselines_result.md).

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

## Result

Mill ranks toxic comments better than every public baseline, in every language, on its own test
split. The margin is widest outside English. On English `toxic` it ties `toxic-bert`.

`toxic` label, all 7 languages, 35,658 rows (AUC):

| Model | AUC | Mill AUC | Mill - model [95% CI] |
|---|---|---|---|
| Detoxify multilingual (`detoxify_multi`) | 0.9692 | 0.9921 | +0.0229 [+0.0214, +0.0243] |
| textdetox XLM-R large (`textdetox_xlmr_large`) | 0.9327 | 0.9921 | +0.0594 [+0.0569, +0.0618] |
| citizenlab mDistilBERT (`citizenlab_mdistilbert`) | 0.8513 | 0.9921 | +0.1409 [+0.1369, +0.1449] |

`toxic` AUC by language:

| Model | en | ru | tr | es | fr | it | pt |
|---|---|---|---|---|---|---|---|
| Mill | 0.9797 | 0.9921 | 0.9897 | 0.9954 | 0.9931 | 0.9947 | 0.9936 |
| Detoxify multilingual | 0.9762 | 0.9222 | 0.9640 | 0.9797 | 0.9786 | 0.9799 | 0.9753 |
| textdetox XLM-R large | 0.9422 | 0.9427 | 0.8986 | 0.9341 | 0.9477 | 0.9568 | 0.9433 |
| citizenlab mDistilBERT | 0.9471 | 0.7970 | 0.7540 | 0.8551 | 0.8836 | 0.8807 | 0.8640 |

All six labels, English only, against `toxic-bert` (4,638 rows):

| Label | toxic-bert AUC | Mill AUC | Mill - model [95% CI] |
|---|---|---|---|
| `toxic` | 0.9797 | 0.9797 | +0.0000 [-0.0026, +0.0026] |
| `severe_toxic` | 0.9592 | 0.9921 | +0.0329 [+0.0257, +0.0402] |
| `obscene` | 0.9797 | 0.9969 | +0.0172 [+0.0143, +0.0204] |
| `threat` | 0.8849 | 0.9881 | +0.1032 [+0.0663, +0.1400] |
| `insult` | 0.9630 | 0.9906 | +0.0276 [+0.0232, +0.0321] |
| `identity_hate` | 0.9838 | 0.9938 | +0.0100 [-0.0008, +0.0179] |
| macro | 0.9584 | 0.9902 | +0.0318 |

### Speed

`baselines.py speed`, one Quadro RTX 6000, fp16, max length 512, wall clock with tokenisation.
Throughput on 2,048 random test rows at batch 64 (median of 3 passes); latency at batch 1 on 280
rows after 20 warm-up rows. Mill was checked to reproduce its saved test scores in fp16 (max gap
0.0023) before timing.

| Model | Params | Comments/s | Latency median | Latency p95 |
|---|---|---|---|---|
| Mill | 565M | 871 | 11.8 ms | 16.0 ms |
| Detoxify multilingual | 278M | 2,424 | 6.2 ms | 9.1 ms |
| textdetox XLM-R large | 560M | 896 | 11.0 ms | 16.6 ms |
| citizenlab mDistilBERT | 135M | 3,700 | 2.9 ms | 4.9 ms |
| toxic-bert | 110M | 1,362 | 5.5 ms | 6.7 ms |

Mill costs what any XLM-R-large model costs: about 2.8x Detoxify multilingual's compute for
+0.023 `toxic` AUC. toxic-bert is slower than its size suggests: its English WordPiece vocabulary
turns a non-English comment into 114 tokens on average, against 72 for the XLM-R tokenizer.

Charts: `baselines.py plot` writes `docs/images/baseline_auc.png` and
`docs/images/baseline_speed.png` (copies in `hf_release/images/`).

### Training-data overlap

Test rows whose text also appears in public Jigsaw data (`baselines.py overlap`). Matching is on
lowercase letters and digits only, because Mill's English text had its punctuation stripped; an
exact match finds only 536 of the English copies.

| Source | en | ru | tr | es | fr | it | pt |
|---|---|---|---|---|---|---|---|
| Jigsaw 2018 `train.csv` (159,571 rows) | 3,011 | 1 | 0 | 2 | 0 | 0 | 1 |
| Jigsaw 2018 `test.csv` (153,164 rows) | 1,139 | 1 | 0 | 0 | 2 | 2 | 2 |
| Civil Comments (1,999,514 rows) | 9 | 10 | 7 | 10 | 21 | 15 | 3 |
| Any of the three | 4,139 of 4,638 | 12 | 7 | 12 | 23 | 16 | 6 |

With those 4,215 rows removed, Mill's lead holds or grows: Detoxify multilingual +0.0253
[+0.0238, +0.0269], textdetox +0.0608, citizenlab +0.1586 on `toxic`; English `toxic` against
`toxic-bert` goes from a tie to +0.0332 [+0.0193, +0.0478]. Full tables in `baselines_result.md`.

### Caveats

| Caveat | Effect |
|---|---|
| Home-field advantage | Mill trained on the same distribution, label definitions and translation pipeline as this test split; the baselines did not. The lead is partly that, not model quality alone |
| Label definitions differ | textdetox and citizenlab were trained on other definitions of toxic, so part of their gap is definition mismatch |
| English overlap is large | 89% of English test rows are in Jigsaw 2018 data that `toxic-bert` and Detoxify trained on, which inflates their English scores |
| Post-exclusion English is not a clean set | The 499 English rows left include 51 of Mill's synthetic `threat` rows (51 of the 52 `threat` positives), so the English six-label numbers after exclusion say little |
| Translated overlap is not detected | Mill's non-English rows appear to be machine translations of English comments; exact matching cannot find a translated copy of a Jigsaw comment |
| 2020 multilingual release not checked | Its non-English validation and test files need a Kaggle login and have no public mirror; its English training files (Jigsaw 2018 train and test, Civil Comments) are covered above |
| textdetox training data not checked | `textdetox/multilingual_toxicity_dataset` was not compared against the test split |
| F1 @0.5 is not a result | Reported in `baselines_result.md` for reference only; no model was threshold-tuned for it |

Label indices were re-checked on four inputs per model (two toxic, two benign; en, es, fr): every
toxic input scored above 0.5 on the index in `MODELS` and every benign input below 0.02.

## Caveats as designed, before the run

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
uv run python experiments/baselines.py overlap PATH/TO/DOWNLOADS
uv run python experiments/baselines.py score > experiments/baselines_result.md
uv run python experiments/baselines.py speed
uv run python experiments/baselines.py plot
```

Each `run` writes `experiments/baselines_out/KEY.npz`. Uses the GPU in fp16 when one is available;
all four together take about two minutes on one Quadro RTX 6000.

`overlap` expects, in one directory: `jigsaw2018_train.csv` and `jigsaw2018_test.csv` (the Kaggle
2018 challenge files; mirrored at `thesofakillers/jigsaw-toxic-comment-classification-challenge`
on the HF Hub) and `civil_comments*.parquet` (`google/civil_comments`, `data/*.parquet`).
