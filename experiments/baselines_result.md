# All test rows

## `toxic` label, all languages (AUC; Mill scored on the same rows)

| Model | Rows | AUC | Mill AUC | Mill - model [95% CI] | F1 @0.5 | Mill F1 @0.5 |
|---|---|---|---|---|---|---|
| citizenlab_mdistilbert | 35658 | 0.8513 | 0.9921 | +0.1409 [+0.1369, +0.1449] | 0.5351 | 0.9642 |
| detoxify_multi | 35658 | 0.9692 | 0.9921 | +0.0229 [+0.0214, +0.0243] | 0.8521 | 0.9642 |
| textdetox_xlmr_large | 35658 | 0.9327 | 0.9921 | +0.0594 [+0.0569, +0.0618] | 0.8549 | 0.9642 |

## `toxic` AUC by language

| Model | en | ru | tr | es | fr | it | pt |
|---|---|---|---|---|---|---|---|
| citizenlab_mdistilbert | 0.9471 | 0.7970 | 0.7540 | 0.8551 | 0.8836 | 0.8807 | 0.8640 |
| Mill | 0.9797 | 0.9921 | 0.9897 | 0.9954 | 0.9931 | 0.9947 | 0.9936 |
| detoxify_multi | 0.9762 | 0.9222 | 0.9640 | 0.9797 | 0.9786 | 0.9799 | 0.9753 |
| textdetox_xlmr_large | 0.9422 | 0.9427 | 0.8986 | 0.9341 | 0.9477 | 0.9568 | 0.9433 |

## All six labels, English only (toxic_bert, 4638 rows)

| Label | Positives | Model AUC | Mill AUC | Mill - model [95% CI] |
|---|---|---|---|---|
| `toxic` | 2227 | 0.9797 | 0.9797 | +0.0000 [-0.0026, +0.0026] |
| `severe_toxic` | 197 | 0.9592 | 0.9921 | +0.0329 [+0.0257, +0.0402] |
| `obscene` | 1230 | 0.9797 | 0.9969 | +0.0172 [+0.0143, +0.0204] |
| `threat` | 120 | 0.8849 | 0.9881 | +0.1032 [+0.0663, +0.1400] |
| `insult` | 1144 | 0.9630 | 0.9906 | +0.0276 [+0.0232, +0.0321] |
| `identity_hate` | 215 | 0.9838 | 0.9938 | +0.0100 [-0.0008, +0.0179] |
| **macro** | | 0.9584 | 0.9902 | +0.0318 |

# Excluding 4215 rows whose text appears in public Jigsaw data

## `toxic` label, all languages (AUC; Mill scored on the same rows)

| Model | Rows | AUC | Mill AUC | Mill - model [95% CI] | F1 @0.5 | Mill F1 @0.5 |
|---|---|---|---|---|---|---|
| citizenlab_mdistilbert | 31443 | 0.8346 | 0.9932 | +0.1586 [+0.1545, +0.1626] | 0.4788 | 0.9691 |
| detoxify_multi | 31443 | 0.9679 | 0.9932 | +0.0253 [+0.0238, +0.0269] | 0.8431 | 0.9691 |
| textdetox_xlmr_large | 31443 | 0.9324 | 0.9932 | +0.0608 [+0.0583, +0.0637] | 0.8510 | 0.9691 |

## `toxic` AUC by language

| Model | en | ru | tr | es | fr | it | pt |
|---|---|---|---|---|---|---|---|
| citizenlab_mdistilbert | 0.9003 | 0.7974 | 0.7548 | 0.8557 | 0.8839 | 0.8808 | 0.8643 |
| Mill | 0.9910 | 0.9921 | 0.9898 | 0.9954 | 0.9931 | 0.9946 | 0.9936 |
| detoxify_multi | 0.9098 | 0.9219 | 0.9641 | 0.9799 | 0.9785 | 0.9798 | 0.9752 |
| textdetox_xlmr_large | 0.8703 | 0.9429 | 0.8986 | 0.9341 | 0.9480 | 0.9568 | 0.9432 |

## All six labels, English only (toxic_bert, 499 rows)

| Label | Positives | Model AUC | Mill AUC | Mill - model [95% CI] |
|---|---|---|---|---|
| `toxic` | 143 | 0.9578 | 0.9910 | +0.0332 [+0.0193, +0.0478] |
| `severe_toxic` | 6 | 0.9936 | 1.0000 | +0.0064 [+0.0000, +0.0191] |
| `obscene` | 46 | 0.9823 | 0.9949 | +0.0126 [+0.0054, +0.0218] |
| `threat` | 52 | 0.8632 | 1.0000 | +0.1368 [+0.0925, +0.1870] |
| `insult` | 34 | 0.9786 | 0.9966 | +0.0180 [+0.0047, +0.0351] |
| `identity_hate` | 3 | 0.9839 | 1.0000 | +0.0161 [+0.0000, +0.0542] |
| **macro** | | 0.9599 | 0.9971 | +0.0372 |
