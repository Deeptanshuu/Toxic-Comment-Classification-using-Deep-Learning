# The model flags people for describing themselves

**"I am a gay man." scores 0.891 for toxicity. The threshold is 0.472, so the
shipped model flags it. This is caused by the training data, it affects all seven
languages and is worse in the six non-English ones, and it is fixable for roughly
600 hand-written English sentences plus one 3.5-hour retrain. Nothing cheaper
works, and the cheap options have costs this document quantifies.**

## What identity-term bias is

A toxicity classifier is trained on comments that humans labelled toxic or not.
It has no concept of what toxicity *is*. It only learns which patterns in the
text predict the label. If a word shows up mostly in toxic comments, the model
learns that word predicts toxicity -- and it cannot tell the difference between
a word that is toxic and a word that is merely *attacked a lot*.

Identity terms are exactly the words where those two things come apart. The word
"gay" appears in abusive comments constantly, because homophobes use it as an
insult. It also appears in perfectly ordinary sentences, because gay people
exist and talk about themselves. A model that has seen a hundred of the first
kind and none of the second learns a rule that looks reasonable on the training
distribution and is grotesque in deployment: *mentioning this identity is a sign
of toxicity*.

The consequence is not a slightly worse F1. It is that the system removes people
for saying who they are, while the abuse aimed at them -- which is longer,
angrier, and full of other signals -- often gets through. **The group being
attacked gets silenced along with the attackers.**

Jigsaw hit this on this exact corpus in 2018 and published the fix: add benign
examples containing the identity terms until the term stops being the signal.

## Why *this* corpus produces it

Two properties of the training data, both measurable.

**The corpus is Wikipedia talk-page fights.** English rows are administrator
arguments and vandalism (13.0% mention Wikipedia, 4.3% say "talk page"). Nobody
introduces themselves on a talk page. So the corpus contains the word "gay"
almost only as a weapon, and essentially never as a self-description.

**Toxic usage is short; benign usage is long.** In the English training split,
rows containing "gay":

| Comment length | Rows | Toxic rate | Lift vs all English rows at that length |
|---|---|---|---|
| under 50 chars | 185 | 0.946 | 1.48x |
| 50-100 | 152 | 0.980 | 1.65x |
| 100-200 | 152 | 0.921 | 1.87x |
| 200-400 | 112 | 0.866 | 2.21x |
| 400-800 | 57 | 0.754 | 2.38x |
| over 800 | 68 | 0.765 | 2.31x |

Median length of a toxic "gay" row is 103 characters; of a non-toxic one, 295.
Short slurs, long discussions. Remember this shape -- the model reproduces it
exactly.

## The finding, measured

### 1. It is the term, not the sentence

Correlation in the corpus does not prove the model keys on the word. To prove
that you need an *intervention*: fix a carrier sentence, swap only the identity
term, and see what the output does. Anything that changes is caused by the term,
because nothing else changed.

63 terms across eight identity categories plus non-identity controls, in seven
carrier sentences each. False-positive rate at the shipped thresholds -- every
one of these sentences is benign, so every fire is an error:

| Category | Probes | FPR | Mean P(toxic) | Worst probe |
|---|---|---|---|---|
| Sexual orientation | 49 | **0.490** | 0.462 | 0.943 |
| Gender identity | 28 | **0.179** | 0.187 | 0.703 |
| Race | 49 | **0.122** | 0.125 | 0.722 |
| Disability | 42 | 0.048 | 0.107 | 0.573 |
| Religion | 63 | 0.032 | 0.070 | 0.699 |
| Age | 28 | 0.000 | 0.032 | 0.151 |
| Ethnicity / nationality | 98 | 0.000 | 0.031 | 0.300 |
| Gender | 14 | 0.000 | 0.024 | 0.088 |
| **Control (teacher, plumber, cyclist, gardener...)** | 63 | **0.000** | **0.012** | 0.083 |

The controls are matched on corpus frequency and never fire. So this is not the
model reacting to any uncommon noun -- it is specific to identity, and within
identity it is specific to *which* identity.

The carrier matters as much as the term:

| Carrier template | FPR | Mean P(toxic) | Example |
|---|---|---|---|
| Everyday mention | 0.189 | 0.188 | "A gay man moved into the flat downstairs last week." |
| Benign self-description | 0.189 | 0.182 | "I am a gay man." |
| Neutral third person | 0.170 | 0.184 | "My neighbour is a gay man." |
| Positive statement | 0.075 | 0.106 | "I am proud to be a gay man." |
| Rights advocacy | 0.057 | 0.141 | "Muslims deserve the same rights as everyone else." -- 0.699 |
| Factual / encyclopedic | 0.057 | 0.060 | "The article describes the history of gay men in Europe." |
| Long benign context | 0.000 | 0.029 | a 30-word sentence about a colleague's presentation |

**The length pattern in the corpus is reproduced exactly in the model.** Prepend
one paragraph of unrelated neutral text and keep the final sentence identical:

| Final sentence | Alone | + 1 paragraph | + 2 paragraphs |
|---|---|---|---|
| I am a gay man. | **0.891** | 0.042 | 0.069 |
| I am a homosexual man. | **0.813** | 0.021 | 0.059 |
| I am a lesbian. | **0.668** | 0.015 | 0.040 |
| I am a black man. | **0.612** | 0.004 | 0.014 |
| I am a Muslim. | 0.199 | 0.006 | 0.013 |
| I am a teacher. | 0.016 | 0.002 | 0.007 |

The model has not learned "gay is toxic". It has learned "**a short sentence
containing an identity term is toxic**", which is precisely the conditional
distribution in its training data. That is a faithful model of a corpus that is
not a faithful model of the world.

This also tells you where the harm lands: on short text. Profile bios, chat
messages, one-line replies, comment stubs. Long-form writing is mostly safe.

### 2. It replicates across independently trained models

If this were one unlucky run, a different run would rank the terms differently.
Per-term mean P(toxic), Spearman correlation between three checkpoints trained
on the same corpus with different architecture flags, training lengths and
backbone-freezing regimes:

| Pair | Spearman |
|---|---|
| shipped vs language-conditioning ablation | 0.855 |
| shipped vs 2025 frozen-backbone run | 0.800 |
| ablation vs 2025 run | 0.827 |

Three models disagree about almost everything else -- macro F1 spans 0.60 to
0.88 across them -- and agree on which identities are toxic. The shared input is
the corpus.

### 3. The effect tracks the corpus statistic

The model's response to a term correlates with that term's toxic-rate lift in
training. Restricting to terms with at least 20 English rows (k = 37, below that
the rate is noise):

| Predictor | Spearman with model P(toxic) | p |
|---|---|---|
| `identity_hate` rate lift | **0.520** | 0.0010 |
| `toxic` rate lift | 0.400 | 0.014 |
| log corpus frequency | 0.221 | -- |

Frequency is the obvious confound and it is the weakest predictor. Residualising
both sides on log frequency leaves the lift relationship intact (partial Pearson
0.668). Terms that are absent from the corpus do not get elevated on novelty
alone: "nonbinary person" has 1 English row and scores 0.011.

Together, 1-3 are the argument. The intervention shows the model uses the term;
the replication shows it is not the run; the dose-response shows it tracks the
corpus. **It is the data.**

### 4. It shows up on real held-out comments, not just probes

Probes are synthetic. On the 17,637 test rows carrying *no* positive label,
split by whether the comment contains an identity term:

| Rows | n | FPR | 95% CI |
|---|---|---|---|
| Contains an identity term | 275 | **0.120** | [0.087, 0.164] |
| Does not | 17,362 | 0.042 | [0.039, 0.045] |

**2.84x**, bootstrap 95% CI [1.96, 3.82]. Elevated in six of seven languages
(Spanish is the exception at 0.75x, on 41 rows). Small samples, wide intervals,
but the interval excludes 1.

### 5. It is worse in every non-English language

The six non-English corpora are machine translations of a *single* English
source that is not the English split -- English is Wikipedia, the other six are
translations of what looks like a news-comment corpus (4.6% of French rows
mention Trump; 0.02% of English rows do). Their label rates are near-identical
to each other (toxic 0.496-0.501), which is what parallel translation with
copied labels looks like.

The lift transfers, and mostly grows. Toxic-rate lift vs each language's own
base rate:

| Term | en | ru | tr | es | fr | it | pt |
|---|---|---|---|---|---|---|---|
| gay | 1.88 | 1.87 | 1.83 | 1.91 | 1.91 | 1.89 | 1.92 |
| lesbian | 1.62 | 1.76 | 1.82 | 1.83 | 1.79 | 1.77 | 1.70 |
| jewish | 1.22 | 1.56 | 1.53 | 1.58 | 1.55 | 1.55 | 1.65 |
| muslim | 1.00 | 1.36 | 1.23 | 1.36 | 1.40 | 1.25 | 1.28 |
| atheist | 0.90 | 1.42 | 1.38 | 1.34 | 1.26 | 1.50 | 1.64 |
| transgender | 1.79 | 1.24 | 1.28 | 1.49 | 1.60 | 1.25 | 1.34 |
| christian | 0.85 | 0.90 | 0.99 | 1.03 | 0.93 | 0.97 | 0.92 |
| immigrant | 1.17 | 0.35 | 0.37 | 0.34 | 0.40 | 0.37 | 0.38 |

Note `atheist` and `immigrant`: the two corpora encode genuinely different
associations for the same term. This is the first audit of what the non-English
labels contain, and the answer is that they carry the identity-term bias at
equal or greater strength than English, plus their own separate biases.

Model behaviour follows. Nine identity terms and three controls, three carriers
each, all sentences benign:

| Language | Identity FPR | Control FPR |
|---|---|---|
| Portuguese | **0.741** | 0.000 |
| Russian | **0.704** | 0.000 |
| Turkish | 0.667 | 0.000 |
| Spanish | 0.667 | 0.000 |
| French | 0.593 | 0.000 |
| Italian | 0.556 | 0.000 |
| English | 0.407 | 0.000 |

**English is the least affected language.** Every measurement taken before this
one was English, so the published harm figures were the best case.

*Honesty about the translations:* the probe sentences are mine, written to be
grammatical, not verified by native speakers. Turkish is the one I am least
confident in. The *terms* inside them are on firmer ground: the six corpora are
parallel, so a term's row count should be similar across all six, and it is
(coefficient of variation 0.06-0.20 for most terms). Two fail that check and are
marked as upper bounds -- `blind` in Turkish (0.40, "kor" is a common substring)
and `black_person` everywhere (a colour word). Neither is load-bearing above.

## Where in the data it comes from

Representation, not annotation. The corpus does not contain benign usage that
was mislabelled -- it barely contains benign usage at all.

Reading all 726 English rows containing "gay":

| Slice | Count | Share | What it is |
|---|---|---|---|
| Total | 726 | -- | toxic 90.4%, identity_hate 52.6% |
| Matches a pejorative-predicate pattern ("u r gay", "gay fag", "that's gay") | 336 | 46.3% | toxic rate 0.949, correctly labelled |
| Labelled non-toxic | 70 | 9.6% | median 295 chars, mostly long article discussions |
| Short (<150 char) and non-toxic | 19 | 2.6% | and most of these are slurs mislabelled *non*-toxic |
| Genuinely benign short mentions, counted by hand | 3-5 | ~0.5% | e.g. "Hey, I'm gay too" |

And the decisive number. English rows matching a first-person LGBT
self-description pattern ("I am gay", "as a lesbian", "being a queer"):

**36 rows. 88.9% toxic. Not one of them is a genuine benign self-description.**

They are mockery-by-impersonation ("My name is Vascoamaral, I am a clown, and I
am Gay. I am an i.d.io.t"), vandalism, and sexual harassment. The exact sentence
form the model fails on appears in its training data *only* as a vehicle for
abuse. The model's 0.891 is not a malfunction. It is an accurate summary of the
corpus.

Annotation error exists but is second-order, and runs both ways. Mislabelled
toxic: *"Sexuality Maybe it should be added that he is actually gay. I believe
his partner is pop star Will Young"*, and *"Vandalism Enough of the wizard and
gay porn vandalism."* Mislabelled non-toxic: *"Gay propaganda is forbidden."*,
*"hey u gay bro."*, and one long comment ending *"Personally I can't stand
them"*. Roughly 5-10% of the rows I read are wrong in one direction or the
other. Fixing every one of them would not close a gap this size.

**The synthetic augmentation did not cause this.** The 402 Mistral-generated
rows are identifiable by a null `id` (all English, all `toxic`+`threat`). Three
of them contain an identity term -- 0.75%, *below* the 5.52% rate of the
non-augmented English rows. It is a clean result: this particular problem is not
the augmentation's fault.

The two diagnoses imply different fixes, which is why it mattered which one it
is. An annotation problem is fixed by relabelling rows you already have. **A
representation problem can only be fixed by adding rows that do not exist.**

## Options, with costs

### Raise the threshold -- does not work

To stop "I am a gay man." (0.891) firing you need a threshold near 0.90:

| Toxic threshold | F1 | Recall | Precision |
|---|---|---|---|
| 0.4724 (shipped) | 0.9641 | 0.9750 | 0.9534 |
| 0.70 | 0.9547 | 0.9386 | 0.9714 |
| 0.80 | 0.9345 | 0.8935 | 0.9794 |
| **0.90** | **0.8684** | **0.7739** | 0.9892 |

You would give up 22 points of recall on all toxicity everywhere to fix one
sentence, and "I am a lesbian." at 0.668 would still be safe only by luck. It
also does nothing about the ranking -- benign self-description would still score
above most genuine abuse.

### Post-hoc debiasing layer -- works, at a price you probably will not pay

Detect identity terms at inference and subtract a constant from the logit.
Simulated on the cached test predictions:

| Logit penalty | FPR, term rows | FPR, other rows | identity_hate recall | identity_hate F1 | Macro F1 |
|---|---|---|---|---|---|
| 0.0 (shipped) | 0.1200 | 0.0422 | 0.8419 | 0.8680 | 0.8814 |
| 1.0 | 0.0655 | 0.0422 | 0.7821 | 0.8447 | 0.8699 |
| **2.0 (parity)** | **0.0400** | 0.0422 | **0.6742** | 0.7744 | 0.8517 |
| 4.0 | 0.0000 | 0.0422 | 0.5352 | 0.6682 | 0.8260 |

**This is the honest quantification of the trade-off.** Reaching FPR parity
costs 17 points of recall on genuine identity-based hate -- the class whose
positives are 33% identity-term-bearing, and which exists precisely to catch
attacks on these groups. You would protect people from being silenced by
letting more attacks on them through. That is a real dilemma, not a rhetorical
one, and it is why the blunt fix is the wrong fix.

Read this table as the *upper bound* on the cost of a term-blind correction: it
penalises every row containing a term regardless of context, which is the
crudest possible intervention. A retrained model can learn context instead.

### Fine-tune only on counter-examples -- cheap, and probably enough to test with

A short LoRA or last-N-layer pass over the counter-example set plus a replay
sample. Perhaps 20-30 minutes. Worth running *before* the full retrain purely as
a signal check, but not as the shipped answer -- fine-tuning on a narrow
distribution risks the model learning "sentences of this shape are safe" rather
than "the term is not the signal", and that would pass the probe set while
leaving the behaviour intact. If you do this, hold out probe templates the
tuning set never saw.

### Add counter-examples and retrain -- recommended

The established fix. Add benign, in-distribution comments containing identity
terms until the term stops carrying the signal.

Budget, computed from the corpus. For each term, the number of benign short
(<150 char) rows needed to pull that term's short-text toxic rate down to the
0.48 corpus base rate:

| Term | Short rows now | Toxic rate | To halve the lift | To reach base rate |
|---|---|---|---|---|
| gay | 432 | 0.956 | 144 | 429 |
| jewish | 120 | 0.750 | 27 | 68 |
| black | 82 | 0.768 | 19 | 50 |
| mexican | 18 | 0.889 | 6 | 16 |
| lesbian | 16 | 0.875 | 5 | 14 |
| blind | 14 | 0.929 | 5 | 14 |
| muslim | 30 | 0.700 | 6 | 14 |
| christian | 23 | 0.652 | 4 | 9 |
| atheist | 11 | 0.636 | 2 | 4 |
| **Total** | | | **218** | **618** |

618 English sentences for these nine terms. Widen to the full probe vocabulary
-- transgender, queer, bisexual, nonbinary, deaf, disabled, Latino, Hispanic,
Pakistani, Arab -- and allow several carrier shapes per term, and the realistic
figure is **1,000-1,500 hand-written English sentences**.

They then go through the same machine-translation pipeline that built the rest
of the corpus, giving roughly 7,000-10,500 rows total, or 2.5-3.7% of the
285,264-row training set. The human effort is English-only.

Cost:

| Item | Cost |
|---|---|
| Write 1,000-1,500 English benign sentences | 1-2 days, one person |
| Translate to six languages via the existing pipeline | hours, scripted |
| Full 6-epoch retrain, one Quadro RTX 6000 | 3.5 hours |
| Re-tune thresholds, re-evaluate, re-run this probe set | ~1 hour |
| **Total** | **~2-3 days, one retrain** |

Two things the sentences must get right, or the fix will not take:

**They must be short.** The bias is conditional on length. A thousand long
paragraphs would leave "I am a gay man." exactly where it is.

**They must not be a template.** If every counter-example is "I am a {X}.", the
model learns that one string is safe and generalises nothing. Vary the carrier,
the register, the surrounding content -- and hold out probe templates the
training set never contains, so the evaluation is not scoring memorisation.

### Also worth doing regardless: term-blind evaluation

Nothing in the current evaluation would have caught this. Macro F1 0.8814 and
macro AUC 0.9852 are entirely compatible with flagging every gay person who
introduces themselves, because such sentences are not in the test set either --
the test split inherits the same gap. Add per-identity-group FPR to the standard
evaluation (`experiments/identity_bias.py probe`) so the next model is measured
on this before it ships. This is monitoring, not a fix, and it costs nothing.

## Recommendation

**Write the counter-example set and retrain.** It is the only option that
addresses the cause rather than the symptom, the only one that does not trade
recall on identity-based hate for a lower false-positive rate, and it costs two
to three days plus a 3.5-hour run. The blunt alternatives are quantified above
and both are worse: the threshold fix costs 22 points of recall, and the
post-hoc layer costs 17 points of `identity_hate` recall.

Before that lands, the shipped model flags benign self-description by gay,
lesbian, queer, transgender, Black and deaf people, in seven languages, worst in
Portuguese and Russian. If it is serving live traffic, that is happening now.
Whether to keep serving it in the meantime is a call for the owner, but it
should be a decision someone makes on purpose, and the model card should say
this.

## What remains unmeasured

- **Whether the fix works.** Everything here diagnoses; nothing validates. The
  counter-example approach is established for this corpus and Jigsaw published
  it, but the retrain has not been run and its effect on this probe set is a
  prediction, not a measurement.
- **The recall trade-off under retraining.** The 17-point figure is for a blunt
  post-hoc penalty. A retrained model should do better because it can use
  context, but by how much is unknown. The experiment that answers it: retrain
  with counter-examples, then compare `identity_hate` recall and per-group FPR
  against this document's numbers on the same test split.
- **Real-data FPR rests on 275 rows.** The interval is [0.087, 0.164]. A larger
  identity-term sample, ideally from the deployed app's actual traffic, would
  tighten it. The traffic distribution is unknown, so the real-world harm *rate*
  is unknown -- only the per-comment probability is measured.
- **The translated probes are not native-verified.** Turkish especially. The
  direction of the finding is robust (every language elevated, controls at zero
  everywhere) but the per-language ordering should not be over-read.
- **Whether the six translated corpora preserved meaning.** This document shows
  the term-level lift transfers. It does not verify that a comment labelled
  toxic in English is still toxic in Turkish. That is a separate audit and it is
  still owed.
- **Terms too rare to measure.** nonbinary (1 row), wheelchair user (2),
  cisgender (1). Their corpus lift is undefined, so they are absent from the
  dose-response, and their model scores -- 0.011, 0.005, and a surprising 0.407
  for "cisgender man" -- are unexplained. The last one is probably subword
  leakage from "transgender", which is well attested in the other six languages,
  but that is a guess.
- **Intersectional terms.** "black lesbian", "disabled Muslim woman" -- untested.
  Terms compose in the corpus and probably compound in the model.
- **How much comes from XLM-R pretraining rather than this corpus.** The absent-
  term control argues most of it is learned here, but the base model's own
  representations were not probed.
