"""Write the benign identity sentences Civil Comments could not supply.

`scripts/build_counterexamples.py` mines real benign identity-term comments out of
google/civil_comments. It works, and 570 rows survived review, but the agent that
built it measured two hard limits:

  * only 35 of the 570 are first-person self-description -- the exact carrier
    the model fails on ("I am a gay man." -> 0.891 against a 0.472 threshold).
    News-site comment sections do not contain many people introducing themselves.
  * about 60% of the rows scoring `toxicity < 0.2` are hostile with no annotator
    flag, so the Civil Comments toxicity score cannot be trusted unread.

So the failing carrier has to be *written*. That is what this file does. See
`experiments/identity_bias.md` for the diagnosis these constraints come from.

  LENGTH   The single most important property, and the easiest to get wrong.
           The bias is conditional on length: "I am a gay man." scores 0.891
           alone and 0.042 with one paragraph of neutral text in front of it. A
           set of natural-sounding twenty-word sentences trains the model on
           inputs it already handles correctly. Target here is a median of 8-10
           words, at least 40% under 8, at least 70% under 15, hard cap 22.
           "I'm a lesbian." and "My brother is gay." are the most valuable rows
           in the file, and short IS the natural register for this form -- they
           do not need padding out into paragraphs to sound real.

  CARRIER  Six carrier forms, not one. If every row is "I am a {X}." the model
           learns that one string is safe and generalises nothing. First-person
           self-description leads because it is the failing form, but it is
           under half the set.

  REGISTER Plain, casual and slightly formal, mixed, and NOT all affirming.
           "I'm gay and I hate cilantro" teaches that the term is orthogonal to
           toxicity, which is the thing being taught. "I am a proud gay man
           living my truth" teaches one narrow register. Negative content is
           allowed and wanted, but it attaches to a situation -- a delay, a
           bill, the rent -- never to the person.

  WEIGHT   Group shares track measured false-positive rate, not how easy a group
           is to write for. Sexual orientation is 0.490 and takes about half;
           ethnicity and nationality is 0.000 and takes a token slice.

Two sources, mixed on purpose:

  hand     Hand-written frames combined combinatorially, in this file. Short
           frames outnumber long ones roughly 3:2 by design.
  ollama   A local model, for surrounding content more varied than a fixed frame
           list can be.

MODEL CHOICE: gemma4:26b, after measuring three.

  gemma4:26b -- chosen. Best output of the three: blunt, short, plain, varied
    grammar, and it writes "a trans woman" rather than "a transgender
    individual" without being asked twice. ~57s for 16 usable sentences.
    The catch, and it took three attempts to find: it is a thinking model with
    no thinking budget, and ANY prompt containing an enumerated list of
    constraints -- a word-count range, sixteen settings to map one-to-one, a
    "vary the grammar" meta-rule -- sends it into unbounded self-verification.
    The first attempt spent 7.5k reasoning tokens re-counting words by hand, hit
    the token cap and returned empty content. Prompts here are therefore prose
    plus example shapes with zero enumerated constraints; variety is imposed
    from outside by rotating term, form and a single place hint per batch.
  deepseek-r1:32b -- close on quality, 4m38s for the same batch, wraps its
    output in a preamble and a numbered list, and produced one broken sentence
    ("The neighbor down the street gardens as a trans woman."). Rejected on cost.
  qwen2.5vl:32b -- a 2024 vision-language model, tried first and rejected. Its
    output ran 21 words median with nothing under 10, latched onto one register,
    dropped the required identity term from 15% of rows, and wrote stilted
    phrasing ("both were visibly transgender men"). Everything the length and
    plainness requirements exclude.
  The abliterated / `-unc` variants were not used: nothing here would be refused
  by a standard model, and they are weaker at instruction-following, which is
  the only capability this task actually needs.

The model is not trusted to comply. `generate()` drops any line missing its own
assigned identity term, or outside 2-22 words, *before* it reaches the pool. And
nothing is trusted unread after that either: every row surviving the automatic
filters was printed by `review` and read, with the casualties in `DROPPED`.

Usage (from the repo root, PYTHONPATH=.):
    python scripts/generate_counterexamples.py generate   # ollama -> raw pool
    python scripts/generate_counterexamples.py review     # dump kept rows to read
    python scripts/generate_counterexamples.py build      # curate, dedup, write
    python scripts/generate_counterexamples.py score      # score vs the current model
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "dataset" / "counterexamples"
SPLIT_DIR = REPO_ROOT / "dataset" / "split"
RAW_POOL = OUT_DIR / "generated_raw_pool.jsonl"

TARGET_COLS = ["id", "comment_text", "toxic", "severe_toxic", "obscene",
               "threat", "insult", "identity_hate", "lang"]
CLASSES = TARGET_COLS[2:8]

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:26b"
NUM_PREDICT = 5000      # gemma4 thinks before it writes; this is mostly headroom

MIN_WORDS = 2           # "I'm gay." is two words and is the point of the exercise
MAX_WORDS = 22          # the bias has vanished well before here

RANDOM_SEED = 20260830
HELDOUT_SHARE = 0.12
# Fraction of a candidate's own 5-word sequences that a single existing row has
# to cover before the two count as the same sentence. Same value and same test
# as build_counterexamples.py, so both sets are checked on identical terms.
NEARDUP_SHARE = 0.5

N_HAND = 620        # hand-written frame fills to attempt
N_OLLAMA = 1100     # ollama lines to attempt (the gates reject a lot)

# The .bin, not the directory: OptimizedToxicityClassifier only accepts a
# directory that holds a "latest" symlink or checkpoint_epoch* subdirs, and
# best_model/ holds the weights flat. Same path experiments/identity_bias.py uses.
CKPT = (REPO_ROOT / "weights" / "toxic_classifier_xlmr_v2" / "best_model"
        / "pytorch_model.bin")

# Shipped per-class thresholds -- hf_release/thresholds.json. A class fires at or
# above. Used only to report what share of these benign rows the current model
# would flag.
TH = {"toxic": 0.4724489795918367, "severe_toxic": 0.4724489795918367,
      "obscene": 0.5275510204081633, "threat": 0.5275510204081633,
      "insult": 0.5642857142857143, "identity_hate": 0.5642857142857143}


# ---------------------------------------------------------------------------
# Group budget. Shares track the per-category false-positive rate measured in
# experiments/identity_bias.md, normalised, with a floor so the two groups at or
# near zero still contribute something rather than nothing.
#
#   sexual orientation      FPR 0.490
#   gender identity             0.179
#   race                        0.122
#   disability                  0.048
#   religion                    0.032   -- nearly unbiased already
#   ethnicity / nationality     0.000   -- minimal
# ---------------------------------------------------------------------------
GROUP_SHARE = {
    "sexual_orientation": 0.48,
    "gender_identity": 0.20,
    "race": 0.16,
    "disability": 0.08,
    "religion": 0.05,
    "ethnicity_nationality": 0.03,
}

# (noun phrase after a/an, predicate form after "is"/"I'm", plural, gender)
# Gender is here only so the hand-written frames agree with the relative they
# are attached to -- "my brother is a lesbian" is not a benign sentence, it is a
# broken one. Kept aligned with the probe vocabulary in
# experiments/identity_bias.py so these rows cover the terms the probe measures.
# See `probe_sentences()`: no generated row is allowed to BE a probe sentence.
TERMS = {
    "sexual_orientation": [
        ("gay man", "gay", "gay men", "m"),
        ("gay woman", "gay", "gay women", "f"),
        ("lesbian", "a lesbian", "lesbians", "f"),
        ("bisexual woman", "bisexual", "bisexual people", "f"),
        ("bisexual man", "bi", "bisexual people", "m"),
        ("queer person", "queer", "queer people", "n"),
        ("homosexual man", "homosexual", "homosexual men", "m"),
        ("straight man", "straight", "straight men", "m"),
        ("heterosexual woman", "heterosexual", "heterosexual women", "f"),
    ],
    "gender_identity": [
        ("transgender woman", "transgender", "transgender women", "f"),
        ("transgender man", "transgender", "transgender men", "m"),
        ("trans man", "trans", "trans men", "m"),
        ("trans woman", "trans", "trans women", "f"),
        ("nonbinary person", "nonbinary", "nonbinary people", "n"),
        ("cisgender man", "cisgender", "cisgender men", "m"),
    ],
    "race": [
        ("black man", "black", "black people", "m"),
        ("black woman", "black", "black women", "f"),
        ("white man", "white", "white people", "m"),
        ("Asian woman", "Asian", "Asian people", "f"),
        ("African American man", "African American", "African Americans", "m"),
        ("Native American woman", "Native American", "Native Americans", "f"),
        ("Latino man", "Latino", "Latinos", "m"),
        ("Hispanic woman", "Hispanic", "Hispanic people", "f"),
    ],
    "disability": [
        ("deaf man", "deaf", "deaf people", "m"),
        ("blind woman", "blind", "blind people", "f"),
        ("autistic person", "autistic", "autistic people", "n"),
        ("disabled man", "disabled", "disabled people", "m"),
        ("wheelchair user", "a wheelchair user", "wheelchair users", "n"),
        ("dyslexic person", "dyslexic", "dyslexic people", "n"),
    ],
    "religion": [
        ("Muslim", "Muslim", "Muslims", "n"),
        ("Jewish man", "Jewish", "Jewish people", "m"),
        ("Christian", "Christian", "Christians", "n"),
        ("Hindu", "Hindu", "Hindus", "n"),
        ("Sikh", "Sikh", "Sikhs", "n"),
        ("Buddhist", "a Buddhist", "Buddhists", "n"),
        ("atheist", "an atheist", "atheists", "n"),
        ("Catholic", "Catholic", "Catholics", "n"),
        ("Mormon", "a Mormon", "Mormons", "n"),
    ],
    "ethnicity_nationality": [
        ("Mexican", "Mexican", "Mexicans", "n"),
        ("Indian", "Indian", "Indians", "n"),
        ("Chinese woman", "Chinese", "Chinese people", "f"),
        ("Arab man", "an Arab", "Arabs", "m"),
        ("Pakistani woman", "Pakistani", "Pakistanis", "f"),
        ("Irish man", "Irish", "Irish people", "m"),
        ("Polish woman", "Polish", "Polish people", "f"),
        ("Nigerian man", "Nigerian", "Nigerians", "m"),
    ],
}

# Carrier forms and the share of the set each is aimed at. First-person
# self-description leads because it is the failing carrier, but it is
# deliberately under half: the lesson is that the term is safe, not that one
# sentence shape is safe.
FORM_SHARE = {
    "self_desc": 0.22,           # "I'm gay."
    "self_desc_clause": 0.20,    # "I'm deaf and I teach maths."
    "third_person_known": 0.16,  # "My sister is a lesbian."
    "incidental": 0.18,          # identity present, not the point of the sentence
    "midsentence": 0.12,         # identity is not the predicate
    "factual": 0.12,             # neutral factual / descriptive
}


# ---------------------------------------------------------------------------
# Hand-written material.
#
# Two frame lists per carrier -- short and medium -- sampled roughly 3:2 in
# favour of short, because the length distribution is the property this set
# lives or dies on. The short lists carry no trailing clause at all.
#
# These are frames, not a template: ~180 distinct surface shapes, with the term
# in different syntactic positions, and trailing content drawn from MUNDANE so
# no two fills share both a shape and a subject. The result is combinatorial but
# does not read as one string with a slot in it, which is the requirement.
#
# Register is deliberately mixed British and American: a set that is all
# allotments and hospital car parks is its own kind of monoculture.
# ---------------------------------------------------------------------------

# Two to four words, for the short frames.
MUNDANE_SHORT = [
    "I'm broke", "I'm tired", "I hate cilantro", "I can't cook", "I'm always late",
    "I don't drive", "I hate mornings", "my back hurts", "I'm hungry", "I snore",
    "I overslept", "my car died", "I burn toast", "I can't swim", "I don't drink",
    "I lost my keys", "I hate flying", "the rent went up", "I'm out of coffee",
    "my phone's dead", "I'm bad at chess", "I forgot the milk", "I need a haircut",
    "I can't whistle", "my knee hurts", "I hate parking", "I've got a cold",
    "I owe my dentist money", "I miss my dog", "I'm out of milk",
]

# Longer, still ordinary, sometimes mildly negative -- at a situation, never a
# person. For the medium frames.
MUNDANE = [
    "I still can't parallel park",
    "I burn every pan I own",
    "I'm terrible at remembering names",
    "I drive a fourteen-year-old Corolla",
    "my commute is fifty minutes each way",
    "I keep forgetting to water the basil",
    "I've been on hold with the bank since ten",
    "my knees are shot from running",
    "I put too much salt in everything",
    "I've lost three umbrellas this year",
    "I can't sleep without a fan on",
    "I'm allergic to almost every cat I meet",
    "I've never once finished a crossword",
    "the guy upstairs plays drums at seven",
    "I've been meaning to fix the gate since March",
    "I only ever cook two things",
    "I'm the slowest person on the group ride",
    "I still use a paper calendar",
    "my phone battery dies by two",
    "I forgot to renew the parking permit",
    "I overwater every plant I touch",
    "I lose at chess to a nine-year-old",
    "the coffee machine at work is broken again",
    "I've been putting off the tax forms",
    "I sing badly and loudly in the car",
    "I never learned to swim properly",
    "I have strong opinions about vacuum cleaners",
    "I keep buying books I don't read",
    "my sourdough starter died twice",
    "I don't understand the offside rule",
    "I'm the designated driver most weekends",
    "my car failed its inspection twice",
    "I've been late to work three times this week",
    "I still owe my brother forty bucks",
    "the landlord hasn't fixed the heat",
    "I got a parking ticket on Tuesday",
    "the DMV line took two hours",
    "I dropped my phone in the sink",
]

# Two to seven words. This is where most of the mass belongs.
SELF_SHORT = [
    "I'm {p}.",
    "I'm {p}, by the way.",
    "Yeah, I'm {p}.",
    "Me, I'm {p}.",
    "Hi, I'm {p}.",
    "I'm {a_n}.",
    "I'm {p}, obviously.",
    "I'm {p} and {ms}.",
    "I'm {p}, thanks for asking.",
    "I'm {p}, thirty-two.",
    "I'm {p}. {MS}.",
    "So, I'm {p}.",
    "I'm {p} and single.",
    "I'm {p} and skint.",
    "I'm {a_n}, apparently.",
    "I'm {p} and I bake.",
    "For the record, I'm {p}.",
    "I'm {p} and it's fine.",
    "I'm {p} and I teach.",
    "I'm {a_n} from Leeds.",
    "I'm {a_n} and a bad driver.",
    "I'm {p}, if it matters.",
    "I'm {p} and I drive a van.",
    "I'm {p}, still renting.",
    "Look, I'm {p}.",
    "I'm {p} and I work nights.",
    "I've always been {p}.",
    "I'm {p} and I'm forty.",
    "I'm {p}. That's it.",
    "I'm {p} and I need coffee.",
    "I'm {a_n}, new here.",
    "I'm {p} and I fix bikes.",
    "Guess I'm {p}.",
    "I'm {p} and I own a cat.",
]

# Eight to fifteen words.
SELF_MED = [
    "I'm {p} and {m}.",
    "I'm {p}, and honestly {m}.",
    "{M}, and I'm {p}.",
    "{M}. Also I'm {p}.",
    "Just so it's on the table, I'm {p}.",
    "I am {p}, if that changes anything.",
    "I'm {p} and I live two streets over.",
    "Speaking as {a_n}, {m}.",
    "As {a_n}, {m}.",
    "I'm {a_n} in my forties.",
    "I'm {a_n}, thirty-two, and {ms}.",
    "I'm {p}, which is not the interesting part.",
    "I'm {p} and I've lived here since 2011.",
    "Not that anyone asked, but I'm {p}.",
    "I'm {p} and I run a shop on the high street.",
    "I'm {p}, married, two kids, one dog.",
    "I'm {p} and I mostly lurk here.",
    "I'm {a_n} who cannot cook.",
    "I'm {p} and I've got nothing clever to add.",
    "I'm {p} and I've been reading this thread all week.",
    "I'm {p} and I moved here for the job.",
    "I'm {p} and I hate group photos.",
    "I'm {p}, currently unemployed, currently fine.",
    "I'm {p} and my apartment is far too small.",
    "I'm {a_n} and I bake at weekends.",
    "I'm {p} and I've never been to a game.",
    "I'm {p} and this is my first post here.",
    "I'm {p} and I own far too many mugs.",
    "I'm {p}, and I'd rather talk about the weather.",
    "I'm {p} and I can't spell to save my life.",
    "I'm {a_n} living in a town of nine thousand.",
    "I'm {p} and I've had the same haircut for a decade.",
    "I'm {p} and I signed up to complain about the bins.",
    "I'm {p}, forty-one, and {m}.",
]

# People the writer knows. Deliberately not "my neighbour" -- that is the
# probe's third-person carrier and it stays out of the training set.
KNOWN = {
    "m": ["my brother", "my uncle", "my dad", "my son", "my nephew",
          "my best mate", "my brother-in-law", "the guy at the corner shop",
          "my father-in-law", "my roommate Dan"],
    "f": ["my sister", "my aunt", "my mum", "my daughter", "my sister-in-law",
          "the woman who cuts my hair", "my niece", "my mother-in-law",
          "the woman at the post office"],
    "n": ["my cousin", "my flatmate", "my oldest friend", "my landlord",
          "my dentist", "my boss", "my coworker Ravi", "my mechanic", "my doctor",
          "my running partner", "my old roommate", "one of my coworkers",
          "our team lead", "my climbing partner", "my line manager"],
}

THIRD_SHORT = [
    "{k} is {p}.",
    "{k}'s {p}, I think.",
    "{k} is {p} and {m3s}.",
    "{k} is {p} too.",
    "{k} is {p} and broke.",
    "{k} is {p}, apparently.",
    "{k} is {p} and lives nearby.",
    "{k} is {p} and hates flying.",
    "{k} is {p} and never calls.",
    "Turns out {k} is {p}.",
    "{k} is {p} and always late.",
    "{k} is {p}, if you must know.",
]

# No gendered pronouns -- the relative's gender is fixed by KNOWN and a
# mismatched pronoun would make the row ungrammatical.
THIRD_MED = [
    "{k} is {p} and {m3}.",
    "{k} is {p} and has terrible taste in films.",
    "{k} is {p} and works longer hours than anyone I know.",
    "{k}, who is {p}, lent me a ladder in April.",
    "{k} is {p} and lives about an hour north.",
    "{k} is {p} and never answers the phone.",
    "{k} is {p} and does the books for a small charity.",
    "{k} is {p} and has been fixing that fence for two years.",
    "{k} is {p}, and between us we own one working car.",
    "{k} is {p} and makes the worst coffee in the building.",
    "{k} is {p} and turned fifty last month.",
    "{k} is {p} and reads three books a week.",
    "{k} is {p} and hates the new opening hours.",
    "{k} is {p} and still drives like a maniac.",
    "{k} is {p}, and we still argue about the thermostat.",
    "{k} is {p} and moved to Leeds in the spring.",
]

MUNDANE3S = [
    "keeps chickens", "still smokes", "is always late", "won't eat mushrooms",
    "is always cold", "hates mornings", "drives a van", "plays the trumpet",
    "grows tomatoes", "does the crossword in pen", "snores", "can't cook",
]

MUNDANE3 = [
    "has a boat that never leaves the drive", "runs the quiz on Thursdays",
    "can't stand the new supermarket", "collects stamps, genuinely",
    "drives a van with a dent in the side",
    "walks the dog past our window at six", "is learning the trumpet badly",
    "grows more tomatoes than anyone can eat", "took up swimming last year",
    "has never once been on time", "burns toast every single morning",
]

# Attributive form of each term, for the incidental and mid-sentence frames
# where the identity modifies a role noun rather than sitting in the predicate.
# `None` where the term has no attributive use -- "a wheelchair user plumber" is
# not English, so that term simply does not take those frames.
ATT = {
    "gay man": "gay", "gay woman": "gay", "lesbian": "lesbian",
    "bisexual woman": "bisexual", "bisexual man": "bisexual",
    "queer person": "queer", "homosexual man": "homosexual",
    "straight man": "straight", "heterosexual woman": "heterosexual",
    "transgender woman": "transgender", "transgender man": "transgender",
    "trans man": "trans", "trans woman": "trans",
    "nonbinary person": "nonbinary", "cisgender man": "cisgender",
    "black man": "black", "black woman": "black", "white man": "white",
    "Asian woman": "Asian", "African American man": "African American",
    "Native American woman": "Native American", "Latino man": "Latino",
    "Hispanic woman": "Hispanic",
    "deaf man": "deaf", "blind woman": "blind", "autistic person": "autistic",
    "disabled man": "disabled", "wheelchair user": None,
    "dyslexic person": "dyslexic",
    "Muslim": "Muslim", "Jewish man": "Jewish", "Christian": "Christian",
    "Hindu": "Hindu", "Sikh": "Sikh", "Buddhist": "Buddhist",
    "atheist": "atheist", "Catholic": "Catholic", "Mormon": "Mormon",
    "Mexican": "Mexican", "Indian": "Indian", "Chinese woman": "Chinese",
    "Arab man": "Arab", "Pakistani woman": "Pakistani", "Irish man": "Irish",
    "Polish woman": "Polish", "Nigerian man": "Nigerian",
}

# Role nouns the attributive form attaches to. Gendered, so "a gay woman bloke"
# cannot happen.
ROLES = {
    "m": ["man", "guy", "bloke", "chap", "lad"],
    "f": ["woman", "lady"],
    "n": ["person", "regular", "volunteer"],
}

# The identity is present and irrelevant; the sentence is about the bakery, the
# compost, the boiler. This is the carrier the diagnosis calls "everyday
# mention" and measures at FPR 0.189, the joint worst.
INCIDENTAL_SHORT = [
    "The {att} {role} next door keeps chickens.",
    "A {att} {role} fixed our router.",
    "The {att} {role} at the bakery makes good bread.",
    "A {att} {role} sold me this ladder.",
    "The {att} {role} upstairs plays piano.",
    "Our plumber is {p} and he's quick.",
    "A {att} {role} runs the corner shop.",
    "The {att} {role} at the gym uses my locker.",
    "A {att} {role} taught my brother to drive.",
    "The {att} {role} down the road has three cats.",
    "A {att} {role} left an umbrella here.",
    "The {att} {role} at reception found my parcel.",
    "A {att} {role} drove the school bus today.",
    "The {att} {role} at the garage said it's fine.",
]

INCIDENTAL_MED = [
    "The {att} {role} who runs the bakery makes very good sourdough.",
    "A {att} {role} in the queue ahead of me had the timetable memorised.",
    "The {att} {role} at the garden centre sold me the wrong compost.",
    "A {att} {role} on my train reads the same newspaper every morning.",
    "The {att} {role} two doors down keeps chickens and sells the eggs.",
    "Our plumber is {p} and he found the leak in four minutes.",
    "The {att} {role} behind the counter remembered my order.",
    "The {att} {role} who cut my hair talked about football for ages.",
    "A {att} {role} from the running club lent me a spare pair of shoes.",
    "A {att} {role} in my building keeps propping the fire door open.",
    "A {att} {role} on the ferry said the crossing is always like that.",
    "The {att} {role} running the pub quiz asked a question about moths.",
]

# The identity sits inside the clause and is never the predicate.
MID_SHORT = [
    "Two {plural} on my team run the tab.",
    "Half the {plural} here drive trucks.",
    "My cousin, {a_n}, sells cars.",
    "The {att} {role} who fixed our boiler was cheap.",
    "Most {plural} I know take the early train.",
    "One of the {plural} in my club always wins.",
    "Three {plural} on the committee voted no.",
    "A few {plural} at work cycle in.",
    "The {att} {role} upstairs practises at seven.",
    "Two {plural} in the choir play piano.",
]

MID_MED = [
    "Two of the {plural} on my five-a-side team organise the food.",
    "Half the {plural} at the allotment have given up on courgettes.",
    "My editor, {a_n}, gets the copy back faster than anyone here.",
    "The {att} {role} who fixed our boiler charged less than the last one.",
    "A couple of {plural} at the chess club still use a physical clock.",
    "My cousin, {a_n}, has been trying to sell that car since March.",
    "The {att} {role} at reception found my parcel behind a filing cabinet.",
    "Several {plural} in the choir have complained about the rehearsal times.",
]

# Flat, neutral, listing-style statements. Local news, a schedule, a notice.
FACTUAL_SHORT = [
    "The second speaker was {a_n}.",
    "Two of the six candidates were {plural}.",
    "About four hundred {plural} live here.",
    "The committee appointed {a_n} as treasurer.",
    "The club has forty {plural} on its books.",
    "Three {plural} applied for the post.",
    "The survey counted {plural} separately.",
    "The panel included {a_n}.",
    "Roughly a fifth of applicants were {plural}.",
    "The register lists nine {plural}.",
]

FACTUAL_MED = [
    "The clinic employs {a_n} two days a week.",
    "The library runs a Thursday reading group for {plural}.",
    "The survey counted {plural} separately from the rest of the sample.",
    "The 2019 census recorded a small increase in the number of {plural}.",
    "The panel included {a_n}, a retired engineer, and a historian.",
    "The charity has run a weekly drop-in for {plural} since 2015.",
    "The article lists every {noun} elected to the council since 1998.",
    "The report notes that {plural} made up eleven percent of applicants.",
]

# form -> (short frames, medium frames, share of the hand budget, short fraction)
HAND_MIX = {
    "self_desc":          (SELF_SHORT, SELF_MED, 0.36, 0.66),
    "third_person_known": (THIRD_SHORT, THIRD_MED, 0.22, 0.60),
    "incidental":         (INCIDENTAL_SHORT, INCIDENTAL_MED, 0.18, 0.62),
    "midsentence":        (MID_SHORT, MID_MED, 0.12, 0.62),
    "factual":            (FACTUAL_SHORT, FACTUAL_MED, 0.12, 0.62),
}


def _a(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _fill(frame: str, noun: str, pred: str, rng: random.Random, **extra) -> str:
    s = (frame.replace("{p}", pred)
              .replace("{noun}", noun)
              .replace("{a_n}", f"{_a(noun)} {noun}"))
    for tag, pool, cap in [("{ms}", MUNDANE_SHORT, False), ("{MS}", MUNDANE_SHORT, True),
                           ("{m}", MUNDANE, False), ("{M}", MUNDANE, True),
                           ("{m3s}", MUNDANE3S, False), ("{m3}", MUNDANE3, False)]:
        while tag in s:
            v = rng.choice(pool)
            s = s.replace(tag, _cap(v) if cap else v, 1)
    for k, v in extra.items():
        s = s.replace("{" + k + "}", v)
    return re.sub(r"\s+", " ", s).strip()


def hand_written(rng: random.Random, n_by_group: dict) -> list[dict]:
    """Fill the hand-written frames. Each (frame, term) pair is used at most once."""
    out = []
    for group, n_target in n_by_group.items():
        terms = TERMS[group]
        for form, (short, med, share, short_frac) in HAND_MIX.items():
            n_form = int(round(n_target * share))
            n_short = round(n_form * short_frac)
            for frames, n in [(short, n_short), (med, n_form - n_short)]:
                pairs = [(f, t) for f in frames for t in terms
                         if "{att}" not in f or ATT[t[0]]]
                rng.shuffle(pairs)
                for frame, (noun, pred, plural, g) in pairs[:n]:
                    extra = {"plural": plural}
                    if "{k}" in frame:
                        extra["k"] = rng.choice(
                            KNOWN["n"] if g == "n" else KNOWN[g] + KNOWN["n"])
                    if "{att}" in frame:
                        extra["att"] = ATT[noun]
                        extra["role"] = rng.choice(ROLES[g])
                    out.append(dict(text=_cap(_fill(frame, noun, pred, rng, **extra)),
                                    group=group, term=noun, form=form,
                                    source="hand", frame=frame))
    return out


# ---------------------------------------------------------------------------
# ollama generation.
#
# One carrier form, one term and one place hint per batch. Variety is imposed by
# rotating those three across batches, NOT by asking the model to vary itself --
# a "vary the grammar" instruction is an enumerated constraint and sends gemma4
# into unbounded self-verification (see the model note at the top). For the same
# reason the prompts are prose plus example shapes, with no numeric length range
# and no list of settings to satisfy one-to-one.
# ---------------------------------------------------------------------------

FORM_PROMPT = {
    "self_desc": (
        "Write {n} short, plain English sentences. In each one the writer says they are "
        "{desc}, the way someone would in a chat message or a forum introduction.\n\n"
        "Like this:\n"
        "  I'm bi and I'm broke.\n"
        "  Yeah, I'm Catholic.\n"
        "  I'm a wheelchair user, by the way.\n"
        "  I'm deaf. Nice to meet you.\n"
        "  Hi, I'm autistic and I fix bikes.\n"
    ),
    "self_desc_clause": (
        "Write {n} short, plain English sentences. In each one the writer mentions being "
        "{desc} and then says something ordinary about their job or their day.\n\n"
        "Like this:\n"
        "  I'm a deaf teacher and I've been at it nine years.\n"
        "  I'm Sikh and I drive a delivery van.\n"
        "  I'm bisexual and my oven is broken.\n"
        "  I'm blind and I take the 8:15 in.\n"
        "  I'm Hindu and I work weekends.\n"
    ),
    "third_person_known": (
        "Write {n} short, plain English sentences. In each one somebody the writer knows -- "
        "a sister, a coworker, a landlord -- is {desc}, and the sentence says something "
        "ordinary about them.\n\n"
        "Like this:\n"
        "  My sister is a lesbian and she still owes me forty quid.\n"
        "  My coworker is Hindu and he runs the fantasy league.\n"
        "  My landlord is blind. Nice guy.\n"
        "  My cousin is trans and drives a taxi.\n"
        "  Turns out my dentist is Mormon.\n"
    ),
    "incidental": (
        "Write {n} short, plain English sentences. In each one somebody is {desc}, but that "
        "is a passing detail and the sentence is really about something else.\n\n"
        "Like this:\n"
        "  The trans woman next door has three cats.\n"
        "  A blind man fixed our router.\n"
        "  Two Sikh guys run the corner shop.\n"
        "  I bought the ladder off a gay man in Hull.\n"
        "  The autistic kid down the road knows every train.\n"
    ),
    "midsentence": (
        "Write {n} short, plain English sentences. In each one the fact that somebody is "
        "{desc} sits in the middle of the sentence and is never the last thing in it.\n\n"
        "Like this:\n"
        "  Two of the Sikh guys on my team organise the food.\n"
        "  My editor, an atheist, still edits the religion column.\n"
        "  Half the deaf students in that class cycle in.\n"
        "  The blind woman who fixed our boiler was cheap.\n"
        "  Three trans men on the committee voted no.\n"
    ),
    "factual": (
        "Write {n} short, plain English sentences. Each is a flat factual statement "
        "involving {desc} -- the kind of line in a local news item, a schedule or a "
        "notice.\n\n"
        "Like this:\n"
        "  The clinic employs a deaf liaison two days a week.\n"
        "  A fifth of this year's intake were Chinese students.\n"
        "  The second speaker was a nonbinary researcher.\n"
        "  Two of the six candidates were Muslims.\n"
        "  The register lists nine wheelchair users.\n"
    ),
}

RULES = (
    "Keep them blunt and short, most under ten words. Flat and unremarkable. "
    "No politics, no pride, no rights, no praise, no drama, no bravery. "
    "Nothing negative about the person -- if a sentence grumbles, it grumbles at a "
    "thing, a bill or a delay or the rain. Write plainly the way people actually "
    "speak: say \"{plain}\", never \"individual\", \"identifies as\" or \"openly\". "
    "{steer}\n\n"
    "One per line. Nothing else."
)

# Steers, one per batch. The first-person forms take a SUBJECT rather than a
# place: pinning a place on "I'm a lesbian and ..." makes the model emit sixteen
# complaints about the same laundromat, all with the same opening. The
# third-party forms take a place, where it buys variety instead of costing it.
SUBJECTS = [
    "money", "sleep", "the weather", "their job", "food", "a pet", "the commute",
    "a hobby", "being tired", "the rent", "an old car", "the news", "a game",
    "cooking", "the gym", "a broken phone", "the shopping", "a hangover",
    "a house plant", "the football", "a bad haircut", "the heating bill",
    "a long shift", "the bus", "a sore knee", "a new job", "the washing up",
]
PLACE_FORMS = {"incidental", "midsentence", "factual", "third_person_known"}

# One place hint per batch, rotated. Deliberately not all British domestic -- an
# earlier attempt produced nothing but allotments and hospital car parks.
PLACES = [
    "a hardware store", "a swimming pool", "a bus stop", "a wedding", "a garage",
    "a book club", "a hospital ward", "a campsite", "a radio phone-in", "a bakery",
    "a yard sale", "a chess club", "a long train ride", "a dentist's waiting room",
    "a chip shop", "a farm", "a laundromat", "a school pickup", "a food truck",
    "a warehouse night shift", "a community pool", "a bowling alley", "a bike shop",
    "a diner at 6am", "a rented flat with bad heating", "a call centre",
    "a supermarket queue", "a car park", "a village hall", "a university library",
    "a building site", "a taxi rank", "an airport gate", "a corner shop",
    "a family barbecue", "a doctor's surgery", "a running club", "a knitting group",
    "a pub quiz", "a music rehearsal", "a museum", "a post office", "a ferry",
    "a small office", "a golf course", "a climbing gym", "a nursery school",
    "a fishing trip", "a bus depot", "a hotel lobby",
]


def ollama_batch(model: str, term: tuple, form: str, place: str,
                 n: int, temperature: float) -> list[str]:
    import requests

    noun, _pred, _plural, _g = term
    desc = noun if noun[0].isupper() else f"{_a(noun)} {noun}"
    if form == "self_desc":
        # No steer. On this form -- and only this one -- a subject to weave in
        # sends gemma4 into unbounded self-critique about whether its twelve
        # mentions of the subject are too alike. Measured: 14.7k reasoning
        # characters, done_reason "length", empty content, 81 seconds wasted.
        steer = ""
    elif form in PLACE_FORMS:
        steer = f"Set them around {place}."
    else:
        steer = f"Have them mention {place} in passing."
    prompt = (FORM_PROMPT[form].format(n=n, desc=f'"{desc}"')
              + "\n" + RULES.format(plain=noun, steer=steer))
    r = requests.post(OLLAMA, json={
        "model": model, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95,
                    "num_predict": NUM_PREDICT},
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=1800)
    r.raise_for_status()
    return _split_lines(r.json()["message"]["content"])


def _split_lines(blob: str) -> list[str]:
    out = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        line = (line.replace("’", "'").replace("‘", "'")
                    .replace("“", '"').replace("”", '"')
                    .replace("—", " - ").replace("–", "-"))
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line.endswith(":"):
            continue
        if re.match(r"^(here are|here's|sure|certainly|note|output|sentences|each )",
                    line, re.I):
            continue
        out.append(line)
    return out


PROGRESS = OUT_DIR / "generated_raw_pool.progress"


def generate(budget_s: float = float("inf")) -> None:
    """Fill the raw pool. Two hard mechanical gates before anything is written.

    The model is not trusted to comply with the prompt. An earlier run with a
    different model put 15% of rows into the pool carrying no identity term at
    all -- just sentences, teaching nothing and diluting the set -- with a length
    distribution entirely outside the range where the bias lives. Both are
    checked here rather than downstream, so a non-compliant line never reaches
    the pool in the first place.
    """
    rng = random.Random(RANDOM_SEED)
    per_batch = 12

    # Batches are allocated by group x form share, then dealt round-robin across
    # that group's terms. A per-term floor would pin the job count to
    # 6 forms x 46 terms regardless of N_OLLAMA, which is four hours of GPU for
    # a set that needs about one.
    n_batches = round(N_OLLAMA / per_batch)
    jobs = []
    for group, gshare in GROUP_SHARE.items():
        terms = TERMS[group]
        for form, fshare in FORM_SHARE.items():
            k = round(n_batches * gshare * fshare)
            order = list(terms)
            rng.shuffle(order)
            for j in range(k):
                jobs.append((group, order[j % len(order)], form))
    rng.shuffle(jobs)
    # Resumable: the job order is deterministic under RANDOM_SEED, so a run that
    # stops on its wall-clock budget continues from the next index.
    start = int(PROGRESS.read_text()) if PROGRESS.exists() else 0
    print(f"{len(jobs)} batches of {per_batch}, resuming at {start}", flush=True)

    seen = set()
    if RAW_POOL.exists():
        seen = {json.loads(x)["text"] for x in RAW_POOL.read_text().splitlines()}
    RAW_POOL.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(seen)} already in the pool", flush=True)

    n_seen = n_noterm = n_len = n_dup = n_empty = 0
    t0 = time.time()
    with RAW_POOL.open("a") as fh:
        for i, (group, term, form) in enumerate(jobs[start:], start + 1):
            if time.time() - t0 > budget_s:
                print(f"  budget spent, stopping after batch {i - 1}", flush=True)
                break
            steer = rng.choice(PLACES if form in PLACE_FORMS else SUBJECTS)
            try:
                lines = ollama_batch(MODEL, term, form, steer,
                                     per_batch, rng.choice([0.95, 1.0, 1.05, 1.1]))
                # An empty batch means the model spent its whole budget thinking
                # and emitted nothing. Retry once, smaller and unsteered.
                if not lines:
                    n_empty += 1
                    lines = ollama_batch(MODEL, term, form, "", 8, 0.95)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [{i}] FAILED {group}/{form}: {exc}", flush=True)
                continue
            for text in lines:
                n_seen += 1
                # Gate 1: the row must actually carry its own assigned term.
                if not re.search(TERM_RE[term[0]], text, re.I):
                    n_noterm += 1
                    continue
                # Gate 2: the row must sit in the band where the bias lives.
                if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
                    n_len += 1
                    continue
                if text in seen:
                    n_dup += 1
                    continue
                seen.add(text)
                fh.write(json.dumps(dict(text=text, group=group, term=term[0],
                                         form=form, source="ollama")) + "\n")
            fh.flush()
            PROGRESS.write_text(str(i))
            if i % 5 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] pool={len(seen)}  dropped: "
                      f"no-term {n_noterm}, length {n_len}, dup {n_dup} "
                      f"of {n_seen} lines; empty batches {n_empty}", flush=True)


# ---------------------------------------------------------------------------
# Automatic rejection. A pre-filter for the hand review, not a substitute for it
# -- the whole reason this file exists is that the Civil Comments filter could
# not be trusted unread.
# ---------------------------------------------------------------------------

# Advocacy, affirmation and struggle framing. Not toxic, but a narrow register,
# and a set made only of these teaches "this identity appears in earnest
# political prose", which is a different wrong lesson.
REJECT_REGISTER = re.compile(
    r"\b(proud(ly)?|pride|deserve|right(s)?|equality|discriminat\w+|prejudic\w+|"
    r"brave|bravery|courage\w*|inspir\w+|struggl\w+|oppress\w+|marginali[sz]\w+|"
    r"stigma\w*|acceptance|tolerance|visibility|representation|"
    r"coming out|came out|living (my|her|his|their) truth|identifies as|"
    r"happens to be|openly|authentic self|true self|empower\w+|"
    r"celebrat\w+|advocat\w+|activis[tm]|ally|allyship|"
    r"despite (her|his|their|the)|even though (she|he|they)|overcome|barriers)\b",
    re.I)

# Hostile or sexual content, slurs, and digs at the person. Nothing here should
# survive a benign prompt; the whole point is not to trust that. The last
# alternation exists because an earlier model, told to be "mildly negative",
# made every single subject incompetent.
REJECT_HOSTILE = re.compile(
    r"\b(fag+(ot)?s?|dyke|tranny|trannie|nigg\w+|spic|chink|kike|"
    r"raghead|towelhead|paki|wetback|retard\w*|cripple|spaz|coon|"
    r"abomination|degenerat\w+|pervert\w*|predator|groom\w+|mentally ill|"
    r"disgust\w+|freak|sinful|deviant|agenda|propaganda|indoctrinat\w+|"
    r"real (man|woman|men|women)|so-called|allegedly|claims to be|"
    r"stupid|idiot|lazy|useless|incompetent|nasty|creepy|weird|rude|"
    r"annoying|obnoxious|clueless)\b", re.I)

REJECT_SEXUAL = re.compile(
    r"\b(sex|sexual|sleeping with|hook ?up|dating app|porn\w*|naked|kink\w*|"
    r"bedroom|lovers?)\b", re.I)

# Meta-text the model leaks now and then. "individual" is here because it is the
# stilted register the prompt bans and the reason qwen was dropped.
REJECT_META = re.compile(
    r"\b(sentence|word count|the term|the identity|as requested|rule \d|prompt|"
    r"here are|instruction|individual)\b", re.I)

FIRST_PERSON = re.compile(
    r"^(i|i'm|im|i've|i am|as a|as an|speaking as|yeah,? i|hi,? i|me,? i|"
    r"just so|not that|for what|for the record|guess i|look,? i|so,? i|"
    r"honestly,? i|my )", re.I)

TERM_RE = {
    "gay man": r"\bgay\b", "gay woman": r"\bgay\b", "lesbian": r"\blesbian(s)?\b",
    "bisexual woman": r"\bbisexual(s)?\b|\bbi\b", "bisexual man": r"\bbisexual(s)?\b|\bbi\b",
    "queer person": r"\bqueer\b", "homosexual man": r"\bhomosexual(s|ity)?\b",
    "straight man": r"\bstraight\b", "heterosexual woman": r"\bheterosexual(s|ity)?\b",
    "transgender woman": r"\btrans(gender)?\b", "transgender man": r"\btrans(gender)?\b",
    "trans man": r"\btrans(gender)?\b", "trans woman": r"\btrans(gender)?\b",
    "nonbinary person": r"\bnon[- ]?binary\b", "cisgender man": r"\bcis(gender)?\b",
    "black man": r"\bblack\b", "black woman": r"\bblack\b", "white man": r"\bwhite\b",
    "Asian woman": r"\basian(s)?\b", "African American man": r"\bafrican[- ]?american(s)?\b",
    "Native American woman": r"\bnative[- ]?american(s)?\b",
    "Latino man": r"\blatin(o|a|x|os|as)\b", "Hispanic woman": r"\bhispanic(s)?\b",
    "deaf man": r"\bdeaf\b", "blind woman": r"\bblind\b", "autistic person": r"\bautis(tic|m)\b",
    "disabled man": r"\bdisab(led|ility|ilities)\b", "wheelchair user": r"\bwheelchair\b",
    "dyslexic person": r"\bdyslexi(a|c)\b",
    "Muslim": r"\bmuslim(s)?\b", "Jewish man": r"\bjew(s|ish)?\b",
    "Christian": r"\bchristian(s|ity)?\b", "Hindu": r"\bhindu(s|ism)?\b",
    "Sikh": r"\bsikh(s)?\b", "Buddhist": r"\bbuddhist(s|m)?\b",
    "atheist": r"\batheist(s|ic)?\b", "Catholic": r"\bcatholic(s|ism)?\b",
    "Mormon": r"\bmormon(s|ism)?\b",
    "Mexican": r"\bmexican(s)?\b", "Indian": r"\bindian(s)?\b",
    "Chinese woman": r"\bchinese\b", "Arab man": r"\barab(s|ic)?\b",
    "Pakistani woman": r"\bpakistani(s)?\b", "Irish man": r"\birish\b",
    "Polish woman": r"\bpolish\b|\bpoles\b", "Nigerian man": r"\bnigerian(s)?\b",
}


def auto_reject(rec) -> str | None:
    """Return a reason if the row should not reach the hand review."""
    t = rec["text"]
    n = len(t.split())
    # MIN_WORDS is 2, not 4. "I'm gay." is two words and is the single most
    # on-target row shape in the file.
    if n < MIN_WORDS:
        return "too short"
    if n > MAX_WORDS:
        return "too long"
    if not re.search(TERM_RE[rec["term"]], t, re.I):
        return "term missing"
    if not re.search(r"[.!?]$", t):
        return "unterminated"
    if REJECT_HOSTILE.search(t):
        return "hostile, slur, or a dig at the person"
    if REJECT_SEXUAL.search(t):
        return "sexual"
    if REJECT_REGISTER.search(t):
        return "advocacy register"
    if REJECT_META.search(t):
        return "meta or stilted"
    if t.count(",") > 2 or ";" in t:
        return "overwrought"
    if (rec["form"] in ("self_desc", "self_desc_clause")
            and not FIRST_PERSON.match(t)
            and not re.search(r"\bI('m| am| have been| was|'ve been)\b", t)):
        return "not first person"
    if rec["form"] == "third_person_known" and not re.search(
            r"\b(my|our|the (woman|man|guy|person|kid|bloke|lad|lady) who)\b", t, re.I):
        return "no known-person anchor"
    if rec["form"] == "midsentence" and re.search(
            TERM_RE[rec["term"]], " ".join(t.split()[-3:]), re.I):
        return "identity in final position"
    return None


# ---------------------------------------------------------------------------
# Hand review.
#
# Every row that survived `auto_reject` was printed by `review` and read. Keys
# are md5(text)[:8], the same convention as build_counterexamples.py, so the two
# review records read the same way.
# ---------------------------------------------------------------------------
DROPPED: dict[str, str] = {}


def key_of(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def probe_normalise(text: str) -> str:
    """normalise(), plus contraction expansion.

    "I'm a lesbian." and "I am a lesbian." are different strings and the second
    is a probe. Training on the first is training on a paraphrase of the thing
    the probe measures, so both collapse to one key here and both get dropped.
    """
    t = " " + text.lower() + " "
    for a, b in [("i'm", "i am"), ("we're", "we are"), ("he's", "he is"),
                 ("she's", "she is"), ("they're", "they are"), ("it's", "it is"),
                 ("that's", "that is"), ("isn't", "is not"), ("neighbor", "neighbour")]:
        t = t.replace(a, b)
    return normalise(t)


def ngrams(text: str, n: int = 5) -> set:
    w = normalise(text).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


# ---------------------------------------------------------------------------
# Overlap checks.
# ---------------------------------------------------------------------------

def probe_sentences() -> set[str]:
    """Every sentence experiments/identity_bias.py evaluates on.

    If a generated row duplicated one of these, the probe would be scoring
    memorisation and the evaluation would be worthless. Imported live rather
    than copied, so it cannot drift out of sync with the probe script.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from experiments import identity_bias as ib

    out = set()
    for _c, _d, s, p, _r in ib.TERMS:
        for _name, fn in ib.TEMPLATES:
            out.add(fn(s, p))
            out.add(ib.FILLER + fn(s, p))
            out.add(ib.FILLER + ib.FILLER + fn(s, p))
    for _lg, d in ib.ML_PROBES.items():
        for _t, sents in d.items():
            out.update(sents)
    return out


def check_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Four checks: within-set, vs the Civil Comments set, vs the splits, vs probes."""
    df = df.copy()
    df["norm"] = df.text.map(normalise)

    n0 = len(df)
    df = df.drop_duplicates("norm").reset_index(drop=True)
    print(f"  [1a] within-set exact duplicates dropped: {n0 - len(df)}")

    # Near-duplicates inside the generated set itself. Combinatorial frames make
    # this the likeliest place for the set to be quietly repetitive.
    idx: dict[str, list[int]] = {}
    drop = set()
    for i, t in enumerate(df.text):
        g = ngrams(t)
        counts: dict[int, int] = {}
        for x in g:
            for j in idx.get(x, ()):
                counts[j] = counts.get(j, 0) + 1
        if counts and max(counts.values()) / len(g) >= 0.8:
            drop.add(i)
            continue
        for x in g:
            idx.setdefault(x, []).append(i)
    print(f"  [1b] within-set near-duplicates (>=80% 5-gram cover) dropped: {len(drop)}")
    df = df[~df.index.isin(drop)].reset_index(drop=True)

    # Against the existing Civil Comments counter-example set.
    cc_norm, cc_grams = set(), set()
    for name in ["counterexamples_train_en", "counterexamples_heldout_en"]:
        p = OUT_DIR / f"{name}.csv"
        if not p.exists():
            continue
        for t in pd.read_csv(p).comment_text.astype(str):
            cc_norm.add(normalise(t))
            cc_grams |= ngrams(t)
    hit = df.norm.isin(cc_norm) | df.text.map(lambda t: bool(ngrams(t) & cc_grams))
    print(f"  [2]  overlapping the Civil Comments counter-example set: {int(hit.sum())}")
    df = df[~hit].reset_index(drop=True)

    # Against train / val / test.
    existing_norm: set[str] = set()
    index: dict[str, dict[int, int]] = {}
    row_id = 0
    for name in ["train", "val", "test"]:
        d = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        txt = d.comment_text.dropna().astype(str)
        existing_norm |= set(txt.map(normalise))
        for t in d[d.lang == "en"].comment_text.dropna().astype(str):
            for g in ngrams(t):
                index.setdefault(g, {})[row_id] = 1
            row_id += 1
        print(f"       {name}.csv: {len(d):,} rows")
    print(f"       indexed {row_id:,} English rows, {len(index):,} distinct 5-grams")

    def best_overlap(text: str) -> float:
        grams = ngrams(text)
        counts: dict[int, int] = {}
        for g in grams:
            for rid in index.get(g, ()):
                counts[rid] = counts.get(rid, 0) + 1
        return (max(counts.values()) / len(grams)) if counts else 0.0

    df["overlap"] = df.text.map(best_overlap)
    n_exact = int(df.norm.isin(existing_norm).sum())
    n_near = int((df.overlap >= NEARDUP_SHARE).sum())
    print(f"  [3]  exact matches in train/val/test: {n_exact}")
    print(f"       near-duplicates (>={NEARDUP_SHARE:.0%} of own 5-grams in one row): "
          f"{n_near}, max overlap {df.overlap.max():.2f}")
    df = df[~(df.norm.isin(existing_norm) | (df.overlap >= NEARDUP_SHARE))]
    df = df.reset_index(drop=True)

    # Against the probe set. Must end at zero or the evaluation is void.
    probes = probe_sentences()
    pnorm = {probe_normalise(s) for s in probes}
    pgrams: set[str] = set()
    for s in probes:
        pgrams |= ngrams(s)
    exact = df.text.map(probe_normalise).isin(pnorm)
    near = df.text.map(lambda t: len(ngrams(t) & pgrams) / max(1, len(ngrams(t))) >= 0.6)
    print(f"  [4]  probe sentences found and removed: {int(exact.sum())} exact "
          f"(contraction-insensitive), {int(near.sum())} near (>=60% 5-gram cover)")
    df = df[~(exact | near)].reset_index(drop=True)
    left = int(df.text.map(probe_normalise).isin(pnorm).sum())
    print(f"       probe sentences remaining in the shipped set: {left}")
    return df


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------

def _load_pool() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    recs = hand_written(rng, {g: int(round(N_HAND * s)) for g, s in GROUP_SHARE.items()})
    if RAW_POOL.exists():
        recs += [json.loads(x) for x in RAW_POOL.read_text().splitlines()]
    df = pd.DataFrame(recs)
    if "frame" not in df.columns:
        df["frame"] = None
    df["text"] = df.text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def _lenstats(n) -> str:
    import numpy as np
    n = np.asarray(n)
    return (f"min {n.min()}  p25 {np.percentile(n, 25):.0f}  median {np.median(n):.0f}  "
            f"p75 {np.percentile(n, 75):.0f}  max {n.max()}   "
            f"under 8: {(n < 8).mean():.1%}   under 15: {(n < 15).mean():.1%}")


def build(write: bool = True) -> pd.DataFrame:
    df = _load_pool()
    print(f"raw pool: {len(df)} ({(df.source == 'hand').sum()} hand, "
          f"{(df.source == 'ollama').sum()} ollama)\n")

    df["reason"] = df.apply(auto_reject, axis=1)
    print("auto-rejected:")
    for reason, n in df[df.reason.notna()].reason.value_counts().items():
        print(f"  {n:5d}  {reason}")
    df = df[df.reason.isna()].drop(columns="reason").reset_index(drop=True)
    print(f"  -> {len(df)} survive the automatic filters\n")

    print("overlap and dedup checks:")
    df = check_overlap(df)
    print(f"  -> {len(df)} survive\n")

    n_before = len(df)
    df = _cap_shapes(df)
    print(f"shape cap: {n_before - len(df)} dropped, {len(df)} kept\n")

    df["key"] = df.text.map(key_of)
    n_before = len(df)
    df = df[~df.key.isin(DROPPED)].reset_index(drop=True)
    print(f"hand review: {n_before - len(df)} dropped by hand, {len(df)} kept\n")

    df["nwords"] = df.text.str.split().str.len()
    df = _split_heldout(df)

    print("\nby group:")
    print(df.groupby(["group", "slice"]).size().unstack(fill_value=0))
    print("\nby carrier form:")
    print(df.groupby(["form", "slice"]).size().unstack(fill_value=0))
    print("\nby source:")
    print(df.groupby(["source", "slice"]).size().unstack(fill_value=0))
    print("\nlength: " + _lenstats(df.nwords))
    print(pd.cut(df.nwords, [1, 4, 8, 13, 21, 35], right=False,
                 labels=["2-3", "4-7", "8-12", "13-20", "21-22"])
          .value_counts().sort_index())

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _to_schema(df[df["slice"] == "train"]).to_csv(
            OUT_DIR / "counterexamples_generated_en.csv", index=False)
        _to_schema(df[df["slice"] == "heldout"]).to_csv(
            OUT_DIR / "counterexamples_generated_heldout_en.csv", index=False)
        meta = df[["text", "group", "term", "form", "source", "slice",
                   "nwords", "key"]].copy()
        meta.insert(0, "id", _ids(df.text))
        meta.rename(columns={"text": "comment_text"}).to_csv(
            OUT_DIR / "counterexamples_generated_metadata.csv", index=False)
        print(f"\nwrote {(df['slice'] == 'train').sum()} train / "
              f"{(df['slice'] == 'heldout').sum()} heldout rows to {OUT_DIR}")
    return df


# No more than this many rows may share one term and one three-word opening.
# Two rows differing only in the identity term are fine and wanted -- that is how
# the model learns the term is not the signal. What is not fine is fifty rows
# that are all "I'm a lesbian." plus a different complaint, which is a template
# wearing a disguise. The local model produces exactly that if left alone: a
# batch pinned to one place came back as sixteen consecutive "I'm a lesbian.
# <something about the laundromat>."
SHAPE_CAP = 9


def _cap_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Cap rows per (term, first three normalised words)."""
    sig = df.term + "|" + df.text.map(lambda t: " ".join(normalise(t).split()[:3]))
    keep = sig.groupby(sig).cumcount() < SHAPE_CAP
    over = sig[~keep].value_counts()
    for s_, n in over.head(8).items():
        print(f"  over cap by {n}: {s_}")
    return df[keep].reset_index(drop=True)


def _split_heldout(df: pd.DataFrame) -> pd.DataFrame:
    """Held out by sentence SHAPE, not by random row.

    A random row split does not work on a combinatorial set: every held-out fill
    of "I'm {p} and {m}." shares five-word sequences with a training fill of the
    same frame, so the leak check drags almost all of them back and the held-out
    slice collapses. Reserving whole frames instead means held-out rows are
    carriers the training set never contained, which is what the slice is for --
    generalisation, not recall of a memorised shape.

    ollama rows have no frame, so they are held out at random and leak-checked
    in the usual way.
    """
    df = df.copy()
    df["slice"] = "train"
    rng = random.Random(RANDOM_SEED)

    by_form: dict[str, list[str]] = {}
    for f in sorted(df.frame.dropna().unique()):
        by_form.setdefault(df.loc[df.frame == f, "form"].iloc[0], []).append(f)
    held_frames: set[str] = set()
    for form, fl in by_form.items():
        k = max(1, round(len(fl) * HELDOUT_SHARE))
        held_frames |= set(rng.sample(sorted(fl), k))
        print(f"  held-out frames, {form}: {k} of {len(fl)}")
    df.loc[df.frame.isin(held_frames), "slice"] = "heldout"

    gen = df[df.frame.isna()]
    for _k, idx in gen.groupby(["group", "form"], observed=True).groups.items():
        if len(idx) < 4:
            continue
        k = max(1, round(len(idx) * HELDOUT_SHARE))
        df.loc[df.loc[idx].sample(n=k, random_state=RANDOM_SEED).index, "slice"] = "heldout"

    tr: set[str] = set()
    for t in df[df["slice"] == "train"].text:
        tr |= ngrams(t)
    leak = df[(df["slice"] == "heldout") & df.text.map(lambda t: bool(ngrams(t) & tr))].index
    df.loc[leak, "slice"] = "train"
    print(f"  moved {len(leak)} rows back to train (5-gram overlap with train)")
    return df


def _ids(texts) -> list[str]:
    # {index}_{lang}_{labelpattern}_{hash}, matching utils/add_ids.py. The index
    # field is "gx" -- generated counter-examples -- so these rows are greppable
    # and removable independently of the "cx" Civil Comments ones:
    #   df[df.id.str.startswith("gx_")]
    return [f"gx_en_000000_{hashlib.md5(t.encode()).hexdigest()[:6]}" for t in texts]


def _to_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"id": _ids(df.text), "comment_text": df.text.values})
    for c in CLASSES:
        out[c] = 0
    out["toxic"] = 0.0        # train.csv carries toxic as float64
    out["lang"] = "en"
    return out[TARGET_COLS]


# ---------------------------------------------------------------------------
# Review dump and scoring.
# ---------------------------------------------------------------------------

def review() -> None:
    df = build(write=False)
    for _, r in df.sort_values(["group", "form", "nwords"]).iterrows():
        print(f"{r.key}\t{r.group[:12]:12s}\t{r.form[:18]:18s}\t{r.source[:6]:6s}\t{r.text}")


def score() -> None:
    """Score the shipped rows with the current model.

    This is the falsifiable part. If these sentences all score near zero already
    they are not exercising the failure and are close to useless. A large share
    above the tuned thresholds is the evidence that they target the bias the
    retrain is meant to remove.
    """
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT))
    from model.inference_optimized import OptimizedToxicityClassifier

    frames = []
    for name, sl in [("counterexamples_generated_en", "train"),
                     ("counterexamples_generated_heldout_en", "heldout")]:
        d = pd.read_csv(OUT_DIR / f"{name}.csv")
        d["slice"] = sl
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    meta = pd.read_csv(OUT_DIR / "counterexamples_generated_metadata.csv")
    df = df.merge(meta[["id", "group", "form", "source", "nwords"]], on="id", how="left")

    clf = OptimizedToxicityClassifier(pytorch_path=str(CKPT), device="cuda")
    texts = df.comment_text.astype(str).tolist()
    probs = []
    for i in range(0, len(texts), 32):
        chunk = texts[i:i + 32]
        res = clf.predict(chunk, langs=["en"] * len(chunk))
        probs += [[r["probabilities"][c] for c in CLASSES] for r in res]
        print(f"  {min(i + 32, len(texts))}/{len(texts)}", flush=True)
    p = np.array(probs)
    for j, c in enumerate(CLASSES):
        df[f"p_{c}"] = p[:, j]
    th = np.array([TH[c] for c in CLASSES])
    df["fires"] = (p >= th).any(axis=1)

    print(f"\nrows scored: {len(df)}")
    print(f"P(toxic): mean {df.p_toxic.mean():.3f}  median {df.p_toxic.median():.3f}  "
          f"max {df.p_toxic.max():.3f}")
    print(f"fires on any class at the shipped thresholds: {int(df.fires.sum())} / "
          f"{len(df)}  ({df.fires.mean():.1%})")
    bins = [0, .1, .25, .4724, .7, .9, 1.01]
    labels = ["<0.10", "0.10-0.25", "0.25-0.47", "0.47-0.70 FIRES",
              "0.70-0.90 FIRES", ">0.90 FIRES"]
    print("\nP(toxic) distribution:")
    print(pd.cut(df.p_toxic, bins, labels=labels, right=False).value_counts().sort_index())
    for col in ["group", "form", "source", "slice"]:
        print(f"\nby {col}:")
        print(df.groupby(col).agg(n=("id", "size"), mean_p=("p_toxic", "mean"),
                                  fires=("fires", "mean"))
              .sort_values("fires", ascending=False))
    print("\nby length bucket:")
    b = pd.cut(df.nwords, [1, 4, 8, 13, 21, 35], right=False,
               labels=["2-3", "4-7", "8-12", "13-20", "21-22"])
    print(df.groupby(b, observed=True).agg(n=("id", "size"), mean_p=("p_toxic", "mean"),
                                           fires=("fires", "mean")))
    print("\nhighest-scoring rows:")
    for _, r in df.nlargest(25, "p_toxic").iterrows():
        print(f"  {r.p_toxic:.3f}  {r.comment_text}")
    df.to_csv(OUT_DIR / "counterexamples_generated_scores.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'counterexamples_generated_scores.csv'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "generate":
        generate(float(sys.argv[2]) if len(sys.argv) > 2 else float("inf"))
    else:
        {"build": build, "review": review, "score": score}[cmd]()
