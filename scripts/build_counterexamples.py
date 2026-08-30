#!/usr/bin/env python
"""Build an English identity-term counter-example set from google/civil_comments.

Why this exists
---------------
`experiments/identity_bias.md` shows the shipped classifier scores "I am a gay
man." at 0.891 against a 0.472 threshold. The cause is representation: in the
English training split 701 rows contain "gay" and 91.2% are labelled toxic, and
of 36 rows matching first-person LGBT self-description not one is a genuine
benign self-description. The model never saw the benign form.

The failure is conditional on LENGTH. Prepending one paragraph of neutral filler
drops "I am a gay man." from 0.891 to 0.042, matching the corpus conditional
(short rows containing "gay" are 95.6% toxic vs 76.0% for long ones). So the
counter-examples have to be short. Everything below is selected inside a 3-35
word window with the mass pushed under 16 words.

What this script does
---------------------
Mines Civil Comments for short, genuinely benign comments that mention an
identity term in its *person-denoting* sense, and emits them in the exact schema
of dataset/split/train.csv with all-zero labels.

Every shipped row was read by hand. The rules alone are nowhere near enough:
`toxicity < 0.2` in Civil Comments still admits "Is Trudeau GAY ?", "LGBTQ is a
bell curve fad", "we know he was a muslim", and "pot calling the kettle black".
Measured precision of the full rule stack, before review, was 25-45% depending on
the group. So the rules only PROPOSE; the verdicts live in two lists below:
CURATED (accepted ids, for the four groups where accepts are the minority) and
CURATED_REJECT (rejected ids, for the two where accepts are the majority).
CURATED_REJECT is applied after the quotas are filled, never before, so removing
a row can never pull in an unread replacement.

Allocation follows measured harm, not corpus supply. Probe FPR by group from
identity_bias.md is sexual orientation 0.490, gender identity 0.179, race 0.122,
disability 0.048, religion 0.032, ethnicity/nationality 0.000, and the row counts
are ordered to match. Religion and ethnicity could each have supplied a thousand
rows and are deliberately held to a few dozen; sexual orientation is at its
corpus ceiling. See the report accompanying this script for the measured ceiling
per group and why the set is ~570 rows rather than the 1,000-1,500 the diagnosis
costed for hand-written sentences.

Outputs (dataset/counterexamples/):
  counterexamples_train_en.csv    <- merge this into dataset/split/train.csv
  counterexamples_heldout_en.csv  <- do NOT merge; generalisation check
  counterexamples_metadata.csv    <- sidecar: group, term, form, scores, slice

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. ./.venv-uv/bin/python scripts/build_counterexamples.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "dataset" / "counterexamples"
SPLIT_DIR = REPO_ROOT / "dataset" / "split"

CC_REPO = "google/civil_comments"
CC_FILES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
CC_SCORES = ["toxicity", "severe_toxicity", "obscene", "threat", "insult",
             "identity_attack", "sexual_explicit"]

# Civil Comments -> repo schema. sexual_explicit has no target column; it is used
# as an exclusion filter only (see SCORE GATE) and then dropped, because a row
# with sexual content labelled all-zero would teach the model something false.
LABEL_MAP = {"toxicity": "toxic", "severe_toxicity": "severe_toxic",
             "obscene": "obscene", "threat": "threat", "insult": "insult",
             "identity_attack": "identity_hate"}
TARGET_COLS = ["id", "comment_text", "toxic", "severe_toxic", "obscene",
               "threat", "insult", "identity_hate", "lang"]

# Binarisation. Civil Comments scores are the fraction of annotators who applied
# the tag, quantised at 1/N. Jigsaw's own convention for this corpus (the
# Unintended Bias competition) is >= 0.5 -> positive, and that is the rule used
# here. It never actually binds, because the selection gates below are far
# stricter than 0.5, so every emitted row maps to all-zero labels under any
# threshold in (0.17, 0.5].
BINARISE_THRESHOLD = 0.5

# Rule-selected rows: not one annotator flagged the comment on any of the seven
# axes. Scores are quantised at 1/N, so this is the only gate strictly below the
# smallest non-zero value, and it leaves 0.5 of headroom under BINARISE_THRESHOLD.
GATE_STRICT = 0.0
# Hand-read rows: at most one annotator in ~six flagged toxicity, and nothing on
# the other six axes. Loosened only for rows a human read, and only because the
# scarce groups need it: "I am so glad that I am gay!" carries toxicity 0.167 in
# Civil Comments, and a lone annotator flagging benign self-description is an
# instance of the very bias this set exists to correct, not evidence against the
# row. Still 0.33 clear of BINARISE_THRESHOLD, so the emitted label is still 0.
GATE_CURATED_TOXICITY = 1 / 6

WORD_MIN, WORD_MAX = 3, 35

# ---------------------------------------------------------------------------
# Identity vocabulary, following experiments/identity_bias.py TERMS.
# ---------------------------------------------------------------------------
GROUPS = {
    # Widened well past the probe vocabulary. The probe uses gay/lesbian/queer/
    # bisexual/homosexual, but restricting the mine to those terms leaves most of
    # the corpus's benign LGBT usage on the table, and this is the group the model
    # harms most (FPR 0.490), so supply here matters more than anywhere else.
    # Terms deliberately NOT included, measured and discarded: "closet" (193
    # all-zero short rows, of which essentially none are LGBT-sense -- they are
    # "closet leftist", "skeletons in the closet", "hid in a closet"), "pride
    # parade" (mostly sneering at the event), and "coming out" (1 usable row).
    "sexual_orientation": {
        "gay": r"\bgay(s)?\b", "lesbian": r"\blesbian(s)?\b", "queer": r"\bqueer\b",
        "homosexual": r"\bhomosexual(s|ity)?\b",
        "bisexual": r"\bbisexual(s)?\b|\bbi[- ]sexual\b",
        "lgbt": r"\bLGBT\w*\b",
        "same_sex": r"\bsame[- ]sex\b",
        "drag_queen": r"\bdrag (queen|king)(s)?\b",
        "homophobia": r"\bhomophobi\w*\b",
        "heterosexual": r"\bheterosexual\w*\b",
        "orientation": r"\bsexual orientation\b",
        "straight": r"\bstraight (man|men|woman|women|guy|guys|people|person|couple)\b",
    },
    "gender_identity": {
        "transgender": r"\btransgender\w*\b",
        "trans_person": (r"\btrans[- ](man|men|woman|women|person|people|kid|kids|"
                         r"youth|folk|guy|girl)\b"),
        "nonbinary": r"\bnon[- ]?binary\b",
        "transsexual": r"\btranssexual\w*\b",
        "gender_identity": r"\bgender identity\b",
        "cisgender": r"\bcis(gender)?\b",
        "two_spirit": r"\btwo[- ]spirit\b",
        "gender_fluid": r"\bgender[- ]fluid\b",
        "trans_rights": r"\btrans rights\b",
    },
    "disability": {
        "deaf": r"\bdeaf\b", "blind": r"\bblind\b", "autistic": r"\bautis(tic|m)\b",
        "disabled": r"\bdisab(led|ility|ilities)\b", "wheelchair": r"\bwheelchair\b",
        "dyslexic": r"\bdyslexi(a|c)\b",
    },
    "race": {
        "black": r"\bblack\b", "white": r"\bwhite\b", "asian": r"\basian(s)?\b",
        "latino": r"\blatino(s|a|as)?\b", "hispanic": r"\bhispanic(s)?\b",
        "african_american": r"\bafrican[- ]?american(s)?\b",
        "native_american": r"\bnative[- ]?american(s)?\b|\bindigenous\b|\bfirst nations\b",
    },
    "ethnicity_nationality": {
        "immigrant": r"\bimmigrant(s)?\b", "refugee": r"\brefugee(s)?\b",
        "mexican": r"\bmexican(s)?\b", "arab": r"\barab(s|ic)?\b",
        "african": r"\bafrican(s)?\b", "pakistani": r"\bpakistani(s)?\b",
        "indian": r"\bindian(s)?\b", "chinese": r"\bchinese\b",
        "japanese": r"\bjapanese\b",
    },
    "religion": {
        "muslim": r"\bmuslim(s)?\b", "jewish": r"\bjew(s|ish)?\b",
        "christian": r"\bchristian(s|ity)?\b", "catholic": r"\bcatholic(s|ism)?\b",
        "hindu": r"\bhindu(s|ism)?\b", "sikh": r"\bsikh(s)?\b",
        "buddhist": r"\bbuddhist(s|m)?\b", "atheist": r"\batheist(s|ic)?\b",
        "mormon": r"\bmormon(s|ism)?\b",
    },
}

# Group priority when a row mentions several: scarcer / worse-affected wins.
# FPR by group from identity_bias.md: sexual orientation 0.490, gender identity
# 0.179, race 0.122, disability 0.048, religion 0.032, ethnicity 0.000.
GROUP_ORDER = ["sexual_orientation", "gender_identity", "disability", "race",
               "ethnicity_nationality", "religion"]

# Terms with a common non-identity sense; these must be shown to denote a person.
AMBIGUOUS = {"black", "white", "blind", "deaf", "gay", "asian", "indian",
             "chinese", "japanese", "african", "arab", "native_american",
             "disabled", "queer"}

PERSON_NOUNS = (
    r"(?:man|men|woman|women|person|persons|people|peoples|guy|guys|kid|kids|child|children|"
    r"boy|boys|girl|girls|student|students|teacher|teachers|doctor|doctors|nurse|nurses|"
    r"worker|workers|employee|employees|soldier|soldiers|troop|troops|athlete|athletes|"
    r"player|players|voter|voters|parent|parents|mother|mothers|father|fathers|son|sons|"
    r"daughter|daughters|brother|brothers|sister|sisters|family|families|friend|friends|"
    r"neighbour|neighbours|neighbor|neighbors|couple|couples|community|communities|"
    r"population|populations|folk|folks|american|americans|canadian|canadians|citizen|citizens|"
    r"immigrant|immigrants|resident|residents|patient|patients|colleague|colleagues|"
    r"priest|priests|clergy|nun|nuns|bishop|bishops|leader|leaders|group|groups|youth|youths|"
    r"adult|adults|senior|seniors|veteran|veterans|candidate|candidates|mayor|governor|"
    r"actor|actress|writer|writers|artist|artists|singer|author|authors|scientist|scientists|"
    r"lawyer|judge|officer|officers|coach|classmate|classmates|roommate|applicant|applicants|"
    r"customer|customers|member|members|partner|spouse|wife|husband|teen|teens|teenager|"
    r"teenagers|baby|babies|toddler|elder|elders|cousin|uncle|aunt|niece|nephew|grandmother|"
    r"grandfather|grandparent|grandparents|boss|manager|owner|driver|farmer|engineer|"
    r"refugee|refugees|tribe|tribes|nation|nations)"
)


def person_sense(term_regex: str) -> re.Pattern:
    """Match only where the identity term denotes a person."""
    return re.compile(
        rf"(?:{term_regex})[- ]{PERSON_NOUNS}\b"
        rf"|\b(?:is|am|are|'m|'re|'s|was|were|be|been|being|become|becomes|became|"
        rf"identifies as|identify as|born)\s+"
        rf"(?:a\s+|an\s+|the\s+|also\s+|not\s+|now\s+|still\s+|openly\s+|proudly\s+|"
        rf"legally\s+|totally\s+|partially\s+|half\s+|part\s+|very\s+)*(?:{term_regex})\b"
        rf"|\bas\s+an?\s+(?:openly\s+|proudly\s+|young\s+|old\s+)*(?:{term_regex})\b"
        rf"|\b(?:who|that)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+)?(?:{term_regex})\b"
        rf"|\bfor\s+the\s+(?:{term_regex})\b"
        rf"|\b(?:a|an|the|this|that|these|those|my|his|her|their|our|its|some|many|most|"
        rf"all|other|another|every|each|two|three|few|several|young|old|elderly|fellow|"
        rf"openly|proudly|first|only)\s+(?:openly\s+|proudly\s+|young\s+|old\s+|first\s+)*"
        rf"(?:{term_regex})\s+{PERSON_NOUNS}\b",
        re.I,
    )


PERSON_PAT = {t: person_sense(rx) for terms in GROUPS.values() for t, rx in terms.items()}

# ---------------------------------------------------------------------------
# Rejection patterns. Every one of these was added after reading candidates that
# passed the score gate; toxicity==0 is not by itself evidence of a benign row.
# ---------------------------------------------------------------------------
IDIOM = re.compile(
    r"\bblack\s?(market|hole|holes|box|friday|ice|gold|bear|bears|lab|belt|sheep|book|tie|"
    r"out|death|smith|berry|board|list|magic|op|ops|swan|jack|eye|eyed|top|bird|birds|"
    r"letter|coffee|tea|pepper|powder|water|widow|hawk|panther|panthers|history month|"
    r"mold|mould|spot|day|days|cat|cats|spruce|diamond|olives|dog|comics)\b"
    r"|\bblack(list|listed|listing|out|outs|board|smith|berry|jack|mail|mailed|ened|ening)\b"
    r"|\bin the black\b|\bblack and white\b|\bblack or white\b|\bkettle black\b"
    r"|\bpot calling\b|\bblack lives matter\b|\borange is the new black\b|\bnew black\b"
    r"|\bblack and blue\b|\bwear(s|ing)? black\b|\bdressed in black\b"
    r"|\bwhite\s?(house|paper|papers|water|noise|wine|wash|washing|out|board|collar|space|"
    r"flag|knuckle|lie|lies|elephant|smoke|christmas|sox|castle|hat|hats|powder|courtesy|"
    r"north|shark|blood|rabbit|walls|picket|snow|sale|goods|light)\b"
    r"|\bwhite(wash|washed|washing|board|out|head)\b|\bgreat white\b"
    r"|\bwhite (privilege|guilt|nationalis|nativis|supremac|sheets?)\b|\balt[- ]white\b"
    r"|\bwhite man'?s burden\b|\bit'?s? ok(ay)? to be white\b|\bmulatto\b|\bhalf[- ]breed\b"
    r"|\bblind\s?(spot|spots|eye|eyes|faith|trust|luck|date|dates|alley|side|sided|item|"
    r"obedience|allegiance|squirrel|rage|panic|ambition|acceptance|loyalty|adherence|"
    r"support|supporter|supporters|devotion|leading|hatred|fury|taste|test|tests|study)\b"
    r"|\bblind(ly|ed|ing|fold|folded|side|sided|sight)\b|\bcolou?r[- ]?blind\b"
    r"|\b(turn|turned|turns|turning)\s+a\s+blind\b|\blove is blind\b|\bjustice is (often )?blind\b"
    r"|\bwil+fully blind\b|\bblind to\b|\bnone (so|are so) blind\b|\bwhole world blind\b"
    r"|\bflying blind\b|\bstone blind\b|\bblind man\b|\bblind partisan\b"
    r"|\bdeaf ear(s)?\b|\bfall(s|en|ing)? on deaf\b|\btone[- ]?deaf\b|\bdeaf to\b"
    r"|\bindian\s?(summer|ocean|giver|curry|food|restaurant)\b|\bcowboys and indians\b"
    r"|\bchinese\s?(food|restaurant|restaurants|takeout|take[- ]out|checkers|whispers|"
    r"new year|wall|import|imports|export|exports|goods|steel|money|currency|yuan|"
    r"government|communist|army|navy|economy|market|markets|company|companies|firm|firms|"
    r"investor|investors|state|city|cities|made)\b"
    r"|\bjapanese\s?(car|cars|food|restaurant|import|imports|model|models|company|companies|"
    r"knotweed|maple|garden|whisky|yen|economy|market|steel|beetle|beetles)\b"
    r"|\basian\s?(elephant|elephants|carp|longhorn|market|markets|currenc|flu|cuisine)\b"
    r"|\btrans(port|fer|form|late|lation|action|it|mission|parent|ition|cript|cend|plant|"
    r"fat|fats|gress|ient|istor)"
    r"|\bstraight (up|forward|away|face|line|answer|talk|shooter|ahead|out|to the)\b"
    r"|\bgay\s?(colou?rs|lick|paree|nineties)\b|\bben gay\b|\benola gay\b"
    r"|\bafrican\s?(union|nation|nations|country|countries|continent|violet|swallow|elephant|"
    r"safari|savanna|savannah|wildlife|art|dictator|dictators)\b"
    r"|\barab\s?(spring|league|world|state|states|oil|peninsula|emirates|sea|numerals)\b"
    r"|\bdisabled the\b|\bdisable[ds]? (account|feature|button|comment|comments|option|"
    r"setting|settings|javascript|cookies|link|links|it|them|system|doors)\b"
    r"|\bnative (plant|plants|species|speaker|speakers|land|title|advertising|app)\b",
    re.I,
)

HOSTILE = re.compile(
    # sexualising / criminalising an identity
    r"\bp[ae]edophil|\bpedo\b|\bgroom(ing|er|ers)\b|\bmolest|\brapist|\brape\b|\bpederast"
    r"|\bsinful\b|\ba sin\b|\bis sin\b|\bsin of\b|\babomination|\bimmoral|\bunnatural"
    r"|\bdeviant|\bperver(t|ted|sion|ts)|\bdegenerate|\bdisgusting\b|\bfilthy\b|\bsodom"
    r"|\bbestiality|\bincest|\bslut|\bwhore\b|\bporn\b"
    # pathologising
    r"|\bmental(ly)? (ill|illness|disorder|defect)|\bdisorder\b|\bdelusional\b|\bdelusion\b"
    r"|\brecovering (homosexual|alcoholic)|\bhomosexual (behavio|tendenc|act|activity|lifestyle)"
    r"|\b(deep[- ]seated|intrinsically) (homosexual )?(tendenc|disorder)|\bdisordered\b"
    r"|\bdysphori|\bconversion therapy\b|\bsuicid|\bpsychiatri|\bdiagnos|\bafflict"
    r"|\bsyndrome\b|\bdisease\b|\bDSM\b"
    # dehumanising register
    r"|\btransgenders\b|\bthe transgender\b|\ba transgender\b|\bthe gays\b|\bthe blacks\b"
    r"|\bthe jews\b|\bthe muslims\b|\bthe whites\b|\bthe mexicans\b|\bthe asians\b"
    r"|\bblacks and\b|\bwhites and\b|\bvermin\b|\bparasit|\bscum\b|\bsavage(s)?\b"
    # culture-war framing
    r"|\b(gay|homosexual|trans|transgender|lgbt\w*) agenda\b|\bgay lifestyle\b"
    r"|\blifestyle choice\b|\bindoctrinat|\bpropaganda\b|\bbrainwash|\bnormies\b|\bmutilat"
    r"|\bspecial rights\b|\bmore equal\b|\bsore winner|\bgold[- ]?digger|\bsjw\b|\bsnowflake"
    r"|\bvirtue signal|\bpolitically correct\b|\bpc police\b|\blibtard|\bsocial justice warrior"
    r"|\bidentity politics\b|\bvictim(hood| card)\b|\bwar on\b|\bthe left\b|\bthe right\b"
    r"|\bregressive left\b|\balt[- ]right\b|\balt[- ]reich\b|\bfake news\b|\bcrowd\b"
    r"|\bactivists\b|\blobby\b|\bmilitant|\bcafeteria catholic|\bin name only\b|\bcinos?\b"
    r"|\bchinos?\b|\bpapist|\bholier than thou\b"
    # slurs
    r"|\bfag|\bdyke\b|\btrann|\bshemale|\bqueers\b|\bnigg|\bkike\b|\bspic\b|\bchink"
    r"|\bwetback|\btowel head|\braghead|\bcoon\b|\bgook\b|\bretard|\bspastic\b|\bcripple(s|d)?\b"
    r"|\bmoron|\bidiot|\bstupid\b|\bdumb\b|\bcolored folk"
    # violence / atrocity / crime vocabulary
    r"|\babuse[ds]?\b|\babusive\b|\bassault|\bmurder|\bkill(ed|ing|s)?\b|\bviolen|\bcrime"
    r"|\bcriminal|\bprison|\bjail\b|\bslaver?y?\b|\bslaves\b|\bholocaust\b|\blynch|\bshoot(ing|er)?\b"
    r"|\bmassacre|\bterror|\bvictims?\b|\bpersecut|\bgenocide\b|\bnazi\b|\bhitler\b|\bkkk\b"
    r"|\bklan\b|\bswastika|\bzundel|\bmossad\b|\bdeep state\b|\btaqiyya|\bjihad|\bsharia\b"
    r"|\bisis\b|\bal[- ]qaeda\b|\bextremist|\bradicaliz|\bmartyr|\bkalashnikov|\bsuspect\b"
    r"|\bgunman\b|\battacker\b"
    # immigration framing that would poison an all-zero label
    r"|\billegal|\bundocumented|\balien(s)?\b|\bmigrant(s)?\b|\banchor bab|\binvaders\b"
    r"|\binfest|\bshithole|\bthug(s)?\b|\bdeport|\bgo back to\b|\bbuild the wall\b"
    # thread abuse
    r"|\bnews flash\b|\bspare me\b|\bgive me a break\b|\boh brother\b|\bget a grip\b"
    r"|\bgrow up\b|\bwake up\b|\bnice try\b|\bwhat a joke\b|\bnonsense\b|\bbaloney\b"
    r"|\bhogwash\b|\bgarbage\b|\brubbish\b|\bpathetic\b|\bridiculous\b|\babsurd\b|\bhypocri"
    r"|\bbigot|\bracist\b|\bracism\b|\bsexist\b|\bhomophob|\btransphob|\bphobic\b|\bslur"
    r"|\bshame on\b|\bdisgrace\b|\bcowar|\bliar\b|\blies\b|\blying\b|\bdishonest"
    r"|\b(i|we|they) hate\b|\bhate(s)? (gay|jew|black|muslim|white|trans)"
    r"|\bthat'?s so gay\b|\bso gay\b|\byou'?re gay\b|\bur gay\b|\bconfused (kid|kids|child|"
    r"children|people|man|woman|boy|girl)|\breverse racis|\bplaying the (race|victim) card",
    re.I,
)

SNARK = re.compile(
    r"\b(angry|old|rich|privileged|entitled|wrinkled|fat) (old )?white (guy|guys|man|men|"
    r"male|males|woman|women|boy|lady|dude)\b"
    r"|\bkeep dreaming\b|\bjust saying\b|\bjust sayin\b|\bgo figure\b|\bwhat a surprise\b"
    r"|\bno surprise there\b|\bsurprise surprise\b|\byeah,? (but|right)\b|\boh please\b"
    r"|\bgive it up\b|\bcry me\b|\bso it'?s (just )?fine\b|\bit'?s all good\b"
    r"|\bhow'?s that working\b|\b/s\b|\bsarcasm\b|\blmao\b|\brofl\b|\beye ?roll\b|\bsmh\b"
    r"|\bmasquerading as\b|\bso[- ]called\b|\bmust be (white|black|gay|muslim)\b"
    r"|\bprobably (white|black|gay|muslim)\b|\bnuff said\b|\bdo tell\b|\bpssst?\b"
    r"|\bcome on\.|\bwe all know\b|\bimagine the outrage\b|\bwhat could go wrong\b"
    r"|\bwho would have guessed\b|\bno doubt\b|\bmethinks\b|\bgood grief\b|\bjesus wept\b"
    r"|\bgee+sh\b|\buh oh\b|\bwhine fest\b|\bwhat about\b|\bhow about\b|\bimagine if\b"
    r"|\bgiving odds\b|\bof course (he|she|they|it|the)\b|\breally aren'?t\b"
    r"|:\)|:\-\)|;\)|:o\)|:D|\?!|!!!",
    re.I,
)

POLITICAL = re.compile(
    r"\btrump\b|\bclinton\b|\bobama\b|\btrudeau\b|\bhillary\b|\bbiden\b|\bpence\b"
    r"|\bdemocrat|\brepublican|\bgop\b|\bndp\b|\bliberals\b|\bconservatives\b"
    r"|\belection|\bvoter|\bvotes\b|\bballot|\bsenate\b|\bcongress\b|\bpolitician"
    r"|\bharper\b|\bwynne\b|\bnotley\b|\bkenney\b|\bmulcair\b|\bleitch\b",
    re.I,
)

SECOND_PERSON = re.compile(r"\byou\b|\byour\b|\byou'?re\b|\byou'?ll\b|\byou'?ve\b|\bur\b|\byourself\b", re.I)
URLISH = re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b|\.html\b|@\w+\.", re.I)
FRAGMENTARY = re.compile(
    r"^(and|but|or|so|yet|because|which|that|then|also|plus|although|however|though|"
    r"as opposed to|including|yes|no|nope|yep|correct|right|true|false|exactly|indeed|"
    r"agreed|ditto|same|amen|lol|ha|haha|hmm|huh|wow|oh|ah|well|ok|okay)\b", re.I)
QUOTED = re.compile(r'^\s*["“‘\']|["“].{15,}["”]')

FIRST_PERSON = re.compile(
    r"\b(i am|i'm|im)\s+(a|an|also|not|so|proudly|openly|still|now)?\s*"
    r"(a|an|proudly|openly|gay|lesbian|queer|bisexual|homosexual|trans|transgender|"
    r"non[- ]?binary|black|white|asian|latino|latina|hispanic|muslim|jewish|jew|christian|"
    r"catholic|hindu|sikh|buddhist|mormon|atheist|deaf|blind|autistic|disabled|dyslexic|"
    r"immigrant|refugee|mexican|arab|indian|chinese|japanese|african|native)\b"
    r"|\bas an?\s+(proudly\s+|openly\s+|white\s+|black\s+)?"
    r"(gay|lesbian|queer|bisexual|homosexual|trans|transgender|non[- ]?binary|black|white|"
    r"asian|latino|latina|hispanic|muslim|jewish|jew|christian|catholic|hindu|sikh|buddhist|"
    r"mormon|atheist|deaf|blind|autistic|disabled|dyslexic|immigrant|refugee|mexican|arab|"
    r"indian|chinese|japanese|african|native)\b"
    r"|\bmy (son|daughter|brother|sister|wife|husband|partner|spouse|friend|friends|mother|mom|"
    r"father|dad|niece|nephew|neighbou?r|neighbou?rs|colleague|co[- ]?worker|cousin|boss|doctor|"
    r"teacher|kid|kids|child|children|uncle|aunt|grandson|granddaughter|grandmother|grandfather|"
    r"parents|family|roommate|student|students|ancestors)\s+"
    r"(is|are|was|were|came)\s+(an?\s+)?(openly\s+|proudly\s+)?"
    r"(gay|lesbian|queer|bisexual|homosexual|trans|transgender|non[- ]?binary|black|white|asian|"
    r"latino|latina|hispanic|muslim|jewish|jew|christian|catholic|hindu|sikh|buddhist|mormon|"
    r"atheist|deaf|blind|autistic|disabled|dyslexic|immigrant|refugee|mexican|arab|indian)\b"
    r"|\bbeing (gay|lesbian|queer|bisexual|trans|transgender|black|deaf|blind|disabled|autistic|"
    r"muslim|jewish|hindu|sikh|catholic|an immigrant|a refugee)\b", re.I)

THIRD_PERSON = re.compile(
    r"\b(he|she|they|his|her|their|the|a|an)\s+\w*\s*(is|are|was|were)\s+(an?\s+)?(openly\s+)?"
    r"(gay|lesbian|queer|bisexual|homosexual|trans|transgender|non[- ]?binary|black|white|asian|"
    r"latino|hispanic|muslim|jewish|christian|catholic|hindu|sikh|buddhist|atheist|mormon|deaf|"
    r"blind|autistic|disabled|immigrant|refugee|mexican|arab|indian)\b"
    r"|\b(openly )?(gay|lesbian|queer|bisexual|transgender|black|deaf|blind|disabled|muslim|"
    r"jewish|hindu|sikh|latino|hispanic|asian) (man|woman|men|women|person|people|couple|"
    r"student|students|teacher|doctor|neighbou?r|friend|kid|kids|family|community|parents)\b", re.I)

# ---------------------------------------------------------------------------
# Hand review. Keys are md5(text)[:8].
#
# For these four groups the rules alone give 25-40% precision, so every candidate
# in the pool was read and only these were kept. The counts are the honest ceiling
# of what Civil Comments contains at this length, not a quota that was hit.
# ---------------------------------------------------------------------------
CURATED = {
    "sexual_orientation": """
        82459a64 3b61c229 290ec700 06973da3 8ae22915 e423a824 9f92c0d4 a5d2296e c1b85c1d
        37440d8c 80aba646 606584da 826f84cf a6dda2e2 2b3987f4 16c004be eaa3f60d a6c0081a
        092cb31b ac429d2a 2c1e25bf 54816658 e4908a25 a8475972 5579ec2f 07e7f998 f70a7f34
        90c8441f 56420225 38c85c34 4a5f79e3 32bb2e1c 5f2db1ee 1a7a8857 000c5b7a 7ed39652
        117b55b1 c6affabd b04145a7 ed4febaa 2a05bfba f6d7e5a6 06611179 731b9b76 17c360d8
        4e5bb732 dea196ce 469ed8f8 445d4ef4 e26cb0d6 48320822 7a483e6b 31fdd44a e06228c3
        41821e34 a4e608b3 2afd9f3e
        313cb319 c1fe0b7e 9164a37a 5220461e 0832086d 5dcb3327 3037a35f 9a60147a 2b9134bb
        5c0dda65 90fd8f29 b5643591 fdbe7dcb d6da91cb 8c96a25f 7d8671a4 73727d0b 42424536
        fce28b46 77a62726 c8ee0623 c4553d69 7a20c50b cecbef40 8d3acc35 96bad82b 2a49edb5
        1c760f9c f4fbfbb1 4dc37a19 a6702c26 4161fc46 93c3ea30 62be017b 152c1676 635ff67a
        40d7f504 d198a391 4005a310 b9562053 87c4ea59 776044b4 b964aa38 f26d551f dabe1c3a
        0a62eec2 c7bc8298 491accad 27ae46ad aec6c355 22264d44 4240530d 822db9ea dc3d35ed
        bd15251c d8f04e64 69a884e2
        90edea29 08b44916 47172df4 b4ace609 ec25924c f4490823 81c82e14 dd7c3536 129641eb
        f11ddc20 988c30aa f24fedc3 64f0a6ab 10cad288 125884f7 e77b01d2 7a07cd0b 06d88b61
        d4c5fd7e 4605576f af611413 4927eef5 d8451e5e f177b649 1d2ebe0c e85024d3 1e64f0d7
        88d3639e 049a4a81 7e8ad1b0 a0d3756d 9e061728 afbafccc c964ca03 f049101c 1d90626e
        d21d254d 8a39f314 bc4d111f 4b63931a b82d3dff 3e7dcb06 1b8eef46 9ff1833b c726f42e
        52ab9a61 4356f220 8451d706 eeb02b74 de4399b9 81db7655 9dc3ce2b 7cf6a070 a36de707
        927ba3c5 0153c169 7c9040f8 2fb2f6f7 7e9a49b2 55515871 f571c3dd d8c4fd7f 5853ee0b
        00d20e74 1cb5ce3b f5d8ddd5 2bb85adc 44813bd9 45d727e9 e1541326 c2aac910 30958b75
        74007449 fcf4ef80 50fab5a4 10723669 92c2e316
    """,
    "gender_identity": """
        17809746 265f92ce 493c29f6 87b2bd60 dfcce4c0 33044216 994598e6 70d58ce2 22bb6e1f
        fe8af69e 049bcf2a edd7eeaf 73627c31
        ec73ea9e 3d936dc5 b3396f5a 97fca67c f8390d14 0fdb8a00 56405a0a adb808cd 53ae183a
        89d438bc c176deec a8e61cb5
    """,
    "disability": """
        d3ca25a4 c840327c ff9870f7 44a4b1e1 13623464 2ac4d8a2 f97bc375 06b886c4 3cbed841
        c87ad0cd 5c92a62e 7baf73f9 585165dc 17ca0d70 ed4b4934 3843419e 09bba45a f66217e8
    """,
    "race": """
        32825b4f 6321ff95 fa8eeb1e 279cb3c8 623169dc b6b73050 0d542a54 f6a94d5f c6e86153
        194a27b3 22c49ee9 7b083d61 5744b7ed ed75d99f 67b6eb70 85802ec7 42c00fd3 2180375e
        13af6a20 93a8f088 4bb2d827 336ead03 7a84e8ac 061cf0f5 c075acb0 c11d4366 f8751fbd
        a8e392d2 2364eae6 a3b1b929 ce2f3bb3 8b92a51a 7ff59d0d 3f6822d1 36a226ed 3e1b76ca
        e3bb2ce4 43c58a0b 12bdbe4e 194e1a5a 66680206 09b4d3d4 b3b8e4ec 01dad1b1 fab1903e
        5de11447 c8783a89 9ec46393 78f5038f b1869f0d 91865393 5f9f121a 0e3d9f7e d122dd82
        b610f891 d7a2d724 22e59a5e 1dc98012 59dab1ed 7d7d9a56 0a53ced5 035cd0bd 45734719
        4fc616c7 498c4bc3 fbf17528 6132330d 37af7a8e 7bcbccdd 702af034 394b1bbf 1ac1a965
        e48b27fb 4c9f87c3 a53b9cfd e5d30787 547609d7 7471f655 c7af8472 3cf5f908 ab303980
        1753ec0d 3ebc7b61 3732e038 1bb3abe9 e0afeae0 f9a48dc1 e296fcad 368e4044 047a17d2
        f6428419 e5277b82 3850a2b9 f54bda0c 6c280038 da1fe761 178be70b 937b48a8 3457751a
        7d8aa75b f6eb92be c44ce50c 7619d700 2c5b1322 252c2dfe 07bb1e95 9c607434 0e59ee96
        2feac6b1 899d2881 3b963e8c 7fcab261 343fd953 31eb411a ad92ced8 6404acc0 5480cf6c
        401f0969 620d16f5 088315b6 6f430887 17b84a94 1357bae7 87860992 d0f031b2 26ae494d
        e00111fa
        0778a6ce 1b8cda11 8c51ba92 44bbf910 e7f87d02 184aaca8 cdbabdbc f867b4d5 7b75c06f
        38597a94 c329a62f 86a85b60 e37ae753 e6b5752d 68767ee3 192c6697 3597d340 55186fdd
        99b85d4e af6512a5 78b2c1f0 9d90a6bc 73390588 a638e665 dcdd0dd0 81debd74 6a79c382
        d90d348d 3238dbd4 b3a72707 674c4df2
    """,
}
CURATED = {g: set(v.split()) for g, v in CURATED.items()}
CURATED_GROUPS = set(CURATED)

# Rows that pass every rule but should not be shipped. Found by reading the
# rule-selected output row by row: sarcasm ("Very Christian of him."), sectarian
# jabs, name puns ("Sikh and ye shall find."), insinuation ("we know he was a
# muslim"), the "illegal/legal immigrant" framing, mock accents, and fragments.
# About 45% of the rule-selected religion rows and 55% of the ethnicity rows
# landed here, which is the measured ceiling of what the regexes can do alone.
CURATED_REJECT = set("""
    00ea0156 00f8d557 02245ee4 02c790d7 05571fb6 0c3c80f9 0cfc9af0 0ee8765f 10901987
    118fe72f 12096b53 14e35dda 158a9694 15c788d1 17a5b30c 17f8e484 1849bd60 184dd937
    1cf715d2 208612c8 23331880 2475eee2 24ac9d19 2679359a 26d098e3 2bad5296 2c7f3482
    2d13eb32 2df6332d 335aca82 337c539c 34b6e59d 36166ccb 3850a368 38e86c23 3b285255
    3cd2e6fc 3d0d33dd 3e01be5b 3f67e639 3f6d8bc6 442a3816 442f9693 444e372e 45983cfb
    4771d34a 4870d605 49954fd7 522ecea2 52cb5bc1 5326fa1b 53f349b0 53ff1122 546778df
    55d0587b 56081281 583aff77 589f5a15 5c682dbd 5d9056b9 5dbdc9a9 5f6258f2 5f93737d
    6378ae92 664a3876 67bc8e29 6928d7b9 693b410a 6b447f12 6cb91bea 726c6919 7422529d
    74579fce 74b48982 75de55ca 7765e7ad 779e1ac1 7a8d8e93 7c67ae17 7d6ac0c3 7e533b32
    82a5397d 8372f2f1 871535f0 8824c540 889c0769 8a50ba7c 8b01c7f2 8ea2d2b6 917e43a7
    92f3502c 935dbcd4 94aee971 95bf0eb6 993120eb 9c6e821c a33d26e3 a34fcc7a a554c3a0
    a74df948 acd8f1e8 ad276c03 ad3223d4 aed493bd aef0cc39 af505f43 afac58e5 b047fd57
    b227fd74 b2801453 bcd7062a bdca0c0d c143c907 c65ce1ea c7333ab2 c7471f30 c9a0de76
    cb583363 cba80d32 cbc3592f ced2b059 d538f5e3 d826be92 dacd4844 db107f2c dc0d7ca4
    df8b85f4 e026af86 e2ce1d49 e30b434f e5acb85c e75061b1 e94a080b ea5bab90 eb243887
    eefee4b4 f37d191d f4e8d0fa f5bc49de f6ac7e52 f7468b9a fb54f563 fc072bb4 fd9470e7
    ff4c93d0
    c731fa65 827aaedc 22b55df6 d9fcc6db e3dbf58a e5144951 9b718d56 0d1008f4 fb659e5a
    3cc17883 ee53c78e ec839be2 002293f6 676fdaf5 5a987656 021e4190 0a814b37 c8bd1009
    1eb345e1 1051c506 1ef2f0d1 c1395693 c758e00b 1947c5d0 807146ee 02526d99 f0168ca0
    2441cfc1 351fbe84 8f62b5e2 83eea4d2 dd5e3ae3 f1e89ad7 236175d1 0db15228 100e4acf
    960dce36 2eb17df3 40687875 434e32fd eea7b754 d7edd805 40bce8e4 78ae9d1f 2951d194
    255add8b ac41fe5f 7451057b 0bddf782 42dc96b1 b6df7f6b 35388347 aa79ced9 95aa274d
    ab5673fa dac6b9c4 fa19be19 ade7a2eb 4115115c 045c4b4f e9c63642 77d1eb2c ed571fb2
    d6314ea1 1ddaf1bb 9825a538 a58ec229
    58daed19 52f081f3 ffb8a9ed 78009d1d 3676ae7a 93946718 1c59bce1 c6059a2c 3963fcfd
    3ad44efd 2c986ffe b8c34ca0 6d62068c 38974cb3 b53fdfc8 8ead34d9 042410c6 26bd4f57
    f7266619 32003a15 7be012cd b3935ab0 804d621b f8f3d573 db9d02ce a92097d7 581f7000
    27451e26 7dd47998 c88baeee 00ad1541 7c17ee2c 7854235a 4365ada0 6614b0ce b8444369
    f59a4aa5 2f7e5598 cf680f8a f7f72e58 c537c3d5 54ea8eef f4fe5ab7 a4ba616f 745531bc
    f54f7111 1963d243 eefff251 a6d8b199 c2d875d3 da3a71ae d1de9605 24b28c4a 61f12465
    c0631df2 60853ded d5392653 b34cf3cd bf18670a 1dc145ab 6dec47d7 deb3dcd8 2745c10d
    934ab587 b692ae13 7f82bb92 d2af896b ac972268 3e12d7b3 51c0b973 df51a2c4 6362dbd7
    216648f3 9adea3ed dd621ac5 80938f0d 2f8c2249 646f631a 79c1e4f7 548dc6d0
    34c745e4 a154255f 5824f8ff c05d7f44 28a494de 43cfc0e2 d21d9646 8f46235e 41464dcd
    e5528e6c d9fd4546 0b857a64 e399cc79 224a0222 e6a426c3
""".split())

# Per-term share cap inside a group, so one newspaper's comment section (this
# corpus is heavy on Catholic-vs-Catholic and Canadian indigenous-affairs threads)
# cannot dominate a group.
TERM_CAP_SHARE = 0.28
# No single term may exceed this share of its group in the shipped set.
TERM_MAX_SHARE = 0.45
# ...but only for groups at least this big; scarce groups are left intact.
TERM_CAP_MIN_GROUP = 80

# Rule-selected quotas, per group per length bucket. These are how many rows the
# rules PROPOSE for review, not how many ship; roughly half are then rejected by
# hand (see CURATED_REJECT). The short buckets are drawn near-exhaustively from
# the available pool, because that is where the model actually fails and where
# the reject rate is highest, so a proportional draw would starve them.
#
# These two groups are deliberately held SMALL. Their measured probe FPR is 0.032
# and 0.000 -- the model barely mishandles them -- and Civil Comments could supply
# thousands of rows for each. Taking that supply would have inverted the set
# against the harm it is meant to fix, so the quotas here are set by what the
# model needs, not by what the corpus happens to contain.
# They are also drawn ONLY from the two short buckets. The hand-curated groups
# skew long -- in this corpus, as in the training corpus, short identity mentions
# are disproportionately hostile, so what survives review is the longer material
# -- and these two groups are the only place with enough supply to put the set's
# mass back under 16 words, which is where the model actually fails.
RULE_QUOTA = {
    "religion":              {(3, 10): 130, (11, 15): 110, (16, 25): 0, (26, 35): 0},
    "ethnicity_nationality": {(3, 10):  85, (11, 15):  85, (16, 25): 0, (26, 35): 0},
}

# Length quotas. identity_bias.md defines "short" as <150 chars (~25 words) and
# the probe that fires at 0.891 is five words, so the mass belongs under 16.
LENGTH_BUCKETS = [(3, 10), (11, 15), (16, 25), (26, 35)]


# A candidate is a near-duplicate of an existing row if it shares this fraction
# of its own 5-word sequences with one single existing row. A shared 5-gram on
# its own is just ordinary English ("the fact that they are") and flags nothing.
NEARDUP_SHARE = 0.5

HELDOUT_SHARE = 0.15
RANDOM_SEED = 20260830


def key_of(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def ngrams(text: str, n: int = 5) -> set:
    w = normalise(text).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_civil_comments() -> pd.DataFrame:
    parts = [pq.read_table(hf_hub_download(CC_REPO, f, repo_type="dataset")).to_pandas()
             for f in CC_FILES]
    cc = pd.concat(parts, ignore_index=True)
    cc["text"] = cc["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return cc


def tag_terms(text: str, require_person_sense: bool = True) -> list:
    """Identity terms present, after stripping non-identity senses."""
    stripped = IDIOM.sub(" ", text)
    hits = []
    for g, terms in GROUPS.items():
        for t, rx in terms.items():
            if not re.search(rx, stripped, re.I):
                continue
            if require_person_sense and t in AMBIGUOUS and not PERSON_PAT[t].search(stripped):
                continue
            hits.append((g, t))
    return hits


def build_pool(cc: pd.DataFrame, curated: bool = False) -> pd.DataFrame:
    """Candidate pool. `curated=True` relaxes the two gates that a human read
    already covers: the hard-zero toxicity score, and the person-sense
    requirement (which correctly kills "Ben Gay" and "pot calling the kettle
    black", but also kills "I see nothing wrong with gay weddings")."""
    df = cc.copy()
    df["nwords"] = df.text.str.split().str.len()
    df = df[(df.nwords >= WORD_MIN) & (df.nwords <= WORD_MAX)]
    print(f"  {WORD_MIN}-{WORD_MAX} words: {len(df):,}")

    other = [c for c in CC_SCORES if c != "toxicity"]
    if curated:
        df = df[(df.toxicity <= GATE_CURATED_TOXICITY + 1e-9) & (df[other] == 0).all(axis=1)]
        print(f"  toxicity <= 1/6, other six == 0: {len(df):,}")
    else:
        df = df[(df[CC_SCORES] == GATE_STRICT).all(axis=1)]
        print(f"  all seven scores == 0: {len(df):,}")

    df = df[~df.text.str.contains(URLISH, na=False)]
    df = df[df.text.map(lambda t: sum(c.isalpha() or c.isspace() for c in t) / max(len(t), 1)) >= 0.80]
    df = df[df.text.map(lambda t: (lambda w: sum(x.isupper() for x in w) / max(len(w), 1))(
        [x for x in t.split() if len(x) >= 3 and x.isalpha()])) < 0.4]
    df = df[df.text.str.contains(r"[a-z]")]
    df = df[~df.text.str.contains(r"\.{4,}|-{3,}|_{3,}|\*{3,}")]
    print(f"  after format filters: {len(df):,}")

    df["hits"] = df.text.map(lambda t: tag_terms(t, require_person_sense=not curated))
    df = df[df.hits.map(len) > 0].copy()
    print(f"  with an identity term{'' if curated else ' in person sense'}: {len(df):,}")

    # These four are heuristics standing in for a human. They are applied to the
    # rule-selected groups only; the curated groups were read row by row, and a
    # human verdict outranks a regex, so `clean` is a column, not a filter.
    df["second"] = df.text.str.contains(SECOND_PERSON, na=False)
    df["question"] = df.text.str.contains(r"\?")
    df["clean"] = ~(
        df.text.str.contains(HOSTILE, na=False)
        | df.text.str.contains(SNARK, na=False)
        | (df.second & df.question)
        | df.text.str.contains(FRAGMENTARY, na=False)
        | df.text.str.contains(QUOTED, na=False)
    )
    print(f"  passing hostile/snark/argument filters: {df.clean.sum():,}")

    df["group"] = df.hits.map(lambda h: next(g for g in GROUP_ORDER if g in {x for x, _ in h}))
    df["term"] = df.apply(lambda r: next(t for g, t in r.hits if g == r.group), axis=1)
    df["political"] = df.text.str.contains(POLITICAL, na=False)
    df["form"] = "other"
    df.loc[df.text.str.contains(THIRD_PERSON, na=False), "form"] = "third_person"
    df.loc[df.text.str.contains(FIRST_PERSON, na=False), "form"] = "self_desc"
    df["key"] = df.text.map(key_of)
    df["norm"] = df.text.map(normalise)
    df = df.drop_duplicates("norm").drop_duplicates("key")
    print(f"  after dedupe: {len(df):,}")
    return df


def select(pool: pd.DataFrame, curated_pool: pd.DataFrame) -> pd.DataFrame:
    """Curated allowlist for the scarce groups; ranked quotas for the rest."""
    chosen = []

    for g, keys in CURATED.items():
        got = curated_pool[curated_pool.key.isin(keys)].copy()
        got["group"] = g          # curated rows keep the group they were read under
        got["selection"] = "manual"
        missing = keys - set(got.key)
        if missing:
            print(f"  WARNING {g}: {len(missing)} curated keys not in pool: {sorted(missing)[:5]}")
        print(f"  {g:<22} curated {len(got):>4} / {len(keys)}")
        chosen.append(got)

    # CURATED_REJECT is applied AFTER the quotas are filled, not before. Filtering
    # first would let the quota backfill with rows nobody has read; filtering last
    # means the shipped set is exactly (what the rules proposed) minus (what the
    # review threw out), and every row in it has been read.
    rest = pool[pool.clean & ~pool.group.isin(CURATED_GROUPS)].copy()
    rest = rest[~rest.second & ~rest.question & ~rest.political]
    rest["score"] = (
        3.0 * rest.form.eq("self_desc")
        + 2.0 * rest.form.eq("third_person")
        + 1.0 * rest.text.str.contains(r"[.!]$")
        + 0.5 * rest.text.str.contains(r"^[A-Z]")
        - 0.5 * rest.text.str.contains(r"\d")
    )
    for g, quota in RULE_QUOTA.items():
        sub = rest[rest.group == g]
        cap = max(3, int(sum(quota.values()) * TERM_CAP_SHARE))
        picked = []
        for lo, hi in LENGTH_BUCKETS:
            want = quota[(lo, hi)]
            band = sub[(sub.nwords >= lo) & (sub.nwords <= hi)]
            band = band.sort_values(["score", "nwords"], ascending=[False, True])
            taken, per_term = [], {}
            for _, row in band.iterrows():
                if len(taken) >= want:
                    break
                if per_term.get(row.term, 0) >= cap:
                    continue
                per_term[row.term] = per_term.get(row.term, 0) + 1
                taken.append(row)
            picked.extend(taken)
            print(f"  {g:<22} {lo:>2}-{hi:<2}w  want {want:>3}  got {len(taken):>3}"
                  f"  (pool {len(band)})")
        got = pd.DataFrame(picked)
        got["selection"] = "rule"
        chosen.append(got)

    out = pd.concat(chosen, ignore_index=True).drop_duplicates("key")
    rejected = out.key.isin(CURATED_REJECT)
    print(f"  hand-rejected after review: {rejected.sum()}")
    out = out[~rejected]

    # Final per-term balance. This corpus is Canadian/US news comments, so
    # "catholic" (one newspaper's comment section) and "native_american"
    # (indigenous-affairs threads) each ran to ~51% of their group. Trim the
    # excess, longest first, so trimming also helps the length profile. This only
    # ever REMOVES rows; backfilling would pull in candidates nobody has read.
    # Only groups with ample supply are balanced. In the scarce groups the supply
    # IS the ceiling -- there is no deeper pool to draw from -- so trimming them
    # for balance would just throw away the rows the fix needs most.
    keep = []
    for g, sub in out.groupby("group"):
        if len(sub) < TERM_CAP_MIN_GROUP:
            keep.append(sub)
            continue
        for term, n in sub.term.value_counts().items():
            rows = sub[sub.term == term]
            others = len(sub) - n
            cap = int(TERM_MAX_SHARE * others / (1 - TERM_MAX_SHARE)) if others else n
            if n > cap:
                print(f"  {g}: trimming {term} {n} -> {cap}")
                rows = rows.sort_values("nwords").head(cap)
            keep.append(rows)
    return pd.concat(keep, ignore_index=True).reset_index(drop=True)


def check_contamination(sel: pd.DataFrame) -> pd.DataFrame:
    """Exact-normalised match, plus a real near-duplicate test.

    The near-dup test builds an inverted index from 5-word sequences to the
    existing English rows containing them, then asks, for each candidate, what
    fraction of its own 5-grams the single best-matching existing row covers.
    Thresholding on that (NEARDUP_SHARE) is the difference between "these two
    comments are the same comment" and "these two comments are both English".
    """
    existing_norm = set()
    index: dict[str, dict[int, int]] = {}
    row_id = 0
    for name in ["train", "val", "test"]:
        df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        txt = df.comment_text.dropna().astype(str)
        existing_norm |= set(txt.map(normalise))
        en = df[df.lang == "en"].comment_text.dropna().astype(str)
        for t in en:
            for g in ngrams(t):
                index.setdefault(g, {})[row_id] = 1
            row_id += 1
        print(f"  {name}.csv: {len(df):,} rows ({len(en):,} en)")
    print(f"  indexed {row_id:,} English rows, {len(index):,} distinct 5-grams")

    def best_overlap(text: str) -> float:
        grams = ngrams(text)
        if not grams:
            return 0.0
        counts: dict[int, int] = {}
        for g in grams:
            for rid in index.get(g, ()):
                counts[rid] = counts.get(rid, 0) + 1
        return (max(counts.values()) / len(grams)) if counts else 0.0

    exact = sel.norm.isin(existing_norm)
    sel = sel.copy()
    sel["overlap"] = sel.text.map(best_overlap)
    near = sel.overlap >= NEARDUP_SHARE
    print(f"  exact normalised-text matches: {exact.sum()}")
    print(f"  rows sharing >=1 5-gram with some en row: {(sel.overlap > 0).sum()}")
    print(f"  near-duplicates (>= {NEARDUP_SHARE:.0%} of own 5-grams in one row): {near.sum()}")
    print(f"  max overlap observed: {sel.overlap.max():.2f}")
    for _, r in sel[exact | near].head(10).iterrows():
        print(f"    dropped ({r.overlap:.2f}): {r.text[:90]}")
    return sel[~(exact | near)].reset_index(drop=True)


def split_heldout(sel: pd.DataFrame) -> pd.DataFrame:
    """Stratified by (group, length bucket), then de-overlapped from the train slice.

    The held-out slice measures whether the fix generalises to counter-examples
    the model never trained on, in-distribution. The synthetic probe set in
    experiments/identity_bias.py is the out-of-distribution check and stays clean
    because none of these rows are probe sentences.
    """
    sel = sel.copy()
    sel["bucket"] = pd.cut(sel.nwords, [2, 10, 15, 25, 35],
                           labels=["3-10", "11-15", "16-25", "26-35"])
    sel["slice"] = "train"
    for (_, _), idx in sel.groupby(["group", "bucket"], observed=True).groups.items():
        n = len(idx)
        if n < 3:
            continue
        k = max(1, round(n * HELDOUT_SHARE))
        pick = sel.loc[idx].sample(n=k, random_state=RANDOM_SEED).index
        sel.loc[pick, "slice"] = "heldout"

    # A held-out row that shares a 5-gram with a training row would measure
    # memorisation, so move those back into train.
    tr_grams = set()
    for t in sel[sel["slice"] == "train"].text:
        tr_grams |= ngrams(t)
    leak = sel[(sel["slice"] == "heldout")
               & sel.text.map(lambda t: bool(ngrams(t) & tr_grams))].index
    sel.loc[leak, "slice"] = "train"
    print(f"  moved {len(leak)} held-out rows back to train (5-gram overlap)")
    return sel


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"comment_text": df.text.values})
    for src, dst in LABEL_MAP.items():
        out[dst] = (df[src].values >= BINARISE_THRESHOLD).astype(int)
    out["lang"] = "en"
    # id format {index}_{lang}_{labelpattern}_{hash}, matching utils/add_ids.py.
    # The index field is "cx" so counter-examples are greppable and removable:
    #   df[df.id.str.startswith("cx_")]
    pat = out[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    out["id"] = [f"cx_en_{''.join(str(v) for v in row)}_{hashlib.md5(t.encode()).hexdigest()[:6]}"
                 for row, t in zip(pat.values, out.comment_text, strict=True)]
    out["toxic"] = out["toxic"].astype(float)   # train.csv has toxic as float64
    return out[TARGET_COLS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading google/civil_comments ...")
    cc = load_civil_comments()
    print(f"  {len(cc):,} rows\n")

    print("Building rule-selection pool ...")
    pool = build_pool(cc)
    print("\nBuilding hand-review pool ...")
    curated_pool = build_pool(cc, curated=True)

    print("\nSelecting ...")
    sel = select(pool, curated_pool)
    print(f"  selected {len(sel)}")

    print("\nContamination check against dataset/split/*.csv ...")
    sel = check_contamination(sel)
    print(f"  {len(sel)} rows survive")

    print("\nHeld-out split ...")
    sel = split_heldout(sel)

    train = sel[sel["slice"] == "train"]
    held = sel[sel["slice"] == "heldout"]
    to_schema(train).to_csv(OUT_DIR / "counterexamples_train_en.csv", index=False)
    to_schema(held).to_csv(OUT_DIR / "counterexamples_heldout_en.csv", index=False)

    meta = to_schema(sel)[["id"]].copy()
    meta["comment_text"] = sel.text.values
    meta["group"] = sel.group.values
    meta["term"] = sel.term.values
    meta["form"] = sel.form.values
    meta["selection"] = sel.selection.values
    meta["slice"] = sel["slice"].values
    meta["nwords"] = sel.nwords.values
    for c in CC_SCORES:
        meta[f"cc_{c}"] = sel[c].values
    meta.to_csv(OUT_DIR / "counterexamples_metadata.csv", index=False)

    print(f"\nWrote {len(train)} train + {len(held)} held-out rows to {OUT_DIR}")
    print("\nBy group:")
    print(pd.crosstab(sel.group, sel["slice"], margins=True))
    print("\nBy length bucket:")
    print(pd.crosstab(sel.bucket, sel["slice"], margins=True))
    print("\nBy form:")
    print(sel.form.value_counts().to_string())
    print(f"\nwords: median {sel.nwords.median():.0f}  mean {sel.nwords.mean():.1f} "
          f" <16w {(sel.nwords < 16).mean():.1%}  <26w {(sel.nwords < 26).mean():.1%}")


if __name__ == "__main__":
    main()
