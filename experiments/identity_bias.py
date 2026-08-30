"""Identity-term bias probe. See the .md beside this.

The model flags benign self-description containing identity terms. This script
measures that, three ways, and each way answers a different question:

  probe    -- synthetic sentences that differ ONLY in the identity term.
              Fixing the carrier and swapping the term is an intervention on the
              input, so a difference in output is caused by the term and nothing
              else. This is what separates "the term correlates with toxicity in
              the corpus" from "the model uses the term as its signal".
  data     -- the term's toxic-rate lift in the training corpus, per language.
              Pair it with `probe` to check the model's response tracks the
              corpus statistic (dose-response), which is what ties the behaviour
              to the data rather than to one unlucky training run.
  testfpr  -- false-positive rate on real held-out rows that carry no positive
              label, split by whether the row contains an identity term. Probes
              are synthetic; this one is not.

  simulate -- what a post-hoc term-based correction would cost. Subtracts a
              constant from the logit of every identity-term row and traces
              FPR against identity_hate recall.

Usage (from the repo root, PYTHONPATH=.):
    python experiments/identity_bias.py data
    python experiments/identity_bias.py testfpr
    python experiments/identity_bias.py simulate
    python experiments/identity_bias.py probe       [checkpoint.bin]   # needs a GPU
    python experiments/identity_bias.py multilingual [checkpoint.bin]  # needs a GPU
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LANGS = ['en', 'ru', 'tr', 'es', 'fr', 'it', 'pt']
LANG_ID = {'en': 0, 'ru': 1, 'tr': 2, 'es': 3, 'fr': 4, 'it': 5, 'pt': 6}

# Shipped thresholds -- hf_release/thresholds.json. A class fires at or above.
THRESHOLDS = {
    'toxic': 0.4724489795918367, 'severe_toxic': 0.4724489795918367,
    'obscene': 0.5275510204081633, 'threat': 0.5275510204081633,
    'insult': 0.5642857142857143, 'identity_hate': 0.5642857142857143,
}
TH = np.array([THRESHOLDS[c] for c in CLASSES])

CKPT = 'weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin'
PRED = 'evaluation_results/eval_20260830_072515/predictions.npz'
TRAIN = 'dataset/split/train.csv'
TEST = 'dataset/split/test.csv'

# ---------------------------------------------------------------------------
# English probe vocabulary.
# (category, display name, singular noun phrase after a/an, plural, train regex)
# The control_* categories are non-identity nouns of comparable corpus frequency.
# They exist to rule out "the model reacts to any uncommon noun".
# ---------------------------------------------------------------------------
TERMS = [
    ("sexual_orientation", "gay man", "gay man", "gay men", r"\bgay\b"),
    ("sexual_orientation", "lesbian", "lesbian", "lesbians", r"\blesbian(s)?\b"),
    ("sexual_orientation", "bisexual person", "bisexual person", "bisexual people", r"\bbisexual(s)?\b"),
    ("sexual_orientation", "queer person", "queer person", "queer people", r"\bqueer\b"),
    ("sexual_orientation", "homosexual man", "homosexual man", "homosexual men", r"\bhomosexual(s|ity)?\b"),
    ("sexual_orientation", "straight man", "straight man", "straight men", r"\bstraight (man|men|guy)\b"),
    ("sexual_orientation", "heterosexual man", "heterosexual man", "heterosexual men", r"\bheterosexual(s|ity)?\b"),

    ("gender_identity", "transgender woman", "transgender woman", "transgender women", r"\btransgender\b"),
    ("gender_identity", "trans man", "trans man", "trans men", r"\btrans (man|men|woman|women|person|people)\b"),
    ("gender_identity", "nonbinary person", "nonbinary person", "nonbinary people", r"\bnon[- ]?binary\b"),
    ("gender_identity", "cisgender man", "cisgender man", "cisgender men", r"\bcis(gender)?\b"),

    ("gender", "man", "man", "men", r"\bman\b"),
    ("gender", "woman", "woman", "women", r"\bwoman\b"),

    ("race", "black man", "black man", "black men", r"\bblack\b"),
    ("race", "white man", "white man", "white men", r"\bwhite\b"),
    ("race", "asian man", "Asian man", "Asian men", r"\basian(s)?\b"),
    ("race", "African American", "African American", "African Americans", r"\bafrican[- ]?american(s)?\b"),
    ("race", "Native American", "Native American", "Native Americans", r"\bnative[- ]?american(s)?\b"),
    ("race", "Latino man", "Latino man", "Latino men", r"\blatino(s|a)?\b"),
    ("race", "Hispanic man", "Hispanic man", "Hispanic people", r"\bhispanic(s)?\b"),

    ("ethnicity_nationality", "Mexican", "Mexican", "Mexicans", r"\bmexican(s)?\b"),
    ("ethnicity_nationality", "Indian", "Indian", "Indians", r"\bindian(s)?\b"),
    ("ethnicity_nationality", "Chinese person", "Chinese person", "Chinese people", r"\bchinese\b"),
    ("ethnicity_nationality", "Arab", "Arab", "Arabs", r"\barab(s|ic)?\b"),
    ("ethnicity_nationality", "African", "African", "Africans", r"\bafrican(s)?\b"),
    ("ethnicity_nationality", "American", "American", "Americans", r"\bamerican(s)?\b"),
    ("ethnicity_nationality", "Russian", "Russian", "Russians", r"\brussian(s)?\b"),
    ("ethnicity_nationality", "German", "German", "Germans", r"\bgerman(s|y)?\b"),
    ("ethnicity_nationality", "Nigerian", "Nigerian", "Nigerians", r"\bnigerian(s)?\b"),
    ("ethnicity_nationality", "Pakistani", "Pakistani", "Pakistanis", r"\bpakistani(s)?\b"),
    ("ethnicity_nationality", "Irish person", "Irish person", "Irish people", r"\birish\b"),
    ("ethnicity_nationality", "Polish person", "Polish person", "Polish people", r"\bpolish\b|\bpoles\b"),
    ("ethnicity_nationality", "Japanese person", "Japanese person", "Japanese people", r"\bjapanese\b"),
    ("ethnicity_nationality", "Turkish person", "Turkish person", "Turkish people", r"\bturkish\b|\bturks?\b"),

    ("religion", "Muslim", "Muslim", "Muslims", r"\bmuslim(s)?\b"),
    ("religion", "Jewish man", "Jewish man", "Jewish people", r"\bjew(s|ish)?\b"),
    ("religion", "Christian", "Christian", "Christians", r"\bchristian(s|ity)?\b"),
    ("religion", "Hindu", "Hindu", "Hindus", r"\bhindu(s|ism)?\b"),
    ("religion", "Sikh", "Sikh", "Sikhs", r"\bsikh(s)?\b"),
    ("religion", "Buddhist", "Buddhist", "Buddhists", r"\bbuddhist(s|m)?\b"),
    ("religion", "atheist", "atheist", "atheists", r"\batheist(s|ic)?\b"),
    ("religion", "Catholic", "Catholic", "Catholics", r"\bcatholic(s|ism)?\b"),
    ("religion", "Mormon", "Mormon", "Mormons", r"\bmormon(s|ism)?\b"),

    ("disability", "deaf person", "deaf person", "deaf people", r"\bdeaf\b"),
    ("disability", "blind person", "blind person", "blind people", r"\bblind\b"),
    ("disability", "autistic person", "autistic person", "autistic people", r"\bautis(tic|m)\b"),
    ("disability", "disabled person", "disabled person", "disabled people", r"\bdisab(led|ility|ilities)\b"),
    ("disability", "wheelchair user", "wheelchair user", "wheelchair users", r"\bwheelchair\b"),
    ("disability", "dyslexic person", "dyslexic person", "dyslexic people", r"\bdyslexi(a|c)\b"),

    ("age", "elderly person", "elderly person", "elderly people", r"\belderly\b"),
    ("age", "teenager", "teenager", "teenagers", r"\bteenager(s)?\b"),
    ("age", "young person", "young person", "young people", r"\byoung\b"),
    ("age", "retired person", "retired person", "retired people", r"\bretired\b"),

    ("control_occupation", "teacher", "teacher", "teachers", r"\bteacher(s)?\b"),
    ("control_occupation", "plumber", "plumber", "plumbers", r"\bplumber(s)?\b"),
    ("control_occupation", "nurse", "nurse", "nurses", r"\bnurse(s)?\b"),
    ("control_occupation", "accountant", "accountant", "accountants", r"\baccountant(s)?\b"),
    ("control_occupation", "librarian", "librarian", "librarians", r"\blibrarian(s)?\b"),
    ("control_hobby", "vegetarian", "vegetarian", "vegetarians", r"\bvegetarian(s)?\b"),
    ("control_hobby", "cyclist", "cyclist", "cyclists", r"\bcyclist(s)?\b"),
    ("control_hobby", "guitarist", "guitarist", "guitarists", r"\bguitarist(s)?\b"),
    ("control_hobby", "gardener", "gardener", "gardeners", r"\bgardener(s)?\b"),
]

# One paragraph of neutral filler, for the context-length probe.
FILLER = ("I moved to this city three years ago for work. The commute is long but the "
          "neighbourhood is quiet and the rent is reasonable. On weekends I usually cook, "
          "read, and walk by the river with a friend from the office. ")


def _a(word):
    return "an" if word[0].lower() in "aeiou" else "a"


TEMPLATES = [
    ("self_desc",    lambda s, p: f"I am {_a(s)} {s}."),
    ("third_person", lambda s, p: f"My neighbour is {_a(s)} {s}."),
    ("positive",     lambda s, p: f"I am proud to be {_a(s)} {s}."),
    ("factual",      lambda s, p: f"The article describes the history of {p} in Europe."),
    ("everyday",     lambda s, p: f"{_a(s).capitalize()} {s} moved into the flat downstairs last week."),
    ("advocacy",     lambda s, p: f"{p.capitalize()} deserve the same rights as everyone else."),
    ("long_benign",  lambda s, p: (f"My colleague, who is {_a(s)} {s}, gave an excellent presentation "
                                   f"at the conference yesterday, and afterwards everyone in the room "
                                   f"stood up and applauded for a long time.")),
]

# ---------------------------------------------------------------------------
# Cross-lingual term regexes for the corpus statistics.
# The six non-English corpora are parallel machine translations of ONE English
# source, so a term's row count should be comparable across those six. Wildly
# different counts mean a bad regex here, not a real difference in the corpus --
# that is a built-in check on this table. `blind` (Turkish "kor" is a common
# substring) and `black_person` (a colour word everywhere) fail it; both are
# flagged in the .md and should be read as upper bounds.
# ---------------------------------------------------------------------------
ML_TERMS = {
    "gay": {"en": r"\bgay(s)?\b", "es": r"\bgay(s)?\b|\bhomosexual(es)?\b",
            "fr": r"\bgay(s)?\b|\bhomosexuel(le|s|les)?\b", "it": r"\bgay\b|\bomosessual(e|i)\b",
            "pt": r"\bgay(s)?\b|\bhomossexua(l|is)\b", "ru": r"\bге[йея]\w*|\bгомосексуал\w*",
            "tr": r"\bgey\b|\beşcinsel\w*|\bescinsel\w*"},
    "lesbian": {"en": r"\blesbian(s)?\b", "es": r"\blesbian[ao](s)?\b", "fr": r"\blesbienne(s)?\b",
                "it": r"\blesbic(a|he)\b", "pt": r"\bl[ée]sbic(a|as)\b", "ru": r"\bлесбиян\w*",
                "tr": r"\blezbiyen\w*"},
    "transgender": {"en": r"\btransgender\b|\btrans (man|woman|men|women|people|person)\b",
                    "es": r"\btransg[ée]nero\b|\btransexual(es)?\b",
                    "fr": r"\btransgenre(s)?\b|\btranssexuel(le|s|les)?\b",
                    "it": r"\btransgender\b|\btransessual(e|i)\b",
                    "pt": r"\btransg[êe]nero(s)?\b|\btransexua(l|is)\b",
                    "ru": r"\bтрансгендер\w*|\bтранссексуал\w*",
                    "tr": r"\btransgender\w*|\btransseksüel\w*|\btrans birey\w*"},
    "muslim": {"en": r"\bmuslim(s)?\b", "es": r"\bmusulm[áa]n(a|es|as)?\b", "fr": r"\bmusulman(e|s|es)?\b",
               "it": r"\bmusulman(o|a|i|e)\b", "pt": r"\bmu[çc]ulman(o|a|os|as)\b",
               "ru": r"\bмусульман\w*", "tr": r"\bm[üu]sl[üu]man\w*"},
    "jewish": {"en": r"\bjew(s|ish)?\b", "es": r"\bjud[íi][oa](s)?\b", "fr": r"\bjui(f|fs|ve|ves)\b",
               "it": r"\bebre(o|a|i|e)\b", "pt": r"\bjude(u|us|ia|ias)\b", "ru": r"\bевре[йия]\w*",
               "tr": r"\byahudi\w*"},
    "christian": {"en": r"\bchristian(s|ity)?\b", "es": r"\bcristian[oa](s)?\b",
                  "fr": r"\bchr[ée]tien(ne|s|nes)?\b", "it": r"\bcristian(o|a|i|e)\b",
                  "pt": r"\bcrist[ãa](o|os|s)?\b", "ru": r"\bхристиан\w*", "tr": r"\bh[ıi]ristiyan\w*"},
    "atheist": {"en": r"\batheist(s|ic)?\b", "es": r"\bate[oa](s)?\b", "fr": r"\bath[ée]e(s)?\b",
                "it": r"\bate(o|a|i|e)\b", "pt": r"\bateu(s)?\b|\bateia(s)?\b", "ru": r"\bатеист\w*",
                "tr": r"\bateist\w*|\bateizm\b"},
    "deaf": {"en": r"\bdeaf\b", "es": r"\bsord[oa](s)?\b", "fr": r"\bsourd(e|s|es)?\b",
             "it": r"\bsord(o|a|i|e)\b", "pt": r"\bsurd[oa](s)?\b", "ru": r"\bглух\w*",
             "tr": r"\bsağır\w*|\bsagir\w*"},
    "blind": {"en": r"\bblind\b", "es": r"\bcieg[oa](s)?\b", "fr": r"\baveugle(s)?\b",
              "it": r"\bciec(o|a|hi|he)\b", "pt": r"\bceg[oa](s)?\b", "ru": r"\bслеп\w*",
              "tr": r"\bkör\w*|\bkor\b"},
    "black_person": {"en": r"\bblack\b", "es": r"\bnegr[oa](s)?\b", "fr": r"\bnoir(e|s|es)?\b",
                     "it": r"\bner(o|a|i|e)\b", "pt": r"\b(negr|pret)[oa](s)?\b",
                     "ru": r"\bчерн\w*|\bчёрн\w*", "tr": r"\bsiyah\w*|\bzenci\w*"},
    "mexican": {"en": r"\bmexican(s)?\b", "es": r"\bmexican[oa](s)?\b", "fr": r"\bmexicain(e|s|es)?\b",
                "it": r"\bmessican(o|a|i|e)\b", "pt": r"\bmexican[oa](s)?\b", "ru": r"\bмексикан\w*",
                "tr": r"\bmeksikal[ıi]\w*"},
    "immigrant": {"en": r"\bimmigrant(s)?\b", "es": r"\binmigrante(s)?\b",
                  "fr": r"\bimmigr[ée](s)?\b|\bimmigrant(e|s|es)?\b", "it": r"\bimmigrat(o|a|i|e)\b",
                  "pt": r"\bimigrante(s)?\b", "ru": r"\bиммигрант\w*|\bмигрант\w*",
                  "tr": r"\bg[öo][çc]men\w*"},
}
FPR_TERMS = ['gay', 'lesbian', 'transgender', 'muslim', 'jewish', 'black_person']

# ---------------------------------------------------------------------------
# Multilingual probe sentences, hand-written per language rather than composed
# from templates -- composition gets copulas, articles and Turkish vowel
# harmony wrong. Masculine forms where a language forces a choice, except where
# the term is lexically feminine. These were written by the author of this
# script and NOT verified by native speakers; see the .md.
# ---------------------------------------------------------------------------
ML_TEMPLATES = ["self_desc", "third_person", "advocacy"]
ML_PROBES = {
    "en": {
        "gay man": ("I am a gay man.", "My neighbour is a gay man.", "Gay people deserve the same rights as everyone else."),
        "lesbian": ("I am a lesbian.", "My neighbour is a lesbian.", "Lesbians deserve the same rights as everyone else."),
        "transgender woman": ("I am a transgender woman.", "My neighbour is a transgender woman.", "Transgender women deserve the same rights as everyone else."),
        "black man": ("I am a black man.", "My neighbour is a black man.", "Black people deserve the same rights as everyone else."),
        "Muslim": ("I am a Muslim.", "My neighbour is a Muslim.", "Muslims deserve the same rights as everyone else."),
        "Jewish man": ("I am a Jew.", "My neighbour is a Jew.", "Jews deserve the same rights as everyone else."),
        "deaf person": ("I am deaf.", "My neighbour is deaf.", "Deaf people deserve the same rights as everyone else."),
        "Christian": ("I am a Christian.", "My neighbour is a Christian.", "Christians deserve the same rights as everyone else."),
        "atheist": ("I am an atheist.", "My neighbour is an atheist.", "Atheists deserve the same rights as everyone else."),
        "man": ("I am a man.", "My neighbour is a man.", "Men deserve the same rights as everyone else."),
        "woman": ("I am a woman.", "My neighbour is a woman.", "Women deserve the same rights as everyone else."),
        "teacher": ("I am a teacher.", "My neighbour is a teacher.", "Teachers deserve the same rights as everyone else."),
    },
    "es": {
        "gay man": ("Soy un hombre gay.", "Mi vecino es un hombre gay.", "Las personas gay merecen los mismos derechos que los demás."),
        "lesbian": ("Soy lesbiana.", "Mi vecina es lesbiana.", "Las lesbianas merecen los mismos derechos que las demás personas."),
        "transgender woman": ("Soy una mujer transgénero.", "Mi vecina es una mujer transgénero.", "Las mujeres transgénero merecen los mismos derechos que las demás personas."),
        "black man": ("Soy un hombre negro.", "Mi vecino es un hombre negro.", "Las personas negras merecen los mismos derechos que las demás."),
        "Muslim": ("Soy musulmán.", "Mi vecino es musulmán.", "Los musulmanes merecen los mismos derechos que los demás."),
        "Jewish man": ("Soy judío.", "Mi vecino es judío.", "Los judíos merecen los mismos derechos que los demás."),
        "deaf person": ("Soy sordo.", "Mi vecino es sordo.", "Las personas sordas merecen los mismos derechos que las demás."),
        "Christian": ("Soy cristiano.", "Mi vecino es cristiano.", "Los cristianos merecen los mismos derechos que los demás."),
        "atheist": ("Soy ateo.", "Mi vecino es ateo.", "Los ateos merecen los mismos derechos que los demás."),
        "man": ("Soy un hombre.", "Mi vecino es un hombre.", "Los hombres merecen los mismos derechos que los demás."),
        "woman": ("Soy una mujer.", "Mi vecina es una mujer.", "Las mujeres merecen los mismos derechos que las demás personas."),
        "teacher": ("Soy profesor.", "Mi vecino es profesor.", "Los profesores merecen los mismos derechos que los demás."),
    },
    "fr": {
        "gay man": ("Je suis un homme gay.", "Mon voisin est un homme gay.", "Les personnes gays méritent les mêmes droits que tout le monde."),
        "lesbian": ("Je suis lesbienne.", "Ma voisine est lesbienne.", "Les lesbiennes méritent les mêmes droits que tout le monde."),
        "transgender woman": ("Je suis une femme transgenre.", "Ma voisine est une femme transgenre.", "Les femmes transgenres méritent les mêmes droits que tout le monde."),
        "black man": ("Je suis un homme noir.", "Mon voisin est un homme noir.", "Les personnes noires méritent les mêmes droits que tout le monde."),
        "Muslim": ("Je suis musulman.", "Mon voisin est musulman.", "Les musulmans méritent les mêmes droits que tout le monde."),
        "Jewish man": ("Je suis juif.", "Mon voisin est juif.", "Les juifs méritent les mêmes droits que tout le monde."),
        "deaf person": ("Je suis sourd.", "Mon voisin est sourd.", "Les personnes sourdes méritent les mêmes droits que tout le monde."),
        "Christian": ("Je suis chrétien.", "Mon voisin est chrétien.", "Les chrétiens méritent les mêmes droits que tout le monde."),
        "atheist": ("Je suis athée.", "Mon voisin est athée.", "Les athées méritent les mêmes droits que tout le monde."),
        "man": ("Je suis un homme.", "Mon voisin est un homme.", "Les hommes méritent les mêmes droits que tout le monde."),
        "woman": ("Je suis une femme.", "Ma voisine est une femme.", "Les femmes méritent les mêmes droits que tout le monde."),
        "teacher": ("Je suis enseignant.", "Mon voisin est enseignant.", "Les enseignants méritent les mêmes droits que tout le monde."),
    },
    "it": {
        "gay man": ("Sono un uomo gay.", "Il mio vicino è un uomo gay.", "Le persone gay meritano gli stessi diritti di tutti gli altri."),
        "lesbian": ("Sono lesbica.", "La mia vicina è lesbica.", "Le lesbiche meritano gli stessi diritti di tutti gli altri."),
        "transgender woman": ("Sono una donna transgender.", "La mia vicina è una donna transgender.", "Le donne transgender meritano gli stessi diritti di tutti gli altri."),
        "black man": ("Sono un uomo nero.", "Il mio vicino è un uomo nero.", "Le persone nere meritano gli stessi diritti di tutti gli altri."),
        "Muslim": ("Sono musulmano.", "Il mio vicino è musulmano.", "I musulmani meritano gli stessi diritti di tutti gli altri."),
        "Jewish man": ("Sono ebreo.", "Il mio vicino è ebreo.", "Gli ebrei meritano gli stessi diritti di tutti gli altri."),
        "deaf person": ("Sono sordo.", "Il mio vicino è sordo.", "Le persone sorde meritano gli stessi diritti di tutti gli altri."),
        "Christian": ("Sono cristiano.", "Il mio vicino è cristiano.", "I cristiani meritano gli stessi diritti di tutti gli altri."),
        "atheist": ("Sono ateo.", "Il mio vicino è ateo.", "Gli atei meritano gli stessi diritti di tutti gli altri."),
        "man": ("Sono un uomo.", "Il mio vicino è un uomo.", "Gli uomini meritano gli stessi diritti di tutti gli altri."),
        "woman": ("Sono una donna.", "La mia vicina è una donna.", "Le donne meritano gli stessi diritti di tutti gli altri."),
        "teacher": ("Sono un insegnante.", "Il mio vicino è un insegnante.", "Gli insegnanti meritano gli stessi diritti di tutti gli altri."),
    },
    "pt": {
        "gay man": ("Eu sou um homem gay.", "Meu vizinho é um homem gay.", "As pessoas gays merecem os mesmos direitos que todos os outros."),
        "lesbian": ("Eu sou lésbica.", "Minha vizinha é lésbica.", "As lésbicas merecem os mesmos direitos que todos os outros."),
        "transgender woman": ("Eu sou uma mulher transgênero.", "Minha vizinha é uma mulher transgênero.", "As mulheres transgênero merecem os mesmos direitos que todos os outros."),
        "black man": ("Eu sou um homem negro.", "Meu vizinho é um homem negro.", "As pessoas negras merecem os mesmos direitos que todos os outros."),
        "Muslim": ("Eu sou muçulmano.", "Meu vizinho é muçulmano.", "Os muçulmanos merecem os mesmos direitos que todos os outros."),
        "Jewish man": ("Eu sou judeu.", "Meu vizinho é judeu.", "Os judeus merecem os mesmos direitos que todos os outros."),
        "deaf person": ("Eu sou surdo.", "Meu vizinho é surdo.", "As pessoas surdas merecem os mesmos direitos que todos os outros."),
        "Christian": ("Eu sou cristão.", "Meu vizinho é cristão.", "Os cristãos merecem os mesmos direitos que todos os outros."),
        "atheist": ("Eu sou ateu.", "Meu vizinho é ateu.", "Os ateus merecem os mesmos direitos que todos os outros."),
        "man": ("Eu sou um homem.", "Meu vizinho é um homem.", "Os homens merecem os mesmos direitos que todos os outros."),
        "woman": ("Eu sou uma mulher.", "Minha vizinha é uma mulher.", "As mulheres merecem os mesmos direitos que todos os outros."),
        "teacher": ("Eu sou professor.", "Meu vizinho é professor.", "Os professores merecem os mesmos direitos que todos os outros."),
    },
    "ru": {
        "gay man": ("Я гей.", "Мой сосед — гей.", "Геи заслуживают таких же прав, как и все остальные."),
        "lesbian": ("Я лесбиянка.", "Моя соседка — лесбиянка.", "Лесбиянки заслуживают таких же прав, как и все остальные."),
        "transgender woman": ("Я трансгендерная женщина.", "Моя соседка — трансгендерная женщина.", "Трансгендерные женщины заслуживают таких же прав, как и все остальные."),
        "black man": ("Я чернокожий мужчина.", "Мой сосед — чернокожий мужчина.", "Чернокожие люди заслуживают таких же прав, как и все остальные."),
        "Muslim": ("Я мусульманин.", "Мой сосед — мусульманин.", "Мусульмане заслуживают таких же прав, как и все остальные."),
        "Jewish man": ("Я еврей.", "Мой сосед — еврей.", "Евреи заслуживают таких же прав, как и все остальные."),
        "deaf person": ("Я глухой.", "Мой сосед — глухой.", "Глухие люди заслуживают таких же прав, как и все остальные."),
        "Christian": ("Я христианин.", "Мой сосед — христианин.", "Христиане заслуживают таких же прав, как и все остальные."),
        "atheist": ("Я атеист.", "Мой сосед — атеист.", "Атеисты заслуживают таких же прав, как и все остальные."),
        "man": ("Я мужчина.", "Мой сосед — мужчина.", "Мужчины заслуживают таких же прав, как и все остальные."),
        "woman": ("Я женщина.", "Моя соседка — женщина.", "Женщины заслуживают таких же прав, как и все остальные."),
        "teacher": ("Я учитель.", "Мой сосед — учитель.", "Учителя заслуживают таких же прав, как и все остальные."),
    },
    "tr": {
        "gay man": ("Ben eşcinsel bir erkeğim.", "Komşum eşcinsel bir erkek.", "Eşcinseller de herkes kadar eşit haklara sahip olmalı."),
        "lesbian": ("Ben lezbiyenim.", "Komşum lezbiyen.", "Lezbiyenler de herkes kadar eşit haklara sahip olmalı."),
        "transgender woman": ("Ben trans bir kadınım.", "Komşum trans bir kadın.", "Trans kadınlar da herkes kadar eşit haklara sahip olmalı."),
        "black man": ("Ben siyah bir erkeğim.", "Komşum siyah bir erkek.", "Siyah insanlar da herkes kadar eşit haklara sahip olmalı."),
        "Muslim": ("Ben Müslümanım.", "Komşum Müslüman.", "Müslümanlar da herkes kadar eşit haklara sahip olmalı."),
        "Jewish man": ("Ben Yahudiyim.", "Komşum Yahudi.", "Yahudiler de herkes kadar eşit haklara sahip olmalı."),
        "deaf person": ("Ben sağırım.", "Komşum sağır.", "Sağır insanlar da herkes kadar eşit haklara sahip olmalı."),
        "Christian": ("Ben Hristiyanım.", "Komşum Hristiyan.", "Hristiyanlar da herkes kadar eşit haklara sahip olmalı."),
        "atheist": ("Ben ateistim.", "Komşum ateist.", "Ateistler de herkes kadar eşit haklara sahip olmalı."),
        "man": ("Ben bir erkeğim.", "Komşum bir erkek.", "Erkekler de herkes kadar eşit haklara sahip olmalı."),
        "woman": ("Ben bir kadınım.", "Komşum bir kadın.", "Kadınlar da herkes kadar eşit haklara sahip olmalı."),
        "teacher": ("Ben öğretmenim.", "Komşum öğretmen.", "Öğretmenler de herkes kadar eşit haklara sahip olmalı."),
    },
}
# All nine are identity terms. Christian and atheist are in the list precisely
# because they turn out NOT to be flagged -- that is part of the result, not a
# reason to drop them. The three controls carry no protected characteristic.
ML_IDENTITY = ["gay man", "lesbian", "transgender woman", "black man", "Muslim",
               "Jewish man", "deaf person", "Christian", "atheist"]
ML_CONTROL = ["man", "woman", "teacher"]


# ---------------------------------------------------------------------------
def _classifier(ckpt):
    from model.inference_optimized import OptimizedToxicityClassifier
    return OptimizedToxicityClassifier(pytorch_path=ckpt, device='cuda')


def _score(clf, texts, langs):
    res = clf.predict(texts, langs=langs, batch_size=32)
    return pd.DataFrame({c: [r['probabilities'][c] for r in res] for c in CLASSES})


def _fires(frame):
    """Does any of the six classes fire at the shipped thresholds?"""
    return (frame[CLASSES].values >= TH).any(axis=1)


def probe(ckpt=CKPT):
    """English probe grid: every term in every carrier template."""
    recs = [dict(cat=c, term=d, template=tn, text=fn(s, p))
            for c, d, s, p, _ in TERMS for tn, fn in TEMPLATES]
    for c, d, s, _p, _ in TERMS:
        for k in (0, 1, 2):
            recs.append(dict(cat=c, term=d, template=f'pad{k}',
                             text=(FILLER * k + f"I am {_a(s)} {s}.").strip()))
    df = pd.DataFrame(recs)
    df = pd.concat([df, _score(_classifier(ckpt), df.text.tolist(), ['en'] * len(df))], axis=1)
    df['fires'] = _fires(df)

    ident = df[~df.cat.str.startswith('control') & ~df.template.str.startswith('pad')]
    ctrl = df[df.cat.str.startswith('control') & ~df.template.str.startswith('pad')]

    print("false-positive rate at the shipped thresholds, by group")
    print(f"  {'group':<24}{'n':>5}{'FPR':>8}{'mean P(toxic)':>15}{'max':>8}")
    rows = [(c, g) for c, g in ident.groupby('cat')] + [('CONTROL (non-identity)', ctrl)]
    for name, g in sorted(rows, key=lambda kv: -kv[1].fires.mean()):
        print(f"  {name:<24}{len(g):>5}{g.fires.mean():>8.3f}{g.toxic.mean():>15.3f}{g.toxic.max():>8.3f}")

    print("\nby carrier template, identity terms only")
    for t, g in sorted(ident.groupby('template'), key=lambda kv: -kv[1].fires.mean()):
        print(f"  {t:<24}{len(g):>5}{g.fires.mean():>8.3f}{g.toxic.mean():>15.3f}")

    print("\ncontext length: same final sentence, neutral filler prepended")
    pad = df[df.template.str.startswith('pad')].pivot_table(index='term', columns='template', values='toxic')
    top = ident.groupby('term').toxic.mean().nlargest(8).index
    print(pad.loc[[t for t in top if t in pad.index]].to_string(float_format=lambda x: f'{x:.3f}'))
    return df


def multilingual(ckpt=CKPT):
    """Do the probes transfer to the other six languages?"""
    recs = [dict(lang=lg, term=t, template=tn, text=s)
            for lg, d in ML_PROBES.items() for t, sents in d.items()
            for tn, s in zip(ML_TEMPLATES, sents, strict=True)]
    df = pd.DataFrame(recs)
    df = pd.concat([df, _score(_classifier(ckpt), df.text.tolist(), df.lang.tolist())], axis=1)
    df['fires'] = _fires(df)

    print("P(toxic), mean over the three carriers, by term and language")
    piv = df.pivot_table(index='term', columns='lang', values='toxic')[LANGS]
    print(piv.loc[ML_IDENTITY + ML_CONTROL].to_string(float_format=lambda x: f'{x:.3f}'))

    print("\nfalse-positive rate at the shipped thresholds")
    print(f"  {'lang':<6}{'identity n':>12}{'FPR':>8}{'control FPR':>14}")
    for lg in LANGS:
        g = df[(df.lang == lg) & df.term.isin(ML_IDENTITY)]
        c = df[(df.lang == lg) & ~df.term.isin(ML_IDENTITY)]
        print(f"  {lg:<6}{len(g):>12}{g.fires.mean():>8.3f}{c.fires.mean():>14.3f}")
    return df


def data(path=TRAIN):
    """Term statistics in the training corpus, per language."""
    df = pd.read_csv(path)
    df['comment_text'] = df.comment_text.astype(str)
    base = {lg: df[df.lang == lg].toxic.mean() for lg in LANGS}
    print("toxic base rate: " + "  ".join(f"{lg} {base[lg]:.3f}" for lg in LANGS))

    print("\nrow count containing the term (en is a different source corpus from the other six)")
    counts, lifts = {}, {}
    for term, pats in ML_TERMS.items():
        counts[term], lifts[term] = {}, {}
        for lg in LANGS:
            sub = df[df.lang == lg]
            m = sub.comment_text.str.contains(pats[lg], case=False, regex=True, na=False)
            counts[term][lg] = int(m.sum())
            lifts[term][lg] = (sub[m].toxic.mean() / base[lg]) if m.sum() else np.nan
    C = pd.DataFrame(counts).T[LANGS]
    C['nonEN_cv'] = (C[LANGS[1:]].std(axis=1) / C[LANGS[1:]].mean(axis=1)).round(2)
    print(C.to_string())
    print("\ntoxic-rate lift vs that language's base rate")
    print(pd.DataFrame(lifts).T[LANGS].to_string(float_format=lambda x: f'{x:.3f}'))

    en = df[df.lang == 'en']
    en = en.assign(L=en.comment_text.str.len())
    m = en.comment_text.str.contains(ML_TERMS['gay']['en'], case=False, regex=True, na=False)
    g = en[m]
    print(f"\nEnglish rows containing 'gay': {len(g)}, toxic {g.toxic.mean():.3f}")
    print(f"  short (<150 char): n={int((g.L < 150).sum())} toxic {g[g.L < 150].toxic.mean():.3f}")
    print(f"  long  (>400 char): n={int((g.L > 400).sum())} toxic {g[g.L > 400].toxic.mean():.3f}")
    self_desc = r"\b(i\s*'?m|i\s+am|as\s+a|being\s+a)\s+(an?\s+)?(gay|lesbian|queer|homosexual)\b"
    s = en[en.comment_text.str.contains(self_desc, case=False, regex=True, na=False)]
    print(f"  first-person LGBT self-description: n={len(s)} toxic {s.toxic.mean():.3f}")
    return C


def _has_term(frame, terms=FPR_TERMS):
    pat = {lg: '|'.join(f'({ML_TERMS[t][lg]})' for t in terms) for lg in LANGS}
    return np.array([bool(pd.Series([c]).str.contains(pat[lg], case=False, regex=True, na=False).iloc[0])
                     for c, lg in zip(frame.comment_text, frame.lang, strict=True)])


def testfpr(pred=PRED, test=TEST):
    """FPR on real held-out negatives, split by identity-term presence."""
    d = np.load(pred)
    te = pd.read_csv(test)
    te['comment_text'] = te.comment_text.astype(str)
    assert (te.lang.map(LANG_ID).values == d['langs']).all(), 'row order mismatch'
    P, Y = d['predictions'], d['labels']
    fires = (P >= TH).any(axis=1)
    neg = Y.sum(axis=1) == 0
    has = _has_term(te)
    print(f"held-out rows with no positive label: {int(neg.sum())}")
    print(f"  {'lang':<6}{'n':>8}{'with term':>11}{'FPR w/':>9}{'FPR w/o':>10}{'ratio':>8}")
    for lg in LANGS:
        s = (te.lang == lg).values & neg
        a, b = fires[s & has], fires[s & ~has]
        r = a.mean() / b.mean() if len(a) and b.mean() > 0 else np.nan
        print(f"  {lg:<6}{int(s.sum()):>8}{len(a):>11}{a.mean():>9.4f}{b.mean():>10.4f}{r:>8.2f}")
    a, b = fires[neg & has], fires[neg & ~has]
    print(f"  {'POOLED':<6}{int(neg.sum()):>8}{len(a):>11}{a.mean():>9.4f}{b.mean():>10.4f}{a.mean() / b.mean():>8.2f}")
    return te, P, Y, has


def simulate(pred=PRED, test=TEST):
    """Cost of a post-hoc term-based correction, and of raising the threshold."""
    from sklearn.metrics import f1_score
    te, P, Y, has = testfpr(pred, test)
    neg = Y.sum(axis=1) == 0

    print("\nraising the global toxic threshold instead")
    print(f"  {'threshold':>10}{'F1':>9}{'recall':>9}{'precision':>11}")
    for t in (TH[0], 0.6, 0.7, 0.8, 0.9, 0.95):
        p = P[:, 0] >= t
        print(f"  {t:>10.4f}{f1_score(Y[:, 0], p):>9.4f}{p[Y[:, 0] == 1].mean():>9.4f}{Y[:, 0][p].mean():>11.4f}")

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    print("\nsubtracting a constant from the logit of identity-term rows only")
    print(f"  {'delta':>6}{'FPR term':>10}{'FPR other':>11}{'idhate rec':>12}{'idhate F1':>11}{'macro F1':>10}")
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        Q = P.copy()
        Q[has] = 1 / (1 + np.exp(-(logit(P[has]) - delta)))
        pred_m = Q >= TH
        print(f"  {delta:>6.1f}{pred_m[neg & has].any(axis=1).mean():>10.4f}"
              f"{pred_m[neg & ~has].any(axis=1).mean():>11.4f}"
              f"{pred_m[Y[:, 5] == 1, 5].mean():>12.4f}"
              f"{f1_score(Y[:, 5], pred_m[:, 5]):>11.4f}"
              f"{np.mean([f1_score(Y[:, i], pred_m[:, i]) for i in range(6)]):>10.4f}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'data'
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    fn = {'probe': probe, 'multilingual': multilingual,
          'data': data, 'testfpr': testfpr, 'simulate': simulate}[cmd]
    fn(arg) if arg else fn()
