#!/usr/bin/env python3
"""
Bauplan der Promptvarianten (subsec:varianten).

Ein Prompt wird nicht als Zeichenkette erzeugt, sondern als Folge von
Segmenten, von denen jedes mit der Herkunft (kern oder redundanz) und dem
Promptabschnitt (instruktion, demonstration, testfrage, filler) versehen ist.
Der Prompttext entsteht durch Konkatenation, die Zeichenoffsets bleiben
erhalten. Damit ist die Auszeichnung nach subsubsec:auszeichnung Bestandteil
der Konstruktion und muss nach der Kompression nicht rekonstruiert werden.

Alle fuenf Varianten werden aus demselben Material gebaut. Der Basisprompt ist
die Variante ohne Einfuegung; er wird nicht erst erzeugt und danach wieder
zerlegt. Dadurch ist ausgeschlossen, dass sich Basisprompt und Redundanzfassung
in etwas anderem als der Einfuegung unterscheiden.

Kalibrierung (subsubsec:kalibrierung)
-------------------------------------
Die eingefuegte Textmenge ist nicht fest, sondern wird je Aufgabe so gewaehlt,
dass eq:kalibrierung erfuellt ist. Jede Variante stellt dafuer eine Menge
austauschbarer Einfuegeeinheiten bereit:

    semantic       bis zu fuenf Paraphrasen (Testfrage und vier Demonstrationen)
    instruction    ein Paar Umformulierungen aus INSTRUCTION_POOL
    demonstration  bis zu vier zusaetzliche Demonstrationen (Slots 5 bis 8)
    filler         Saetze aus FILLER_POOL

Welche Teilmenge verwendet wird, entscheidet 03_build_redundancy.py anhand der
in Modelltoken gemessenen Laenge. Diese Datei liefert dazu die Einheiten
(einheiten) und baut zu einer gewaehlten Teilmenge den Prompt (baue). Die
Groesse der wiederholten Einheit bleibt damit das unterscheidende Merkmal der
Varianten, das eingefuegte Volumen wird zwischen ihnen angeglichen.
"""

import itertools
import re
import unicodedata

KERN = "kern"
RED = "redundanz"

A_INSTRUKTION = "instruktion"
A_DEMONSTRATION = "demonstration"
A_TESTFRAGE = "testfrage"
A_FILLER = "filler"

# ---------------------------------------------------------------------------
# Promptformat
# ---------------------------------------------------------------------------

INSTRUCTION = ("Answer the question with the name of the entity only. "
               "Do not explain.")

QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = "A: "
PARAPHRASE_PREFIX = "Q: "
BLOCK_SEPARATOR = "\n\n"
ZEILE = "\n"
ANSWER_CUE = "A:"

N_DEMOS_BASIS = 4

# ---------------------------------------------------------------------------
# Stufe instruction
# ---------------------------------------------------------------------------
# Umformulierungen derselben Instruktion in gestaffelter Laenge. Nach
# subsubsec:redundanztypen werden genau zwei davon eingefuegt, eine vor dem
# Demonstrationsblock und eine danach. Welches Paar, entscheidet die
# Kalibrierung. Keine Formulierung wiederholt die Instruktion woertlich, damit
# die Wiederholung nicht lexikalisch erkennbar wird (subsubsec:auszeichnung).
INSTRUCTION_POOL = [
    "Answer with the entity name only.",
    "Reply with the entity name and nothing else.",
    "State only the name of the entity, with no explanation.",
    "Give only the name of the entity as your answer, without any further text.",
    "Your reply should contain the name of the entity and no other words at all.",
    "Respond with the name of the entity by itself, leaving out any sentence "
    "that would surround it.",
    "Write down nothing but the name of the entity; explanations, full "
    "sentences and additional remarks are not wanted here.",
    "The expected answer is the bare name of the entity, so do not add a "
    "sentence, a reason, or any surrounding text to it.",
    "Keep the response to the entity name alone, since any explanation, "
    "restatement or additional sentence would not be part of the expected "
    "answer.",
]

# ---------------------------------------------------------------------------
# Stufe filler
# ---------------------------------------------------------------------------
# Einfuegestelle nach subsubsec:filler: der Absatz zwischen Instruktion und
# Demonstrationsblock.
FILLER_POSITION = "nach_instruktion"

# Aufgabenirrelevanter Fliesstext ohne Wiederholungscharakter. Der Pool ist
# ueber alle Aufgaben identisch; die Kalibrierung waehlt daraus eine Teilmenge.
# Jeder Satz ist eine eigene Einheit, damit der Block satzweise und nicht
# mitten im Satz zugeschnitten wird. Der Text enthaelt bewusst keinen
# Eigennamen, damit keine Zeichenkette als Goldantwort in Betracht kommt (V14).
FILLER_POOL = [
    "The tide rises and falls twice each day along most coastlines.",
    "The pull of the moon drives it, and to a smaller degree the pull of the sun.",
    "Harbour wardens keep tables that list the predicted height for every hour.",
    "Small boats time their departures by them.",
    "In shallow estuaries the water can drain far enough to expose wide banks of mud.",
    "These banks are then crossed on foot by people gathering shellfish.",
    "Mapmakers record the line reached by the mean high water and print it on their charts.",
    "That line, and not the visible edge of the water, marks the legal shore.",
    "Surveyors once measured it with poles and chains over many seasons.",
    "Today a network of gauges reports the level continuously.",
    "The long series is averaged over nineteen years, the period after which the lunar cycle repeats.",
    "Wind and air pressure shift the actual level away from the prediction.",
    "The tables therefore carry a warning that they describe an average sea.",
    "Sailors learn to read the difference from the colour of the water.",
]

# Hoechstzahl der Einheiten, ueber die die Kalibrierung sucht. Fuer filler
# begrenzt sie den Suchraum, fuer die uebrigen Varianten ist sie ohne Wirkung.
MAX_FILLER_SAETZE = 4


# ---------------------------------------------------------------------------
# Zusammenbau
# ---------------------------------------------------------------------------

class Assembler:
    """Sammelt Segmente und fuehrt die Zeichenoffsets mit."""

    def __init__(self):
        self.stuecke = []

    def add(self, label, herkunft, abschnitt, text, sep=BLOCK_SEPARATOR,
            slot=None):
        if not text:
            return
        self.stuecke.append({
            "label": label, "herkunft": herkunft, "abschnitt": abschnitt,
            "text": text, "slot": slot,
            "sep": "" if not self.stuecke else sep,
        })

    def build(self):
        text = ""
        segmente = []
        for ord_, s in enumerate(self.stuecke):
            text += s["sep"]
            start = len(text)
            text += s["text"]
            segmente.append({
                "ord": ord_, "label": s["label"], "herkunft": s["herkunft"],
                "abschnitt": s["abschnitt"], "slot": s["slot"],
                "start_zeichen": start, "ende_zeichen": len(text),
            })
        return text, segmente


def _demo_paar(asm, nr, demo, paraphrase, herkunft, sep=BLOCK_SEPARATOR):
    """Ein Frage-Antwort-Paar, optional mit nachgestellter Paraphrase."""
    label = "demonstration_%d" % nr if herkunft == KERN \
        else "demonstration_red_%d" % nr
    asm.add(label + "_frage", herkunft, A_DEMONSTRATION,
            QUESTION_PREFIX + demo["frage"], sep=sep, slot=demo.get("slot"))
    if paraphrase:
        asm.add("paraphrase_" + label, RED, A_DEMONSTRATION,
                PARAPHRASE_PREFIX + paraphrase, sep=ZEILE,
                slot=demo.get("slot"))
    asm.add(label + "_antwort", herkunft, A_DEMONSTRATION,
            ANSWER_PREFIX + demo["antwort"], sep=ZEILE, slot=demo.get("slot"))


# ---------------------------------------------------------------------------
# Einfuegeeinheiten und Bauparameter
# ---------------------------------------------------------------------------
#
# Ein Bauparameter ("bau") ist ein dict und beschreibt die gewaehlte Teilmenge:
#
#   semantic       {"slots": (0, 1, 3)}   0 = Testfrage, 1..4 = Demonstration
#   instruction    {"paar": (2, 5)}       Indizes in INSTRUCTION_POOL
#   demonstration  {"slots": (0, 2, 3)}   Indizes in material["demos_red"]
#   filler         {"saetze": (0, 4, 9)}  Indizes in FILLER_POOL
#
# einheiten() liefert je moeglicher Einheit ihren Text, damit die Kalibrierung
# die Laengen einmal messen und danach nur noch summieren muss.

def einheiten(variante, material):
    """{kennung: text} aller einfuegbaren Einheiten dieser Variante."""
    if variante == "semantic":
        aus = {}
        if material.get("para_frage"):
            aus[0] = PARAPHRASE_PREFIX + material["para_frage"]
        for nr in range(1, len(material["demos"]) + 1):
            p = material.get("para_demos", {}).get(nr)
            if p:
                aus[nr] = PARAPHRASE_PREFIX + p
        return aus
    if variante == "instruction":
        return {i: t for i, t in enumerate(INSTRUCTION_POOL)}
    if variante == "demonstration":
        return {i: QUESTION_PREFIX + d["frage"] + ZEILE + ANSWER_PREFIX
                + d["antwort"]
                for i, d in enumerate(material.get("demos_red", []))}
    if variante == "filler":
        return {i: t for i, t in enumerate(FILLER_POOL)}
    return {}


def kandidaten(variante, material):
    """Alle zulaessigen Bauparameter, aufsteigend nach Zahl der Einheiten."""
    e = einheiten(variante, material)
    schluessel = sorted(e)
    if variante == "instruction":
        # genau zwei Formulierungen, eine vor und eine nach dem Block
        return [{"paar": (i, j)}
                for i, j in itertools.permutations(schluessel, 2)]
    if variante == "filler":
        aus = []
        for r in range(1, MAX_FILLER_SAETZE + 1):
            aus += [{"saetze": c} for c in itertools.combinations(schluessel, r)]
        return aus
    aus = []
    for r in range(1, len(schluessel) + 1):
        aus += [{"slots": c} for c in itertools.combinations(schluessel, r)]
    return aus


def bau_einheiten(variante, bau):
    """Kennungen der in bau gewaehlten Einheiten."""
    if not bau:
        return ()
    if variante == "instruction":
        return tuple(bau["paar"])
    if variante == "filler":
        return tuple(bau["saetze"])
    return tuple(bau["slots"])


# ---------------------------------------------------------------------------

def baue(variante, material, bau=None):
    """Baut eine Variante. Rueckgabe: (text, segmente) oder None.

    material erwartet:
        frage            Text der Testfrage
        demos            Liste der vier Basisdemonstrationen
        demos_red        Liste der Redundanzdemonstrationen
        para_frage       Paraphrase der Testfrage
        para_demos       {nr: Paraphrase} der Basisdemonstrationen
    bau  Bauparameter nach kandidaten(); None nur fuer die Variante basis.
    """
    gewaehlt = set(bau_einheiten(variante, bau))
    if variante != "basis" and not gewaehlt:
        return None

    asm = Assembler()
    asm.add("instruktion", KERN, A_INSTRUKTION, INSTRUCTION)

    if variante == "instruction":
        vor, nach = bau["paar"]
        asm.add("instruktion_red_vor", RED, A_INSTRUKTION,
                INSTRUCTION_POOL[vor], sep=ZEILE)

    if variante == "filler" and FILLER_POSITION == "nach_instruktion":
        for i in sorted(gewaehlt):
            asm.add("filler_%d" % i, RED, A_FILLER, FILLER_POOL[i],
                    sep=BLOCK_SEPARATOR if i == min(gewaehlt) else " ")

    for nr, demo in enumerate(material["demos"], start=1):
        p = material.get("para_demos", {}).get(nr) \
            if (variante == "semantic" and nr in gewaehlt) else None
        _demo_paar(asm, nr, demo, p, KERN)

    if variante == "demonstration":
        for nr, i in enumerate(sorted(gewaehlt), start=1):
            _demo_paar(asm, nr, material["demos_red"][i], None, RED)

    if variante == "instruction":
        asm.add("instruktion_red_nach", RED, A_INSTRUKTION,
                INSTRUCTION_POOL[bau["paar"][1]])

    if variante == "filler" and FILLER_POSITION == "vor_frage":
        for i in sorted(gewaehlt):
            asm.add("filler_%d" % i, RED, A_FILLER, FILLER_POOL[i],
                    sep=BLOCK_SEPARATOR if i == min(gewaehlt) else " ")

    asm.add("frage", KERN, A_TESTFRAGE, QUESTION_PREFIX + material["frage"])
    if variante == "semantic" and 0 in gewaehlt:
        if not material.get("para_frage"):
            return None
        asm.add("paraphrase_frage", RED, A_TESTFRAGE,
                PARAPHRASE_PREFIX + material["para_frage"], sep=ZEILE)
    asm.add("frage_cue", KERN, A_TESTFRAGE, ANSWER_CUE, sep=ZEILE)

    return asm.build()


# ---------------------------------------------------------------------------
# Kennzahlen der Konstruktion
# ---------------------------------------------------------------------------

WORTFORM_RE = re.compile(r"\w+", re.UNICODE)


def segmenttext(text, seg):
    return text[seg["start_zeichen"]:seg["ende_zeichen"]]


def ws_token_nach_herkunft(text, segmente):
    """Whitespace-Token getrennt nach Kern und Redundanz."""
    kern = sum(len(segmenttext(text, s).split())
               for s in segmente if s["herkunft"] == KERN)
    red = sum(len(segmenttext(text, s).split())
              for s in segmente if s["herkunft"] == RED)
    return kern, red


def lexikalische_ueberlappung(text, segmente):
    """Anteil der Wortformen der Redundanz, die auch im Kern vorkommen (V13).

    Gemessen als Typenanteil, nicht als Vorkommensanteil: gefragt ist, wie
    weit sich die eingefuegte Einheit lexikalisch mit dem Kern deckt, nicht
    wie oft ein geteiltes Funktionswort auftritt.
    """
    kern = set()
    red = set()
    for s in segmente:
        ziel = kern if s["herkunft"] == KERN else red
        ziel.update(w.casefold()
                    for w in WORTFORM_RE.findall(segmenttext(text, s)))
    if not red:
        return None
    return len(red & kern) / len(red)


def normalisiere(text):
    """Vereinheitlicht fuer den Abgleich: Kleinschreibung, keine Satzzeichen,
    keine englischen Artikel (subsubsec:korrektheit)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def antwortleckage(text, segmente, antworten):
    """Kommt eine Goldantwort ausserhalb des Frageblocks vor (V14)?

    Rueckgabe: (0/1, Abschnitt des ersten Fundes oder None).
    """
    for s in segmente:
        if s["abschnitt"] == A_TESTFRAGE:
            continue
        norm = " %s " % normalisiere(segmenttext(text, s))
        for a in antworten:
            an = normalisiere(a)
            if an and (" %s " % an) in norm:
                return 1, s["abschnitt"]
    return 0, None
