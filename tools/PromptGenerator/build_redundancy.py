#!/usr/bin/env python3
"""
Redundanzvarianten fuer die PopQA-Prompts (Abschnitt 4.5).

Liest alle fertigen Basisprompts eines Ordners, zerlegt sie in Instruktion,
Demonstrationen und Testfrage und schreibt je Variante eine eigene Fassung mit
eingefuegter Redundanz in einen Unterordner des Ausgabeordners.

Varianten:
  basis          unveraendert, Referenz
  semantisch     jeder Frage wird eine Paraphrase aus PopQA-TP nachgestellt
  instruktion    zwei zusaetzliche Formulierungen der Instruktion
  demonstration  vier zusaetzliche Frage-Antwort-Paare (Slots 5 bis 8)
  filler         aufgabenirrelevanter Fliesstext gleicher Tokenmenge

Je Variante entsteht eine manifest.tsv mit Zeichenoffsets und der Zuordnung
jedes Segments zu Kern oder Redundanz. Das ist die Grundlage fuer den
Selektivitaetsindex S(rho) = R_kern(rho) / R_red(rho).

Aufruf: python3 build_redundancy.py
Alle Einstellungen stehen im Konfigurationsblock.
"""

import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Ordner mit den Basisprompts (Ausgabe von build_prompts.py).
INPUT_DIR = "../../experiments/PopQA/subset_A/1_prompts"

# Zielordner. Je Variante entsteht darin ein Unterordner.
OUTPUT_DIR = "../../experiments/PopQA/subset_A/2_redundancy"

# Zu bauende Varianten. filler wird stets zuletzt gebaut, weil seine Laenge
# aus den anderen Stufen folgt.
VARIANTS = ("basis", "semantisch", "instruktion", "demonstration", "filler")

# Materialdateien. Relative Pfade beziehen sich auf das Skriptverzeichnis.
DEMO_FILE = "demo_assignments.tsv"
PARAPHRASE_FILE = "../../datasets/popQA/popQA_template_paraphrases.csv"

# ---- Stufe instruktion ----------------------------------------------------
# Zwei zusaetzliche Formulierungen derselben Instruktion, der Originalinstruktion
# nachgestellt.
INSTRUCTION_VARIANTS = [
    "Reply with the entity name and nothing else. No explanation is required.",
    "Give only the name of the entity as your answer, without any further text.",
]

# ---- Stufe demonstration --------------------------------------------------
# Slots der Redundanzdemonstrationen in demo_assignments.tsv.
DEMO_SLOTS_RED = (5, 6, 7, 8)

# ---- Stufe semantisch -----------------------------------------------------
# Auch den Demonstrationsfragen wird eine Paraphrase nachgestellt.
PARAPHRASE_DEMOS = True
# template_id 0 ist die Originalformulierung und scheidet als Paraphrase aus.
EXCLUDE_TEMPLATE_IDS = (0,)
# In PopQA-TP ist template_id 4 bei 838 Fragen textgleich zur Originalfrage
# (betrifft vor allem die Kategorie country). Solche Kandidaten verwerfen,
# sonst stuende die Frage zweimal wortgleich im Prompt.
DROP_IDENTICAL_TO_ORIGINAL = True
# Auswahl je Frage: "rotate" waehlt deterministisch ueber die Frage-ID,
# "first" nimmt stets die kleinste zulaessige template_id.
PARAPHRASE_PICK = "rotate"
# Praefix der nachgestellten Paraphrasezeile.
PARAPHRASE_PREFIX = "Q: "
# popQA_template_paraphrases.csv liegt in verstuemmelter Kodierung vor: die
# UTF-8-Bytes der Namen wurden einmal als MacRoman gelesen, aus Detiege wird
# Deti√®ge. popQA.tsv ist davon nicht betroffen. Der Schalter macht diesen
# Schritt beim Einlesen rueckgaengig.
REPAIR_MOJIBAKE = True

# ---- Stufe filler ---------------------------------------------------------
# Zieltokenzahl des Fliesstextes:
#   "mittel"        Mittel der eingefuegten Tokenmengen der drei anderen Stufen
#   "demonstration" so viele Token wie der Demonstrationsblock derselben Aufgabe
#   ganze Zahl      feste Tokenzahl fuer alle Aufgaben
FILLER_TARGET = "mittel"
# Einfuegestelle: "vor_frage" oder "nach_instruktion".
FILLER_POSITION = "vor_frage"
# Aufgabenirrelevanter Fliesstext, zyklisch auf die Zieltokenzahl zugeschnitten.
FILLER_TEXT = (
    "The tide rises and falls twice each day along most coastlines, driven by "
    "the pull of the moon and, to a smaller degree, of the sun. Harbour "
    "wardens keep tide tables that list the predicted heights for every hour, "
    "and small boats time their departures by them. In shallow estuaries the "
    "water can drain far enough to expose wide banks of mud, which are then "
    "crossed on foot by people gathering shellfish. Cartographers record the "
    "line reached by the mean high water and print it on their charts, "
    "because that line and not the visible edge of the water marks the legal "
    "shore. Surveyors once measured it with poles and chains over many "
    "seasons; today a network of gauges reports the level continuously and "
    "the long series is averaged over nineteen years, the period after which "
    "the lunar cycle repeats. Wind and air pressure shift the actual water "
    "level away from the prediction, sometimes by more than a metre, so the "
    "tables carry a warning that they describe an average sea and not the sea "
    "of any particular morning. Sailors learn to read the difference between "
    "the two from the colour of the water and the set of the buoys."
)

# ---- Promptformat (muss zu build_prompts.py passen) -----------------------
QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = "A: "
BLOCK_SEPARATOR = "\n\n"
ANSWER_CUE = "A:"

INPUT_SUFFIX = ".txt"
OUTPUT_SUFFIX = ".txt"
MANIFEST_FILE = "manifest.tsv"

# Abbruch statt Warnung bei fehlendem Material fuer eine Aufgabe.
STRICT = True

ENCODING = "utf-8"

# ---------------------------------------------------------------------------
# Ab hier keine Einstellungen mehr
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KERN = "kern"
RED = "redundanz"


def resolve(path_value):
    p = Path(path_value)
    return p if p.is_absolute() else BASE_DIR / p


def read_tsv(path):
    with open(path, encoding=ENCODING) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield dict(zip(header, line.split("\t")))


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def load_demos(path):
    """task_id -> {"red": Liste der Redundanzslots, "ids": Frage -> demo_id}."""
    out = {}
    for row in read_tsv(path):
        eintrag = out.setdefault(row["task_id"], {"red": [], "ids": {}})
        eintrag["ids"][row["demo_question"]] = row["demo_id"]
        if int(row["slot"]) in DEMO_SLOTS_RED:
            eintrag["red"].append({
                "slot": int(row["slot"]),
                "prop": row["demo_prop"],
                "question": row["demo_question"],
                "answer": row["demo_answer"],
            })
    for k in out:
        out[k]["red"].sort(key=lambda r: r["slot"])
    return out


def repariere_kodierung(text):
    """MacRoman-Fehllesung rueckgaengig machen. Rueckgabe: (text, geaendert)."""
    if not REPAIR_MOJIBAKE or not any(c in text for c in "\u221a\u00ac\u00c3\u00e2"):
        return text, False
    try:
        neu = text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False
    return neu, neu != text


def load_paraphrases(path):
    """Frage-ID -> Liste von (template_id, paraphrase), ohne die Ausschluesse.

    Erwartet die Spalten paraphrase, template_id und id, also das Format von
    popQA_template_paraphrases.csv (PopQA-TP, Rabinovich et al. 2023).
    """
    out = {}
    with open(path, encoding=ENCODING, newline="") as fh:
        leser = csv.DictReader(fh)
        for feld in ("paraphrase", "template_id", "id"):
            if feld not in leser.fieldnames:
                sys.exit("Spalte %s fehlt in %s" % (feld, path))
        original = {}
        repariert = 0
        for row in leser:
            tid = int(row["template_id"])
            text, geaendert = repariere_kodierung(row["paraphrase"])
            repariert += 1 if geaendert else 0
            if tid == 0:
                original[row["id"]] = text
            if tid in EXCLUDE_TEMPLATE_IDS:
                continue
            out.setdefault(row["id"], []).append((tid, text))
        if repariert:
            print("Kodierung repariert: %d Paraphrasen in %s"
                  % (repariert, Path(path).name))
    for k in out:
        if DROP_IDENTICAL_TO_ORIGINAL and k in original:
            out[k] = [t for t in out[k] if t[1] != original[k]]
        out[k].sort(key=lambda t: t[0])
    return {k: v for k, v in out.items() if v}


def paraphrase_fuer(frage_id, paraphrasen):
    """Deterministisch eine Paraphrase zu einer Frage-ID waehlen."""
    kandidaten = paraphrasen.get(str(frage_id))
    if not kandidaten:
        return None
    if PARAPHRASE_PICK == "first":
        return kandidaten[0][1]
    try:
        schluessel = int(frage_id)
    except ValueError:
        schluessel = sum(ord(c) for c in str(frage_id))
    return kandidaten[schluessel % len(kandidaten)][1]


# ---------------------------------------------------------------------------
# Prompt zerlegen und wieder zusammensetzen
# ---------------------------------------------------------------------------

def parse_prompt(text):
    """Basisprompt in Instruktion, Demonstrationen und Frage zerlegen."""
    bloecke = [b for b in text.split(BLOCK_SEPARATOR) if b.strip()]
    if not bloecke:
        return None, [], None
    instruktion = None
    if not bloecke[0].startswith(QUESTION_PREFIX):
        instruktion = bloecke.pop(0)
    if not bloecke:
        return instruktion, [], None
    frage_block = bloecke.pop()
    demos = []
    for b in bloecke:
        zeilen = b.split("\n")
        antwort = ""
        for z in zeilen[1:]:
            if z.startswith(ANSWER_PREFIX):
                antwort = z[len(ANSWER_PREFIX):]
        demos.append({"question": zeilen[0][len(QUESTION_PREFIX):],
                      "answer": antwort})
    frage = frage_block.split("\n")[0][len(QUESTION_PREFIX):]
    return instruktion, demos, frage


class Assembler:
    """Sammelt Textstuecke mit Rolle und fuehrt die Zeichenoffsets mit."""

    def __init__(self):
        self.stuecke = []

    def add(self, label, rolle, text, sep=BLOCK_SEPARATOR):
        self.stuecke.append({"label": label, "rolle": rolle, "text": text,
                             "sep": "" if not self.stuecke else sep})

    def build(self):
        text = ""
        segmente = []
        for s in self.stuecke:
            text += s["sep"]
            start = len(text)
            text += s["text"]
            segmente.append({"label": s["label"], "rolle": s["rolle"],
                             "start": start, "end": len(text)})
        return text, segmente


def demo_block(asm, index, demo, paraphrase=None, rolle=KERN, label=None):
    """Ein Frage-Antwort-Paar anhaengen, optional mit nachgestellter Paraphrase."""
    label = label or ("demonstration_%d" % index)
    asm.add(label, rolle, QUESTION_PREFIX + demo["question"])
    if paraphrase:
        asm.add("paraphrase_" + label, RED, PARAPHRASE_PREFIX + paraphrase,
                sep="\n")
    asm.add(label + "_antwort", rolle, ANSWER_PREFIX + demo["answer"], sep="\n")


def filler_block(n_token):
    """Fliesstext auf n_token Whitespace-Token zuschneiden, zyklisch ab Offset 0."""
    token = FILLER_TEXT.split()
    if n_token <= 0 or not token:
        return ""
    return " ".join(token[i % len(token)] for i in range(n_token))


def baue_variante(variante, instruktion, demos, frage, ctx):
    """Prompt einer Variante bauen. Rueckgabe: (text, segmente) oder None."""
    asm = Assembler()

    if instruktion is not None:
        asm.add("instruktion", KERN, instruktion)
        if variante == "instruktion":
            for i, zusatz in enumerate(INSTRUCTION_VARIANTS, start=1):
                asm.add("instruktion_red_%d" % i, RED, zusatz, sep="\n")

    if variante == "filler" and FILLER_POSITION == "nach_instruktion":
        asm.add("filler", RED, ctx["filler_text"])

    for i, demo in enumerate(demos, start=1):
        p = (ctx.get("para_demos", {}).get(i)
             if variante == "semantisch" and PARAPHRASE_DEMOS else None)
        demo_block(asm, i, demo, paraphrase=p)

    if variante == "demonstration":
        for j, demo in enumerate(ctx["demos_red"], start=1):
            demo_block(asm, j, demo, rolle=RED, label="demonstration_red_%d" % j)

    if variante == "filler" and FILLER_POSITION == "vor_frage":
        asm.add("filler", RED, ctx["filler_text"])

    asm.add("frage", KERN, QUESTION_PREFIX + frage)
    if variante == "semantisch":
        if not ctx.get("para_frage"):
            return None
        asm.add("paraphrase_frage", RED, PARAPHRASE_PREFIX + ctx["para_frage"],
                sep="\n")
    asm.add("frage_cue", KERN, ANSWER_CUE, sep="\n")
    return asm.build()


def kennzahlen(text, segmente):
    kern = sum(len(text[s["start"]:s["end"]].split())
               for s in segmente if s["rolle"] == KERN)
    red = sum(len(text[s["start"]:s["end"]].split())
              for s in segmente if s["rolle"] == RED)
    return kern, red


# ---------------------------------------------------------------------------

def main():
    in_dir = resolve(INPUT_DIR)
    out_root = resolve(OUTPUT_DIR)
    if not in_dir.is_dir():
        sys.exit("Eingabeordner nicht gefunden: %s" % in_dir)

    varianten = list(VARIANTS)
    warnungen = []

    demos_tab = {}
    if "demonstration" in varianten or "semantisch" in varianten:
        p = resolve(DEMO_FILE)
        if not p.is_file():
            sys.exit("Demonstrationstabelle nicht gefunden: %s" % p)
        demos_tab = load_demos(p)

    paraphrasen = {}
    if "semantisch" in varianten:
        pp = resolve(PARAPHRASE_FILE)
        if pp.is_file():
            paraphrasen = load_paraphrases(pp)
        else:
            varianten.remove("semantisch")
            warnungen.append("Stufe semantisch uebersprungen, Paraphrasentabelle "
                             "fehlt: %s" % pp)

    if "filler" in varianten:
        varianten = [v for v in varianten if v != "filler"] + ["filler"]

    dateien = sorted(p for p in in_dir.iterdir()
                     if p.is_file() and p.suffix == INPUT_SUFFIX
                     and p.name != MANIFEST_FILE)
    if not dateien:
        sys.exit("Keine Dateien mit Endung %s in %s" % (INPUT_SUFFIX, in_dir))

    for v in varianten:
        (out_root / v).mkdir(parents=True, exist_ok=True)

    manifeste = {v: [] for v in varianten}

    for src in dateien:
        task_id = src.stem
        instruktion, demos, frage = parse_prompt(src.read_text(encoding=ENCODING))
        if frage is None:
            warnungen.append("%s ohne erkennbare Frage, uebersprungen" % src.name)
            continue

        ctx = {}
        eintrag = demos_tab.get(task_id, {"red": [], "ids": {}})

        if "demonstration" in varianten:
            if len(eintrag["red"]) < len(DEMO_SLOTS_RED):
                text = ("Aufgabe %s hat nur %d der Slots %s in %s"
                        % (task_id, len(eintrag["red"]), DEMO_SLOTS_RED, DEMO_FILE))
                if STRICT:
                    sys.exit(text)
                warnungen.append(text)
            ctx["demos_red"] = eintrag["red"][:len(DEMO_SLOTS_RED)]

        if "semantisch" in varianten:
            ctx["para_frage"] = paraphrase_fuer(task_id, paraphrasen)
            if ctx["para_frage"] is None:
                text = "keine Paraphrase zu Aufgabe %s in %s" % (task_id,
                                                                 PARAPHRASE_FILE)
                if STRICT:
                    sys.exit(text)
                warnungen.append(text)
            para_demos = {}
            if PARAPHRASE_DEMOS:
                for i, d in enumerate(demos, start=1):
                    demo_id = eintrag["ids"].get(d["question"])
                    if demo_id is None:
                        warnungen.append("Demonstration %d der Aufgabe %s nicht "
                                         "in %s gefunden" % (i, task_id, DEMO_FILE))
                        continue
                    p = paraphrase_fuer(demo_id, paraphrasen)
                    if p:
                        para_demos[i] = p
                    else:
                        warnungen.append("keine Paraphrase zu Demonstration %s "
                                         "der Aufgabe %s" % (demo_id, task_id))
            ctx["para_demos"] = para_demos

        gebaut = {}
        for v in varianten:
            if v == "filler":
                continue
            ergebnis = baue_variante(v, instruktion, demos, frage, ctx)
            if ergebnis is None:
                warnungen.append("Variante %s fuer Aufgabe %s nicht baubar"
                                 % (v, task_id))
                continue
            gebaut[v] = ergebnis

        if "filler" in varianten:
            eingefuegt = {v: kennzahlen(t, s)[1] for v, (t, s) in gebaut.items()
                          if v != "basis"}
            if isinstance(FILLER_TARGET, int):
                ziel = FILLER_TARGET
            elif FILLER_TARGET == "demonstration":
                ziel = eingefuegt.get("demonstration", 0)
            else:
                ziel = (round(sum(eingefuegt.values()) / len(eingefuegt))
                        if eingefuegt else 0)
            ctx["filler_text"] = filler_block(ziel)
            gebaut["filler"] = baue_variante("filler", instruktion, demos,
                                             frage, ctx)

        basis_token = (len(gebaut["basis"][0].split())
                       if "basis" in gebaut else None)

        for v, (text, segmente) in gebaut.items():
            (out_root / v / (task_id + OUTPUT_SUFFIX)).write_text(
                text, encoding=ENCODING)
            kern, red = kennzahlen(text, segmente)
            manifeste[v].append({
                "prompt_id": task_id,
                "variante": v,
                "n_zeichen": len(text),
                "n_ws_token": len(text.split()),
                "n_ws_token_kern": kern,
                "n_ws_token_red": red,
                "delta_ws_token": (len(text.split()) - basis_token
                                   if basis_token is not None else ""),
                "segmente": json.dumps(segmente, ensure_ascii=False),
            })

    for v in varianten:
        rows = manifeste[v]
        if not rows:
            continue
        spalten = list(rows[0].keys())
        with open(out_root / v / MANIFEST_FILE, "w", encoding=ENCODING) as fh:
            fh.write("\t".join(spalten) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in spalten) + "\n")

    for w in warnungen:
        print("Warnung: " + w, file=sys.stderr)
    print("Eingabe: %s (%d Prompts)" % (in_dir, len(dateien)))
    print("Ausgabe: %s" % out_root)
    print("%-14s %8s %8s %8s" % ("variante", "n", "median", "median_red"))
    for v in varianten:
        rows = manifeste[v]
        if not rows:
            continue
        ges = sorted(r["n_ws_token"] for r in rows)
        red = sorted(r["n_ws_token_red"] for r in rows)
        print("%-14s %8d %8d %8d" % (v, len(rows), ges[len(ges) // 2],
                                     red[len(red) // 2]))


if __name__ == "__main__":
    main()
