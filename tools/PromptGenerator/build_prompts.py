#!/usr/bin/env python3
"""
Promptbau fuer die PopQA-Stichprobe (Abschnitt 4.4.x).

Liest alle Fragedateien eines Eingabeordners, erzeugt je Frage einen Prompt aus
Instruktion, N Demonstrationen und der Testfrage und schreibt ihn als Textdatei
in den Ausgabeordner. Zusaetzlich entsteht eine Manifestdatei mit Kennzahlen und
den Zeichenoffsets der Segmente (Instruktion, Demonstration 1..N, Frage).

Die Demonstrationen stammen aus der einmal gezogenen und ueber alle Varianten
konstanten Zuweisung in demo_assignments.tsv. Slots werden aufsteigend belegt,
Slots 1..4 gehoeren zum Basisprompt, Slots 5..8 zur Redundanzstufe
"demonstration". N_DEMOS = 4 erzeugt also den Basisprompt, N_DEMOS = 8 die
Redundanzvariante.

Aufruf: python3 build_prompts.py
Alle Einstellungen stehen im Konfigurationsblock.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Ordner mit den Fragedateien. Dateiname ohne Endung = Aufgaben-ID,
# jede nicht leere Zeile in der Datei gilt als eine Frage.
INPUT_DIR = "../../experiments/PopQA/subset_A/0_questions"

# Zielordner fuer die fertigen Prompts. Wird angelegt, falls nicht vorhanden.
OUTPUT_DIR = "../../experiments/PopQA/subset_A/1_prompts"

# Tabelle mit der fixierten Demonstrationszuweisung.
DEMO_FILE = "demo_assignments.tsv"

# Anzahl der Demonstrationen je Prompt. 0 bis 8.
N_DEMOS = 4

# Instruktion. Steht als erster Block im Prompt. Leerer String laesst sie weg.
INSTRUCTION = "Answer the question with the name of the entity only. Do not explain."

# Aufbau der Frage-Antwort-Paare und des Abschlusses.
QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = "A: "
BLOCK_SEPARATOR = "\n\n"      # zwischen Instruktion, Demonstrationen und Frage
APPEND_ANSWER_CUE = True      # haengt "A:" nach der Testfrage an
ANSWER_CUE = "A:"
TRAILING_NEWLINE = False      # Zeilenumbruch am Dateiende

# Dateiendungen im Eingabeordner und Endung der Ausgabedateien.
INPUT_SUFFIX = ".txt"
OUTPUT_SUFFIX = ".txt"

# Name der Manifestdatei im Ausgabeordner. Leerer String unterdrueckt sie.
MANIFEST_FILE = "manifest.tsv"

# Abbruch statt Warnung, wenn eine Aufgabe keine oder zu wenige
# Demonstrationen in DEMO_FILE hat.
STRICT = True

ENCODING = "utf-8"

# ---------------------------------------------------------------------------
# Ab hier keine Einstellungen mehr
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def resolve(path_value):
    """Relative Pfade werden auf das Skriptverzeichnis bezogen."""
    p = Path(path_value)
    return p if p.is_absolute() else BASE_DIR / p


def read_demo_assignments(path):
    """demo_assignments.tsv einlesen, Rueckgabe: task_id -> Liste je Slot."""
    demos = {}
    with open(path, encoding=ENCODING) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        needed = ["task_id", "slot", "demo_id", "demo_prop",
                  "demo_question", "demo_answer"]
        missing = [c for c in needed if c not in header]
        if missing:
            sys.exit("Spalten fehlen in %s: %s" % (path, ", ".join(missing)))
        idx = {name: header.index(name) for name in header}
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cells = line.split("\t")
            row = {
                "slot": int(cells[idx["slot"]]),
                "demo_id": cells[idx["demo_id"]],
                "demo_prop": cells[idx["demo_prop"]],
                "question": cells[idx["demo_question"]],
                "answer": cells[idx["demo_answer"]],
            }
            demos.setdefault(cells[idx["task_id"]], []).append(row)
    for task_id in demos:
        demos[task_id].sort(key=lambda r: r["slot"])
    return demos


def read_questions(path):
    """Jede nicht leere Zeile einer Fragedatei ist eine Frage."""
    with open(path, encoding=ENCODING) as fh:
        return [line.strip() for line in fh if line.strip()]


def build_prompt(instruction, demo_rows, question):
    """Prompt zusammensetzen und die Zeichenoffsets der Segmente mitfuehren.

    Rueckgabe: (prompttext, segmentliste). Ein Segment ist
    {"label": ..., "start": ..., "end": ...} mit Offsets in Zeichen.
    """
    parts = []
    segments = []
    cursor = 0

    def add(label, text):
        nonlocal cursor
        if parts:
            cursor += len(BLOCK_SEPARATOR)
        parts.append(text)
        segments.append({"label": label, "start": cursor,
                         "end": cursor + len(text)})
        cursor += len(text)

    if instruction:
        add("instruktion", instruction)

    for i, row in enumerate(demo_rows, start=1):
        block = "%s%s\n%s%s" % (QUESTION_PREFIX, row["question"],
                                ANSWER_PREFIX, row["answer"])
        add("demonstration_%d" % i, block)

    frage = QUESTION_PREFIX + question
    if APPEND_ANSWER_CUE:
        frage += "\n" + ANSWER_CUE
    add("frage", frage)

    text = BLOCK_SEPARATOR.join(parts)
    if TRAILING_NEWLINE:
        text += "\n"
    return text, segments


def main():
    if not 0 <= N_DEMOS <= 8:
        sys.exit("N_DEMOS muss zwischen 0 und 8 liegen, ist %r" % (N_DEMOS,))

    in_dir = resolve(INPUT_DIR)
    out_dir = resolve(OUTPUT_DIR)
    if not in_dir.is_dir():
        sys.exit("Eingabeordner nicht gefunden: %s" % in_dir)

    demos = {}
    if N_DEMOS > 0:
        demo_path = resolve(DEMO_FILE)
        if not demo_path.is_file():
            sys.exit("Demonstrationstabelle nicht gefunden: %s" % demo_path)
        demos = read_demo_assignments(demo_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix == INPUT_SUFFIX)
    if not files:
        sys.exit("Keine Dateien mit Endung %s in %s" % (INPUT_SUFFIX, in_dir))

    manifest = []
    warnungen = []
    n_prompts = 0

    for src in files:
        task_id = src.stem
        fragen = read_questions(src)
        if not fragen:
            warnungen.append("%s enthaelt keine Frage, uebersprungen" % src.name)
            continue

        demo_rows = []
        if N_DEMOS > 0:
            verfuegbar = demos.get(task_id, [])
            if len(verfuegbar) < N_DEMOS:
                text = ("Aufgabe %s hat nur %d Demonstrationen in %s, "
                        "benoetigt werden %d"
                        % (task_id, len(verfuegbar), DEMO_FILE, N_DEMOS))
                if STRICT:
                    sys.exit(text)
                warnungen.append(text)
            demo_rows = verfuegbar[:N_DEMOS]

        for k, frage in enumerate(fragen, start=1):
            name = task_id if len(fragen) == 1 else "%s_%02d" % (task_id, k)
            prompt, segments = build_prompt(INSTRUCTION, demo_rows, frage)
            (out_dir / (name + OUTPUT_SUFFIX)).write_text(prompt,
                                                          encoding=ENCODING)
            manifest.append({
                "prompt_id": name,
                "task_id": task_id,
                "frage_nr": k,
                "quelldatei": src.name,
                "n_demos": len(demo_rows),
                "demo_ids": ",".join(r["demo_id"] for r in demo_rows),
                "demo_props": ",".join(r["demo_prop"] for r in demo_rows),
                "n_zeichen": len(prompt),
                "n_ws_token": len(prompt.split()),
                "segmente": json.dumps(segments, ensure_ascii=False),
            })
            n_prompts += 1

    if MANIFEST_FILE and manifest:
        spalten = list(manifest[0].keys())
        with open(out_dir / MANIFEST_FILE, "w", encoding=ENCODING) as fh:
            fh.write("\t".join(spalten) + "\n")
            for row in manifest:
                fh.write("\t".join(str(row[c]) for c in spalten) + "\n")

    for w in warnungen:
        print("Warnung: " + w, file=sys.stderr)
    print("%d Prompts aus %d Dateien, %d Demonstrationen je Prompt"
          % (n_prompts, len(files), N_DEMOS))
    print("Ausgabe: %s" % out_dir)
    if manifest:
        laengen = [r["n_ws_token"] for r in manifest]
        print("Whitespace-Token je Prompt: min %d, median %d, max %d"
              % (min(laengen), sorted(laengen)[len(laengen) // 2], max(laengen)))


if __name__ == "__main__":
    main()
