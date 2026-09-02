#!/usr/bin/env python3
"""
Schritt 1: Datenbank anlegen und Materialbasis importieren.

Liest die Ergebnisse der Stichprobenziehung (subsubsec:stichprobe) und legt sie
in der Datenbank ab:

    sample_a.tsv          -> frage           (300 Aufgaben der Teilstudie A)
    sample_b.tsv          -> frage.in_teilstudie_b = 1 (30 Aufgaben)
    demo_assignments.tsv  -> demonstration   (300 x 8 fixierte Slots)
    popQA_template_paraphrases.csv -> paraphrase (Material der Stufe semantic)

Der Import ist idempotent: ein zweiter Lauf ersetzt den Bestand.

Aufruf: python3 01_init_db.py [--ohne-paraphrasen]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio                                              # noqa: E402

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPO = BASE_DIR.parent

SAMPLE_A = REPO / "sample_a.tsv"
SAMPLE_B = REPO / "sample_b.tsv"
DEMO_FILE = REPO / "tools" / "PromptGenerator" / "demo_assignments.tsv"
PARAPHRASE_FILE = REPO / "datasets" / "popQA" / "popQA_template_paraphrases.csv"

# Slots 1 bis 4 gehoeren zum Basisprompt, 5 bis 8 zur Redundanzstufe.
SLOTS_BASIS = (1, 2, 3, 4)
SLOTS_RED = (5, 6, 7, 8)

# template_id 0 ist die Originalformulierung und scheidet als Paraphrase aus.
ORIGINAL_TEMPLATE_ID = 0

# popQA_template_paraphrases.csv liegt in verstuemmelter Kodierung vor: die
# UTF-8-Bytes wurden einmal als MacRoman gelesen, aus Detiege wird Deti√®ge.
REPAIR_MOJIBAKE = True
MOJIBAKE_MARKER = "√¬Ãâ"

ENCODING = "utf-8"


# ---------------------------------------------------------------------------

def repariere_kodierung(text):
    """MacRoman-Fehllesung rueckgaengig machen. Rueckgabe: (text, geaendert)."""
    if not REPAIR_MOJIBAKE or not any(c in text for c in MOJIBAKE_MARKER):
        return text, False
    try:
        neu = text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False
    return neu, neu != text


def lies_tsv(pfad):
    with open(pfad, newline="", encoding=ENCODING) as fh:
        for zeile in csv.DictReader(fh, delimiter="\t"):
            yield zeile


def antwortformen(rohwert):
    """possible_answers steht als JSON-Liste in der TSV."""
    try:
        werte = json.loads(rohwert)
    except (ValueError, TypeError):
        return [rohwert] if rohwert else []
    return [str(w) for w in werte] if isinstance(werte, list) else [str(werte)]


# ---------------------------------------------------------------------------

def importiere_fragen(con, lauf_id):
    if not SAMPLE_A.is_file():
        sys.exit("Stichprobe nicht gefunden: %s" % SAMPLE_A)

    b_ids = set()
    if SAMPLE_B.is_file():
        b_ids = {z["id"].strip() for z in lies_tsv(SAMPLE_B)}

    zeilen = []
    for z in lies_tsv(SAMPLE_A):
        fid = z["id"].strip()
        antworten = antwortformen(z.get("possible_answers", ""))
        kuerzeste = min((len(a.split()) for a in antworten), default=None)
        zeilen.append((
            fid,
            1,
            1 if fid in b_ids or z.get("in_subset_b") == "1" else 0,
            z.get("prop", ""),
            z.get("prop_id", ""),
            z.get("subj", ""),
            z.get("obj", ""),
            int(z["s_pop"]) if z.get("s_pop", "").strip().isdigit() else None,
            int(z["pop_tercile"]) if z.get("pop_tercile", "").strip().isdigit() else None,
            z.get("question", ""),
            json.dumps(antworten, ensure_ascii=False),
            len(antworten),
            kuerzeste,
            int(z["n_paraphrases"]) if z.get("n_paraphrases", "").strip().isdigit() else None,
            dbio.ws_token(z.get("question", "")),
            lauf_id,
        ))

    con.execute("DELETE FROM frage")
    con.executemany(
        "INSERT INTO frage (frage_id, in_teilstudie_a, in_teilstudie_b, "
        "relation, relation_id, subjekt, objekt, s_pop, pop_terzil, fragetext, "
        "antworten, n_antwortformen, laenge_goldantwort_ws, n_paraphrasen, "
        "n_ws_token_frage, lauf_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", zeilen)
    con.commit()
    return len(zeilen)


def importiere_demonstrationen(con, lauf_id):
    if not DEMO_FILE.is_file():
        sys.exit("Demonstrationstabelle nicht gefunden: %s" % DEMO_FILE)

    bekannte = {r[0] for r in con.execute("SELECT frage_id FROM frage")}
    zeilen = []
    fremd = 0
    for z in lies_tsv(DEMO_FILE):
        tid = z["task_id"].strip()
        if tid not in bekannte:
            fremd += 1
            continue
        slot = int(z["slot"])
        zeilen.append((
            tid, slot,
            "base" if slot in SLOTS_BASIS else "demonstration",
            z["demo_id"].strip(), z["demo_prop"],
            z["demo_question"], z["demo_answer"],
        ))

    con.execute("DELETE FROM demonstration")
    con.executemany(
        "INSERT INTO demonstration (frage_id, slot, rolle, demo_id, "
        "demo_relation, demo_frage, demo_antwort) VALUES (?,?,?,?,?,?,?)",
        zeilen)
    con.commit()
    if fremd:
        print("Hinweis: %d Zeilen ohne zugehoerige Aufgabe uebersprungen"
              % fremd, file=sys.stderr)
    return len(zeilen)


def importiere_paraphrasen(con, lauf_id):
    """Nur die Paraphrasen zu Test- und Demonstrationsfragen der Stichprobe.

    Die vollstaendige Datei umfasst ueber 118000 Zeilen. Gebraucht wird davon
    nur, was in einem Prompt auftauchen kann.
    """
    if not PARAPHRASE_FILE.is_file():
        print("Hinweis: Paraphrasentabelle fehlt, Stufe semantic nicht baubar: "
              "%s" % PARAPHRASE_FILE, file=sys.stderr)
        return 0

    gebraucht = {r[0] for r in con.execute("SELECT frage_id FROM frage")}
    gebraucht |= {r[0] for r in con.execute("SELECT demo_id FROM demonstration")}

    original = {}
    kandidaten = []
    repariert = 0

    with open(PARAPHRASE_FILE, encoding=ENCODING, newline="") as fh:
        leser = csv.DictReader(fh)
        for feld in ("paraphrase", "template_id", "id"):
            if feld not in (leser.fieldnames or []):
                sys.exit("Spalte %s fehlt in %s" % (feld, PARAPHRASE_FILE))
        for z in leser:
            qid = str(z["id"]).strip()
            if qid not in gebraucht:
                continue
            tid = int(z["template_id"])
            text, geaendert = repariere_kodierung(z["paraphrase"])
            repariert += 1 if geaendert else 0
            if tid == ORIGINAL_TEMPLATE_ID:
                original[qid] = text
            kandidaten.append((qid, tid, text, 1 if geaendert else 0))

    zeilen = []
    for qid, tid, text, geaendert in kandidaten:
        identisch = 1 if original.get(qid) == text and tid != ORIGINAL_TEMPLATE_ID else 0
        verwendbar = 0 if (tid == ORIGINAL_TEMPLATE_ID or identisch) else 1
        zeilen.append((qid, tid, text, geaendert, identisch, verwendbar))

    con.execute("DELETE FROM paraphrase")
    con.executemany(
        "INSERT OR REPLACE INTO paraphrase (quelle_id, template_id, text, "
        "kodierung_repariert, identisch_zum_original, verwendbar) "
        "VALUES (?,?,?,?,?,?)", zeilen)
    con.commit()

    if repariert:
        print("Kodierung repariert: %d Paraphrasen" % repariert)
    ohne = sum(1 for f in gebraucht
               if not con.execute("SELECT 1 FROM paraphrase WHERE quelle_id = ? "
                                  "AND verwendbar = 1 LIMIT 1", (f,)).fetchone())
    if ohne:
        print("Achtung: %d Fragen ohne verwendbare Paraphrase" % ohne,
              file=sys.stderr)
    return len(zeilen)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None, help="Pfad der Datenbankdatei")
    p.add_argument("--ohne-paraphrasen", action="store_true")
    args = p.parse_args()

    con = dbio.verbinde(args.db)
    dbio.schema_anlegen(con)

    lauf_id = dbio.lauf_beginnen(con, "init", __file__, konfiguration={
        "sample_a": str(SAMPLE_A), "sample_b": str(SAMPLE_B),
        "demo_file": str(DEMO_FILE), "paraphrase_file": str(PARAPHRASE_FILE),
        "repair_mojibake": REPAIR_MOJIBAKE,
    })

    n_f = importiere_fragen(con, lauf_id)
    n_d = importiere_demonstrationen(con, lauf_id)
    n_p = 0 if args.ohne_paraphrasen else importiere_paraphrasen(con, lauf_id)

    dbio.lauf_beenden(con, lauf_id, n_f + n_d + n_p)

    n_b = dbio.skalar(con, "SELECT COUNT(*) FROM frage WHERE in_teilstudie_b = 1")
    print("Datenbank: %s" % (args.db or dbio.DB_PFAD))
    print("Fragen:          %5d  (davon Teilstudie B: %d)" % (n_f, n_b))
    print("Demonstrationen: %5d  (%d je Aufgabe)"
          % (n_d, n_d // n_f if n_f else 0))
    print("Paraphrasen:     %5d" % n_p)
    print()
    print("%-14s %6s %6s %6s %6s" % ("Relation", "n", "T1", "T2", "T3"))
    for z in con.execute(
            "SELECT relation, COUNT(*) n, "
            "SUM(pop_terzil=1) t1, SUM(pop_terzil=2) t2, SUM(pop_terzil=3) t3 "
            "FROM frage GROUP BY relation ORDER BY relation"):
        print("%-14s %6d %6d %6d %6d" % (z["relation"], z["n"], z["t1"],
                                         z["t2"], z["t3"]))
    con.close()


if __name__ == "__main__":
    main()
