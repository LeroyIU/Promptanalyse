#!/usr/bin/env python3
"""
Qualitaetssicherung der erzeugten Prompts (Basisprompts und Redundanzstufen).

Prueft Dateibestand, Kodierung, Aufbau, Uebereinstimmung mit den Quelldateien,
Manifestkonsistenz, Kalibrierung der Stufe filler, Antwortleckage und
Reproduzierbarkeit. Schreibt einen Bericht als Markdown und meldet ueber den
Rueckgabewert, ob Fehler gefunden wurden.

Aufruf: python3 qa_prompts.py
"""

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

BASIS = "../../experiments/PopQA/subset_A"
QUESTIONS_DIR = BASIS + "/0_questions"
PROMPTS_DIR = BASIS + "/1_prompts"
REDUNDANCY_DIR = BASIS + "/2_redundancy"
SAMPLE_FILE = BASIS + "/sample_a.tsv"
DEMO_FILE = "demo_assignments.tsv"
PARAPHRASE_FILE = "../../datasets/popQA/popQA_template_paraphrases.csv"
BUILDER = "build_redundancy.py"

REPORT_FILE = BASIS + "/2_redundancy/qa_report.md"

VARIANTS = ("basis", "semantisch", "instruktion", "demonstration", "filler")
N_DEMOS_BASE = 4
N_DEMOS_RED = 4
N_INSTRUCTION_RED = 2
BEISPIELE = 3          # so viele Fundstellen je Befund im Bericht
PRUEFE_REPRODUKTION = True

BASE_DIR = Path(__file__).resolve().parent
MOJIBAKE = ("√", "Ã", "â€", "¬")


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else BASE_DIR / p


class Bericht:
    def __init__(self):
        self.pruefungen = []

    def add(self, nummer, titel, treffer, n_geprueft, stufe="fehler"):
        """stufe: fehler (muss behoben werden), befund (zu bewerten),
        hinweis (nur zur Kenntnis)."""
        self.pruefungen.append({
            "nummer": nummer, "titel": titel, "treffer": treffer,
            "n": n_geprueft, "stufe": stufe,
        })

    @property
    def fehler(self):
        return sum(len(p["treffer"]) for p in self.pruefungen
                   if p["stufe"] == "fehler")

    @property
    def befunde(self):
        return sum(len(p["treffer"]) for p in self.pruefungen
                   if p["stufe"] == "befund")


def read_tsv_simple(path):
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield dict(zip(header, line.split("\t")))


def read_tsv_quoted(path):
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            yield row


def blocks(text):
    return text.split("\n\n")


def load_builder():
    spec = importlib.util.spec_from_file_location("builder", resolve(BUILDER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    b = Bericht()
    red_root = resolve(REDUNDANCY_DIR)
    builder = load_builder()

    # ---------------- Quellen -------------------------------------------
    sample = {}
    for row in read_tsv_quoted(resolve(SAMPLE_FILE)):
        try:
            antworten = json.loads(row["possible_answers"])
        except (json.JSONDecodeError, TypeError):
            antworten = []
        sample[row["id"]] = {"prop": row["prop"], "subj": row["subj"],
                             "obj": row["obj"], "question": row["question"],
                             "antworten": antworten}

    demos = defaultdict(dict)
    for row in read_tsv_simple(resolve(DEMO_FILE)):
        demos[row["task_id"]][int(row["slot"])] = row

    paraphrasen = builder.load_paraphrases(resolve(PARAPHRASE_FILE))
    para_alle = defaultdict(dict)
    with open(resolve(PARAPHRASE_FILE), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text, _ = builder.repariere_kodierung(row["paraphrase"])
            para_alle[row["id"]][int(row["template_id"])] = text

    ids = sorted(sample)
    text = {}       # (variante, id) -> Prompttext
    manifest = {}   # (variante, id) -> Manifestzeile

    # ---------------- 1 Dateibestand ------------------------------------
    treffer = []
    for v in VARIANTS:
        ordner = red_root / v
        if not ordner.is_dir():
            treffer.append("Ordner fehlt: %s" % v)
            continue
        vorhanden = sorted(p.stem for p in ordner.glob("*.txt"))
        fehlend = set(ids) - set(vorhanden)
        ueber = set(vorhanden) - set(ids)
        if fehlend:
            treffer.append("%s: %d Prompts fehlen (%s)"
                           % (v, len(fehlend), sorted(fehlend)[:BEISPIELE]))
        if ueber:
            treffer.append("%s: %d unerwartete Dateien (%s)"
                           % (v, len(ueber), sorted(ueber)[:BEISPIELE]))
        if not (ordner / "manifest.tsv").is_file():
            treffer.append("%s: manifest.tsv fehlt" % v)
        for i in vorhanden:
            text[(v, i)] = (ordner / (i + ".txt")).read_text(encoding="utf-8")
        for row in read_tsv_simple(ordner / "manifest.tsv"):
            manifest[(v, row["prompt_id"])] = row
    b.add(1, "Dateibestand je Variante", treffer, len(VARIANTS) * len(ids))

    # ---------------- 2 Kodierung ---------------------------------------
    treffer = []
    for (v, i), t in sorted(text.items()):
        befunde = []
        if "�" in t:
            befunde.append("Ersatzzeichen U+FFFD")
        if "﻿" in t:
            befunde.append("BOM")
        if "\r" in t:
            befunde.append("Wagenruecklauf")
        steuer = {c for c in t if ord(c) < 32 and c != "\n"}
        if steuer:
            befunde.append("Steuerzeichen %s" % sorted(hex(ord(c)) for c in steuer))
        sig = [s for s in MOJIBAKE if s in t]
        if sig:
            befunde.append("Mojibake-Signatur %s" % sig)
        if befunde:
            treffer.append("%s/%s: %s" % (v, i, ", ".join(befunde)))
    b.add(2, "Kodierung, keine verstuemmelten Zeichen", treffer, len(text))

    treffer = []
    for (v, i), t in sorted(text.items()):
        befunde = []
        if "\u00a0" in t:
            befunde.append("geschuetztes Leerzeichen aus der Quelle")
        if t != unicodedata.normalize("NFC", t):
            befunde.append("nicht NFC-normalisiert")
        if befunde:
            treffer.append("%s/%s: %s" % (v, i, ", ".join(befunde)))
    b.add(3, "Zeichenartefakte aus PopQA", treffer, len(text), stufe="befund")

    # ---------------- 3 Aufbau ------------------------------------------
    treffer = []
    erwartet_q = {"basis": 1 + N_DEMOS_BASE, "instruktion": 1 + N_DEMOS_BASE,
                  "filler": 1 + N_DEMOS_BASE,
                  "semantisch": 2 * (1 + N_DEMOS_BASE),
                  "demonstration": 1 + N_DEMOS_BASE + N_DEMOS_RED}
    for (v, i), t in sorted(text.items()):
        befunde = []
        if not t.endswith("\nA:"):
            befunde.append("endet nicht auf Antwortmarke")
        if t != t.rstrip():
            befunde.append("abschliessender Leerraum")
        if "\n\n\n" in t:
            befunde.append("Leerzeile zu viel")
        if any(z != z.rstrip() for z in t.split("\n")):
            befunde.append("Zeile mit abschliessendem Leerzeichen")
        if "  " in t:
            befunde.append("doppeltes Leerzeichen")
        q = [z for z in t.split("\n") if z.startswith("Q: ")]
        if len(q) != erwartet_q[v]:
            befunde.append("%d statt %d Fragezeilen" % (len(q), erwartet_q[v]))
        if len(set(q)) != len(q):
            befunde.append("wortgleiche Fragezeilen")
        a = [z for z in t.split("\n") if z.startswith("A: ")]
        erwartet_a = N_DEMOS_BASE + (N_DEMOS_RED if v == "demonstration" else 0)
        if len(a) != erwartet_a:
            befunde.append("%d statt %d Antwortzeilen" % (len(a), erwartet_a))
        if v == "instruktion":
            kopf = blocks(t)[0].split("\n")
            if len(kopf) != 1 + N_INSTRUCTION_RED:
                befunde.append("%d statt %d Instruktionszeilen"
                               % (len(kopf), 1 + N_INSTRUCTION_RED))
        if befunde:
            treffer.append("%s/%s: %s" % (v, i, ", ".join(befunde)))
    b.add(4, "Aufbau und Formatierung", treffer, len(text))

    # ---------------- 4 Kern ueber alle Varianten identisch --------------
    treffer = []
    for i in ids:
        kerne = {}
        for v in VARIANTS:
            row = manifest.get((v, i))
            if not row:
                continue
            t = text[(v, i)]
            segs = json.loads(row["segmente"])
            kerne[v] = "\n".join(t[s["start"]:s["end"]] for s in segs
                                 if s["rolle"] == "kern")
        if len(set(kerne.values())) > 1:
            abweichend = [v for v in kerne if kerne[v] != kerne["basis"]]
            treffer.append("%s: Kerninhalt weicht ab in %s" % (i, abweichend))
    b.add(5, "Kerninhalt in allen Varianten identisch", treffer, len(ids))

    # ---------------- 5 Uebereinstimmung mit den Quellen ------------------
    treffer = []
    for i in ids:
        meta = sample[i]
        frage_datei = (resolve(QUESTIONS_DIR) / (i + ".txt")).read_text(
            encoding="utf-8").strip()
        t = text[("basis", i)]
        bl = blocks(t)
        gestellt = bl[-1].split("\n")[0][3:]
        if gestellt != meta["question"]:
            treffer.append("%s: Testfrage weicht von sample_a ab" % i)
        if frage_datei != meta["question"]:
            treffer.append("%s: 0_questions weicht von sample_a ab" % i)
        for k in range(N_DEMOS_BASE):
            zeilen = bl[1 + k].split("\n")
            soll = demos[i].get(k + 1)
            if soll is None:
                treffer.append("%s: Slot %d fehlt in demo_assignments" % (i, k + 1))
                continue
            if zeilen[0][3:] != soll["demo_question"]:
                treffer.append("%s: Demonstration %d weicht ab" % (i, k + 1))
            if zeilen[-1][3:] != soll["demo_answer"]:
                treffer.append("%s: Antwort der Demonstration %d weicht ab"
                               % (i, k + 1))
        bl_red = blocks(text[("demonstration", i)])
        for k in range(N_DEMOS_RED):
            zeilen = bl_red[1 + N_DEMOS_BASE + k].split("\n")
            soll = demos[i].get(5 + k)
            if soll is None:
                treffer.append("%s: Slot %d fehlt in demo_assignments" % (i, 5 + k))
                continue
            if zeilen[0][3:] != soll["demo_question"]:
                treffer.append("%s: Redundanzdemonstration %d weicht ab" % (i, k + 1))
    b.add(6, "Uebereinstimmung mit 0_questions, sample_a und demo_assignments",
          treffer, len(ids))

    # ---------------- 6 Kategorien der Demonstrationen --------------------
    treffer = []
    id_menge = set(ids)
    for i in ids:
        props = [demos[i][s]["demo_prop"] for s in sorted(demos[i])]
        if len(set(props)) != len(props):
            treffer.append("%s: Kategorien nicht paarweise verschieden (%s)"
                           % (i, [p for p, n in Counter(props).items() if n > 1]))
        if sample[i]["prop"] in props:
            treffer.append("%s: Demonstration aus der Kategorie der Testfrage" % i)
        dids = {demos[i][s]["demo_id"] for s in demos[i]}
        if dids & id_menge:
            treffer.append("%s: Demonstration aus der Stichprobe A (%s)"
                           % (i, sorted(dids & id_menge)))
    b.add(7, "Demonstrationen aus verschiedenen Kategorien, Pool disjunkt",
          treffer, len(ids))

    # ---------------- 7 Paraphrasen --------------------------------------
    treffer = []
    geprueft = 0
    for i in ids:
        t = text[("semantisch", i)]
        bl = blocks(t)
        paare = [(demos[i][k + 1]["demo_id"], bl[1 + k]) for k in range(N_DEMOS_BASE)]
        paare.append((i, bl[-1]))
        for frage_id, block in paare:
            zeilen = block.split("\n")
            if len(zeilen) < 2:
                treffer.append("%s: Block ohne Paraphrase" % i)
                continue
            para = zeilen[1][3:]
            geprueft += 1
            kandidaten = para_alle.get(str(frage_id), {})
            tids = [tid for tid, txt in kandidaten.items() if txt == para]
            if not tids:
                treffer.append("%s: Paraphrase gehoert nicht zu Frage %s (%r)"
                               % (i, frage_id, para[:60]))
                continue
            if 0 in tids:
                treffer.append("%s: Originalformulierung als Paraphrase (Frage %s)"
                               % (i, frage_id))
            if para == zeilen[0][3:]:
                treffer.append("%s: Paraphrase wortgleich zur Frage %s"
                               % (i, frage_id))
    b.add(8, "Paraphrasen stammen aus PopQA-TP und sind keine Originalfrage",
          treffer, geprueft)

    # ---------------- 8 Manifestkonsistenz --------------------------------
    treffer = []
    for (v, i), row in sorted(manifest.items()):
        t = text[(v, i)]
        segs = json.loads(row["segmente"])
        if int(row["n_zeichen"]) != len(t):
            treffer.append("%s/%s: n_zeichen falsch" % (v, i))
        if int(row["n_ws_token"]) != len(t.split()):
            treffer.append("%s/%s: n_ws_token falsch" % (v, i))
        kern = sum(len(t[s["start"]:s["end"]].split()) for s in segs
                   if s["rolle"] == "kern")
        red = sum(len(t[s["start"]:s["end"]].split()) for s in segs
                  if s["rolle"] == "redundanz")
        if (kern, red) != (int(row["n_ws_token_kern"]), int(row["n_ws_token_red"])):
            treffer.append("%s/%s: Kern- oder Redundanztoken falsch" % (v, i))
        if kern + red != len(t.split()):
            treffer.append("%s/%s: Summe Kern und Redundanz ungleich Gesamtzahl"
                           % (v, i))
        if {s["rolle"] for s in segs} - {"kern", "redundanz"}:
            treffer.append("%s/%s: unbekannte Rolle" % (v, i))
        if segs[0]["start"] != 0 or segs[-1]["end"] != len(t):
            treffer.append("%s/%s: Segmente decken den Text nicht ab" % (v, i))
        for x, y in zip(segs, segs[1:]):
            if t[x["end"]:y["start"]] not in ("\n", "\n\n"):
                treffer.append("%s/%s: Luecke zwischen %s und %s"
                               % (v, i, x["label"], y["label"]))
        delta = len(t.split()) - len(text[("basis", i)].split())
        if int(row["delta_ws_token"]) != delta:
            treffer.append("%s/%s: delta_ws_token falsch" % (v, i))
        if v == "basis" and red != 0:
            treffer.append("%s: Basisprompt enthaelt Redundanzsegmente" % i)
    b.add(9, "Manifest stimmt mit den Dateien ueberein", treffer, len(manifest))

    # ---------------- 9 Kalibrierung filler -------------------------------
    treffer = []
    for i in ids:
        red = {v: int(manifest[(v, i)]["n_ws_token_red"])
               for v in ("semantisch", "instruktion", "demonstration")}
        soll = round(sum(red.values()) / len(red))
        ist = int(manifest[("filler", i)]["n_ws_token_red"])
        if ist != soll:
            treffer.append("%s: filler %d Token statt %d" % (i, ist, soll))
    b.add(10, "Laenge der Stufe filler entspricht dem Mittel der anderen Stufen",
          treffer, len(ids))

    # ---------------- 10 Antwortleckage -----------------------------------
    treffer = []
    for i in ids:
        antworten = {a for a in sample[i]["antworten"] + [sample[i]["obj"]]
                     if a and len(a) > 2}
        for v in VARIANTS:
            row = manifest[(v, i)]
            t = text[(v, i)]
            segs = json.loads(row["segmente"])
            rot = " ".join(t[s["start"]:s["end"]] for s in segs
                           if s["rolle"] == "redundanz")
            kern = " ".join(t[s["start"]:s["end"]] for s in segs
                            if s["rolle"] == "kern")
            if not rot:
                continue
            for a in sorted(antworten):
                muster = r"\b%s\b" % re.escape(a)
                if not re.search(muster, rot, re.IGNORECASE):
                    continue
                if re.search(muster, kern, re.IGNORECASE):
                    continue     # steht ohnehin im Kern, kein Zugewinn
                treffer.append("%s/%s: Goldantwort %r nur im eingefuegten Text"
                               % (v, i, a))
    b.add(11, "Goldantwort taucht im eingefuegten Text auf", treffer,
          len(ids) * len(VARIANTS), stufe="befund")

    # ---------------- 12 Goldantwort in den Basisdemonstrationen -----------
    treffer = []
    for i in ids:
        antworten = {a for a in sample[i]["antworten"] + [sample[i]["obj"]]
                     if a and len(a) > 2}
        t = text[("basis", i)]
        demoteil = "\n\n".join(blocks(t)[1:-1])
        for a in sorted(antworten):
            if re.search(r"\b%s\b" % re.escape(a), demoteil, re.IGNORECASE):
                treffer.append("%s: Goldantwort %r in den Basisdemonstrationen"
                               % (i, a))
    b.add(12, "Goldantwort in den vier Basisdemonstrationen", treffer, len(ids),
          stufe="hinweis")

    # ---------------- Lexikalische Erkennbarkeit ---------------------------
    quoten = {}
    for v in VARIANTS:
        if v == "basis":
            continue
        werte = []
        for i in ids:
            t = text[(v, i)]
            segs = json.loads(manifest[(v, i)]["segmente"])
            kern = " ".join(t[s["start"]:s["end"]] for s in segs if s["rolle"] == "kern")
            rot = " ".join(t[s["start"]:s["end"]] for s in segs if s["rolle"] == "redundanz")
            kw = {w.lower().strip(".,:?'\"") for w in kern.split()}
            rw = [w.lower().strip(".,:?'\"") for w in rot.split()]
            if rw:
                werte.append(sum(1 for w in rw if w in kw) / len(rw))
        quoten[v] = sum(werte) / len(werte) if werte else 0.0

    # ---------------- 13 Reproduzierbarkeit --------------------------------
    treffer = []
    n_repro = 0
    if PRUEFE_REPRODUKTION:
        tmp = Path(tempfile.mkdtemp(prefix="qa_repro_"))
        try:
            builder.OUTPUT_DIR = str(tmp)
            builder.main()
            for v in VARIANTS:
                for i in ids:
                    neu = tmp / v / (i + ".txt")
                    if not neu.is_file():
                        treffer.append("%s/%s im zweiten Lauf nicht erzeugt" % (v, i))
                        continue
                    n_repro += 1
                    h1 = hashlib.sha256(neu.read_bytes()).hexdigest()
                    h2 = hashlib.sha256(
                        (red_root / v / (i + ".txt")).read_bytes()).hexdigest()
                    if h1 != h2:
                        treffer.append("%s/%s nicht bitgleich reproduziert" % (v, i))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    b.add(13, "Zweiter Lauf erzeugt bitgleiche Redundanzdateien", treffer, n_repro)

    # ---------------- 14 Basisprompts reproduzierbar ----------------------
    treffer = []
    n_basis = 0
    if PRUEFE_REPRODUKTION:
        spec = importlib.util.spec_from_file_location("bp", resolve("build_prompts.py"))
        bp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bp)
        tmp = Path(tempfile.mkdtemp(prefix="qa_basis_"))
        try:
            bp.OUTPUT_DIR = str(tmp)
            bp.main()
            for i in ids:
                neu = tmp / (i + ".txt")
                alt = resolve(PROMPTS_DIR) / (i + ".txt")
                if not neu.is_file():
                    treffer.append("%s im zweiten Lauf nicht erzeugt" % i)
                    continue
                n_basis += 1
                if neu.read_bytes() != alt.read_bytes():
                    treffer.append("%s weicht von 1_prompts ab" % i)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    b.add(14, "1_prompts aus 0_questions reproduzierbar", treffer, n_basis)

    # ---------------- Bericht ---------------------------------------------
    zeilen = ["# Qualitaetssicherung der Prompts", "",
              "Geprueft: %d Prompts in %d Varianten." % (len(ids), len(VARIANTS)),
              "", "| Nr | Pruefung | Stufe | geprueft | Ergebnis |",
              "|---|---|---|---|---|"]
    for p in b.pruefungen:
        art = ("bestanden" if not p["treffer"]
               else "%d Fundstellen" % len(p["treffer"]))
        zeilen.append("| %d | %s | %s | %d | %s |"
                      % (p["nummer"], p["titel"], p["stufe"], p["n"], art))
    zeilen += ["", "## Kennzahlen je Variante", "",
               "| Variante | Median Token gesamt | Median Token Redundanz | "
               "Wortformen des Einschubs, die auch im Kern vorkommen |", "|---|---|---|---|"]
    for v in VARIANTS:
        ges = sorted(int(manifest[(v, i)]["n_ws_token"]) for i in ids)
        red = sorted(int(manifest[(v, i)]["n_ws_token_red"]) for i in ids)
        q = "%.3f" % quoten[v] if v in quoten else "-"
        zeilen.append("| %s | %d | %d | %s |" % (v, ges[len(ges) // 2],
                                                 red[len(red) // 2], q))
    for p in b.pruefungen:
        if not p["treffer"]:
            continue
        zeilen += ["", "## Pruefung %d: %s" % (p["nummer"], p["titel"]), ""]
        for eintrag in p["treffer"][:20]:
            zeilen.append("- " + eintrag)
        if len(p["treffer"]) > 20:
            zeilen.append("- weitere %d Fundstellen" % (len(p["treffer"]) - 20))
    zeilen += ["", "Fehler gesamt: %d, zu bewertende Befunde: %d"
               % (b.fehler, b.befunde), ""]

    ziel = resolve(REPORT_FILE)
    ziel.write_text("\n".join(zeilen), encoding="utf-8")

    for p in b.pruefungen:
        art = ("ok" if not p["treffer"] else
               {"fehler": "FEHLER", "befund": "BEFUND",
                "hinweis": "Hinweis"}[p["stufe"]])
        print("%-6s Pruefung %2d  %-62s n=%d, Befunde=%d"
              % (art, p["nummer"], p["titel"][:62], p["n"], len(p["treffer"])))
    print("\nBericht: %s" % ziel)
    print("Fehler gesamt: %d, zu bewertende Befunde: %d"
          % (b.fehler, b.befunde))
    return 1 if b.fehler else 0


if __name__ == "__main__":
    sys.exit(main())
