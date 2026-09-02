#!/usr/bin/env python3
"""Unabhaengige Pruefung der Ziehung gegen die Anforderungen aus 4.4.3."""
import csv, json, hashlib, os, subprocess, sys
from collections import Counter, defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "popqa_subset"
SRC = sys.argv[2] if len(sys.argv) > 2 else "popQA.tsv"
ok = True
def chk(cond, msg):
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    ok = ok and cond

src = {r["id"]: r for r in csv.DictReader(open(SRC, newline="", encoding="utf-8"), delimiter="\t")}
man = json.load(open(f"{OUT}/draw_manifest.json", encoding="utf-8"))
A = list(csv.DictReader(open(f"{OUT}/sample_a.tsv", newline="", encoding="utf-8"), delimiter="\t"))
B = list(csv.DictReader(open(f"{OUT}/sample_b.tsv", newline="", encoding="utf-8"), delimiter="\t"))
D = list(csv.DictReader(open(f"{OUT}/demo_assignments.tsv", newline="", encoding="utf-8"), delimiter="\t"))
pool = [l.strip() for l in open(f"{OUT}/demo_pool_ids.txt", encoding="utf-8") if l.strip()]

print("\n[1] Umfang und Eindeutigkeit")
chk(len(A) == 300, f"Grundstichprobe A = 300 (ist {len(A)})")
chk(len(B) == 30, f"Teilstichprobe B = 30 (ist {len(B)})")
chk(len({r['id'] for r in A}) == 300, "A ohne Duplikate")
chk(len({r['id'] for r in B}) == 30, "B ohne Duplikate")
chk(all(r["id"] in src for r in A), "alle A-IDs existieren in PopQA")

print("\n[2] Schachtelung und Disjunktheit")
ida, idb, idp = {r["id"] for r in A}, {r["id"] for r in B}, set(pool)
chk(idb <= ida, "B vollstaendig in A enthalten")
chk(not (ida & idp), "Demonstrationspool disjunkt zu A")
n_excl = sum(1 for r in src.values() if r["prop"] == "capital of")
chk(len(idp) == len(src) - 300 - n_excl,
    f"Pool = nicht gezogene Fragen der Studienkategorien ({len(idp)})")

print("\n[3] Schichtung nach Relationskategorie (A)")
ca = Counter(r["prop"] for r in A)
chk(len(ca) == 15, f"alle 15 Studienkategorien vertreten (ist {len(ca)})")
chk("capital of" not in ca, "'capital of' nicht in der Stichprobe")
chk(set(ca) == set(man["allocation"]), "Kategorienmenge stimmt mit Manifest")
chk(all(ca[c] == man["allocation"][c] for c in ca), "Quoten je Kategorie eingehalten")
chk(max(ca.values()) <= 21 and min(ca.values()) >= 18,
    f"keine Kategorie dominiert (min {min(ca.values())}, max {max(ca.values())})")
print("       " + ", ".join(f"{c}:{ca[c]}" for c in sorted(ca)))

print("\n[4] Schichtung nach Popularitaetsterzil (A)")
tt = defaultdict(Counter)
for r in A: tt[r["prop"]][r["pop_tercile"]] += 1
bad = [c for c in tt if len(set(tt[c].values())) != 1 or len(tt[c]) != 3]
chk(not bad, f"in jeder Kategorie alle 3 Terzile gleich stark besetzt (Abw.: {bad})")
chk(Counter(r["pop_tercile"] for r in A) == Counter({"1": 100, "2": 100, "3": 100}),
    "global 100 je Terzil")

print("\n[5] Terzilgrenzen unabhaengig nachgerechnet")
bycat = defaultdict(list)
for r in src.values(): bycat[r["prop"]].append(r)
ref = {}
for c, lst in bycat.items():
    lst = sorted(lst, key=lambda r: (float(r["s_pop"]), int(r["id"])))
    n = len(lst); cuts = [round(n * i / 3) for i in range(4)]
    for t in range(3):
        for r in lst[cuts[t]:cuts[t+1]]: ref[r["id"]] = t + 1
chk(all(int(r["pop_tercile"]) == ref[r["id"]] for r in A),
    "Terzilzuordnung reproduzierbar aus (s_pop, id)")
sizes = {c: Counter(ref[r["id"]] for r in bycat[c]) for c in bycat}
chk(all(max(s.values()) - min(s.values()) <= 1 for s in sizes.values()),
    "Grundgesamtheit je Kategorie in gleich grosse Terzile geteilt")

print("\n[6] Teilstichprobe B balanciert")
cb = Counter(r["prop"] for r in B)
chk(len(cb) == 15, f"alle 15 Kategorien in B (ist {len(cb)})")
chk("capital of" not in cb, "'capital of' nicht in B")
chk(max(cb.values()) <= 2, f"max 2 je Kategorie (ist {max(cb.values())})")
chk(Counter(r["pop_tercile"] for r in B) == Counter({"1": 10, "2": 10, "3": 10}),
    "10 Fragen je Terzil in B")

print("\n[7] Paraphrasen-Ausschlusskriterium")
pp = man["paraphrases_per_category"]
chk(all(int(r["n_paraphrases"]) >= man["min_paraphrases_sample"] for r in A),
    f">= {man['min_paraphrases_sample']} echte Paraphrasen je A-Frage")
chk(man["excluded_categories"] == ["capital of"],
    f"Ausschluss trifft genau 'capital of' (ist {man['excluded_categories']})")
chk(all(pp[c] < man["min_paraphrases_sample"] for c in man["excluded_categories"]),
    "ausgeschlossene Kategorie verfehlt das Kriterium tatsaechlich")
chk(all(pp[r["prop"]] >= man["min_paraphrases_sample"] for r in A)
    and set(ca) == {c for c, n in pp.items() if n >= man["min_paraphrases_sample"]},
    "Stichprobe = genau die Kategorien, die das Kriterium erfuellen")
chk(all(pp[src[i]["prop"]] >= man["min_paraphrases_pool"] for i in pool),
    "jede Poolfrage hat mind. 1 echte Paraphrase")

print("\n[8] Demonstrationszuweisung")
byt = defaultdict(list)
for d in D: byt[d["task_id"]].append(d)
chk(len(byt) == 300, f"Zuweisung fuer alle 300 Aufgaben (ist {len(byt)})")
chk(all(len(v) == 8 for v in byt.values()), "je 8 Demonstrationen")
chk(all(len({d['demo_prop'] for d in v}) == 8 for v in byt.values()),
    "8 paarweise verschiedene Kategorien je Aufgabe")
chk(not any(d["demo_prop"] == "capital of" for d in D),
    "keine Demonstration aus der ausgeschlossenen Kategorie")
chk(not any(src[i]["prop"] == "capital of" for i in pool),
    "'capital of' auch nicht im Demonstrationspool")
tp = {r["id"]: r["prop"] for r in A}
chk(all(all(d["demo_prop"] != tp[t] for d in v) for t, v in byt.items()),
    "keine Demonstration aus der Kategorie der Testfrage")
chk(all(d["demo_id"] in idp for d in D), "alle Demonstrationen aus dem Pool")
chk(all(len({d['demo_id'] for d in v}) == 8 for v in byt.values()),
    "keine Demonstration doppelt innerhalb einer Aufgabe")
roles = {t: Counter(d["role"] for d in v) for t, v in byt.items()}
chk(all(r["base"] == 4 and r["demonstration"] == 4 for r in roles.values()),
    "je 4 Demonstrationen fuer Basisprompt und Redundanzstufe")
chk(all(d["demo_question"] == src[d["demo_id"]]["question"] for d in D),
    "Demonstrationstexte stimmen mit der Quelle ueberein")

print("\n[9] Prompt-Dateien")
files = sorted(os.listdir(f"{OUT}/prompts"))
chk(len(files) == 300, f"300 Dateien (ist {len(files)})")
chk({f[:-4] for f in files} == ida, "Dateinamen = PopQA-IDs der Stichprobe")
chk(all(f.endswith(".txt") for f in files), "alle Dateien .txt")
mism = [f for f in files
        if open(f"{OUT}/prompts/{f}", encoding="utf-8").read().strip() != src[f[:-4]]["question"].strip()]
chk(not mism, f"Dateiinhalt = Originalfrage (Abw.: {len(mism)})")
chk(all(os.path.getsize(f"{OUT}/prompts/{f}") > 0 for f in files), "keine leere Datei")

print("\n[10] Reproduzierbarkeit")
subprocess.run([sys.executable, "draw_sample.py", SRC, "_repro"], check=True,
               stdout=subprocess.DEVNULL)
def dig(p):
    h = hashlib.sha256()
    for root, _, fs in os.walk(p):
        for f in sorted(fs):
            h.update(f.encode()); h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()
chk(dig(OUT) == dig("_repro"), "zweiter Lauf mit Seed erzeugt bitgleiche Ausgabe")

print("\n" + ("ALLE PRUEFUNGEN BESTANDEN" if ok else "PRUEFUNG FEHLGESCHLAGEN"))
sys.exit(0 if ok else 1)
