#!/usr/bin/env python3
"""
Stichprobenziehung fuer Abschnitt 4.4.3.

Grundstichprobe A: 300 PopQA-Fragen, geschichtet nach Relationskategorie und
innerhalb jeder Kategorie nach Popularitaetsterzil.
Teilstichprobe B: 30 Fragen, vollstaendig in A enthalten.
Demonstrationspool: alle nicht gezogenen Fragen, disjunkt zu A.

Alles deterministisch ueber SEED reproduzierbar.
"""
import csv, json, os, random, sys
from collections import defaultdict, Counter

SEED = 42
IN_TSV = sys.argv[1] if len(sys.argv) > 1 else "popQA.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "popqa_subset"

# --- Anzahl echter Paraphrasen je Relationskategorie in PopQA-TP -----------
# Rabinovich et al. (2023), Tabelle 1: Templates sind relationsspezifisch,
# daher gilt der Wert fuer jede Frage der Kategorie.
# Wert = Anzahl Paraphrasen OHNE die Originalformulierung.
PARAPHRASES = {
    "author": 5, "capital": 6, "capital of": 3, "color": 5, "composer": 5,
    "country": 9, "director": 10, "father": 4, "genre": 6, "mother": 5,
    "occupation": 5, "place of birth": 6, "producer": 10, "religion": 5,
    "screenwriter": 10, "sport": 6,
}
MIN_PARAPHRASES = 4          # Ausschlusskriterium Stichprobe: schliesst
                             # "capital of" (3) als einzige Kategorie aus
MIN_PARAPHRASES_POOL = 1     # Demopool: mind. eine Paraphrase

# --- Allokation -------------------------------------------------------------
# 15 Kategorien, jede Quote durch 3 teilbar (gleich besetzte Terzile).
# 300 = 10 x 21 + 5 x 18; geringstmoegliche Spreizung bei 15 Kategorien.
N_A = 300
BASE_PER_CAT = 18
EXTRA_CATS = ["screenwriter", "director", "genre", "producer", "author",
              "composer", "country", "capital", "place of birth", "father"]
EXTRA = 3                    # 10 x 3 = 30  ->  270 + 30 = 300

N_B = 30                     # 15 Kategorien x 2, global 10 je Terzil
B_SMALL_CATS = []

N_DEMOS = 8                  # 4 Basisprompt + 4 Redundanzstufe "demonstration"

rng = random.Random(SEED)

# --- Einlesen ---------------------------------------------------------------
with open(IN_TSV, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))

for r in rows:
    r["s_pop"] = float(r["s_pop"])
    r["n_paraphrases"] = PARAPHRASES[r["prop"]]

eligible = [r for r in rows if r["n_paraphrases"] >= MIN_PARAPHRASES]
pool_eligible_ids = {r["id"] for r in rows if r["n_paraphrases"] >= MIN_PARAPHRASES_POOL}

by_cat = defaultdict(list)
for r in eligible:
    by_cat[r["prop"]].append(r)
CATS = sorted(by_cat)
assert len(CATS) == 15, CATS
assert "capital of" not in CATS

# --- Terzile: rangbasiert, damit Bindungen in s_pop die Schichten nicht
#     ungleich gross machen. Sortierschluessel (s_pop, id) ist eindeutig.
for cat in CATS:
    lst = sorted(by_cat[cat], key=lambda r: (r["s_pop"], int(r["id"])))
    n = len(lst)
    cuts = [round(n * i / 3) for i in range(4)]
    for t in range(3):
        for r in lst[cuts[t]:cuts[t + 1]]:
            r["pop_tercile"] = t + 1          # 1 = niedrig, 2 = mittel, 3 = hoch
    by_cat[cat] = lst

# --- Ziehung Grundstichprobe A ---------------------------------------------
sample_a = []
for cat in CATS:
    quota = BASE_PER_CAT + (EXTRA if cat in EXTRA_CATS else 0)
    assert quota % 3 == 0
    per_tercile = quota // 3
    for t in (1, 2, 3):
        stratum = sorted([r for r in by_cat[cat] if r["pop_tercile"] == t],
                         key=lambda r: int(r["id"]))
        if len(stratum) < per_tercile:
            raise SystemExit(f"Terzil {t} von {cat} zu klein: {len(stratum)} < {per_tercile}")
        sample_a.extend(rng.sample(stratum, per_tercile))

assert len(sample_a) == N_A, len(sample_a)
sample_a.sort(key=lambda r: (r["prop"], r["pop_tercile"], int(r["id"])))
ids_a = {r["id"] for r in sample_a}
assert len(ids_a) == N_A

# --- Ziehung Teilstichprobe B (geschachtelt in A) ---------------------------
# Quote je Kategorie: 2  ->  15 x 2 = 30.
# Zusaetzlich global exakt 10 Fragen je Popularitaetsterzil.
b_quota = {c: (1 if c in B_SMALL_CATS else 2) for c in CATS}
assert sum(b_quota.values()) == N_B

a_by_cat = defaultdict(list)
for r in sample_a:
    a_by_cat[r["prop"]].append(r)

def draw_b(rnd):
    """Zieht B mit Kategorienquote und global gleicher Terzilbesetzung."""
    for _ in range(200000):
        pick = []
        for cat in CATS:
            pick.extend(rnd.sample(a_by_cat[cat], b_quota[cat]))
        cnt = Counter(r["pop_tercile"] for r in pick)
        if cnt[1] == cnt[2] == cnt[3] == N_B // 3:
            return pick
    raise SystemExit("Keine terzilbalancierte Ziehung fuer B gefunden")

sample_b = draw_b(rng)
sample_b.sort(key=lambda r: (r["prop"], r["pop_tercile"], int(r["id"])))
ids_b = {r["id"] for r in sample_b}
assert ids_b <= ids_a and len(ids_b) == N_B

# --- Demonstrationspool -----------------------------------------------------
# Nicht gezogene Fragen der Studienkategorien. Die ausgeschlossene Kategorie
# bleibt auch hier aussen vor, damit der Kategorienraum der Studie einheitlich
# ist. Die Poolbedingung (mind. eine Paraphrase) ist dadurch stets erfuellt.
demo_pool = [r for r in rows if r["id"] not in ids_a
             and r["id"] in pool_eligible_ids and r["prop"] in CATS]
pool_by_cat = defaultdict(list)
for r in demo_pool:
    pool_by_cat[r["prop"]].append(r)
for c in pool_by_cat:
    pool_by_cat[c].sort(key=lambda r: int(r["id"]))

# --- Demonstrationszuweisung: 8 Kategorien je Aufgabe, paarweise verschieden,
#     keine aus der Kategorie der Testfrage; einmal gezogen, danach konstant.
demo_assign = {}
for r in sample_a:
    other = [c for c in CATS if c != r["prop"]]
    chosen_cats = rng.sample(other, N_DEMOS)
    demos = []
    for i, c in enumerate(chosen_cats):
        d = rng.choice(pool_by_cat[c])
        demos.append({
            "slot": i + 1,
            "role": "base" if i < 4 else "demonstration",
            "demo_id": d["id"], "demo_prop": d["prop"],
            "demo_question": d["question"],
            "demo_answer": json.loads(d["possible_answers"])[0],
        })
    demo_assign[r["id"]] = demos

# --- Schreiben --------------------------------------------------------------
os.makedirs(f"{OUT}/prompts", exist_ok=True)
for r in sample_a:
    with open(f"{OUT}/prompts/{r['id']}.txt", "w", encoding="utf-8") as fh:
        fh.write(r["question"].strip() + "\n")

META = ["id", "prop", "prop_id", "pop_tercile", "s_pop", "subj", "obj",
        "question", "possible_answers", "n_paraphrases", "in_subset_b"]

def write_meta(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(META)
        for r in recs:
            w.writerow([r["id"], r["prop"], r["prop_id"], r["pop_tercile"],
                        int(r["s_pop"]), r["subj"], r["obj"], r["question"],
                        r["possible_answers"], r["n_paraphrases"],
                        int(r["id"] in ids_b)])

write_meta(f"{OUT}/sample_a.tsv", sample_a)
write_meta(f"{OUT}/sample_b.tsv", sample_b)

for name, recs in (("sample_a_ids.txt", sample_a), ("sample_b_ids.txt", sample_b)):
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as fh:
        fh.write("\n".join(r["id"] for r in sorted(recs, key=lambda x: int(x["id"]))) + "\n")

with open(f"{OUT}/demo_pool_ids.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(r["id"] for r in sorted(demo_pool, key=lambda x: int(x["id"]))) + "\n")

with open(f"{OUT}/demo_assignments.tsv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["task_id", "task_prop", "slot", "role", "demo_id", "demo_prop",
                "demo_question", "demo_answer"])
    for r in sample_a:
        for d in demo_assign[r["id"]]:
            w.writerow([r["id"], r["prop"], d["slot"], d["role"], d["demo_id"],
                        d["demo_prop"], d["demo_question"], d["demo_answer"]])

with open(f"{OUT}/draw_manifest.json", "w", encoding="utf-8") as fh:
    json.dump({
        "seed": SEED, "source_file": os.path.basename(IN_TSV),
        "source_n": len(rows), "eligible_n": len(eligible),
        "n_a": len(sample_a), "n_b": len(sample_b), "demo_pool_n": len(demo_pool),
        "allocation": {c: BASE_PER_CAT + (EXTRA if c in EXTRA_CATS else 0) for c in CATS},
        "excluded_categories": sorted(
            c for c, n in PARAPHRASES.items() if n < MIN_PARAPHRASES),
        "b_allocation": b_quota,
        "min_paraphrases_sample": MIN_PARAPHRASES,
        "min_paraphrases_pool": MIN_PARAPHRASES_POOL,
        "paraphrases_per_category": PARAPHRASES,
        "tercile_method": "rangbasiert nach (s_pop, id), gleich grosse Schichten",
        "python": sys.version.split()[0],
    }, fh, indent=2, ensure_ascii=False)

print(f"A={len(sample_a)} B={len(sample_b)} pool={len(demo_pool)} -> {OUT}/")
