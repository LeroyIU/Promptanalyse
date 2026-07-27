# Experiment-Pipeline: Ablage & Auswertung

Ablagesystem für die Prompts entlang der Experiment-Pipeline. Die Pipeline hat
vier Stages, mit fester Verzweigung pro Raw-Prompt:

```
raw prompt  ─▶  variante  ─▶  kompressionsrate  ─▶  inferenz
   1 Raw   =   4 Varianten  ×   4 Kompressionsraten  =  16 Kompressate
```

Die Raw-Prompts kommen aus **MuSiQue**: jede Frage bringt ~20 Wikipedia-Passagen
mit, von denen 2–4 als `is_supporting` (Gold-Evidenz) markiert sind.

Die **4 Varianten** pro Raw-Prompt sind die **Basis** (Kontrolle: der
Raw-Prompt ohne hinzugefügte Redundanz) plus die **3 Redundanztypen**
(`passages`, `demonstrations`, `instructions`) — sie adressieren genau die drei
Bestandteile eines Kontext-Prompts. Alle vier werden bei denselben
4 Kompressionsraten komprimiert.

Jedes **Kompressat** (= 1 Variante bei 1 Kompressionsrate) ist die statistische
**Beobachtungseinheit**. Es gibt genau 16 davon pro Raw-Prompt. In der
Auswertung ist `baseline` das **Referenzlevel** des Faktors `redundancy_type` —
so misst du den Effekt jedes Redundanztyps gegen die redundanzfreie Kontrolle.

## Warum diese Struktur (die eigentliche Design-Entscheidung)

Speicherung und Statistik haben gegensätzliche Anforderungen. Deshalb werden
**zwei Ebenen getrennt**:

| Ebene | Inhalt | Format | Wofür |
|---|---|---|---|
| **Artefakte** | die eigentlichen Prompt-Texte (raw, redundant, komprimiert, Inferenz-Output) | eine **JSON-Datei pro Knoten** im nach Stages partitionierten Verzeichnisbaum | verlustfreie, reproduzierbare, versionierbare Ablage |
| **Messwerte** | IDs, Redundanztyp, Rate, Tokenzahlen, Korrektheit, Metriken | **eine „tidy" Long-Format-Tabelle**: `manifest.jsonl` (verlustfrei) + `manifest.csv` (flach) | direkt in pandas/R für die statistische Auswertung |

Die lange Textmenge liegt also in Dateien; die auswertbaren Zahlen werden in
**eine Zeile pro Kompressat** projiziert. Stabile IDs verbinden beide Ebenen,
sodass jede Manifest-Zeile auf ihre Artefakt-Dateien zeigt und umgekehrt.

**Warum Long-/Tidy-Format?** Eine Zeile = eine Beobachtung ist die kanonische
Form für Statistik: `groupby`, ANOVA, gemischte Modelle (Prompt als
Zufallseffekt), Plots über `redundancy_type × target_ratio` funktionieren ohne
Umbau. Ein „wide" Format (16 Spalten pro Raw-Prompt) müsste man dafür erst
wieder aufschmelzen.

## Verzeichnislayout

```
experiments/<experiment_id>/
├── experiment.json                       # Konfiguration: Modell, Seeds, Enums
├── 01_raw/<prompt_id>.json               # Stage 1: Original-Prompt
├── 02_redundant/<prompt_id>/<typ>.json   # Stage 2: + 1 Redundanztyp
├── 03_compressed/<prompt_id>/<typ>/<rate>.json   # Stage 3: komprimiert
├── 04_inference/<prompt_id>/<typ>/<rate>.json    # Stage 4: Modell-Output + Score
├── manifest.jsonl                        # 1 Datensatz pro Kompressat (verlustfrei)
└── manifest.csv                          # dieselben Daten flach (Auswertung)
```

### ID- und Namensschema

Die IDs sind rein aus ihren Koordinaten abgeleitet (nicht aus Inhalt/Zeit),
damit erneute Läufe **idempotent** sind — dieselbe Koordinate überschreibt am
selben Pfad statt Duplikate anzuhäufen.

- `prompt_id` – z. B. `p-2hop-101-201` (aus der MuSiQue-ID oder einem Zähler)
- `rate_label` – feste Breite, lexikalisch sortierbar: `cr020, cr040, cr060, cr080` (= 20/40/60/80 % behaltene Tokens)
- `compressate_id` – `p-2hop-101-201__passages__cr040` → lässt sich zurück in Pfad und Koordinaten zerlegen

## End-to-End-Lauf (echter Redundanzgenerator)

`run_pipeline.py` verdrahtet den echten `redundanzgenerator` dieses Repos mit dem
Store: es baut Kontext-Prompts aus MuSiQue, erzeugt die 4 Varianten (Basis + 3
Redundanztypen), komprimiert jede bei 4 Raten, bewertet sie und schreibt den
gesamten Stage-Baum plus Manifest.

```bash
pip install -e tools/PipelineStore          # Speicherung
pip install -e tools/Redundanzgenerator     # Redundanzgenerator (Stages 1–2)
git lfs pull                                # Datensatz aus Git LFS auschecken
python experiments/run_pipeline.py --experiment run-2026-07 --n-prompts 20
```

Ohne `--query-ids` werden die ersten `--n-prompts` Datensätze verwendet — ein Lauf
braucht also keine vorher bekannten IDs.

Die beiden Stages, die keine Prompt-Konstruktion sind — **Kompression** und
**Inferenz** — sind als Callables **einsteckbar**. Standard sind offline
Baselines (kontextschonendes Token-Kürzen bzw. Gold-Containment-Scoring), damit
der Lauf ohne Modelle/Netz durchläuft; die echten Methoden (z. B. LLMLingua,
Claude API) setzt man mit gleicher Signatur ein:

```python
from pipelinestore import PipelineStore
from pipelinestore.redundancy_pipeline import run_pipeline, RedundancyCounts

def my_compress(text: str, target_ratio: float) -> str: ...      # z. B. LLMLingua-2
def my_infer(text: str, gold: list[str]): ...                    # z. B. Claude API

run_pipeline(
    PipelineStore("experiments", "run-2026-07"),
    musique_source="datasets/musique/musique_ans_v1.0_dev.jsonl",
    query_ids=["2hop__128801_205185", "3hop1__..."],
    compression_rates=[0.2, 0.4, 0.6, 0.8],
    counts=RedundancyCounts(passages=3, demonstrations=3, instructions=2),
    compress=my_compress, infer=my_infer,
)
```

## Nutzung direkt über die Store-API

Falls du die Prompts anderweitig erzeugst — die Store-Aufrufe bleiben gleich:

```python
from pipelinestore import (
    PipelineStore, RawPrompt, RedundantVariant, Compressate, InferenceResult,
    ids, whitespace_tokens,
)

store = PipelineStore("experiments", "run-2026-07")
store.store_experiment({"inference_model": "...", "seed": 42,
                        "redundancy_types": [...], "compression_rates": [0.2, 0.4, 0.6, 0.8]})

store.store_raw(RawPrompt(prompt_id=ids.prompt_id("2hop__101_201"), text=raw_text,
                          n_tokens=whitespace_tokens(raw_text), source="musique",
                          source_id="2hop__101_201", source_category="2"))
# ... store_redundant / store_compressate / store_inference für alle 16 ...

store.build_manifest()   # baut manifest.jsonl + manifest.csv aus dem Baum neu
```

Das Manifest wird **aus dem Baum neu gebaut**, nicht mitgeschrieben — so ist es
immer ein getreuer Index dessen, was tatsächlich auf der Platte liegt, und kann
nie „auseinanderlaufen". Neu bauen jederzeit auch per CLI:

```bash
pipelinestore build-manifest --root experiments --experiment run-2026-07
```

## Statistische Auswertung

`manifest.csv` direkt laden. Jede Zeile ist ein Kompressat; Faktoren sind
`redundancy_type` und `target_ratio`, Kovariaten die `n_tokens_*`-Spalten,
Zielgrößen `is_correct` / `metric_*`.

```python
import pandas as pd
df = pd.read_csv("experiments/musique-dev-demo/manifest.csv")

# Genauigkeit je Redundanztyp × Kompressionsrate
pivot = df.pivot_table(index="redundancy_type", columns="target_ratio",
                       values="is_correct", aggfunc="mean")

# tatsächlich erreichte vs. angezielte Kompression
df.groupby("target_ratio")["achieved_ratio"].mean()
```

`prompt_id` bleibt in jeder Zeile erhalten und dient in gemischten Modellen als
Zufallseffekt (dieselben 16 Kompressate stammen aus einem Prompt).

### Evidenz-Metriken (das, was die MuSiQue-Labels bringen)

Neben `is_correct` liefert jede Zeile drei Kennzahlen darüber, **was** die
Kompression behalten hat — messbar nur, weil MuSiQue Gold-Passagen auszeichnet:

| Spalte | Bedeutung |
|---|---|
| `metric_supporting_retained` | Anteil der Supporting-Passagen, der die Kompression als **geordnete Teilfolge** überlebt hat |
| `metric_distractor_retained` | dasselbe für die Distraktoren |
| `metric_evidence_selectivity` | Differenz der beiden. `> 0` heißt: der Kompressor bevorzugt Evidenz gegenüber Rauschen; `≈ 0` heißt, er kürzt nur, statt auszuwählen |

Damit lässt sich der interessante Fall vom uninteressanten trennen: Ein Abfall
der Genauigkeit bei niedriger Rate ist erst dann ein Kompressions*fehler*, wenn
die Selektivität dabei nicht steigt.

Zur Messung: Die Reihenfolge zählt mit, jedes Token wird höchstens einmal
verbraucht. Reine Mengenzugehörigkeit („kommt dieses Wort irgendwo im
komprimierten Prompt vor?") beantwortet jedes Funktionswort mit ja und meldet
bei 20 Passagen Kontext ~95 % Retention bei 20 % behaltenen Tokens — sie misst
Vokabularüberlappung, nicht Überleben. Redundante Kopien werden auf ihr Original
zurückgefaltet (Wertung: die bestüberlebende Kopie), damit die `passages`-
Bedingung nicht gegen einen größeren Nenner gemessen wird als die anderen drei.

## Beispiele im Repo

- **`musique-dev-demo/`** — echter End-to-End-Lauf über `run_pipeline.py` auf
  den ersten 3 Datensätzen von `musique_ans_v1.0_dev.jsonl` (3 Raw-Prompts →
  48 Kompressate). Kompression und Inferenz sind die offline Baselines, also
  weder LLMLingua noch ein echtes Modell — die Zahlen zeigen das Ablagelayout
  und die Metrik-Mechanik, sie sind kein inhaltliches Ergebnis.
- **`make_demo_experiment.py`** — erzeugt zusätzlich ein rein illustratives
  `demo-experiment/` **ohne** Abhängigkeit vom Redundanzgenerator (nur um das
  Speicher-Layout zu zeigen). Nicht eingecheckt, bei Bedarf lokal ausführen.

In beiden Fällen ist die „Kompression" ein deterministisches Token-Kürzen als
reproduzierbare Baseline — im echten Experiment den tatsächlichen Kompressor und
ein echtes Inferenzmodell über die `compress`/`infer`-Parameter einsetzen.

## Spalten des Manifests

`experiment_id, compressate_id, prompt_id, source, source_id, source_category,
redundancy_type, rate_label, target_ratio, achieved_ratio, n_tokens_raw,
n_tokens_redundant, n_tokens_compressed, redundancy_added_tokens,
compression_removed_tokens, model, output, gold_answer, is_correct, seed,`
Pfad-Spalten (`raw_path` …) sowie pro Metrik eine `metric_<name>`-Spalte.

`source_category` ist der datensatzseitige Stratum-Schlüssel — bei MuSiQue die
Hop-Zahl (2/3/4), also der natürliche Block- bzw. Kovariatenfaktor.

> Hinweis Tokenzählung: Standard ist eine dependency-freie Whitespace-Zählung
> (gut für relative Vergleiche). Für absolute Token-Budgets einen echten
> Tokenizer (z. B. `tiktoken`) als `token_counter` übergeben.
