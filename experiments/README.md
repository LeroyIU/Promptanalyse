# Experiment-Pipeline: Ablage & Auswertung

Ablagesystem für die Prompts entlang der Experiment-Pipeline. Die Pipeline hat
vier Stages, mit fester Verzweigung pro Raw-Prompt:

```
raw prompt  ─▶  variante  ─▶  kompressionsrate  ─▶  inferenz
   1 Raw   =   4 Varianten  ×   4 Kompressionsraten  =  16 Kompressate
```

Die **4 Varianten** pro Raw-Prompt sind die **Basis** (Kontrolle: der
Raw-Prompt ohne hinzugefügte Redundanz) plus die **3 Redundanztypen**
(`lexical`, `demonstrations`, `instructions`). Alle vier werden bei denselben
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

- `prompt_id` – z. B. `p-101` (aus PopQA-ID oder Zähler)
- `rate_label` – feste Breite, lexikalisch sortierbar: `cr020, cr040, cr060, cr080` (= 20/40/60/80 % behaltene Tokens)
- `compressate_id` – `p-101__lexical__cr040` → lässt sich zurück in Pfad und Koordinaten zerlegen

## End-to-End-Lauf (echter Redundanzgenerator)

`run_pipeline.py` verdrahtet den echten `redundanzgenerator` dieses Repos mit dem
Store: es baut Raw-Prompts aus PopQA, erzeugt die 4 Varianten (Basis + 3
Redundanztypen), komprimiert jede bei 4 Raten, bewertet sie und schreibt den
gesamten Stage-Baum plus Manifest.

```bash
pip install -e tools/PipelineStore          # Speicherung
pip install -e tools/Redundanzgenerator     # Redundanzgenerator (Stages 1–2)
python experiments/run_pipeline.py --experiment popqa-occupation-demo
```

Die beiden Stages, die keine Prompt-Konstruktion sind — **Kompression** und
**Inferenz** — sind als Callables **einsteckbar**. Standard sind offline
Baselines (deterministisches Token-Kürzen bzw. Gold-Containment-Scoring), damit
der Lauf ohne Modelle/Netz durchläuft; die echten Methoden (z. B. LLMLingua,
Claude API) setzt man mit gleicher Signatur ein:

```python
from pipelinestore import PipelineStore
from pipelinestore.redundancy_pipeline import run_pipeline, RedundancyCounts

def my_compress(text: str, target_ratio: float) -> str: ...      # z. B. LLMLingua-2
def my_infer(text: str, gold: list[str]): ...                    # z. B. Claude API

run_pipeline(
    PipelineStore("experiments", "run-2026-07"),
    popqa_source="datasets/popQA/test.tsv",
    popqa_tp_source="datasets/popQA/popQA_template_paraphrases.csv",
    query_ids=["4222362", "4725190", "4382392"],
    compression_rates=[0.2, 0.4, 0.6, 0.8],
    counts=RedundancyCounts(lexical=3, demonstrations=3, instructions=2),
    compress=my_compress, infer=my_infer,
)
```

`popqa-occupation-demo/` ist ein vollständig durchgerechneter echter Lauf (3
Raw-Prompts → 48 Kompressate) mit realen Paraphrasen/Demonstrationen aus PopQA.

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

store.store_raw(RawPrompt(prompt_id=ids.prompt_id(101), text=raw_text,
                          n_tokens=whitespace_tokens(raw_text), source="popqa",
                          source_id="101", source_prop="capital"))
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
df = pd.read_csv("experiments/popqa-occupation-demo/manifest.csv")

# Genauigkeit je Redundanztyp × Kompressionsrate
pivot = df.pivot_table(index="redundancy_type", columns="target_ratio",
                       values="is_correct", aggfunc="mean")

# tatsächlich erreichte vs. angezielte Kompression
df.groupby("target_ratio")["achieved_ratio"].mean()
```

`prompt_id` bleibt in jeder Zeile erhalten und dient in gemischten Modellen als
Zufallseffekt (dieselben 16 Kompressate stammen aus einem Prompt).

## Beispiele im Repo

- **`popqa-occupation-demo/`** — echter End-to-End-Lauf über `run_pipeline.py`
  (3 Raw-Prompts → 48 Kompressate) mit realen Paraphrasen und Demonstrationen aus
  PopQA. Kompression/Inferenz sind die offline Baselines.
- **`make_demo_experiment.py`** — erzeugt zusätzlich ein rein illustratives
  `demo-experiment/` **ohne** Abhängigkeit vom Redundanzgenerator (nur um das
  Speicher-Layout zu zeigen). Nicht eingecheckt, bei Bedarf lokal ausführen.

In beiden Fällen ist die „Kompression" ein deterministisches Token-Kürzen als
reproduzierbare Baseline — im echten Experiment den tatsächlichen Kompressor und
ein echtes Inferenzmodell über die `compress`/`infer`-Parameter einsetzen.

## Spalten des Manifests

`experiment_id, compressate_id, prompt_id, source, source_id, source_prop,
redundancy_type, rate_label, target_ratio, achieved_ratio, n_tokens_raw,
n_tokens_redundant, n_tokens_compressed, redundancy_added_tokens,
compression_removed_tokens, model, output, gold_answer, is_correct, seed,`
Pfad-Spalten (`raw_path` …) sowie pro Metrik eine `metric_<name>`-Spalte.

> Hinweis Tokenzählung: Standard ist eine dependency-freie Whitespace-Zählung
> (gut für relative Vergleiche). Für absolute Token-Budgets einen echten
> Tokenizer (z. B. `tiktoken`) als `token_counter` übergeben.
