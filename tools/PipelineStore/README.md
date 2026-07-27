# PipelineStore

Strukturierte Ablage für die Redundanz-/Kompressions-Experiment-Pipeline und
Aufbau eines auswertbaren „tidy" Manifests.

Stages: `raw prompt → redundanztyp → kompressionsrate → inferenz`.
Verzweigung: **1 Raw-Prompt = 4 Redundanztypen × 4 Kompressionsraten = 16 Kompressate.**

Zwei-Ebenen-Modell:

- **Artefakte** (variabel lange Prompt-Texte) → eine JSON-Datei pro Stage-Knoten
  im partitionierten Verzeichnisbaum.
- **Messwerte** (IDs, Tokenzahlen, Scores) → eine Zeile pro Kompressat, als
  `manifest.jsonl` (verlustfrei) und `manifest.csv` (auswertungsfertig).

Das Gesamtkonzept, das Verzeichnislayout und Auswertungs-Snippets stehen in
[`../../experiments/README.md`](../../experiments/README.md).

## Installation

```bash
pip install -e .            # nur Standardbibliothek
pip install -e ".[dev]"     # + pytest
```

## API

```python
from pipelinestore import (
    PipelineStore, RawPrompt, RedundantVariant, Compressate, InferenceResult,
    ids, whitespace_tokens,
)

store = PipelineStore("experiments", "run-2026-07")
store.store_raw(RawPrompt(prompt_id=ids.prompt_id("2hop__101_201"), text=t,
                          n_tokens=whitespace_tokens(t), source="musique"))
# store_redundant / store_compressate / store_inference ...
store.build_manifest()      # -> manifest.jsonl + manifest.csv
```

- `ids` – deterministisches ID-/Pfadschema (`p-2hop-101-201__passages__cr040`), idempotent.
- `models` – Datensätze je Stage (`RawPrompt`, `RedundantVariant`, `Compressate`,
  `InferenceResult`) plus `TidyRow` (flache Auswertungszeile).
- `store` – Pfade, Schreiben der Artefakte, Neubau des Manifests aus dem Baum.

## End-to-End mit dem Redundanzgenerator

`pipelinestore.redundancy_pipeline` verdrahtet den echten `redundanzgenerator`
(Stages 1–2) mit dem Store und fährt die volle Verzweigung (Basis + 3
Redundanztypen × 4 Raten). Kompression und Inferenz sind einsteckbare Callables
mit offline Baselines als Default. Der `redundanzgenerator` wird **lazy**
importiert — das Kern-Package bleibt abhängigkeitsfrei; separat editable
installieren:

```bash
pip install -e ../Redundanzgenerator
```

```python
from pipelinestore import PipelineStore
from pipelinestore.redundancy_pipeline import run_pipeline, RedundancyCounts

run_pipeline(
    PipelineStore("experiments", "run-2026-07"),
    musique_source="datasets/musique/musique_ans_v1.0_dev.jsonl",
    query_ids=["2hop__128801_205185", "3hop1__..."],
    compression_rates=[0.2, 0.4, 0.6, 0.8],
    counts=RedundancyCounts(passages=3, demonstrations=3, instructions=2),
    # compress=..., infer=...  <- echte Methoden hier einsetzen
)
```

Der Default-Kompressor ist `query_preserving_compressor`: Er verteilt das
Token-Budget auf den Kontext und lässt den Frageblock stehen — wie es echte
Prompt-Kompressoren tun. Reines Kürzen von vorn (`head_ratio_compressor`)
schneidet bei Kontext-Prompts die Frage ab, die ganz am Ende steht, und liefert
dann in jeder Bedingung 0 % Genauigkeit aus demselben nichtssagenden Grund.

Neben `is_correct` schreibt jeder Lauf `evidence_retention`-Metriken ins Manifest
(`metric_supporting_retained`, `metric_distractor_retained`,
`metric_evidence_selectivity`): Sie nutzen die Gold-Labels von MuSiQue, um zu
messen, *was* die Kompression behalten hat — nicht nur, ob die Antwort überlebt hat.

Runnable driver: [`../../experiments/run_pipeline.py`](../../experiments/run_pipeline.py).

## CLI

```bash
pipelinestore build-manifest --root experiments --experiment run-2026-07
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
