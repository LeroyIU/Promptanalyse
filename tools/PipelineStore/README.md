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
store.store_raw(RawPrompt(prompt_id=ids.prompt_id(101), text=t,
                          n_tokens=whitespace_tokens(t), source="popqa"))
# store_redundant / store_compressate / store_inference ...
store.build_manifest()      # -> manifest.jsonl + manifest.csv
```

- `ids` – deterministisches ID-/Pfadschema (`p-101__lexical__cr040`), idempotent.
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
    popqa_source="datasets/popQA/test.tsv",
    popqa_tp_source="datasets/popQA/popQA_template_paraphrases.csv",
    query_ids=["4222362", "4725190"],
    compression_rates=[0.2, 0.4, 0.6, 0.8],
    counts=RedundancyCounts(lexical=3, demonstrations=3, instructions=2),
    # compress=..., infer=...  <- echte Methoden hier einsetzen
)
```

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
