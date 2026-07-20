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

## CLI

```bash
pipelinestore build-manifest --root experiments --experiment run-2026-07
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
