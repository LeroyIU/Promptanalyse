# Redundanzgenerator

Dieser Generator erstellt kontrollierte Redundanzen in Few-Shot-Prompts — als Werkzeug für Experimente zur Robustheit von LLMs gegenüber redundanten Prompt-Bestandteilen.

Drei Redundanzarten werden unterstützt:

| Redundanzart | Quelle | Wirkung |
|---|---|---|
| **Lexikalisch / Paraphrasen** | [PopQA-TP](https://huggingface.co/datasets/ibm-research/popqa-tp) | Die Query (optional auch Demonstrations-Fragen) wird zusätzlich in semantisch äquivalenten Umformulierungen gestellt |
| **Demonstrationen** | [PopQA](https://huggingface.co/datasets/akariasai/PopQA) | Zusätzliche, informativ redundante Beispiele aus derselben Relationskategorie (`prop`) wie die Query |
| **Instruktionen** | Regelbasierte Templates oder LLM (Claude API) | Die Instruktion wird wiederholt bzw. umformuliert, am Anfang und/oder Ende des Prompts |

Alle Strategien sind seed-gesteuert reproduzierbar, und jeder Lauf liefert einen `redundancy_report`, der genau dokumentiert, was wo eingefügt wurde.

## Installation

```bash
pip install -e .            # Kern (nur Standardbibliothek)
pip install -e ".[hf]"      # + Hugging-Face-Datasets direkt vom Hub laden
pip install -e ".[llm]"     # + LLM-basierte Instruktions-Umformulierung
pip install -e ".[dev]"     # + pytest
```

## Prompt-Format (JSON)

```json
{
  "instruction": "Answer the following question with a short factual answer.",
  "demonstrations": [
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "meta": {"id": 102, "prop": "capital"}}
  ],
  "query": "What is the capital of France?",
  "meta": {"id": 101, "prop": "capital"}
}
```

`meta.id` (PopQA-ID) und `meta.prop` (Kategorie) sind optional, verbessern aber das Matching: Paraphrasen werden zuerst über die ID gesucht, dann über den Fragetext; die Kategorie der Query wird für redundante Demonstrationen genutzt.

## CLI

Redundanz in einen bestehenden Prompt einfügen:

```bash
redundanzgen generate \
  --input prompt.json \
  --lexical 2 \                # 2 Query-Paraphrasen aus PopQA-TP
  --demos 3 \                  # 3 redundante Demonstrationen aus PopQA
  --instructions 1 \           # 1 redundante Instruktions-Formulierung
  --popqa akariasai/PopQA \    # oder lokale CSV/JSON-Datei
  --popqa-tp ibm-research/popqa-tp \
  --seed 42 \
  --output out.json --text out.txt
```

Weitere Optionen: `--paraphrase-demos` (auch Demonstrations-Fragen paraphrasieren), `--instruction-position start|end|both`, `--llm` (Instruktionen per Claude API umformulieren, benötigt `ANTHROPIC_API_KEY`).

Basis-Few-Shot-Prompt direkt aus PopQA bauen:

```bash
redundanzgen build --popqa akariasai/PopQA --query-id 12345 --n-demos 4 --output prompt.json
```

Als Datenquelle (`--popqa`, `--popqa-tp`) funktioniert jeweils eine Hugging-Face-Dataset-ID (benötigt Extra `[hf]`) **oder** eine lokale CSV-/JSON-/JSONL-Datei mit denselben Spalten.

## Bibliotheks-API

```python
from redundanzgenerator import (
    FewShotPrompt, RedundancyConfig, RedundancyGenerator, render_prompt,
)

prompt = FewShotPrompt.from_json("prompt.json")
generator = RedundancyGenerator.from_config(
    RedundancyConfig(n_paraphrases=2, n_demonstrations=3, n_instructions=1, seed=42),
    popqa="akariasai/PopQA",
    popqa_tp="ibm-research/popqa-tp",
)
redundant_prompt, report = generator.generate(prompt)   # Input bleibt unverändert
print(render_prompt(redundant_prompt))
```

Eigene Strategien lassen sich über das `RedundancyStrategy`-Interface (`strategies/base.py`) ergänzen und direkt an `RedundancyGenerator(strategies=[...], seed=...)` übergeben.

## Beispiel-Output

```
Answer the following question with a short factual answer.
Again, your task is as follows: Answer the following question with a short factual answer.

Q: What is the capital of Japan?
A: Tokyo

Q: What is the capital of Egypt?
A: Cairo

Q: What is the capital of France?
Q: What city is the capital of France?
Q: Name the capital city of France.
A:
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Die Tests laufen offline gegen kleine Fixture-Dateien (`tests/fixtures/`) mit dem echten PopQA-/PopQA-TP-Schema.
