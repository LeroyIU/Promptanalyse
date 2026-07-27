# Redundanzgenerator

Dieser Generator erstellt kontrollierte Redundanzen in Few-Shot-Prompts — als Werkzeug für Experimente zur Robustheit von LLMs gegenüber redundanten Prompt-Bestandteilen.

Drei Redundanzarten werden unterstützt:

| Redundanzart | Quelle | Wirkung |
|---|---|---|
| **Lexikalisch / Paraphrasen** | [PopQA-TP](https://huggingface.co/datasets/ibm-research/popqa-tp) | Die Query (optional auch Demonstrations-Fragen) wird zusätzlich in semantisch äquivalenten Umformulierungen gestellt |
| **Demonstrationen** | [PopQA](https://huggingface.co/datasets/akariasai/PopQA) | Zusätzliche, informativ redundante Beispiele aus derselben Relationskategorie (`prop`) wie die Query |
| **Instruktionen** | Regelbasierte Templates oder LLM (Claude API) | Die Instruktion wird wiederholt bzw. umformuliert, am Anfang und/oder Ende des Prompts |

Alle Strategien sind seed-gesteuert reproduzierbar, und jeder Lauf liefert einen `redundancy_report`, der genau dokumentiert, was wo eingefügt wurde.

Prompts gibt es in zwei Ausprägungen: **ohne Kontext** (PopQA, Closed-Book — das Modell antwortet aus dem Parameterwissen) und **mit Kontext** ([MuSiQue](../../datasets/musique/README.md), 20 Wikipedia-Passagen pro Frage, davon 2–4 mit Gold-Evidence-Label). Siehe [Prompts mit Kontext](#prompts-mit-kontext-musique).

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

Mit Kontext kommt ein `context`-Feld dazu (auf Prompt-Ebene und optional pro Demonstration):

```json
{
  "instructions": ["Answer the question using only the passages provided."],
  "demonstrations": [],
  "query": "In which country is the city where the Eiffel Tower stands?",
  "context": [
    {"text": "The Eiffel Tower is a wrought-iron lattice tower ...", "title": "Eiffel Tower", "is_supporting": true, "meta": {"idx": 0}},
    {"text": "Gustave Eiffel was a civil engineer ...", "title": "Gustave Eiffel", "is_supporting": false, "meta": {"idx": 1}}
  ],
  "meta": {"id": "2hop__101_201", "n_hops": 2, "n_supporting": 2, "n_distractors": 18}
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

Prompt **mit Kontext** aus MuSiQue bauen (20 Passagen pro Frage):

```bash
redundanzgen build-context \
  --musique datasets/musique/musique_ans_v1.0_dev.jsonl \
  --query-id 2hop__128801_205185 \
  --output prompt.json --text prompt.txt
```

Weitere Optionen: `--n-demos N` (Demonstrationen mit derselben Hop-Zahl; Default 0 = Zero-Shot, weil der Kontext den Prompt dominieren soll), `--demo-context` (Demonstrationen bringen ihre eigenen Passagen mit — vervielfacht die Prompt-Länge), `--no-distractors` (nur die Supporting-Passagen, also der Orakel-Kontext, den jede Kompression idealerweise treffen würde).

Als Datenquelle (`--popqa`, `--popqa-tp`, `--musique`) funktioniert jeweils eine Hugging-Face-Dataset-ID (benötigt Extra `[hf]`) **oder** eine lokale CSV-/TSV-/JSON-/JSONL-Datei mit denselben Feldern (das mitgelieferte `datasets/popQA/test.tsv` lädt direkt; MuSiQue wird per `datasets/musique/fetch_musique.py` geholt).

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

## Prompts mit Kontext (MuSiQue)

`FewShotPrompt.context` und `Demonstration.context` nehmen `ContextPassage`-Objekte auf (`text`, `title`, `is_supporting`, `meta`); `render_prompt` setzt sie als nummerierten `Context:`-Block direkt über die zugehörige Frage. `is_supporting` trägt das Gold-Evidence-Label des Datensatzes mit — dadurch lässt sich nach der Kompression nicht nur messen, *ob* die Antwort noch stimmt, sondern auch, *welche* Passagen überlebt haben.

```python
from redundanzgenerator import MuSiQueLoader, render_prompt

musique = MuSiQueLoader("datasets/musique/musique_ans_v1.0_dev.jsonl")
prompt = musique.build_prompt("2hop__128801_205185")
print(render_prompt(prompt))

row = musique.by_id("2hop__128801_205185")
musique.supporting_paragraphs(row)   # 2-4 Gold-Passagen
musique.distractor_paragraphs(row)   # 16-18 Distraktoren
musique.answers_of(row)              # Antwort + Aliase
musique.by_hops(3)                   # Hop-Zahl als Sampling-Kategorie (wie `prop` bei PopQA)
```

Die Nummerierung im Kontext-Block ist bewusst: Eine Passage, die nach der Kompression fehlt, bleibt über ihren Index identifizierbar und lässt sich gegen die Gold-Labels abgleichen.

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

Die Tests laufen offline gegen kleine Fixture-Dateien (`tests/fixtures/`) mit dem echten PopQA-/PopQA-TP-/MuSiQue-Schema.
