# Redundanzgenerator

Dieser Generator erstellt kontrollierte Redundanzen in Kontext-Prompts — als Werkzeug für Experimente zur Robustheit von LLMs gegenüber redundanten Prompt-Bestandteilen unter Kompression.

Grundlage ist [MuSiQue](../../datasets/musique/README.md): jede Frage bringt ~20 Wikipedia-Passagen mit, von denen 2–4 als `is_supporting` (Gold-Evidenz) ausgezeichnet sind.

Drei Redundanzarten werden unterstützt — je eine pro Bestandteil eines Kontext-Prompts:

| Redundanzart | Ziel im Prompt | Wirkung |
|---|---|---|
| **Passagen** | Kontext | Redundante Kopien von Kontext-Passagen, standardmäßig der Gold-Passagen: verbatim (`duplicate`) oder in ein umformulierendes Template gewickelt (`restate`) |
| **Demonstrationen** | Beispiele | Zusätzliche, informativ redundante Beispiele mit derselben Hop-Zahl wie die Query |
| **Instruktionen** | Anweisung | Die Instruktion wird wiederholt bzw. umformuliert, am Anfang und/oder Ende des Prompts (regelbasierte Templates oder Claude API) |

Warum die Redundanz im Kontext liegt: Der Kontext ist bei Weitem der größte Teil des Prompts und damit der Teil, auf den ein Kompressor sein Budget verwendet. Redundanz nur in Query oder Instruktion wäre für die Kompression fast folgenlos. Dupliziert wird bevorzugt Gold-Evidenz, weil ein zweites Exemplar eines Distraktors nur Tokens kostet, ein zweites Exemplar der Evidenz aber die eigentliche Frage stellt: Erkennt der Kompressor die Redundanz — oder gibt er Budget doppelt aus und verliert dafür etwas anderes?

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

`context` gibt es auf Prompt-Ebene und optional pro Demonstration. `meta.id` (MuSiQue-ID) und `meta.n_hops` sind optional, verbessern aber das Matching: die Hop-Zahl der Query steuert, woraus redundante Demonstrationen gezogen werden; ohne sie wird sie über ID bzw. Fragetext nachgeschlagen.

## CLI

Prompt mit Kontext aus MuSiQue bauen:

```bash
redundanzgen build \
  --musique datasets/musique/musique_ans_v1.0_dev.jsonl \
  --query-id 2hop__128801_205185 \
  --output prompt.json --text prompt.txt
```

Weitere Optionen: `--n-demos N` (Demonstrationen mit derselben Hop-Zahl; Default 0 = Zero-Shot, weil der Kontext den Prompt dominieren soll), `--demo-context` (Demonstrationen bringen ihre eigenen Passagen mit — vervielfacht die Prompt-Länge), `--no-distractors` (nur die Supporting-Passagen, also der Orakel-Kontext, den jede Kompression idealerweise treffen würde).

Redundanz in einen bestehenden Prompt einfügen:

```bash
redundanzgen generate \
  --input prompt.json \
  --passages 3 \               # 3 redundante Kopien von Kontext-Passagen
  --demos 3 \                  # 3 redundante Demonstrationen gleicher Hop-Zahl
  --instructions 1 \           # 1 redundante Instruktions-Formulierung
  --musique datasets/musique/musique_ans_v1.0_dev.jsonl \
  --seed 42 \
  --output out.json --text out.txt
```

Weitere Optionen: `--passage-mode duplicate|restate`, `--passage-target supporting|any`, `--passage-position interleave|append`, `--demo-context`, `--instruction-position start|end|both`, `--llm` (Instruktionen per Claude API umformulieren, benötigt `ANTHROPIC_API_KEY`).

Als Datenquelle (`--musique`) funktioniert eine Hugging-Face-Dataset-ID (benötigt Extra `[hf]`) **oder** eine lokale JSONL-/JSON-/CSV-Datei mit denselben Feldern. Der Datensatz ist nicht eingecheckt und wird per `datasets/musique/fetch_musique.py` geholt.

Nur die Passagen-Redundanz braucht überhaupt keine Datenquelle — sie arbeitet auf dem Kontext, den der Prompt schon mitbringt.

## Bibliotheks-API

```python
from redundanzgenerator import (
    MuSiQueLoader, RedundancyConfig, RedundancyGenerator, render_prompt,
)

musique = MuSiQueLoader("datasets/musique/musique_ans_v1.0_dev.jsonl")
prompt = musique.build_prompt("2hop__128801_205185")

generator = RedundancyGenerator.from_config(
    RedundancyConfig(n_passages=3, n_demonstrations=3, n_instructions=1, seed=42),
    musique=musique,
)
redundant_prompt, report = generator.generate(prompt)   # Input bleibt unverändert
print(render_prompt(redundant_prompt))
```

Weitere Zugriffe auf den Datensatz:

```python
row = musique.by_id("2hop__128801_205185")
musique.supporting_paragraphs(row)   # 2-4 Gold-Passagen
musique.distractor_paragraphs(row)   # 16-18 Distraktoren
musique.answers_of(row)              # Antwort + Aliase
musique.by_hops(3)                   # Hop-Zahl als Sampling-Kategorie
```

Eigene Strategien lassen sich über das `RedundancyStrategy`-Interface (`strategies/base.py`) ergänzen und direkt an `RedundancyGenerator(strategies=[...], seed=...)` übergeben.

## Kontext im Prompt-Modell

`FewShotPrompt.context` und `Demonstration.context` nehmen `ContextPassage`-Objekte auf (`text`, `title`, `is_supporting`, `meta`); `render_prompt` setzt sie als nummerierten `Context:`-Block direkt über die zugehörige Frage. `is_supporting` trägt das Gold-Evidence-Label des Datensatzes mit — dadurch lässt sich nach der Kompression nicht nur messen, *ob* die Antwort noch stimmt, sondern auch, *welche* Passagen überlebt haben (siehe die `metric_*_retained`-Spalten in [`experiments/README.md`](../../experiments/README.md)).

Die Nummerierung im Kontext-Block ist bewusst: Eine Passage, die nach der Kompression fehlt, bleibt über ihren Index identifizierbar und lässt sich gegen die Gold-Labels abgleichen. Redundante Kopien tragen zusätzlich `meta.redundant_copy_of` mit dem Index ihres Originals, sodass eingefügte Redundanz später vom Originalmaterial unterscheidbar bleibt.

## Beispiel-Output

```
Answer the question using only the passages provided. Respond with a short factual answer.
Remember: Answer the question using only the passages provided. Respond with a short factual answer.

Context:
[1] Eiffel Tower
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
[2] Gustave Eiffel
Gustave Eiffel was a civil engineer known for iron structures and bridges.
[3] Paris
Paris is the capital and most populous city of France.

Q: In which country is the city where the Eiffel Tower stands?
Again, your task is as follows: Answer the question using only the passages provided. Respond with a short factual answer.
A:
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Die Tests laufen offline gegen kleine Fixture-Dateien (`tests/fixtures/`) im echten MuSiQue-Schema.
