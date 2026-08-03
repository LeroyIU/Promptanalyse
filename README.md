# Promptanalyse

Dieses Repository dokumentiert die im Rahmen meiner Abschlussarbeit durchgeführte Promptanalyse. Es enthält die verwendeten Tools und Skripte zur Erhebung und Auswertung, die zugrunde liegenden Datasets sowie die daraus gewonnenen Ergebnisse. Ziel ist es, den gesamten Analyseprozess nachvollziehbar und reproduzierbar zu dokumentieren.

## Inhalt

- **Tools** – Skripte und Werkzeuge zur Durchführung und Auswertung der Promptanalyse
  - *Redundanzgenerator* – erzeugt kontrollierte Redundanzen in Kontext-Prompts (Passagen, Demonstrationen, Instruktionen)
  - *PipelineStore* – strukturierte Ablage der Pipeline-Prompts + auswertbares Manifest
- **Datasets** – die verwendeten Datengrundlagen
  - *[MuSiQue](datasets/musique/README.md)* – Multi-Hop-QA mit Kontext: 20 Wikipedia-Passagen pro Frage, davon 2–4 mit Gold-Evidence-Label (`is_supporting`)
  - *[LongBench-E TriviaQA (8k+)](datasets/longbench_trivia_qa_e/README.md)* – Few-Shot Open-Domain-QA-Subset (100 Beispiele) aus dem 8k+-Token-Bucket von LongBench-E, für Prompt-Kompressionsexperimente im Long-Context-Regime
- **Experiments** – Ablage der Pipeline-Läufe (`raw → redundanztyp → kompressionsrate → inferenz`) samt Manifest für die statistische Auswertung; siehe [`experiments/README.md`](experiments/README.md)
- **Ergebnisse** – aus der Analyse hervorgegangene Auswertungen und Erkenntnisse

## Setup

Die Datendateien unter `datasets/` liegen in [Git LFS](https://git-lfs.com) (siehe [`.gitattributes`](.gitattributes)). Nach dem Klonen einmalig:

```bash
git lfs install    # macOS: brew install git-lfs, Ubuntu/Debian: sudo apt install git-lfs
git lfs pull
```

Ohne Git LFS enthalten die betroffenen Dateien nur Pointer-Text statt der Daten.

```bash
pip install -e tools/PipelineStore
pip install -e tools/Redundanzgenerator
python experiments/run_pipeline.py --experiment run-2026-07 --n-prompts 20
```

## Kontext

Dieses Projekt entstand im Rahmen einer Bachelor-/Masterarbeit und dient der wissenschaftlichen Auseinandersetzung mit Prompt-Engineering bzw. der Analyse von Prompts.
