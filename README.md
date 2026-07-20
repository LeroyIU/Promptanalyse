# Promptanalyse

Dieses Repository dokumentiert die im Rahmen meiner Abschlussarbeit durchgeführte Promptanalyse. Es enthält die verwendeten Tools und Skripte zur Erhebung und Auswertung, die zugrunde liegenden Datasets sowie die daraus gewonnenen Ergebnisse. Ziel ist es, den gesamten Analyseprozess nachvollziehbar und reproduzierbar zu dokumentieren.

## Inhalt

- **Tools** – Skripte und Werkzeuge zur Durchführung und Auswertung der Promptanalyse
  - *Redundanzgenerator* – erzeugt kontrollierte Redundanzen in Few-Shot-Prompts
  - *PipelineStore* – strukturierte Ablage der Pipeline-Prompts + auswertbares Manifest
- **Datasets** – die verwendeten Datengrundlagen
- **Experiments** – Ablage der Pipeline-Läufe (`raw → redundanztyp → kompressionsrate → inferenz`) samt Manifest für die statistische Auswertung; siehe [`experiments/README.md`](experiments/README.md)
- **Ergebnisse** – aus der Analyse hervorgegangene Auswertungen und Erkenntnisse

## Kontext

Dieses Projekt entstand im Rahmen einer Bachelor-/Masterarbeit und dient der wissenschaftlichen Auseinandersetzung mit Prompt-Engineering bzw. der Analyse von Prompts.
