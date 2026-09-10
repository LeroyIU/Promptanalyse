# Promptanalyse

Werkzeuge und Datenbestand zur Bachelorarbeit ueber den Umgang von
Promptkompression mit struktureller Redundanz in Few-shot-Prompts.

## Aufbau

Werkzeug und Bestand sind getrennt: unter `tools` stehen die Skripte, unter
`results` liegt, was sie erzeugen.

| Ordner | Inhalt |
|---|---|
| `datasets/popQA` | PopQA und PopQA-TP als Rohdaten, dazu die Datensatzkarte. |
| `tools/PromptGenerator` | Ziehung der Stichprobe: `draw_sample.py`, `verify_sample.py` und deren Ergebnis (`sample_a.tsv`, `sample_b.tsv`, ID-Listen, `demo_pool_ids.txt`, `demo_assignments.tsv`, `draw_manifest.json`). |
| `tools/Pipeline` | Die Pipeline beider Teilstudien, `01_init_db.py` bis `09_auswertung.py`, dazu `dbio.py`, `store.py`, `promptbau.py` und `schema.sql`. Ablauf und Begruendung stehen in `tools/Pipeline/README.md`. |
| `results` | `promptanalyse.db` als einziger Bestand des Experiments, dazu die CSV-Exporte fuer die Arbeit und die `schutz_*.csv` zur Pruefung von V4. |

Die frueheren Ordner `0_questions`, `1_prompts`, `2_redundancy` und
`3_compressed` und die dateibasierten Vorlaeuferskripte sind entfallen; ihre
Aufgaben erledigen `tools/Pipeline/02_build_prompts.py` bis
`05_selectivity.py`.

## Reihenfolge

```
cd tools/PromptGenerator
python3 draw_sample.py ../../datasets/popQA/popQA.tsv .   # nur zur Reproduktion der Ziehung
cd ../Pipeline
python3 01_init_db.py                                     # weiter siehe tools/Pipeline/README.md
```

`promptanalyse.db` liegt ausserhalb der Versionsverwaltung.
