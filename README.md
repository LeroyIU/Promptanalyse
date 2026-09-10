# Promptanalyse

Werkzeuge und Datenbestand zur Bachelorarbeit ueber den Umgang von
Promptkompression mit struktureller Redundanz in Few-shot-Prompts.

## Aufbau

| Ordner | Inhalt |
|---|---|
| `datasets/popQA` | PopQA und PopQA-TP als Rohdaten, dazu die Datensatzkarte. |
| `tools/PromptGenerator` | Ziehung der Stichprobe: `draw_sample.py`, `verify_sample.py`, das Ergebnis (`sample_a.tsv`, `sample_b.tsv`, ID-Listen, `demo_pool_ids.txt`, `demo_assignments.tsv`) und `draw_manifest.json`. |
| `results` | Die Pipeline beider Teilstudien und ihr Bestand in `promptanalyse.db`, dazu die Exporte fuer die Arbeit. Ablauf und Begruendung stehen in `results/README.md`. |

Die Datenbank ist die einzige Ablage des Experiments. Die frueheren Ordner
`0_questions`, `1_prompts`, `2_redundancy` und `3_compressed` und die
dateibasierten Vorlaeuferskripte sind entfallen; ihre Aufgaben erledigen
`results/02_build_prompts.py` bis `results/05_selectivity.py`.

## Reihenfolge

```
cd tools/PromptGenerator
python3 draw_sample.py ../../datasets/popQA/popQA.tsv .   # nur zur Reproduktion der Ziehung
cd ../../results
python3 01_init_db.py                                     # weiter siehe results/README.md
```

`promptanalyse.db` liegt ausserhalb der Versionsverwaltung.
