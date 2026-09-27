# PopQA-Stichprobe (Abschnitt 4.4.3)

Geschachtelte, geschichtete Ziehung aus PopQA (14.267 Fragen) mit festem
Zufallsstartwert `SEED = 42`.

## Inhalt

| Datei | Beschreibung |
|---|---|
| `prompts/<id>.txt` | 300 Dateien, eine je Frage der Grundstichprobe. Dateiname = PopQA-ID, Inhalt = Fragetext. |
| `sample_a.tsv` | Grundstichprobe A (300 Zeilen) mit Kategorie, Terzil, Popularität, Antworten, Flag `in_subset_b`. |
| `sample_b.tsv` | Teilstichprobe B (30 Zeilen), vollständig in A enthalten. |
| `sample_a_ids.txt`, `sample_b_ids.txt` | reine ID-Listen. |
| `demo_pool_ids.txt` | Demonstrationspool, 13.604 IDs, disjunkt zu A. |
| `demo_assignments.tsv` | 300 × 8 Demonstrationen im Langformat (`role` = `base` / `demonstration`). |
| `draw_manifest.json` | Seed, Quoten, Paraphrasenzahlen, ausgeschlossene Kategorie, Terzilverfahren. |
| `draw_sample.py` | Ziehungsskript, exakt wiederholbar. |
| `verify_sample.py` | unabhängiges Prüfskript (10 Prüfblöcke). |

## Ausschluss von `capital of`

Die Paraphrasen in PopQA-TP (Rabinovich et al. 2023) sind relationsspezifische
Templates und gelten damit einheitlich für alle Fragen einer Kategorie. Zahl der
Paraphrasen ohne die Originalformulierung: director, producer, screenwriter 10;
country 9; capital, genre, place of birth, sport 6; author, color, composer,
mother, occupation, religion 5; father 4; **capital of 3**.

`capital of` verfehlt als einzige Kategorie die Bedingung von mindestens vier
Paraphrasen und ist deshalb vollständig ausgeschlossen — aus der Stichprobe wie
aus dem Demonstrationspool. Die Studie arbeitet auf 15 Relationskategorien.

## Ziehungsverfahren

**Schichtung nach Relationskategorie.** 300 = 10 × 21 + 5 × 18. Jede Quote ist
durch 3 teilbar, damit die Terzile innerhalb jeder Kategorie exakt gleich
besetzt sind; bei 15 Kategorien ist das die geringstmögliche Spreizung. Die
zehn größten Kategorien erhalten 21, die fünf kleinsten (sport, occupation,
religion, mother, color) 18. Verhältnis größte zu kleinster Kategorie 21:18.

**Schichtung nach Popularität.** Innerhalb jeder Kategorie wird `s_pop`
rangbasiert nach dem eindeutigen Schlüssel `(s_pop, id)` in drei gleich große
Terzile geteilt; aus jedem Terzil wird gleich viel gezogen (6 bzw. 7).
Rangbasiert statt wertbasiert, weil `s_pop` viele Bindungen enthält (z. B. 552
in `author`), die bei Wertgrenzen ungleich große Schichten erzeugen würden.
Global entfallen auf jedes Terzil 100 Fragen.

**Teilstichprobe B.** 30 Fragen, gezogen aus A: genau 2 je Kategorie
(15 × 2 = 30) und global exakt 10 Fragen je Popularitätsterzil.

**Demonstrationspool.** Die 13.604 nicht gezogenen Fragen der 15
Studienkategorien, disjunkt zu A. Je Aufgabe werden 8 Demonstrationen aus 8
paarweise verschiedenen Kategorien gezogen, keine aus der Kategorie der
Testfrage. Slots 1–4 gehören zum Basisprompt, Slots 5–8 zur Redundanzstufe
`demonstration`. Die Zuweisung wird einmal gezogen und ist über alle Varianten
und Reduktionsstufen einer Aufgabe konstant.

## Reproduktion

```
python3 draw_sample.py popQA.tsv popqa_subset
python3 verify_sample.py popqa_subset popQA.tsv
```

Ein zweiter Lauf erzeugt bitgleiche Ausgabe (in `verify_sample.py` per
SHA-256-Vergleich geprüft).
