# Experiment-Pipeline: Ablage & Auswertung

Ablagesystem für die Prompts entlang der Experiment-Pipeline. Die Pipeline hat
vier Stages, mit fester Verzweigung pro Raw-Prompt:

```
raw prompt  ─▶  redundanztyp  ─▶  kompressionsrate  ─▶  inferenz
   1 Raw   =   4 Redundanztypen  ×   4 Kompressionsraten  =  16 Kompressate
```

Jedes **Kompressat** (= 1 Redundanztyp bei 1 Kompressionsrate) ist die
statistische **Beobachtungseinheit**. Es gibt genau 16 davon pro Raw-Prompt.

## Warum diese Struktur (die eigentliche Design-Entscheidung)

Speicherung und Statistik haben gegensätzliche Anforderungen. Deshalb werden
**zwei Ebenen getrennt**:

| Ebene | Inhalt | Format | Wofür |
|---|---|---|---|
| **Artefakte** | die eigentlichen Prompt-Texte (raw, redundant, komprimiert, Inferenz-Output) | eine **JSON-Datei pro Knoten** im nach Stages partitionierten Verzeichnisbaum | verlustfreie, reproduzierbare, versionierbare Ablage |
| **Messwerte** | IDs, Redundanztyp, Rate, Tokenzahlen, Korrektheit, Metriken | **eine „tidy" Long-Format-Tabelle**: `manifest.jsonl` (verlustfrei) + `manifest.csv` (flach) | direkt in pandas/R für die statistische Auswertung |

Die lange Textmenge liegt also in Dateien; die auswertbaren Zahlen werden in
**eine Zeile pro Kompressat** projiziert. Stabile IDs verbinden beide Ebenen,
sodass jede Manifest-Zeile auf ihre Artefakt-Dateien zeigt und umgekehrt.

**Warum Long-/Tidy-Format?** Eine Zeile = eine Beobachtung ist die kanonische
Form für Statistik: `groupby`, ANOVA, gemischte Modelle (Prompt als
Zufallseffekt), Plots über `redundancy_type × target_ratio` funktionieren ohne
Umbau. Ein „wide" Format (16 Spalten pro Raw-Prompt) müsste man dafür erst
wieder aufschmelzen.

## Verzeichnislayout

```
experiments/<experiment_id>/
├── experiment.json                       # Konfiguration: Modell, Seeds, Enums
├── 01_raw/<prompt_id>.json               # Stage 1: Original-Prompt
├── 02_redundant/<prompt_id>/<typ>.json   # Stage 2: + 1 Redundanztyp
├── 03_compressed/<prompt_id>/<typ>/<rate>.json   # Stage 3: komprimiert
├── 04_inference/<prompt_id>/<typ>/<rate>.json    # Stage 4: Modell-Output + Score
├── manifest.jsonl                        # 1 Datensatz pro Kompressat (verlustfrei)
└── manifest.csv                          # dieselben Daten flach (Auswertung)
```

### ID- und Namensschema

Die IDs sind rein aus ihren Koordinaten abgeleitet (nicht aus Inhalt/Zeit),
damit erneute Läufe **idempotent** sind — dieselbe Koordinate überschreibt am
selben Pfad statt Duplikate anzuhäufen.

- `prompt_id` – z. B. `p-101` (aus PopQA-ID oder Zähler)
- `rate_label` – feste Breite, lexikalisch sortierbar: `cr020, cr040, cr060, cr080` (= 20/40/60/80 % behaltene Tokens)
- `compressate_id` – `p-101__lexical__cr040` → lässt sich zurück in Pfad und Koordinaten zerlegen

## Nutzung

```bash
pip install -e tools/PipelineStore          # einmalig
python experiments/make_demo_experiment.py  # erzeugt das Beispiel unten neu
```

Im Pipeline-Code (schematisch — die realen Generatoren/Kompressoren einsetzen,
die Store-Aufrufe bleiben gleich):

```python
from pipelinestore import (
    PipelineStore, RawPrompt, RedundantVariant, Compressate, InferenceResult,
    ids, whitespace_tokens,
)

store = PipelineStore("experiments", "run-2026-07")
store.store_experiment({"inference_model": "...", "seed": 42,
                        "redundancy_types": [...], "compression_rates": [0.2, 0.4, 0.6, 0.8]})

store.store_raw(RawPrompt(prompt_id=ids.prompt_id(101), text=raw_text,
                          n_tokens=whitespace_tokens(raw_text), source="popqa",
                          source_id="101", source_prop="capital"))
# ... store_redundant / store_compressate / store_inference für alle 16 ...

store.build_manifest()   # baut manifest.jsonl + manifest.csv aus dem Baum neu
```

Das Manifest wird **aus dem Baum neu gebaut**, nicht mitgeschrieben — so ist es
immer ein getreuer Index dessen, was tatsächlich auf der Platte liegt, und kann
nie „auseinanderlaufen". Neu bauen jederzeit auch per CLI:

```bash
pipelinestore build-manifest --root experiments --experiment run-2026-07
```

## Statistische Auswertung

`manifest.csv` direkt laden. Jede Zeile ist ein Kompressat; Faktoren sind
`redundancy_type` und `target_ratio`, Kovariaten die `n_tokens_*`-Spalten,
Zielgrößen `is_correct` / `metric_*`.

```python
import pandas as pd
df = pd.read_csv("experiments/demo-experiment/manifest.csv")

# Genauigkeit je Redundanztyp × Kompressionsrate
pivot = df.pivot_table(index="redundancy_type", columns="target_ratio",
                       values="is_correct", aggfunc="mean")

# tatsächlich erreichte vs. angezielte Kompression
df.groupby("target_ratio")["achieved_ratio"].mean()
```

`prompt_id` bleibt in jeder Zeile erhalten und dient in gemischten Modellen als
Zufallseffekt (dieselben 16 Kompressate stammen aus einem Prompt).

## Beispiel

`demo-experiment/` ist ein vollständig durchgerechnetes Beispiel (3 Raw-Prompts
→ 48 Kompressate), erzeugt von `make_demo_experiment.py`. Die dortige
„Kompression" ist ein triviales Token-Abschneiden als Platzhalter, damit das
Beispiel offline ohne Modelle läuft — im echten Lauf den Redundanzgenerator und
den echten Kompressor einsetzen.

## Spalten des Manifests

`experiment_id, compressate_id, prompt_id, source, source_id, source_prop,
redundancy_type, rate_label, target_ratio, achieved_ratio, n_tokens_raw,
n_tokens_redundant, n_tokens_compressed, redundancy_added_tokens,
compression_removed_tokens, model, output, gold_answer, is_correct, seed,`
Pfad-Spalten (`raw_path` …) sowie pro Metrik eine `metric_<name>`-Spalte.

> Hinweis Tokenzählung: Standard ist eine dependency-freie Whitespace-Zählung
> (gut für relative Vergleiche). Für absolute Token-Budgets einen echten
> Tokenizer (z. B. `tiktoken`) als `token_counter` übergeben.
