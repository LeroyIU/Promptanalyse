# Datenhaltung der Promptanalyse

Der gesamte Bestand beider Teilstudien liegt in einer SQLite-Datei,
`promptanalyse.db`. Die frueheren Ordner `0_questions`, `1_prompts`,
`2_redundancy` und `3_compressed` werden nicht mehr beschrieben.

## Warum eine Datenbank

Die Ordnerform hielt den Prompttext, aber nicht seine Eigenschaften. Alles,
was die Arbeit auswertet, musste aus Dateinamen und Manifestdateien
rekonstruiert werden: die Reduktionsstufe aus dem Dateinamen, die Variante aus
dem Ordnernamen, die Auszeichnung von Kern und Redundanz aus einem
nachtraeglichen Abgleich zweier Textfassungen. Jede dieser Rekonstruktionen
ist eine Fehlerquelle und keine davon ist noetig, wenn die Eigenschaft dort
steht, wo sie entsteht.

Drei Dinge werden dadurch erst moeglich:

1. **Die Auszeichnung ist Konstruktion, nicht Rekonstruktion.** Nach
   `subsubsec:auszeichnung` wird jedes Wort schon beim Bau des Prompts als
   Kern oder Redundanz markiert. Das leistet die Tabelle `wort`. Nach der
   Kompression bleibt nur zu bestimmen, welche Positionen erhalten sind.
2. **Kontrollgroessen fallen beim Bau an.** Kalibrierung (V2), lexikalische
   Ueberlappung (V13) und Antwortleckage (V14) werden je Prompt berechnet und
   gespeichert, statt am Ende geschaetzt zu werden.
3. **Statistik ohne Zwischenschritt.** `v_bedingung` ist die flache Tabelle,
   die das gemischte Modell braucht.

## Ablauf

```
python3 01_init_db.py                      # Schema, Stichprobe, Material
python3 02_build_prompts.py    --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
python3 03_build_redundancy.py --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
python3 04a_probelauf.py --n 5             # Strukturschutz pruefen
python3 04_compress.py         --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct
python3 05_selectivity.py
python3 06_inference.py --backend ollama --alle-aufgaben     # Teilstudie B
python3 06_inference.py --backend ollama --wiederholung 2 --limit 20
python3 06_inference.py --pruefe-determinismus
python3 07_export.py
python3 08_nullmodell.py                   # Nullmodell zum Selektivitaetsindex
python3 09_auswertung.py                   # Auswertung fuer den Ergebnisteil
```

`01_init_db.py` liest die Stichprobe (`sample_a.tsv`, `sample_b.tsv`) und die
Demonstrationszuweisung (`demo_assignments.tsv`) aus `tools/PromptGenerator`,
die Paraphrasen aus `datasets/popQA`.

Jeder Schritt ist wiederholbar und ueberschreibt nur, was er selbst erzeugt.
`04_compress.py` und `06_inference.py` arbeiten nur die noch offenen
Bedingungen ab, ein Abbruch kostet also nichts.

## Der Tokenizer ist nicht optional

`eq:kalibrierung` und `eq:istrate` sind in Token des Zielmodells definiert.
`02` und `03` brechen deshalb ohne `--tokenizer` ab, statt die Spalten leer zu
lassen. Ein Lauf ohne Modelltoken waere still unbrauchbar: die Kalibrierung
liesse sich nicht pruefen und V2 nicht belegen.

Bezugsquellen in dieser Reihenfolge:

| Angabe | Wirkung |
|---|---|
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | **empfohlen.** Der Tokenizer von Llama-3.1 ueber `transformers`, aus einer Spiegelung ohne Zugangsbeschraenkung. Exakt, schnell, keine Serveraufrufe. Vokabular 128256. |
| `meta-llama/Llama-3.1-8B-Instruct` | dasselbe aus der Originalablage, verlangt einen angenommenen Lizenzvertrag bei Hugging Face |
| Pfad zu `tokenizer.json` | lokal hinterlegter Tokenizer |
| `ollama:llama3.1:8b` | Rueckfallebene ohne Hugging Face, siehe Vorbehalt unten |

Beim Laden wird die Vokabulargroesse gemeldet. Steht dort nicht 128256, ist ein
anderes Modell geladen als angenommen und die Kalibrierung bezoege sich auf
eine andere Tokenisierung als die Inferenz.

### Vorbehalt gegen den Ollama-Weg

Ollama bietet je nach Fassung `/api/tokenize` an; wo der Endpunkt fehlt, bleibt
nur `prompt_eval_count` eines Durchlaufs, und das ist keine verlaessliche
Tokenzahl. llama.cpp haelt den zuletzt ausgewerteten Prompt vor und wertet bei
einem Prompt mit gleichem Anfang nur den neuen Teil aus. `prompt_eval_count`
zaehlt dann die neu ausgewerteten Token, nicht die des Textes. Da sich die
Prompts dieser Arbeit einen langen gemeinsamen Anfang teilen, waeren die
Laengen still zu klein und die Kalibrierung waertlos, ohne dass irgendetwas
auffiele.

`OllamaTokenizer` prueft das beim Start: derselbe Prompt wird zweimal gezaehlt,
und weichen die Zahlen ab, verweigert der Tokenizer den Dienst mit einem
Hinweis auf den `transformers`-Weg. Ausserdem wird der konstante Aufschlag des
Satzanfangstokens ueber zwei Texte gemessen, deren Token sich addieren; der
leere Prompt taugt dafuer nicht, weil Ollama fuer ihn nichts auswertet und gar
kein `prompt_eval_count` zurueckgibt.

## Kalibrierung

`03_build_redundancy.py` legt die eingefuegte Textmenge nicht fest, sondern
sucht sie. Jede Variante stellt austauschbare Einfuegeeinheiten bereit:

| Variante | Einheiten | eingefuegt werden meist |
|---|---|---|
| `semantic` | bis zu fuenf Paraphrasen (Testfrage, vier Demonstrationen) | vier |
| `instruction` | ein Paar aus neun Umformulierungen | zwei, in der Laenge passend |
| `demonstration` | bis zu vier Zusatzdemonstrationen (Slots 5 bis 8) | drei |
| `filler` | vierzehn Fuellsaetze | zwei bis drei |

Gewaehlt wird die Teilmenge, deren Umfang `L_red / L_basis` dem Zielwert 0,50
am naechsten kommt; bei Gleichstand die kleinere Zahl von Einheiten. Die Suche
laeuft additiv ueber die einmal vermessenen Einheiten und prueft die vier
besten Kandidaten exakt nach. Die gewaehlte Teilmenge steht als JSON in
`prompt.bau_parameter`.

Das Band von 0,075 ist konstruktiv begruendet und nicht frei gewaehlt: die
groesste Einheit, eine vollstaendige Demonstration, misst rund 17 Prozent der
Basispromptlaenge, und eine Auswahl ganzer Einheiten trifft einen Zielwert
hoechstens bis auf die halbe Einheitengroesse. Ein engeres Band zwaenge zum
Ausschluss von Aufgaben und zerstoerte die Schichtung der Stichprobe.

## Ebenen

| Tabelle | Einheit | Inhalt |
|---|---|---|
| `frage` | Aufgabe | 300 PopQA-Fragen, Relation, Popularitaetsterzil, Aliasliste |
| `demonstration` | Aufgabe x Slot | acht fixierte Demonstrationen je Aufgabe |
| `paraphrase` | Frage x Schablone | Material der Stufe `semantic` aus PopQA-TP |
| `prompt` | Aufgabe x Variante | Prompttext, Laengen, Kalibrierung, Bauparameter, Kontrollgroessen |
| `segment` | Textabschnitt | Herkunft (kern/redundanz) und Promptabschnitt |
| `wort` | Token | Auszeichnung je Wort, relative Position |
| `kompressat` | Prompt x Stufe | Kompressat, erreichte Rate, Selektivitaetskennzahlen |
| `abschnitt_erhalt` | Kompressat x Abschnitt | Erhaltungsrate je Promptabschnitt |
| `wort_erhalt` | Kompressat x Wort | erhalten ja/nein, beide Aufloesungen |
| `inferenz` | Kompressat x Lauf | Modellausgabe, Korrektheit, Zeiten |
| `lauf` | Verarbeitungsschritt | Skript, Pruefsumme, Konfiguration, Commit |

Die Stufe `rho = 0` steht als Zeile in `kompressat` und wird ohne Aufruf des
Kompressors angelegt. Damit liegen alle zwanzig Bedingungen einer Aufgabe in
derselben Tabelle und Teilstudie B greift einheitlich darauf zu. Fuer diese
Stufe werden `wort_erhalt` und `abschnitt_erhalt` bewusst nicht befuellt: das
Kompressat ist der Prompt selbst, jede Erhaltungsrate ist eins, und die Zeilen
machten rund vierzig Prozent der groessten Tabelle aus.

## Die Pruefgroesse: S ueber die freien Woerter

`eq:selektivitaet` setzt zwei Erhaltungsraten ins Verhaeltnis. So wie sie
dasteht, misst sie aber zum Teil etwas anderes als die Entscheidung des
Kompressors.

Der Strukturschutz entzieht Marken und Ziffern der Streichung. Sie bleiben
ausnahmslos erhalten, gehen aber in beide Raten ein, und sie sind ungleich
verteilt: sechzehn Prozent der Kernwoerter sind geschuetzt, aber null Prozent
der Redundanz von `filler` und `instruction` und neunzehn Prozent der von
`demonstration`. Ein Regressionstest mit einem Verfahren, das dieselbe Rate
erreicht, dieselben Zeichen schuetzt und im Uebrigen zufaellig streicht, also
die Herkunft vollstaendig ignoriert, ergibt bei `rho = 0,75`:

```
variante         S nach eq:selektivitaet     S_frei
filler                             1.495      0.973
instruction                        1.545      1.002
semantic                           1.053      0.983
demonstration                      0.936      0.983
```

Die rohe Formel liefert die in H2 postulierte Rangfolge, obwohl nichts erkannt
wurde. Ueber die freien Woerter, also die, bei denen der Kompressor ueberhaupt
eine Wahl hatte, liegt der Index dort, wo er hingehoert.

`s_frei` ist deshalb die Pruefgroesse. Sie hat die Stetigkeitskorrektur nach
Haldane und Anscombe eingebaut (0,5 auf jeden Zaehler, 1 auf jeden Nenner),
damit auch Bedingungen mit vollstaendig entfernter Redundanz definiert bleiben;
diese auszuschliessen wuerde die Faelle maximaler Selektivitaet entfernen.
`s_index` bleibt daneben als deskriptive Groesse, die der Formel in der Arbeit
entspricht. Die Zahl der Nullfaelle und die Zahl der Bedingungen, bei denen die
beidseitige Aufloesung der Auszeichnung zu verschiedenen Werten fuehrt, weist
`v_selektivitaet` je Zelle aus.

Die geschuetzten Zeichen stehen als `dbio.FORCE_TOKENS` an einer einzigen
Stelle, von der sowohl `04_compress.py` als auch `05_selectivity.py` sie
beziehen. Eine Aenderung am Schutz wirkt damit zwingend auf beide.

## Sichten

`v_bedingung` ist die Arbeitstabelle der Teilstudie A, `v_studie_b` die der
Teilstudie B. Dazu kommen `v_selektivitaet`, `v_ratenkonformitaet`,
`v_kalibrierung` und `v_genauigkeit` als fertige Aggregate. `v_selektivitaet`
laesst die Variante `basis` aus, da sie keine Redundanz enthaelt und `S` fuer
sie nicht definiert ist; in `v_ratenkonformitaet` ist sie enthalten.

## Strukturschutz

`04a_probelauf.py` komprimiert eine kleine Stichprobe und berichtet, ob Marken
und Ziffern erhalten bleiben, ob die Blocktrennung steht, ob das Kompressat
eine Teilfolge des Originals ist und ob zwei gleiche Aufrufe dasselbe liefern.
Der Schritt ist vor dem vollstaendigen Lauf zwingend. Genau hier ist eine
frueher verwendete Konfiguration gescheitert: geschuetzt waren nur `\n` und
`:`, nicht die Buchstaben der Marken, worauf LLMLingua-2 `Q:` und `A:` zerlegte
und ab `rho = 0,5` kein Prompt mehr eine vollstaendige Marke enthielt. Der
Fehler war erst nach 4500 Kompressionen sichtbar geworden. Die Spalten
`anteil_marken_erhalten` und `anteil_ziffern_erhalten` weisen den Erfolg je
Bedingung nach, sodass V4 und V10 belegbar sind statt behauptet.

## Inferenz

`06_inference.py` kennt zwei Wege. `--backend ollama` rechnet ueber den
lokalen Ollama-Server in Q4_K_M und ist auf Apple Silicon der einzig
praktikable, da `bitsandbytes` dort nicht laeuft. `--backend transformers`
rechnet mit `bitsandbytes` in nf4 auf CUDA. Der Prompt wird mit `raw=True`
uebergeben, also ohne Chatvorlage: eine Vorlage fuegte Steuertoken und
Rollenmarken hinzu und veraenderte genau die Promptstruktur, um die es geht.

`--wiederholung 2` erzeugt einen zweiten Durchgang auf einer Teilmenge,
`--pruefe-determinismus` vergleicht beide und belegt damit V8.

`--alle-aufgaben` laesst Teilstudie B auf allen 300 Aufgaben rechnen und
gehoert zum Hauptlauf. Die urspruengliche Planung sah eine Teilstichprobe von
30 Aufgaben vor, weil Modellinferenz als der teure Schritt galt. Gemessen sind
es 0,29 Sekunden je Inferenz, also rund 29 Minuten fuer alle 6000. Der
Standardfehler einer Genauigkeit sinkt damit von etwa sieben auf zwei
Prozentpunkte, und die geschachtelte Stichprobe entfaellt. Die Spalte
`frage.in_teilstudie_b` bleibt als Nachweis der urspruenglichen Ziehung
erhalten, filtert aber nichts mehr.

## Betriebshinweise

* Die Datei liegt in einem OneDrive-Ordner. SQLite laeuft deshalb im
  Journalmodus `TRUNCATE` und nicht in `WAL`, damit keine Nebendateien
  entstehen, die getrennt synchronisiert werden. Waehrend eines langen Laufs
  sollte die Synchronisierung pausiert werden.
* `--db PFAD` legt die Datenbank an einen beliebigen Ort, etwa auf eine lokale
  Platte fuer die Dauer der Erhebung.
