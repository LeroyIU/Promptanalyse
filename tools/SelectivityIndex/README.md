# Selektivitätsindex

Berechnet den Selektivitätsindex $S(\rho)$ und die beiden Erhaltungsraten
$R_{\text{kern}}$ und $R_{\text{red}}$ für jede Kombination aus Prompt,
Redundanzvariante und Reduktionsstufe. Ohne Modellinferenz, allein aus den
Textdateien. Bezug: `subsubsec:selektivitaet` und `subsubsec:auszeichnung` in
`Dokument.tex`.

## Grundgedanke

Jeder Prompt liegt zweifach vor: als Basisprompt vor der Redundanzeinfügung und
als Gesamtprompt danach. Die eingefügte Redundanz ist genau die Differenz beider
Fassungen.

| Menge | Quelle |
|---|---|
| $W_{\text{kern}}$ | Wörter des Basisprompts, wiedergefunden im Gesamtprompt |
| $W_{\text{red}}$ | Gesamtprompt ohne $W_{\text{kern}}$, also die eingefügten Wörter |
| $W'$ | Wörter, die nach der Kompression verblieben sind |

$$R_{\text{kern}}(\rho)=\frac{|W_{\text{kern}}\cap W'|}{|W_{\text{kern}}|},\qquad
R_{\text{red}}(\rho)=\frac{|W_{\text{red}}\cap W'|}{|W_{\text{red}}|},\qquad
S(\rho)=\frac{R_{\text{kern}}(\rho)}{R_{\text{red}}(\rho)}$$

$S=1$ heißt, das Verfahren streicht Kern und Redundanz in gleichem Maß. $S>1$
heißt, Redundanz wird bevorzugt entfernt, $S<1$ der Kerninhalt.

Gezählt werden Wörter, nicht Zeichen. Satzzeichen bleiben außen vor, weil sie
über die geschützten Zeichen des Kompressors ohnehin nie gestrichen werden und
beide Erhaltungsraten gleichermaßen anheben würden. Die Spalten mit Suffix
`_alle` wiederholen die Rechnung einschließlich Satzzeichen.

## Logarithmierte Auswertung und undefinierte Fälle

Ein Verhältnis darf nicht arithmetisch gemittelt werden: $S=2$ und $S=0{,}5$
beschreiben gleich starke, entgegengesetzte Selektivität und müssen sich zu 1
aufheben. Aggregiert wird deshalb über $\ln S$, ausgewiesen wird das
geometrische Mittel `S_geom` samt Konfidenzintervall.

Ist $R_{\text{red}}=0$, wurde die gesamte eingefügte Redundanz gestrichen und
der Quotient ist nicht gebildet. Solche Zeilen tragen `R_red_null = 1`, bleiben
in `S` leer und gehen in kein Aggregat ein. Die Aggregatdatei führt ihre Zahl
als `n_R_red_null`. Dieser Fall tritt bei hohen Reduktionsstufen gehäuft auf und
ist vor der Interpretation zu prüfen: eine Stufe, in der er die Mehrheit der
Bedingungen betrifft, ist über den Quotienten nicht auswertbar.

## Zuordnung von Wörtern zu Kern und Redundanz

Die Differenz wird positionell gebildet, nicht als Multimenge. Kern und
Redundanz teilen sich Wörter wie `Q`, `A` oder `the`. Eine reine
Multimengenrechnung müsste ein erhaltenes `the` beiden Mengen zurechnen oder
willkürlich aufteilen.

Das Werkzeug bettet stattdessen den Basisprompt als Teilfolge in den
Gesamtprompt ein, womit jede Position eindeutig Kern oder Redundanz ist, und
anschließend das Kompressat als Teilfolge in den Gesamtprompt, womit für jede
Position feststeht, ob sie erhalten blieb. Mehrdeutigkeiten werden nach dem
Prinzip der frühesten noch nicht belegten Position aufgelöst, wie in
`subsubsec:auszeichnung` festgelegt. Der gierige Durchlauf ist für die
Teilfolgesuche vollständig und damit exakt.

Als Gegenprobe wird dieselbe Rechnung mit der spätesten statt der frühesten
Position durchgeführt (`S_spiegel`). Da die Redundanz stets hinter dem
zugehörigen Kernabschnitt steht, rechnet die erste Auflösung ein mehrdeutiges
Wort dem Kern zu und die zweite der Redundanz. Beide Werte schließen den wahren
Index ein. `n_mehrdeutig` zählt die betroffenen Positionen, entscheidend ist
aber erst ein Unterschied zwischen `S` und `S_spiegel`.

`S_bag` ist zusätzlich die reine Multimengenfassung. Geteilte Worttypen werden
dort im Verhältnis ihrer Häufigkeiten aufgeteilt, was bei einer Kompression
ohne Rücksicht auf die Herkunft erwartungstreu $S=1$ ergibt. Robustheitsprüfung,
kein Primärwert.

## Raten

Der Zahlenteil im Dateinamen (`<id>_0.25.txt`) ist der Ratenparameter des
Kompressors, also der Anteil der zu erhaltenden Wörter. Die Arbeit rechnet mit
der Reduktionsrate $\rho = 1 - t$. Das Werkzeug rechnet um: aus `_0.25` wird
`rho_ziel = 0,75`. Beide Werte stehen in der Ausgabe.

`rho_ist_wort` ist die erreichte Reduktion, gemessen in Wörtern.
`eq:istrate` in der Arbeit definiert $\hat\rho$ dagegen über den Tokenizer des
Zielmodells. Die Spalte ist deshalb eine Näherung für die Ratenkonformität und
nicht der in der Arbeit berichtete Wert.

## Erwartete Ordnerstruktur

```
<Studienordner>/
  1_prompts/<id>.txt                     Basisprompt
  2_redundancy/<variante>/<id>.txt       Gesamtprompt
  3_compressed/<variante>/<id>_<t>.txt   Kompressat
  sample_a.tsv                           optional, liefert Relation und Terzil
```

Varianten und Stufen werden aus den Ordner- und Dateinamen abgeleitet,
`manifest.tsv` wird übersprungen. Für die Referenzvariante `basis` ist
$W_{\text{red}}$ leer, `S` bleibt dort leer und nur `R_kern` wird berichtet.

## Aufruf

```
python3 selectivity_index.py
python3 selectivity_index.py --studie "../../experiments/Study B"
python3 selectivity_index.py --varianten semantic instruction --stufen 0.5 0.75
python3 selectivity_index.py --out ../../results/StudyA --stichprobe ""
```

Alle Vorgaben stehen im Konfigurationsblock am Kopf des Skripts.

## Ausgabe

| Datei | Inhalt |
|---|---|
| `selektivitaet_je_datei.csv` | eine Zeile je Prompt, Variante und Stufe |
| `selektivitaet_aggregat.csv` | Kennwerte je Variante und Stufe, inklusive `S_geom` |
| `selektivitaet_kategorie.csv` | dasselbe nach Relationskategorie |
| `ratenkonformitaet.csv` | angezielte gegen erreichte Reduktionsrate |

Spalten von `selektivitaet_je_datei.csv`:

| Spalte | Bedeutung |
|---|---|
| `prompt_id`, `variante` | Kennung des Falls |
| `rho_ziel`, `rate_parameter` | Reduktionsstufe und der Wert aus dem Dateinamen |
| `kategorie`, `terzil` | aus `sample_a.tsv`, leer ohne Join |
| `n_kern`, `n_red`, `n_gesamt` | Mächtigkeit der drei Wortmengen |
| `redundanzanteil` | $n_{\text{red}}/n_{\text{gesamt}}$ |
| `n_komp`, `rho_ist_wort`, `rho_delta_wort` | erreichte Reduktion und Abweichung |
| `R_kern`, `R_red`, `S`, `ln_S` | Primärwerte nach `eq:erhaltungsraten` und `eq:selektivitaet` |
| `R_red_null` | 1, wenn die Redundanz vollständig gestrichen wurde und `S` deshalb leer bleibt |
| `R_kern_alle`, `R_red_alle`, `S_alle` | dieselbe Rechnung einschließlich Satzzeichen |
| `R_kern_spiegel`, `R_red_spiegel`, `S_spiegel` | Gegenprobe bei Auflösung von rechts |
| `R_kern_bag`, `R_red_bag`, `S_bag` | Multimengenfassung als Robustheitsprüfung |
| `n_mehrdeutig`, `anteil_mehrdeutig` | Wortpositionen, die beide Auflösungen verschieden zuordnen |
| `basis_ist_teilfolge` | 1, wenn der Basisprompt unverändert im Gesamtprompt steht |
| `n_basis_unzugeordnet` | Kernwörter ohne Entsprechung im Gesamtprompt |
| `n_komp_unzugeordnet` | Kompressatwörter ohne Entsprechung im Gesamtprompt |

Die letzten drei Spalten sind Prüfgrößen und müssen 1, 0 und 0 sein. Weicht eine
ab, war die Redundanzeinfügung nicht rein additiv oder die Kompression nicht
rein subtraktiv, und die Zeile ist vor der Auswertung zu prüfen.

## Prüfung

```
python3 test_selectivity_index.py
```

Sieben Fälle mit bekanntem Sollwert: reine Redundanzentfernung ($S=4$), blinde
Kompression ($S=1$), umgekehrte Selektivität ($S=0{,}25$), vollständig
gestrichene Redundanz ($S$ undefiniert), geteilte Strukturwörter, wörtliche
Wiederholung (die Schranken müssen auseinanderfallen und die Mehrdeutigkeit muss
ausgewiesen werden) und die Referenzvariante ohne Redundanz. Geprüft wird die
Rechenfunktion und der vollständige Durchlauf über die Ordnerstruktur,
einschließlich der Umrechnung des Ratenparameters in die Reduktionsstufe.

## Hinweis zu OneDrive

Liegen Prompts als Platzhalter in der Cloud, meldet das Werkzeug sie als nicht
lesbar und überspringt sie. Vor dem Lauf im Finder auf dem Ordner `experiments`
die Option "Immer auf diesem Gerät behalten" setzen, sonst fehlen Fälle. Die
Spalte `n` in `selektivitaet_aggregat.csv` zeigt, wie viele Fälle je Bedingung
tatsächlich eingegangen sind. Vollständig sind es 300 je Variante und Stufe.
