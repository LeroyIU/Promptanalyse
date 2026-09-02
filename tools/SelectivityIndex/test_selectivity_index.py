#!/usr/bin/env python3
"""
Pruefskript fuer selectivity_index.py.

Baut Faelle mit bekanntem Sollwert und vergleicht die Ausgabe damit. Geprueft
werden die Rechenfunktion und der vollstaendige Durchlauf ueber die
Ordnerstruktur. Aufruf:

    python3 test_selectivity_index.py
"""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selectivity_index as si


# name, Basisprompt, Gesamtprompt, Kompressat, Sollwerte
FAELLE = [
    # Nur Redundanz wird gestrichen. R_kern = 1, R_red = 1/4.
    {"name": "selektiv",
     "basis": "aa bb cc dd",
     "gesamt": "aa bb cc dd r1 r2 r3 r4",
     "komp": "aa bb cc dd r1",
     "S": 4.0, "S_spiegel": 4.0, "R_red_null": 0, "n_mehrdeutig": 0},
    # Aus beiden Mengen wird gleich viel gestrichen. S muss genau 1 sein.
    {"name": "blind",
     "basis": "aa bb cc dd",
     "gesamt": "aa bb cc dd r1 r2 r3 r4",
     "komp": "aa bb r1 r2",
     "S": 1.0, "S_spiegel": 1.0, "R_red_null": 0, "n_mehrdeutig": 0},
    # Umgekehrte Selektivitaet, der Kern wird bevorzugt gestrichen.
    {"name": "invers",
     "basis": "aa bb cc dd",
     "gesamt": "aa bb cc dd r1 r2 r3 r4",
     "komp": "aa r1 r2 r3 r4",
     "S": 0.25, "S_spiegel": 0.25, "R_red_null": 0, "n_mehrdeutig": 0},
    # Die Redundanz wird vollstaendig gestrichen. Der Quotient ist nicht
    # definiert und der Fall muss als solcher ausgewiesen werden.
    {"name": "nur_kern",
     "basis": "aa bb cc dd",
     "gesamt": "aa bb cc dd r1 r2 r3 r4",
     "komp": "aa bb cc dd",
     "S": None, "S_spiegel": None, "R_red_null": 1, "n_mehrdeutig": 0},
    # Kern und Redundanz teilen die Strukturwoerter Q und A.
    # R_kern = 5/6, R_red = 3/6, also S = 5/3.
    {"name": "teilgeteilt",
     "basis": "Q: who is aa ? A: bb",
     "gesamt": "Q: who is aa ? A: bb Q: who is cc ? A: dd",
     "komp": "Q: who aa ? A: bb Q: cc ? A:",
     "S": 5.0 / 3.0, "S_spiegel": 5.0 / 3.0, "R_red_null": 0,
     "n_mehrdeutig": 0},
    # Woertliche Wiederholung. Welche Haelfte erhalten blieb, ist aus dem Text
    # nicht entscheidbar. Die beiden Schranken muessen maximal auseinander
    # liegen und die Mehrdeutigkeit muss ausgewiesen werden.
    {"name": "woertlich",
     "basis": "a b",
     "gesamt": "a b a b",
     "komp": "a b",
     "S": None, "S_spiegel": 0.0, "R_red_null": 1, "n_mehrdeutig": 2},
    # Referenzvariante ohne Redundanz. S ist nicht definiert, R_kern gilt.
    {"name": "basis",
     "basis": "aa bb cc dd",
     "gesamt": "aa bb cc dd",
     "komp": "aa bb",
     "S": None, "S_spiegel": None, "R_red_null": 0, "n_mehrdeutig": 0,
     "n_red": 0, "R_kern": 0.5},
]


def gleich(ist, soll):
    if soll is None or ist is None:
        return ist is None and soll is None
    return abs(ist - soll) < 1e-9


def baue(wurzel):
    (wurzel / "1_prompts").mkdir(parents=True)
    for fall in FAELLE:
        name = fall["name"]
        (wurzel / "1_prompts" / (name + ".txt")).write_text(
            fall["basis"] + "\n", encoding="utf-8")
        # Der Zahlenteil im Dateinamen ist der Ratenparameter des Kompressors.
        # 0.5 entspricht der Reduktionsstufe rho = 0,5.
        for stufe, text, datei in (("2_redundancy", fall["gesamt"],
                                    name + ".txt"),
                                   ("3_compressed", fall["komp"],
                                    name + "_0.5.txt")):
            ordner = wurzel / stufe / name
            ordner.mkdir(parents=True, exist_ok=True)
            (ordner / datei).write_text(text + "\n", encoding="utf-8")


def pruefe():
    fehler = []

    for fall in FAELLE:
        name = fall["name"]
        ist = si.auswerten(fall["basis"], fall["gesamt"], fall["komp"])
        for feld in ("S", "S_spiegel", "R_kern", "n_red"):
            if feld not in fall:
                continue
            if not gleich(ist[feld], fall[feld]):
                fehler.append("%-12s %-12s soll %s, ist %s"
                              % (name, feld, fall[feld], ist[feld]))
        for feld in ("R_red_null", "n_mehrdeutig"):
            if ist[feld] != fall[feld]:
                fehler.append("%-12s %-12s soll %s, ist %s"
                              % (name, feld, fall[feld], ist[feld]))
        if not ist["basis_ist_teilfolge"]:
            fehler.append("%-12s Basisprompt nicht als Teilfolge erkannt"
                          % name)
        if ist["n_komp_unzugeordnet"]:
            fehler.append("%-12s %d Woerter des Kompressats nicht zugeordnet"
                          % (name, ist["n_komp_unzugeordnet"]))
        if ist["n_kern"] + ist["n_red"] != ist["n_gesamt"]:
            fehler.append("%-12s Kern und Redundanz ergeben nicht den "
                          "Gesamtprompt" % name)
        if ist["S"] is not None and ist["ln_S"] is None:
            fehler.append("%-12s ln_S fehlt trotz definiertem S" % name)

    # Der Ratenparameter 0.5 im Dateinamen muss als Reduktionsstufe
    # rho = 0,5 gefuehrt werden, nicht als Erhaltungsrate.
    with tempfile.TemporaryDirectory() as tmp:
        wurzel = Path(tmp)
        baue(wurzel)
        argv = sys.argv
        sys.argv = ["selectivity_index.py", "--studie", str(wurzel),
                    "--out", str(wurzel / "out"), "--stichprobe", ""]
        try:
            si.main()
        finally:
            sys.argv = argv

        zeilen = {z["variante"]: z for z in csv.DictReader(
            open(wurzel / "out" / "selektivitaet_je_datei.csv",
                 encoding="utf-8"))}
        for fall in FAELLE:
            name = fall["name"]
            if name not in zeilen:
                fehler.append("%-12s fehlt in der Ausgabedatei" % name)
                continue
            z = zeilen[name]
            if z["rho_ziel"] != "0.5" or z["rate_parameter"] != "0.5":
                fehler.append("%-12s rho_ziel %s, rate_parameter %s"
                              % (name, z["rho_ziel"], z["rate_parameter"]))
            soll = fall["S"]
            if soll is None:
                if z["S"] != "":
                    fehler.append("%-12s S muss leer sein, ist %s"
                                  % (name, z["S"]))
            elif abs(float(z["S"]) - soll) > 1e-6:
                fehler.append("%-12s Ausgabedatei weicht ab: %s statt %s"
                              % (name, z["S"], soll))

    return fehler


if __name__ == "__main__":
    probleme = pruefe()
    print()
    if probleme:
        print("Fehlgeschlagen:")
        for f in probleme:
            print("  " + f)
        sys.exit(1)
    print("Alle Pruefungen bestanden (%d Faelle)." % len(FAELLE))
