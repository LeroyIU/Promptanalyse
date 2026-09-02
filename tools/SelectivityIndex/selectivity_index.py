#!/usr/bin/env python3
"""
Selektivitaetsindex S(rho) fuer Teilstudie A (Abschnitt subsubsec:selektivitaet).

Grundgedanke
------------
Jeder Prompt liegt zweifach vor: als Basisprompt vor der Redundanzeinfuegung
und als Gesamtprompt danach. Die eingefuegte Redundanz ist genau die Differenz
beider Fassungen. Daraus folgen die drei Wortmengen der Arbeit:

    W_kern    Woerter des Basisprompts, wiedergefunden im Gesamtprompt
    W_red     Gesamtprompt ohne W_kern, also die eingefuegten Woerter
    W'        Woerter, die nach der Kompression verblieben sind

    R_kern(rho) = |W_kern ^ W'| / |W_kern|     (eq:erhaltungsraten)
    R_red(rho)  = |W_red  ^ W'| / |W_red|
    S(rho)      = R_kern(rho) / R_red(rho)     (eq:selektivitaet)

S = 1 heisst, das Verfahren streicht Kern und Redundanz in gleichem Mass.
S > 1 heisst, Redundanz wird bevorzugt entfernt, S < 1 der Kerninhalt.
Ausgewertet wird der Index logarithmiert, Faelle mit R_red = 0 werden gesondert
ausgewiesen und gehen in die Aggregate nicht ein.

Zuordnung
---------
Die Differenz wird positionell gebildet, nicht als Multimenge. Der Basisprompt
wird als Teilfolge in den Gesamtprompt eingebettet, jede Position ist danach
eindeutig Kern oder Redundanz. Anschliessend wird das Kompressat als Teilfolge
in den Gesamtprompt eingebettet, womit fuer jede Position feststeht, ob sie
erhalten blieb. Mehrdeutigkeiten werden nach dem Prinzip der fruehesten noch
nicht belegten Position aufgeloest. Ihr Anteil wird berichtet, und mit
S_spiegel steht die Gegenprobe bei Aufloesung von rechts daneben.

Erwartete Ordnerstruktur
------------------------
    1_prompts/<id>.txt                    Basisprompt, Quelle von W_kern
    2_redundancy/<variante>/<id>.txt      Gesamtprompt
    3_compressed/<variante>/<id>_<t>.txt  Kompressat, t ist der Ratenparameter
                                          des Kompressors, also 1 - rho

Aufruf
------
    python3 selectivity_index.py
    python3 selectivity_index.py --studie "../../experiments/Study B"
    python3 selectivity_index.py --varianten semantic instruction --stufen 0.5

Ausgabe
-------
    selektivitaet_je_datei.csv     eine Zeile je Prompt, Variante und Stufe
    selektivitaet_aggregat.csv     Kennwerte je Variante und Stufe
    selektivitaet_kategorie.csv    dasselbe nach Relationskategorie
    ratenkonformitaet.csv          angezielte gegen erreichte Reduktionsrate
"""

import argparse
import csv
import difflib
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Studienordner. Relative Pfade beziehen sich auf das Skriptverzeichnis.
STUDIE = "../../experiments/Study A"

BASIS_DIR = "1_prompts"
REDUNDANZ_DIR = "2_redundancy"
KOMPRIMIERT_DIR = "3_compressed"
AUSGABE_DIR = "../../results/Study A"

# Stichprobendatei fuer Relation und Popularitaetsterzil. Leer setzen, um den
# Join zu ueberspringen.
STICHPROBE = "sample_a.tsv"
SP_ID = "id"
SP_KATEGORIE = "prop"
SP_TERZIL = "pop_tercile"

# Varianten. Leere Liste bedeutet, alle Unterordner von REDUNDANZ_DIR nehmen.
VARIANTEN = []

# Reduktionsstufen rho. Leere Liste bedeutet, alle aus den Dateinamen ableiten.
STUFEN = []

# Gross- und Kleinschreibung beim Abgleich ignorieren.
CASEFOLD = True

DATEIENDUNG = ".txt"
MANIFEST_NAME = "manifest.tsv"
ENCODING = "utf-8"

# Ein Token ist eine Wortkette oder ein einzelnes Satzzeichen. Apostrophe
# trennen, weil LLMLingua-2 das Genitiv-s haeufig einzeln entfernt und
# Murnoy's sonst nicht mehr auf Murnoy abbildbar waere.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORT_RE = re.compile(r"\w", re.UNICODE)


# ---------------------------------------------------------------------------
# Tokenisierung und Einbettung
# ---------------------------------------------------------------------------

def tokenisiere(text):
    """Zerlegt Text in Token und liefert Abgleichsschluessel und Wortflagge."""
    roh = TOKEN_RE.findall(text)
    schluessel = [t.casefold() if CASEFOLD else t for t in roh]
    ist_wort = [bool(WORT_RE.search(t)) for t in roh]
    return schluessel, ist_wort


def ausrichten(a, b):
    """Positionspaare der gemeinsamen Bloecke von a und b."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    paare = []
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            paare.append((i + k, j + k))
    return paare


def einbetten_links(klein, gross):
    """Frueheste Einbettung von klein als Teilfolge von gross.

    Liefert die getroffenen Positionen in gross oder None, falls klein keine
    Teilfolge ist. Der gierige Durchlauf von links ist fuer die Teilfolgesuche
    vollstaendig: existiert eine Einbettung, so findet er eine. Das ist das in
    Abschnitt subsubsec:auszeichnung festgelegte Prinzip der fruehesten noch
    nicht belegten Position.
    """
    stellen = []
    j = 0
    n = len(gross)
    for t in klein:
        while j < n and gross[j] != t:
            j += 1
        if j == n:
            return None
        stellen.append(j)
        j += 1
    return stellen


def einbetten_rechts(klein, gross):
    """Spaeteste Einbettung von klein als Teilfolge von gross."""
    n = len(gross)
    stellen = einbetten_links(klein[::-1], gross[::-1])
    if stellen is None:
        return None
    return [n - 1 - i for i in reversed(stellen)]


def einbetten_block(klein, gross):
    """Rueckfall auf die Blockausrichtung, falls keine Teilfolge vorliegt."""
    return [i for j, i in ausrichten(klein, gross)]


def ist_teilfolge(klein, gross):
    return einbetten_links(klein, gross) is not None


def rollen_bestimmen(kern, gesamt):
    """Markiert jede Position des Gesamtprompts als Kern oder Redundanz."""
    stellen = einbetten_links(kern, gesamt)
    if stellen is None:
        stellen = einbetten_block(kern, gesamt)
    rolle = ["red"] * len(gesamt)
    for j in stellen:
        rolle[j] = "kern"
    return rolle, len(kern) - len(stellen)


def ueberleben_bestimmen(gesamt, komp, rechts=False):
    """Markiert jede Position des Gesamtprompts als erhalten oder gestrichen.

    Steht ein erhaltenes Wort an mehreren Stellen des Gesamtprompts, faellt es
    bei rechts=False an die frueheste und damit an den Kern, bei rechts=True an
    die spaeteste und damit an die Redundanz. Da die Redundanz stets hinter dem
    zugehoerigen Kernabschnitt eingefuegt wird, schliessen die beiden Lesarten
    den wahren Wert von S ein.
    """
    stellen = (einbetten_rechts(komp, gesamt) if rechts
               else einbetten_links(komp, gesamt))
    if stellen is None:
        stellen = einbetten_block(komp, gesamt)
    lebt = [False] * len(gesamt)
    for j in stellen:
        lebt[j] = True
    return lebt, len(komp) - len(stellen)


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------

def quotient(r_kern, r_red):
    """S nach eq:selektivitaet. None, wenn der Nenner null oder leer ist."""
    if r_kern is None or r_red is None or r_red == 0:
        return None
    return r_kern / r_red


def erhaltungsraten(rolle, lebt, ist_wort, nur_woerter=True):
    """Liefert n_kern, n_red, R_kern, R_red und S."""
    n_k = n_r = e_k = e_r = 0
    for idx, ro in enumerate(rolle):
        if nur_woerter and not ist_wort[idx]:
            continue
        if ro == "kern":
            n_k += 1
            e_k += 1 if lebt[idx] else 0
        else:
            n_r += 1
            e_r += 1 if lebt[idx] else 0
    r_k = e_k / n_k if n_k else None
    r_r = e_r / n_r if n_r else None
    return n_k, n_r, r_k, r_r, quotient(r_k, r_r)


def multimengen_raten(kern, gesamt, komp, ist_wort_gesamt):
    """Kontrollrechnung als reine Multimengendifferenz.

    W_red ist hier die Multimengendifferenz Gesamtprompt minus Basisprompt.
    Erhaltene Woerter eines Typs, der in beiden Mengen vorkommt, werden im
    Verhaeltnis der Haeufigkeiten aufgeteilt. Diese Aufteilung ist
    erwartungstreu: streicht das Verfahren ohne Ruecksicht auf die Herkunft,
    ergibt sich S = 1.
    """
    woerter = {t for t, w in zip(gesamt, ist_wort_gesamt) if w}
    c_ges = Counter(t for t in gesamt if t in woerter)
    c_kern = Counter(t for t in kern if t in woerter)
    c_komp = Counter(t for t in komp if t in woerter)

    kern_eff, red_eff = {}, {}
    for t, n in c_ges.items():
        k = min(c_kern.get(t, 0), n)
        kern_eff[t] = k
        if n - k > 0:
            red_eff[t] = n - k

    n_k = sum(kern_eff.values())
    n_r = sum(red_eff.values())
    e_k = e_r = 0.0
    for t, c in c_komp.items():
        k = kern_eff.get(t, 0)
        r = red_eff.get(t, 0)
        ges = k + r
        if ges == 0:
            continue
        c = min(c, ges)
        e_k += c * k / ges
        e_r += c * r / ges

    r_k = e_k / n_k if n_k else None
    r_r = e_r / n_r if n_r else None
    return r_k, r_r, quotient(r_k, r_r)


def auswerten(basis_text, gesamt_text, komp_text):
    """Alle Kennzahlen fuer ein Tripel aus Basis, Gesamt und Kompressat."""
    kern, _ = tokenisiere(basis_text)
    gesamt, gesamt_wort = tokenisiere(gesamt_text)
    komp, _ = tokenisiere(komp_text)

    # Rollen einmal festlegen. W_kern ist die Einbettung des Basisprompts,
    # W_red der Rest, also genau die Differenz beider Promptfassungen.
    rolle, basis_fehlend = rollen_bestimmen(kern, gesamt)

    # Erhaltene Woerter von links zugeordnet, Prinzip der fruehesten Position.
    lebt, komp_fremd = ueberleben_bestimmen(gesamt, komp)
    n_k, n_r, r_k, r_r, s = erhaltungsraten(rolle, lebt, gesamt_wort)
    n_ka, n_ra, r_ka, r_ra, s_a = erhaltungsraten(rolle, lebt, gesamt_wort,
                                                  nur_woerter=False)

    # Gegenprobe: dieselben Rollen, erhaltene Woerter von rechts zugeordnet.
    lebt_s, _ = ueberleben_bestimmen(gesamt, komp, rechts=True)
    _, _, r_ks, r_rs, s_s = erhaltungsraten(rolle, lebt_s, gesamt_wort)
    mehrdeutig = sum(1 for a, b in zip(lebt, lebt_s) if a != b) // 2

    r_kb, r_rb, s_b = multimengen_raten(kern, gesamt, komp, gesamt_wort)

    n_wort = sum(1 for w in gesamt_wort if w)
    n_komp_wort = sum(1 for t in komp if WORT_RE.search(t))
    rho_ist = 1 - n_komp_wort / n_wort if n_wort else None

    return {
        "n_kern": n_k,
        "n_red": n_r,
        "n_gesamt": n_wort,
        "redundanzanteil": n_r / n_wort if n_wort else None,
        "n_komp": n_komp_wort,
        "rho_ist_wort": rho_ist,
        "R_kern": r_k,
        "R_red": r_r,
        "S": s,
        "ln_S": math.log(s) if s else None,
        "R_red_null": 1 if (n_r and r_r == 0) else 0,
        "R_kern_alle": r_ka,
        "R_red_alle": r_ra,
        "S_alle": s_a,
        "R_kern_spiegel": r_ks,
        "R_red_spiegel": r_rs,
        "S_spiegel": s_s,
        "R_kern_bag": r_kb,
        "R_red_bag": r_rb,
        "S_bag": s_b,
        "n_mehrdeutig": mehrdeutig,
        "anteil_mehrdeutig": mehrdeutig / n_komp_wort if n_komp_wort else None,
        "basis_ist_teilfolge": int(ist_teilfolge(kern, gesamt)),
        "n_basis_unzugeordnet": basis_fehlend,
        "n_komp_unzugeordnet": komp_fremd,
    }


# ---------------------------------------------------------------------------
# Dateien einsammeln
# ---------------------------------------------------------------------------

def stufe_aus_name(pfad):
    """1234_0.25.txt liefert ('1234', 0.75), also die Reduktionsrate rho.

    Der Zahlenteil im Dateinamen ist der Ratenparameter des Kompressors, also
    der Anteil der zu erhaltenden Woerter. Die Arbeit rechnet durchgaengig mit
    der Reduktionsrate rho = 1 - t (Abschnitt subsubsec:raten).
    """
    stamm = pfad.stem
    if "_" not in stamm:
        return None
    prompt_id, t_txt = stamm.rsplit("_", 1)
    try:
        t = float(t_txt)
    except ValueError:
        return None
    return prompt_id, round(1 - t, 6), t


def varianten_finden(red_root, gewaehlt):
    vorhanden = sorted(d.name for d in red_root.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    if not gewaehlt:
        return vorhanden
    fehlend = [v for v in gewaehlt if v not in vorhanden]
    if fehlend:
        sys.exit("Variante nicht gefunden: %s. Vorhanden: %s"
                 % (", ".join(fehlend), ", ".join(vorhanden)))
    return list(gewaehlt)


def stichprobe_lesen(pfad):
    """Ordnet jeder Prompt-ID Relation und Popularitaetsterzil zu."""
    if not pfad or not Path(pfad).is_file():
        return {}
    zuordnung = {}
    with open(pfad, newline="", encoding=ENCODING) as fh:
        for zeile in csv.DictReader(fh, delimiter="\t"):
            if SP_ID not in zeile:
                return {}
            zuordnung[str(zeile[SP_ID]).strip()] = (
                zeile.get(SP_KATEGORIE, ""), zeile.get(SP_TERZIL, ""))
    return zuordnung


def lies(pfad):
    try:
        return pfad.read_text(encoding=ENCODING)
    except (OSError, UnicodeDecodeError) as e:
        print("nicht lesbar: %s (%s)" % (pfad, e), file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def kennwerte(werte):
    """Mittelwert, Standardabweichung, Median und 95-Prozent-Intervall."""
    werte = [w for w in werte
             if isinstance(w, float) and math.isfinite(w)]
    n = len(werte)
    if n == 0:
        return {"n": 0, "mittel": "", "sd": "", "median": "",
                "ci95_unten": "", "ci95_oben": ""}
    mittel = statistics.fmean(werte)
    sd = statistics.stdev(werte) if n > 1 else 0.0
    halb = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mittel": mittel, "sd": sd,
            "median": statistics.median(werte),
            "ci95_unten": mittel - halb, "ci95_oben": mittel + halb}


def runde(wert, stellen=6):
    if wert is None:
        return ""
    if isinstance(wert, float):
        return "" if not math.isfinite(wert) else round(wert, stellen)
    return wert


def schreibe_csv(pfad, spalten, zeilen):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "w", newline="", encoding=ENCODING) as fh:
        schreiber = csv.DictWriter(fh, fieldnames=spalten)
        schreiber.writeheader()
        for z in zeilen:
            schreiber.writerow({s: runde(z.get(s, "")) for s in spalten})


def aggregiere(zeilen, schluesselspalten, messgroessen):
    """Kennwerte je Gruppe.

    Fuer S wird zusaetzlich das geometrische Mittel ausgewiesen, gebildet als
    Rueckrechnung aus dem Mittel der logarithmierten Werte. Es ist der Kennwert
    der Arbeit, weil ein Verhaeltnis nicht arithmetisch gemittelt werden darf:
    S = 2 und S = 0,5 beschreiben gleich starke, entgegengesetzte Selektivitaet
    und muessen sich zu 1 aufheben.
    """
    gruppen = defaultdict(list)
    for z in zeilen:
        gruppen[tuple(z[s] for s in schluesselspalten)].append(z)
    ergebnis = []
    for schluessel in sorted(gruppen):
        gruppe = gruppen[schluessel]
        zeile = dict(zip(schluesselspalten, schluessel))
        zeile["n"] = len(gruppe)
        zeile["n_R_red_null"] = sum(z["R_red_null"] for z in gruppe)
        for m in messgroessen:
            k = kennwerte([z[m] for z in gruppe])
            zeile["%s_n" % m] = k["n"]
            zeile["%s_mittel" % m] = k["mittel"]
            zeile["%s_sd" % m] = k["sd"]
            zeile["%s_median" % m] = k["median"]
            zeile["%s_ci95_unten" % m] = k["ci95_unten"]
            zeile["%s_ci95_oben" % m] = k["ci95_oben"]
        if "ln_S" in messgroessen and zeile["ln_S_mittel"] != "":
            zeile["S_geom"] = math.exp(zeile["ln_S_mittel"])
            zeile["S_geom_ci95_unten"] = math.exp(zeile["ln_S_ci95_unten"])
            zeile["S_geom_ci95_oben"] = math.exp(zeile["ln_S_ci95_oben"])
        else:
            zeile["S_geom"] = ""
            zeile["S_geom_ci95_unten"] = ""
            zeile["S_geom_ci95_oben"] = ""
        ergebnis.append(zeile)
    return ergebnis


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------

SPALTEN_DATEI = [
    "prompt_id", "variante", "rho_ziel", "rate_parameter",
    "kategorie", "terzil",
    "n_kern", "n_red", "n_gesamt", "redundanzanteil",
    "n_komp", "rho_ist_wort", "rho_delta_wort",
    "R_kern", "R_red", "S", "ln_S", "R_red_null",
    "R_kern_alle", "R_red_alle", "S_alle",
    "R_kern_spiegel", "R_red_spiegel", "S_spiegel",
    "R_kern_bag", "R_red_bag", "S_bag",
    "n_mehrdeutig", "anteil_mehrdeutig",
    "basis_ist_teilfolge", "n_basis_unzugeordnet", "n_komp_unzugeordnet",
]

MESSGROESSEN = ["ln_S", "S", "R_kern", "R_red", "S_alle", "S_spiegel",
                "S_bag", "rho_ist_wort", "rho_delta_wort",
                "anteil_mehrdeutig"]


def main():
    global CASEFOLD

    p = argparse.ArgumentParser(
        description="Berechnet den Selektivitaetsindex je Prompt, Variante "
                    "und Reduktionsstufe.")
    p.add_argument("--studie", default=None,
                   help="Studienordner mit 1_prompts, 2_redundancy, "
                        "3_compressed")
    p.add_argument("--basis", default=None, help="Ordner der Basisprompts")
    p.add_argument("--redundanz", default=None,
                   help="Ordner mit einem Unterordner je Variante")
    p.add_argument("--komprimiert", default=None,
                   help="Ordner mit einem Unterordner je Variante")
    p.add_argument("--out", default=None, help="Ausgabeordner")
    p.add_argument("--varianten", nargs="*", default=None,
                   help="Auswahl der Varianten, sonst alle")
    p.add_argument("--stufen", nargs="*", type=float, default=None,
                   help="Auswahl der Reduktionsstufen rho, sonst alle")
    p.add_argument("--stichprobe", default=None,
                   help="TSV mit Relation und Terzil, leer setzen zum "
                        "Ueberspringen")
    p.add_argument("--exakt", action="store_true",
                   help="Gross- und Kleinschreibung beim Abgleich beachten")
    args = p.parse_args()

    if args.exakt:
        CASEFOLD = False

    hier = Path(__file__).resolve().parent
    studie = (hier / (args.studie or STUDIE)).resolve()
    basis_root = (Path(args.basis).resolve() if args.basis
                  else studie / BASIS_DIR)
    red_root = (Path(args.redundanz).resolve() if args.redundanz
                else studie / REDUNDANZ_DIR)
    komp_root = (Path(args.komprimiert).resolve() if args.komprimiert
                 else studie / KOMPRIMIERT_DIR)
    out_root = (Path(args.out).resolve() if args.out
                else (hier / AUSGABE_DIR).resolve())

    for d, name in ((basis_root, "Basisprompts"), (red_root, "Redundanz"),
                    (komp_root, "Kompressate")):
        if not d.is_dir():
            sys.exit("Ordner fehlt (%s): %s" % (name, d))

    sp_datei = args.stichprobe if args.stichprobe is not None else STICHPROBE
    sp_pfad = ((studie / sp_datei) if sp_datei and
               not Path(sp_datei).is_absolute() else sp_datei)
    stichprobe = stichprobe_lesen(sp_pfad) if sp_datei else {}

    varianten = varianten_finden(red_root,
                                 args.varianten if args.varianten is not None
                                 else VARIANTEN)
    stufen_filter = set(args.stufen if args.stufen is not None else STUFEN)

    zeilen = []
    fehlend_basis = fehlend_komp = unlesbar = 0
    basis_cache = {}

    for variante in varianten:
        red_dir = red_root / variante
        komp_dir = komp_root / variante
        if not komp_dir.is_dir():
            print("kein Kompressatordner fuer Variante %s" % variante,
                  file=sys.stderr)
            continue

        kompressate = defaultdict(list)
        for f in sorted(komp_dir.glob("*" + DATEIENDUNG)):
            if f.name == MANIFEST_NAME:
                continue
            zerlegt = stufe_aus_name(f)
            if zerlegt is None:
                continue
            prompt_id, rho, t = zerlegt
            if stufen_filter and rho not in stufen_filter:
                continue
            kompressate[prompt_id].append((rho, t, f))

        for gesamt_pfad in sorted(red_dir.glob("*" + DATEIENDUNG)):
            if gesamt_pfad.name == MANIFEST_NAME:
                continue
            prompt_id = gesamt_pfad.stem
            basis_pfad = basis_root / (prompt_id + DATEIENDUNG)
            if not basis_pfad.is_file():
                fehlend_basis += 1
                continue
            if prompt_id not in kompressate:
                fehlend_komp += 1
                continue

            if prompt_id not in basis_cache:
                basis_cache[prompt_id] = lies(basis_pfad)
            basis_text = basis_cache[prompt_id]
            gesamt_text = lies(gesamt_pfad)
            if basis_text is None or gesamt_text is None:
                unlesbar += 1
                continue

            kategorie, terzil = stichprobe.get(prompt_id, ("", ""))

            for rho, t, komp_pfad in sorted(kompressate[prompt_id]):
                komp_text = lies(komp_pfad)
                if komp_text is None:
                    unlesbar += 1
                    continue
                zeile = {"prompt_id": prompt_id, "variante": variante,
                         "rho_ziel": rho, "rate_parameter": t,
                         "kategorie": kategorie, "terzil": terzil}
                zeile.update(auswerten(basis_text, gesamt_text, komp_text))
                ist = zeile["rho_ist_wort"]
                zeile["rho_delta_wort"] = (ist - rho if ist is not None
                                           else None)
                zeilen.append(zeile)

    if not zeilen:
        sys.exit("Keine auswertbaren Tripel gefunden.")

    schreibe_csv(out_root / "selektivitaet_je_datei.csv",
                 SPALTEN_DATEI, zeilen)

    aggregat = aggregiere(zeilen, ["variante", "rho_ziel"], MESSGROESSEN)
    schreibe_csv(out_root / "selektivitaet_aggregat.csv",
                 list(aggregat[0].keys()), aggregat)

    if any(z["kategorie"] for z in zeilen):
        nach_kat = aggregiere(zeilen, ["variante", "rho_ziel", "kategorie"],
                              ["ln_S", "R_kern", "R_red"])
        schreibe_csv(out_root / "selektivitaet_kategorie.csv",
                     list(nach_kat[0].keys()), nach_kat)

    konform = [{
        "variante": z["variante"], "rho_ziel": z["rho_ziel"], "n": z["n"],
        "rho_ist_wort_mittel": z["rho_ist_wort_mittel"],
        "rho_ist_wort_sd": z["rho_ist_wort_sd"],
        "rho_delta_wort_mittel": z["rho_delta_wort_mittel"],
        "rho_delta_wort_median": z["rho_delta_wort_median"],
    } for z in aggregat]
    schreibe_csv(out_root / "ratenkonformitaet.csv",
                 list(konform[0].keys()), konform)

    print("Basis:       %s" % basis_root)
    print("Redundanz:   %s" % red_root)
    print("Kompressate: %s" % komp_root)
    print("Ausgabe:     %s" % out_root)
    print("%d Auswertungen aus %d Varianten"
          % (len(zeilen), len(set(z["variante"] for z in zeilen))))
    if fehlend_basis:
        print("ohne Basisprompt: %d" % fehlend_basis, file=sys.stderr)
    if fehlend_komp:
        print("ohne Kompressat: %d" % fehlend_komp, file=sys.stderr)
    if unlesbar:
        print("nicht lesbar: %d" % unlesbar, file=sys.stderr)

    for feld, text in (
            ("basis_ist_teilfolge", "Basisprompt nicht als Teilfolge "
                                    "enthalten"),
            ("n_basis_unzugeordnet", "Kernwoerter ohne Entsprechung im "
                                     "Gesamtprompt"),
            ("n_komp_unzugeordnet", "Kompressatwoerter ohne Entsprechung im "
                                    "Gesamtprompt")):
        treffer = sum(1 for z in zeilen
                      if (not z[feld] if feld == "basis_ist_teilfolge"
                          else z[feld]))
        if treffer:
            print("%s: %d Faelle, Spalte %s pruefen" % (text, treffer, feld),
                  file=sys.stderr)

    def fmt(w):
        return "%.3f" % w if isinstance(w, float) else "-"

    print()
    print("%-14s %6s %6s %8s %8s %8s %8s %8s %6s" %
          ("variante", "rho", "n", "R_kern", "R_red", "S_geom", "S_spieg",
           "S_bag", "R_red0"))
    for z in aggregat:
        print("%-14s %6.2f %6d %8s %8s %8s %8s %8s %6d" % (
            z["variante"], z["rho_ziel"], z["n"],
            fmt(z["R_kern_mittel"]), fmt(z["R_red_mittel"]),
            fmt(z["S_geom"]), fmt(z["S_spiegel_median"]),
            fmt(z["S_bag_median"]), z["n_R_red_null"]))


if __name__ == "__main__":
    main()
