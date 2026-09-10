#!/usr/bin/env python3
"""
Nullmodell zum Selektivitaetsindex (subsubsec:selektivitaet).

Beziffert, welchen Index ein Verfahren erreicht, das dieselbe Rate trifft,
dieselben Zeichen schuetzt und im Uebrigen zufaellig streicht, die Herkunft
der Woerter also vollstaendig ignoriert. Das ist die Begruendung dafuer, dass
eq:selektivitaet als Pruefgroesse nicht ausreicht und die Arbeit stattdessen
eq:selektivitaetfrei verwendet.

Verfahren
---------
Geschuetzte Woerter (die Marken Q und A sowie Ziffern, siehe dbio.SCHUTZ_FORMEN
und dbio.RESERVE_DIGITS) ueberleben sicher. Jedes freie Wort ueberlebt mit
derselben Wahrscheinlichkeit p, unabhaengig davon, ob es zum Kern oder zur
Redundanz gehoert. p wird je Bedingung so gewaehlt, dass die tatsaechlich
beobachtete Zahl erhaltener Woerter getroffen wird. Damit unterscheidet sich
das Nullmodell vom echten Lauf ausschliesslich darin, dass es die Herkunft
nicht kennt.

Gerechnet wird der Erwartungswert und nicht eine Ziehung, weil der
Erwartungswert exakt ist und keinen eigenen Zufallsstartwert braucht.

Ausgabe je Variante und Stufe das geometrische Mittel von
  S  nach eq:selektivitaet        (erwartet deutlich ueber 1, obwohl das
                                   Verfahren nichts erkannt hat)
  S~ nach eq:selektivitaetfrei    (erwartet 1, der Wert der Gleichbehandlung)

Aufruf
------
    python3 08_nullmodell.py
    python3 08_nullmodell.py --csv nullmodell.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

import dbio

# Stetigkeitskorrektur wie in 05_selectivity.py.
KORR_ZAEHLER = 0.5
KORR_NENNER = 1.0


def wortmengen(con):
    """Je Prompt und Herkunft die Zahl geschuetzter und freier Woerter."""
    sql = ("SELECT prompt_id, herkunft, form, ist_ziffer, COUNT(*) n "
           "FROM wort WHERE ist_wort = 1 "
           "GROUP BY prompt_id, herkunft, form, ist_ziffer")
    mengen = defaultdict(lambda: {"kern": [0, 0], "redundanz": [0, 0]})
    for r in con.execute(sql):
        eintrag = mengen[r["prompt_id"]][r["herkunft"]]
        if dbio.ist_geschuetzt(r["form"], r["ist_ziffer"]):
            eintrag[0] += r["n"]
        else:
            eintrag[1] += r["n"]
    return mengen


def erhaltene_woerter(con):
    """Je Kompressat die Zahl erhaltener Woerter, Satzzeichen ausgenommen."""
    sql = ("SELECT we.kompressat_id, SUM(we.erhalten) n "
           "FROM wort_erhalt we JOIN wort w ON w.wort_id = we.wort_id "
           "WHERE w.ist_wort = 1 GROUP BY we.kompressat_id")
    return {r["kompressat_id"]: r["n"] for r in con.execute(sql)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--csv", default=None, help="Ergebnis zusaetzlich als CSV")
    args = p.parse_args()

    con = dbio.verbinde(args.db, readonly=True)
    mengen = wortmengen(con)
    erhalten = erhaltene_woerter(con)

    sql = ("SELECT k.kompressat_id, k.prompt_id, k.rho_ziel, p.variante "
           "FROM kompressat k JOIN prompt p ON p.prompt_id = k.prompt_id "
           "WHERE k.rho_ziel > 0 AND p.variante <> 'basis' "
           "ORDER BY p.variante, k.rho_ziel")

    ln_s = defaultdict(list)
    ln_s_frei = defaultdict(list)
    for a in con.execute(sql):
        m = mengen[a["prompt_id"]]
        kg, kf = m["kern"]
        rg, rf = m["redundanz"]
        frei = kf + rf
        if not frei or not kf or not rf:
            continue
        # Ueberlebenswahrscheinlichkeit der freien Woerter, so kalibriert,
        # dass die beobachtete Kompressatlaenge getroffen wird.
        pw = (erhalten[a["kompressat_id"]] - (kg + rg)) / frei
        pw = min(1.0, max(0.0, pw))

        schluessel = (a["variante"], a["rho_ziel"])

        r_kern = (kg + pw * kf) / (kg + kf)
        r_red = (rg + pw * rf) / (rg + rf)
        if r_red > 0:
            ln_s[schluessel].append(math.log(r_kern / r_red))

        r_kern_frei = (pw * kf + KORR_ZAEHLER) / (kf + KORR_NENNER)
        r_red_frei = (pw * rf + KORR_ZAEHLER) / (rf + KORR_NENNER)
        ln_s_frei[schluessel].append(math.log(r_kern_frei / r_red_frei))

    zeilen = []
    for schluessel in sorted(ln_s_frei, key=lambda x: (x[1], x[0])):
        werte = ln_s[schluessel]
        werte_frei = ln_s_frei[schluessel]
        zeilen.append({
            "variante": schluessel[0],
            "rho_ziel": schluessel[1],
            "n": len(werte_frei),
            "s_geometrisch": math.exp(sum(werte) / len(werte)) if werte else "",
            "s_frei_geometrisch": math.exp(sum(werte_frei) / len(werte_frei)),
        })

    print("%-14s %6s %6s %14s %14s"
          % ("Variante", "rho", "n", "S (eq:sel)", "S~ (frei)"))
    for z in zeilen:
        s = "%14.3f" % z["s_geometrisch"] if z["s_geometrisch"] != "" else " " * 14
        print("%-14s %6.2f %6d %s %14.3f"
              % (z["variante"], z["rho_ziel"], z["n"], s,
                 z["s_frei_geometrisch"]))

    if args.csv:
        with open(args.csv, "w", newline="", encoding=dbio.ENCODING) as fh:
            schreiber = csv.DictWriter(fh, fieldnames=list(zeilen[0]))
            schreiber.writeheader()
            schreiber.writerows(zeilen)
        print("\n%s geschrieben" % args.csv, file=sys.stderr)


if __name__ == "__main__":
    main()
