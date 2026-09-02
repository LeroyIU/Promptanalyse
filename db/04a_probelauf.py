#!/usr/bin/env python3
"""
Schritt 4a: Vorabpruefung der Kompressorkonfiguration (V4, V10, V12).

Komprimiert eine kleine Stichprobe und berichtet, ob die Zusicherungen der
Methodik tatsaechlich eingeloest werden. Es wird nichts in die Datenbank
geschrieben. Der Schritt ist vor dem vollstaendigen Lauf zwingend, denn genau
hier ist die frueher verwendete Konfiguration gescheitert: geschuetzt waren nur
"\\n" und ":", nicht aber die Buchstaben der Marken, worauf LLMLingua-2 die
Marken Q: und A: zerlegte und ab rho = 0,5 kein Prompt mehr eine vollstaendige
Marke enthielt. Der Fehler blieb unbemerkt, weil er erst nach 4500
Kompressionen sichtbar geworden waere.

Geprueft wird je Stufe:

    Marken       Anteil erhaltener Q: und A:            (V4)
    Ziffern      Anteil erhaltener Ziffern               (V10)
    Leerzeile    bleibt die Blocktrennung erhalten       (V4)
    Teilfolge    ist das Kompressat Teilfolge des Originals (V12)
    Determinismus zweimal derselbe Aufruf, dasselbe Ergebnis

Aufruf:
    python3 04a_probelauf.py --n 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio       # noqa: E402
import importlib  # noqa: E402

komp = importlib.import_module("04_compress")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--n", type=int, default=10,
                   help="Prompts je Variante")
    p.add_argument("--geraet", default=None)
    args = p.parse_args()
    if args.geraet:
        komp.DEVICE = args.geraet

    con = dbio.verbinde(args.db, readonly=True)
    prompts = con.execute(
        "SELECT p.prompt_id, p.variante, p.text FROM prompt p "
        "JOIN (SELECT variante, MIN(frage_id) f FROM prompt GROUP BY variante) x "
        "ON 1=1 WHERE p.variante = x.variante "
        "ORDER BY p.variante, p.frage_id").fetchall()

    nach_variante = {}
    for z in prompts:
        nach_variante.setdefault(z["variante"], []).append(z)
    auswahl = [z for v in nach_variante.values() for z in v[:args.n]]
    print("%d Prompts, %d Stufen" % (len(auswahl), len(komp.STUFEN) - 1))

    print()
    print("%-14s %6s %8s %8s %10s %10s %12s" %
          ("variante", "rho", "marken", "ziffern", "leerzeile", "teilfolge",
           "determin."))
    for variante, zeilen in sorted(nach_variante.items()):
        for rho in komp.STUFEN:
            if rho == 0:
                continue
            m = z_ = lz = tf = det = 0
            n = 0
            for z in zeilen[:args.n]:
                erg = komp.komprimiere(z["text"], round(1 - rho, 6))
                text = erg["compressed_prompt"]
                marken, ziffern = komp.strukturkennzahlen(z["text"], text)
                m += marken if marken is not None else 0
                z_ += ziffern if ziffern is not None else 1
                lz += 1 if "\n\n" in text else 0
                _, gross, _ = dbio.tokenisiere(z["text"])
                _, klein, _ = dbio.tokenisiere(text)
                tf += 1 if dbio.einbetten_links(klein, gross) is not None else 0
                zwei = komp.komprimiere(z["text"], round(1 - rho, 6))
                det += 1 if zwei["compressed_prompt"] == text else 0
                n += 1
            print("%-14s %6.2f %8.3f %8.3f %9d/%d %9d/%d %11d/%d" %
                  (variante, rho, m / n, z_ / n, lz, n, tf, n, det, n))

    print()
    print("Erwartet: marken 1.000, ziffern 1.000, leerzeile und teilfolge "
          "vollzaehlig,\nDeterminismus vollzaehlig. Jede Abweichung ist vor dem "
          "vollstaendigen Lauf zu klaeren.")
    con.close()


if __name__ == "__main__":
    main()
