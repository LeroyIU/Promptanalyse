#!/usr/bin/env python3
"""
Schritt 7: Export aus der Datenbank.

Erzeugt die Tabellen, mit denen in R oder pandas weitergerechnet wird, und auf
Wunsch den vollstaendigen Textbestand fuer den Anhang der Arbeit. Die
Datenbank bleibt die einzige Quelle; alles hier Erzeugte ist jederzeit
wiederherstellbar und gehoert nicht in die Versionsverwaltung.

    bedingungen.csv          eine Zeile je Aufgabe x Variante x Stufe
    studie_b.csv             dasselbe mit Modelllauf
    selektivitaet.csv        Aggregat je Variante und Stufe
    ratenkonformitaet.csv    angezielte gegen erreichte Rate
    kalibrierung.csv         Nachweis zu eq:kalibrierung und V13, V14
    abschnitte.csv           Erhaltungsraten je Promptabschnitt
    wortebene.csv            je Wort und Bedingung, nur mit --wortebene
    prompts/, kompressate/   Textdateien, nur mit --texte

Aufruf: python3 07_export.py [--out PFAD] [--texte] [--wortebene]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio   # noqa: E402

AUSGABE = "../../results"

ABFRAGEN = {
    "bedingungen": "SELECT * FROM v_bedingung ORDER BY frage_id, variante, rho_ziel",
    "studie_b": "SELECT * FROM v_studie_b ORDER BY frage_id, variante, rho_ziel",
    "selektivitaet": "SELECT * FROM v_selektivitaet",
    "ratenkonformitaet": "SELECT * FROM v_ratenkonformitaet",
    "kalibrierung": "SELECT * FROM v_kalibrierung",
    "genauigkeit": "SELECT * FROM v_genauigkeit",
    "abschnitte": """SELECT b.frage_id, b.variante, b.rho_ziel, a.abschnitt,
                            a.herkunft, a.n_woerter, a.n_erhalten,
                            a.erhaltungsrate
                     FROM abschnitt_erhalt a
                     JOIN v_bedingung b USING(kompressat_id)
                     ORDER BY b.frage_id, b.variante, b.rho_ziel, a.abschnitt""",
    "laeufe": "SELECT * FROM lauf ORDER BY lauf_id",
}

WORTEBENE = """
SELECT b.frage_id, b.variante, b.rho_ziel, w.pos, w.rel_pos, w.form,
       w.ist_wort, w.ist_ziffer, w.herkunft, w.abschnitt, e.erhalten,
       e.erhalten_spiegel
FROM wort_erhalt e
JOIN wort w USING(wort_id)
JOIN v_bedingung b ON b.kompressat_id = e.kompressat_id
WHERE w.ist_wort = 1
ORDER BY b.frage_id, b.variante, b.rho_ziel, w.pos
"""


def schreibe_csv(con, pfad, sql):
    cur = con.execute(sql)
    spalten = [d[0] for d in cur.description]
    pfad.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(pfad, "w", newline="", encoding="utf-8") as fh:
        schreiber = csv.writer(fh)
        schreiber.writerow(spalten)
        for zeile in cur:
            schreiber.writerow(["" if v is None else v for v in zeile])
            n += 1
    return n


def schreibe_texte(con, wurzel):
    """Textbestand fuer den Anhang, in derselben Ordnerform wie zuvor."""
    n = 0
    for z in con.execute("SELECT frage_id, variante, text FROM prompt"):
        p = wurzel / "prompts" / z["variante"]
        p.mkdir(parents=True, exist_ok=True)
        (p / (z["frage_id"] + ".txt")).write_text(z["text"], encoding="utf-8")
        n += 1
    for z in con.execute(
            "SELECT p.frage_id, p.variante, k.rate_parameter t, k.text "
            "FROM kompressat k JOIN prompt p ON p.prompt_id = k.prompt_id "
            "WHERE k.rho_ziel > 0"):
        p = wurzel / "kompressate" / z["variante"]
        p.mkdir(parents=True, exist_ok=True)
        name = "%s_%g.txt" % (z["frage_id"], z["t"])
        (p / name).write_text(z["text"], encoding="utf-8")
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--out", default=AUSGABE)
    p.add_argument("--texte", action="store_true")
    p.add_argument("--wortebene", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = (Path(__file__).resolve().parent / out).resolve()

    con = dbio.verbinde(args.db, readonly=True)
    for name, sql in ABFRAGEN.items():
        n = schreibe_csv(con, out / (name + ".csv"), sql)
        print("%-22s %7d Zeilen" % (name + ".csv", n))
    if args.wortebene:
        n = schreibe_csv(con, out / "wortebene.csv", WORTEBENE)
        print("%-22s %7d Zeilen" % ("wortebene.csv", n))
    if args.texte:
        n = schreibe_texte(con, out / "texte")
        print("%-22s %7d Dateien" % ("texte/", n))
    print("\nAusgabe: %s" % out)
    con.close()


if __name__ == "__main__":
    main()
