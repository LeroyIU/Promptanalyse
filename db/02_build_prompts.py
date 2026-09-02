#!/usr/bin/env python3
"""
Schritt 2: Basisprompts bauen (subsubsec:basisprompt).

Liest Frage und Demonstrationsslots 1 bis 4 aus der Datenbank, setzt daraus je
Aufgabe den Basisprompt zusammen und legt ihn samt Segmentauszeichnung und
Wortebene in der Tabelle prompt ab. Der Ordner 1_prompts entfaellt.

Aufruf: python3 02_build_prompts.py [--db PFAD] [--limit N] [--ohne-wortebene]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio        # noqa: E402
import promptbau as pb  # noqa: E402
import store       # noqa: E402

STUDIE = "A"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="nur die ersten N Aufgaben, fuer Probelaeufe")
    p.add_argument("--ohne-wortebene", action="store_true")
    p.add_argument("--neu", action="store_true",
                   help="Prompts auch bei unveraendertem Text neu anlegen; verwirft die darauf aufbauenden Kompressate")
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer des Zielmodells; die Laengen des "
                        "Basisprompts sind die Bezugsgroesse der Kalibrierung "
                        "nach eq:kalibrierung und in Modelltoken definiert")
    args = p.parse_args()

    con = dbio.verbinde(args.db)
    tokenizer = dbio.modell_tokenizer(args.tokenizer)
    if tokenizer is None:
        sys.exit("Tokenizer %s nicht ladbar." % args.tokenizer)

    lauf_id = dbio.lauf_beginnen(con, "prompts", __file__, konfiguration={
        "instruktion": pb.INSTRUCTION,
        "n_demos": pb.N_DEMOS_BASIS,
        "question_prefix": pb.QUESTION_PREFIX,
        "answer_prefix": pb.ANSWER_PREFIX,
        "block_separator": pb.BLOCK_SEPARATOR,
        "answer_cue": pb.ANSWER_CUE,
        "tokenizer": args.tokenizer,
        "wortebene": not args.ohne_wortebene,
    })

    sql = "SELECT frage_id FROM frage WHERE in_teilstudie_a = 1 ORDER BY frage_id"
    if args.limit:
        sql += " LIMIT %d" % args.limit
    ids = [z[0] for z in con.execute(sql)]

    n = 0
    neu_gebaut = 0
    fehler = []
    for frage_id in ids:
        material = store.material_laden(con, frage_id)
        if material is None or len(material["demos"]) < pb.N_DEMOS_BASIS:
            fehler.append("Aufgabe %s hat nur %d Basisdemonstrationen"
                          % (frage_id, len(material["demos"]) if material else 0))
            continue
        material["demos"] = material["demos"][:pb.N_DEMOS_BASIS]
        text, segmente = pb.baue("basis", material)
        _, geaendert = store.schreibe_prompt(
            con, frage_id, "basis", text, segmente, lauf_id,
            material["antworten"], basis=None, tokenizer=tokenizer,
            studie=STUDIE, mit_wortebene=not args.ohne_wortebene,
            neu=args.neu)
        n += 1
        neu_gebaut += 1 if geaendert else 0
        if n % 50 == 0:
            con.commit()
    con.commit()

    dbio.lauf_beenden(con, lauf_id, n)

    for f in fehler:
        print("Warnung: " + f, file=sys.stderr)

    z = dbio.eine_zeile(con,
        "SELECT COUNT(*) n, MIN(n_ws_token) mi, MAX(n_ws_token) ma, "
        "AVG(n_ws_token) mw FROM prompt WHERE variante = 'basis'")
    print("Basisprompts: %d  (davon in diesem Lauf neu gebaut: %d)"
          % (z["n"], neu_gebaut))
    print("Whitespace-Token je Prompt: min %d, mittel %.1f, max %d"
          % (z["mi"], z["mw"], z["ma"]))
    print("Antwortleckage im Basisprompt: %d von %d" % (
        dbio.skalar(con, "SELECT SUM(antwortleckage) FROM prompt "
                         "WHERE variante = 'basis'") or 0, z["n"]))
    con.close()


if __name__ == "__main__":
    main()
