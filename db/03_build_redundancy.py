#!/usr/bin/env python3
"""
Schritt 3: Redundanzvarianten bauen (subsubsec:redundanztypen, subsubsec:filler).

Erzeugt je Aufgabe die vier Varianten semantic, instruction, demonstration und
filler und legt sie samt Segmentauszeichnung und Wortebene in der Tabelle
prompt ab.

Anders als eine dateibasierte Fassung zerlegt dieser Schritt den Basisprompt
nicht wieder in seine Bestandteile, sondern baut jede Variante unmittelbar aus
dem Material der Datenbank. Basisprompt und Redundanzfassung unterscheiden sich
damit nachweislich nur in der Einfuegung.

Kalibrierung (subsubsec:kalibrierung, eq:kalibrierung)
------------------------------------------------------
Die eingefuegte Textmenge ist nicht fest, sondern wird je Aufgabe und Variante
so gewaehlt, dass L_red / L_basis moeglichst nahe an KALIBRIERUNG_ZIEL liegt,
gemessen in Token des Zielmodells. Gesucht wird ueber die Teilmengen der
verfuegbaren Einfuegeeinheiten:

    semantic       welche der bis zu fuenf Fragen eine Paraphrase erhalten
    instruction    welches Paar Umformulierungen eingefuegt wird
    demonstration  welche der bis zu vier Zusatzdemonstrationen erscheinen
    filler         welche Saetze des Fuelltextpools verwendet werden

Die Art der eingefuegten Einheit bleibt damit das unterscheidende Merkmal der
Varianten; angeglichen wird allein ihr Volumen. Das ist die Voraussetzung
dafuer, dass H2 die Groesse der wiederholten Einheit prueft und nicht die
Menge des eingefuegten Textes (V1, V2).

Die Suche laeuft zweistufig. Zuerst wird jede Einfuegeeinheit einmal
tokenisiert und die Teilmengen werden ueber die Summe ihrer Laengen bewertet;
das ist schnell und fuer die Vorauswahl genau genug. Die besten Kandidaten
werden anschliessend vollstaendig gebaut und exakt gemessen, denn massgeblich
ist die Zahl der Token, um die die Variante den Basisprompt tatsaechlich
verlaengert.

Aufruf: python3 03_build_redundancy.py --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio        # noqa: E402
import promptbau as pb  # noqa: E402
import store       # noqa: E402

STUDIE = "A"
ALLE = ("semantic", "instruction", "demonstration", "filler")

# Zahl der Kandidaten, die nach der additiven Vorauswahl exakt gemessen werden.
# Klein gehalten, weil eine exakte Messung ueber den Ollama-Tokenizer einen
# Serveraufruf kostet. Die additive Vorauswahl ist genau genug, um die besten
# Kandidaten sicher unter die ersten vier zu bringen.
N_FEINPRUEFUNG = 4


def waehle_bau(variante, material, basis_mt, tokenizer, antworten):
    """Bester Bauparameter nach eq:kalibrierung.

    Rueckgabe: (bau, text, segmente, kal) oder None, wenn nichts baubar ist.
    """
    einheiten = pb.einheiten(variante, material)
    if not einheiten:
        return None
    laenge = {k: dbio.modell_token(t, tokenizer) for k, t in einheiten.items()}

    kand = pb.kandidaten(variante, material)
    if not kand:
        return None

    def schaetzung(bau):
        return sum(laenge[k] for k in pb.bau_einheiten(variante, bau))

    # Vorauswahl ueber die Summe der Einheitenlaengen.
    ziel = store.KALIBRIERUNG_ZIEL * basis_mt
    kand.sort(key=lambda b: (abs(schaetzung(b) - ziel),
                             len(pb.bau_einheiten(variante, b)),
                             pb.bau_einheiten(variante, b)))

    bestes = None
    for bau in kand[:N_FEINPRUEFUNG]:
        gebaut = pb.baue(variante, material, bau)
        if gebaut is None:
            continue
        text, segmente = gebaut
        kal = store.kalibrierung(dbio.modell_token(text, tokenizer), basis_mt)
        if kal is None:
            continue
        schluessel = (abs(kal - store.KALIBRIERUNG_ZIEL),
                      len(pb.bau_einheiten(variante, bau)),
                      pb.bau_einheiten(variante, bau))
        if bestes is None or schluessel < bestes[0]:
            bestes = (schluessel, bau, text, segmente, kal)
    if bestes is None:
        return None
    return bestes[1], bestes[2], bestes[3], bestes[4]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--varianten", nargs="*", default=None,
                   help="Auswahl aus %s" % (", ".join(ALLE),))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ohne-wortebene", action="store_true")
    p.add_argument("--neu", action="store_true",
                   help="Prompts auch bei unveraendertem Text neu anlegen; "
                        "verwirft die darauf aufbauenden Kompressate")
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer des Zielmodells; die Kalibrierung ist in "
                        "Modelltoken definiert und ohne ihn nicht pruefbar")
    args = p.parse_args()

    gewaehlt = list(args.varianten) if args.varianten else list(ALLE)
    unbekannt = [v for v in gewaehlt if v not in ALLE]
    if unbekannt:
        sys.exit("unbekannte Variante: %s" % ", ".join(unbekannt))

    con = dbio.verbinde(args.db)
    tokenizer = dbio.modell_tokenizer(args.tokenizer)
    if tokenizer is None:
        sys.exit("Tokenizer %s nicht ladbar." % args.tokenizer)

    lauf_id = dbio.lauf_beginnen(con, "redundanz", __file__, konfiguration={
        "varianten": gewaehlt,
        "instruction_pool": pb.INSTRUCTION_POOL,
        "filler_position": pb.FILLER_POSITION,
        "filler_pool": pb.FILLER_POOL,
        "max_filler_saetze": pb.MAX_FILLER_SAETZE,
        "kalibrierung_ziel": store.KALIBRIERUNG_ZIEL,
        "kalibrierung_band": store.KALIBRIERUNG_BAND,
        "n_feinpruefung": N_FEINPRUEFUNG,
        "tokenizer": args.tokenizer,
    })

    sql = ("SELECT frage_id FROM prompt WHERE variante = 'basis' AND studie = ? "
           "ORDER BY frage_id")
    if args.limit:
        sql += " LIMIT %d" % args.limit
    ids = [z[0] for z in con.execute(sql, (STUDIE,))]
    if not ids:
        sys.exit("Keine Basisprompts vorhanden. Zuerst 02_build_prompts.py.")

    n = neu_gebaut = ausser_band = 0
    warnungen = []
    for i, frage_id in enumerate(ids, start=1):
        material = store.material_laden(con, frage_id)
        material["demos"] = material["demos"][:pb.N_DEMOS_BASIS]
        basis = store.basis_laengen(con, frage_id, STUDIE)

        for variante in gewaehlt:
            gewaehlt_bau = waehle_bau(variante, material,
                                      basis["n_modell_token"], tokenizer,
                                      material["antworten"])
            if gewaehlt_bau is None:
                warnungen.append("Variante %s fuer %s nicht baubar"
                                 % (variante, frage_id))
                continue
            bau, text, segmente, kal = gewaehlt_bau
            if store.im_band(kal) == 0:
                ausser_band += 1
                warnungen.append(
                    "%s %s ausserhalb des Bandes: %.3f" % (frage_id, variante, kal))
            _, geaendert = store.schreibe_prompt(
                con, frage_id, variante, text, segmente, lauf_id,
                material["antworten"], basis=basis, tokenizer=tokenizer,
                studie=STUDIE, mit_wortebene=not args.ohne_wortebene,
                neu=args.neu, bau=bau)
            neu_gebaut += 1 if geaendert else 0
            n += 1
        if i % 25 == 0:
            con.commit()
            print("  %d / %d Aufgaben" % (i, len(ids)), file=sys.stderr)
    con.commit()
    dbio.lauf_beenden(con, lauf_id, n)

    for w in warnungen[:20]:
        print("Warnung: " + w, file=sys.stderr)
    if len(warnungen) > 20:
        print("... und %d weitere Warnungen" % (len(warnungen) - 20),
              file=sys.stderr)

    print("%d Promptvarianten, davon in diesem Lauf neu gebaut: %d"
          % (n, neu_gebaut))
    print("ausserhalb des Kalibrierungsbandes (%.3f +- %.3f): %d"
          % (store.KALIBRIERUNG_ZIEL, store.KALIBRIERUNG_BAND, ausser_band))
    print()
    print("%-14s %5s %8s %8s %8s %8s %8s %8s %7s" %
          ("variante", "n", "mt_ges", "mt_red", "kal", "kal_min", "kal_max",
           "im_band", "einh"))
    for z in con.execute(
            "SELECT variante, COUNT(*) n, AVG(n_modell_token) ges, "
            "AVG(n_modell_token_red) red, AVG(kalibrierung_token) kal, "
            "MIN(kalibrierung_token) kmin, MAX(kalibrierung_token) kmax, "
            "AVG(kalibrierung_im_band) band, AVG(n_einheiten) einh "
            "FROM prompt WHERE studie = ? AND variante <> 'basis' "
            "GROUP BY variante "
            "ORDER BY CASE variante WHEN 'filler' THEN 1 WHEN 'semantic' "
            "THEN 2 WHEN 'instruction' THEN 3 ELSE 4 END", (STUDIE,)):
        print("%-14s %5d %8.1f %8.1f %8.3f %8.3f %8.3f %8.2f %7.2f" % (
            z["variante"], z["n"], z["ges"], z["red"] or 0, z["kal"] or 0,
            z["kmin"] or 0, z["kmax"] or 0, z["band"] or 0, z["einh"] or 0))
    con.close()


if __name__ == "__main__":
    main()
