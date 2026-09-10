#!/usr/bin/env python3
"""
Schritt 4: Kompression mit LLMLingua-2 (subsec:kompression).

Liest die Prompts aus der Datenbank, komprimiert jeden in den Stufen
rho = 0,25 / 0,50 / 0,75 und schreibt Kompressat, erreichte Rate, die
Selbstauskunft des Kompressors und die Kompressionsdauer in die Tabelle
kompressat. Die Stufe rho = 0 wird ohne Aufruf des Kompressors als Kopie des
Prompts angelegt, damit alle zwanzig Bedingungen je Aufgabe in derselben
Tabelle stehen und Teilstudie B einheitlich darauf zugreifen kann.

Der Ordner 3_compressed entfaellt.

Aufruf:
    python3 04_compress.py                      alle offenen Bedingungen
    python3 04_compress.py --stufen 0.5         nur eine Stufe
    python3 04_compress.py --varianten basis    nur eine Variante
    python3 04_compress.py --nur-b              nur die 30 Aufgaben der Studie B
    python3 04_compress.py --trocken            ohne Modell, nur rho = 0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio   # noqa: E402

# ---------------------------------------------------------------------------
# Konfiguration (subsubsec:konfiguration)
# ---------------------------------------------------------------------------

MODELL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEVICE = "mps"                 # "cpu" | "mps" (Apple Silicon) | "cuda"

# Geschuetzte Zeichen. Zeilenumbrueche und Satzzeichen bleiben erhalten, damit
# die Gliederung des Prompts bei hohen Raten nicht zerfaellt (V4).
#
# WICHTIG gegenueber der frueheren Konfiguration: "Q" und "A" stehen jetzt
# ausdruecklich in der Liste. Zuvor waren nur "\n" und ":" geschuetzt, nicht
# aber die Buchstaben der Marken selbst. LLMLingua-2 hat daraufhin das Q und
# das A gestrichen und nur den Doppelpunkt stehen lassen; ab rho = 0,5 enthielt
# kein einziger Prompt mehr eine vollstaendige Marke. Damit war der in V4
# zugesicherte Strukturschutz faktisch nicht wirksam und ein Leistungseinbruch
# in Teilstudie B waere dem Formatverlust und nicht dem Inhaltsverlust
# zuzuschreiben gewesen. Die Spalte anteil_marken_erhalten weist das je
# Bedingung nach.
FORCE_TOKENS = dbio.FORCE_TOKENS
RESERVE_DIGITS = dbio.RESERVE_DIGITS   # Ziffernschutz (V10)

# Ebenfalls geaendert: drop_consecutive entfernt aufeinanderfolgende gleiche
# Schutzzeichen. Da "\n" geschuetzt ist, hat die Einstellung True die
# Leerzeile zwischen den Promptbloecken zu einem einfachen Umbruch
# zusammengezogen und damit genau die Gliederung beseitigt, die der
# Strukturschutz sichern soll.
DROP_CONSECUTIVE = False

CHUNK_END_TOKENS = [".", "\n"]

# Aufgabenunabhaengige Betriebsart: keine Frage wird uebergeben.
QUESTION = None

STUFEN = (0.0, 0.25, 0.50, 0.75)

# Marken der Promptstruktur, deren Erhalt protokolliert wird.
MARKEN = ("Q:", "A:")

_compressor = None


def get_compressor():
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        print("Lade %s auf %s ..." % (MODELL, DEVICE), file=sys.stderr)
        _compressor = PromptCompressor(model_name=MODELL, use_llmlingua2=True,
                                       device_map=DEVICE)
    return _compressor


def komprimiere(text, rate):
    kwargs = {"rate": rate, "force_tokens": FORCE_TOKENS,
              "force_reserve_digit": RESERVE_DIGITS,
              "drop_consecutive": DROP_CONSECUTIVE,
              "chunk_end_tokens": CHUNK_END_TOKENS}
    if QUESTION:
        kwargs["question"] = QUESTION
    return get_compressor().compress_prompt(text, **kwargs)


# ---------------------------------------------------------------------------

def strukturkennzahlen(original, kompressat):
    """Anteil erhaltener Strukturmarken und Ziffern (V4, V10)."""
    def anteil(zaehle):
        vor = zaehle(original)
        return (zaehle(kompressat) / vor) if vor else None
    marken = anteil(lambda t: sum(t.count(m) for m in MARKEN))
    ziffern = anteil(lambda t: sum(c.isdigit() for c in t))
    return marken, ziffern


def schreibe_kompressat(con, prompt_id, rho, text, prompt_text, lauf_id,
                        tokenizer=None, ergebnis=None, dauer=None):
    n_ws0 = dbio.ws_token(prompt_text)
    n_ws = dbio.ws_token(text)
    n_mt0 = dbio.modell_token(prompt_text, tokenizer)
    n_mt = dbio.modell_token(text, tokenizer)

    rho_ist_wort = 1 - n_ws / n_ws0 if n_ws0 else None
    rho_ist_token = (1 - n_mt / n_mt0) if (n_mt is not None and n_mt0) else None
    marken, ziffern = strukturkennzahlen(prompt_text, text)

    con.execute("DELETE FROM kompressat WHERE prompt_id = ? AND rho_ziel = ?",
                (prompt_id, rho))
    con.execute(
        "INSERT INTO kompressat (prompt_id, rho_ziel, rate_parameter, text, "
        "sha256, n_ws_token, n_modell_token, rho_ist_wort, rho_ist_token, "
        "rho_delta_wort, rho_delta_token, komp_origin_tokens, "
        "komp_compressed_tokens, komp_rate_ist, dauer_kompression_s, "
        "kompressor_modell, geraet, n_zeilenumbrueche, "
        "anteil_marken_erhalten, anteil_ziffern_erhalten, erzeugt, lauf_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (prompt_id, rho, round(1 - rho, 6), text, dbio.sha256_text(text),
         n_ws, n_mt, rho_ist_wort, rho_ist_token,
         (rho_ist_wort - rho) if rho_ist_wort is not None else None,
         (rho_ist_token - rho) if rho_ist_token is not None else None,
         (ergebnis or {}).get("origin_tokens"),
         (ergebnis or {}).get("compressed_tokens"),
         ((ergebnis or {}).get("compressed_tokens") /
          (ergebnis or {}).get("origin_tokens"))
         if ergebnis and ergebnis.get("origin_tokens") else None,
         dauer, MODELL if rho > 0 else None, DEVICE if rho > 0 else None,
         text.count("\n"), marken, ziffern, dbio.jetzt(), lauf_id))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--stufen", nargs="*", type=float, default=None)
    p.add_argument("--varianten", nargs="*", default=None)
    p.add_argument("--nur-b", action="store_true",
                   help="nur Aufgaben der Teilstichprobe B")
    p.add_argument("--neu", action="store_true",
                   help="vorhandene Kompressate ueberschreiben")
    p.add_argument("--trocken", action="store_true",
                   help="nur die Stufe rho = 0 anlegen, kein Modell laden")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--geraet", default=None)
    args = p.parse_args()

    global DEVICE
    if args.geraet:
        DEVICE = args.geraet

    stufen = tuple(args.stufen) if args.stufen else STUFEN
    if args.trocken:
        stufen = tuple(s for s in stufen if s == 0.0)

    con = dbio.verbinde(args.db)
    tokenizer = dbio.modell_tokenizer(args.tokenizer) if args.tokenizer else None

    lauf_id = dbio.lauf_beginnen(con, "kompression", __file__, konfiguration={
        "modell": MODELL, "geraet": DEVICE, "stufen": list(stufen),
        "force_tokens": FORCE_TOKENS, "force_reserve_digit": RESERVE_DIGITS,
        "drop_consecutive": DROP_CONSECUTIVE,
        "chunk_end_tokens": CHUNK_END_TOKENS, "question": QUESTION,
        "varianten": args.varianten, "nur_b": args.nur_b,
        "tokenizer": args.tokenizer,
    })

    sql = ("SELECT p.prompt_id, p.text, p.variante, p.frage_id FROM prompt p "
           "JOIN frage f ON f.frage_id = p.frage_id WHERE 1=1")
    params = []
    if args.nur_b:
        sql += " AND f.in_teilstudie_b = 1"
    if args.varianten:
        sql += " AND p.variante IN (%s)" % ",".join("?" * len(args.varianten))
        params += list(args.varianten)
    sql += " ORDER BY p.frage_id, p.variante"
    prompts = con.execute(sql, params).fetchall()

    vorhanden = set()
    if not args.neu:
        vorhanden = {(z[0], round(z[1], 6)) for z in
                     con.execute("SELECT prompt_id, rho_ziel FROM kompressat")}

    offen = [(pr, rho) for pr in prompts for rho in stufen
             if (pr["prompt_id"], round(rho, 6)) not in vorhanden]
    print("%d Bedingungen offen (%d Prompts x %d Stufen)"
          % (len(offen), len(prompts), len(stufen)), file=sys.stderr)

    n = fehler = 0
    t_start = time.time()
    for i, (pr, rho) in enumerate(offen, start=1):
        if rho == 0.0:
            schreibe_kompressat(con, pr["prompt_id"], 0.0, pr["text"],
                                pr["text"], lauf_id, tokenizer)
            n += 1
        else:
            try:
                t0 = time.perf_counter()
                erg = komprimiere(pr["text"], round(1 - rho, 6))
                dauer = time.perf_counter() - t0
            except Exception as e:                      # noqa: BLE001
                print("FEHLER %s %s rho=%.2f: %s"
                      % (pr["frage_id"], pr["variante"], rho, e),
                      file=sys.stderr)
                fehler += 1
                continue
            schreibe_kompressat(con, pr["prompt_id"], rho,
                                erg["compressed_prompt"], pr["text"], lauf_id,
                                tokenizer, erg, dauer)
            n += 1
        if i % 100 == 0:
            con.commit()
            print("  %d / %d  (%.1f s)" % (i, len(offen), time.time() - t_start),
                  file=sys.stderr)
    con.commit()
    dbio.lauf_beenden(con, lauf_id, n)

    print("%d Kompressate geschrieben, %d Fehler, %.1f s"
          % (n, fehler, time.time() - t_start))
    print()
    print("%-14s %6s %6s %10s %10s %10s" %
          ("variante", "rho", "n", "rho_ist", "delta", "dauer_s"))
    for z in con.execute(
            "SELECT p.variante, k.rho_ziel, COUNT(*) n, "
            "AVG(k.rho_ist_wort) ist, AVG(k.rho_delta_wort) d, "
            "AVG(k.dauer_kompression_s) t FROM kompressat k "
            "JOIN prompt p ON p.prompt_id = k.prompt_id "
            "GROUP BY p.variante, k.rho_ziel ORDER BY p.variante, k.rho_ziel"):
        print("%-14s %6.2f %6d %10.3f %10.3f %10s" % (
            z["variante"], z["rho_ziel"], z["n"], z["ist"] or 0, z["d"] or 0,
            ("%.2f" % z["t"]) if z["t"] else "-"))
    con.close()


if __name__ == "__main__":
    main()
