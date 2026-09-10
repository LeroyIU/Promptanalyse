#!/usr/bin/env python3
"""
Schritt 5: Erhaltungsraten und Selektivitaetsindex (subsubsec:selektivitaet).

    R_kern(rho) = |W_kern & W'| / |W_kern|      (eq:erhaltungsraten)
    R_red(rho)  = |W_red  & W'| / |W_red|
    S(rho)      = R_kern(rho) / R_red(rho)      (eq:selektivitaet)

W_kern und W_red werden nicht rekonstruiert, sondern stammen aus der
Auszeichnung, die beim Bau des Prompts in der Tabelle wort abgelegt wurde
(subsubsec:auszeichnung). Zu bestimmen bleibt allein, welche Positionen des
Gesamtprompts im Kompressat erhalten sind. Da LLMLingua-2 nur streicht und die
Reihenfolge nicht veraendert, ist das Kompressat eine Teilfolge des Originals;
die Einbettung ist damit wohldefiniert.

Mehrdeutigkeiten werden nach dem Prinzip der fruehesten noch nicht belegten
Position aufgeloest. Als Gegenprobe wird dieselbe Rechnung mit der spaetesten
Position gefuehrt. Da Redundanz stets hinter dem zugehoerigen Kernabschnitt
steht, schliessen beide Werte den wahren Index ein.

Ergebnisse werden in kompressat, abschnitt_erhalt und wort_erhalt
zurueckgeschrieben. Der Ordner results entfaellt fuer diesen Schritt; Ausgaben
fuer die Arbeit erzeugt 07_export.py.

Aufruf: python3 05_selectivity.py [--db PFAD] [--neu] [--ohne-wortebene]
"""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio   # noqa: E402

KERN = "kern"
RED = "redundanz"


# ---------------------------------------------------------------------------

def ueberleben(gesamt, komp, rechts=False):
    """Markiert jede Position des Gesamtprompts als erhalten oder gestrichen.

    Rueckgabe: (lebt, teilfolge, unzugeordnet). teilfolge ist 0, wenn das
    Kompressat keine Teilfolge des Originals ist. Dieser Fall widerspricht der
    in subsubsec:auszeichnung getroffenen Annahme ueber LLMLingua-2 und darf
    nicht stillschweigend auf die Blockausrichtung zurueckfallen, sondern wird
    je Bedingung protokolliert und in v_selektivitaet ausgewiesen.
    """
    stellen = (dbio.einbetten_rechts(komp, gesamt) if rechts
               else dbio.einbetten_links(komp, gesamt))
    teilfolge = 1
    unzugeordnet = 0
    if stellen is None:
        teilfolge = 0
        stellen = dbio.einbetten_block(komp, gesamt)
        unzugeordnet = len(komp) - len(stellen)
    lebt = [False] * len(gesamt)
    for j in stellen:
        lebt[j] = True
    return lebt, teilfolge, unzugeordnet


def quotient(r_kern, r_red):
    if r_kern is None or r_red in (None, 0):
        return None
    return r_kern / r_red


def raten(woerter, lebt, nur_frei=False):
    """Zaehlungen und Raten ueber die ausgezeichneten Positionen.

    nur_frei=True beschraenkt auf die Woerter, bei denen der Kompressor
    ueberhaupt entscheiden konnte, laesst also geschuetzte Marken und Ziffern
    aus Zaehler und Nenner heraus.

    Rueckgabe: (n_kern, n_red, e_kern, e_red, R_kern, R_red, S).
    """
    n_k = n_r = e_k = e_r = 0
    for w, l in zip(woerter, lebt):
        if not w["ist_wort"]:
            continue
        if nur_frei and dbio.ist_geschuetzt(w["form"], w["ist_ziffer"]):
            continue
        if w["herkunft"] == KERN:
            n_k += 1
            e_k += 1 if l else 0
        else:
            n_r += 1
            e_r += 1 if l else 0
    r_k = e_k / n_k if n_k else None
    r_r = e_r / n_r if n_r else None
    return n_k, n_r, e_k, e_r, r_k, r_r, quotient(r_k, r_r)


# Stetigkeitskorrektur nach Haldane und Anscombe.
KORR_ZAEHLER = 0.5
KORR_NENNER = 1.0


def freier_index(n_k, n_r, e_k, e_r):
    """Pruefgroesse der Arbeit: R_kern / R_red ueber die freien Woerter.

    Zwei Eingriffe gegenueber eq:selektivitaet, beide notwendig.

    Erstens die Beschraenkung auf die freien Woerter. Der Strukturschutz
    entzieht Marken und Ziffern der Streichung, sie gehen aber in beide
    Erhaltungsraten ein. Da sie zwischen Kern und Redundanz ungleich verteilt
    sind, erzeugt schon ein Verfahren, das die Herkunft vollstaendig ignoriert,
    einen Index deutlich ueber eins. Ueber die freien Woerter ist der Index
    eines indifferenten Verfahrens exakt eins, und nur dann misst er, was er
    messen soll.

    Zweitens die Stetigkeitskorrektur, damit Bedingungen mit vollstaendig
    entfernter Redundanz definiert bleiben. Sie auszuschliessen wuerde die
    Faelle maximaler Selektivitaet entfernen und den Index nach unten
    verzerren.
    """
    if not n_k or not n_r:
        return None, None, None
    r_k = (e_k + KORR_ZAEHLER) / (n_k + KORR_NENNER)
    r_r = (e_r + KORR_ZAEHLER) / (n_r + KORR_NENNER)
    return r_k, r_r, r_k / r_r


def abschnittsraten(woerter, lebt):
    """Erhaltungsraten je Promptabschnitt und Herkunft."""
    zaehler = defaultdict(lambda: [0, 0])
    for w, l in zip(woerter, lebt):
        if not w["ist_wort"]:
            continue
        eintrag = zaehler[(w["abschnitt"], w["herkunft"])]
        eintrag[0] += 1
        eintrag[1] += 1 if l else 0
    return zaehler


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--neu", action="store_true",
                   help="auch bereits berechnete Kompressate neu rechnen")
    p.add_argument("--ohne-wortebene", action="store_true",
                   help="Tabelle wort_erhalt nicht befuellen")
    p.add_argument("--varianten", nargs="*", default=None)
    args = p.parse_args()

    con = dbio.verbinde(args.db)
    lauf_id = dbio.lauf_beginnen(con, "selektivitaet", __file__,
                                 konfiguration={"wortebene":
                                                not args.ohne_wortebene})

    sql = ("SELECT k.kompressat_id, k.prompt_id, k.text, k.rho_ziel, "
           "p.variante FROM kompressat k "
           "JOIN prompt p ON p.prompt_id = k.prompt_id WHERE 1=1")
    params = []
    if not args.neu:
        sql += " AND k.r_kern IS NULL"
    if args.varianten:
        sql += " AND p.variante IN (%s)" % ",".join("?" * len(args.varianten))
        params += list(args.varianten)
    sql += " ORDER BY k.prompt_id, k.rho_ziel"
    aufgaben = con.execute(sql, params).fetchall()
    print("%d Kompressate zu rechnen" % len(aufgaben), file=sys.stderr)

    woerter_cache = {}
    n = 0
    for a in aufgaben:
        pid = a["prompt_id"]
        if pid not in woerter_cache:
            woerter_cache = {pid: con.execute(
                "SELECT wort_id, pos, form, schluessel, ist_wort, ist_ziffer, "
                "herkunft, abschnitt FROM wort WHERE prompt_id = ? "
                "ORDER BY pos", (pid,)).fetchall()}
        woerter = woerter_cache[pid]
        if not woerter:
            print("Warnung: keine Wortebene fuer prompt_id %d" % pid,
                  file=sys.stderr)
            continue

        gesamt = [w["schluessel"] for w in woerter]
        _, komp, _ = dbio.tokenisiere(a["text"])

        lebt, teilfolge, unzugeordnet = ueberleben(gesamt, komp)
        lebt_s, _, _ = ueberleben(gesamt, komp, rechts=True)

        n_k, n_r, e_k, e_r, r_k, r_r, s = raten(woerter, lebt)
        *_, r_ks, r_rs, s_sp = raten(woerter, lebt_s)
        fk, fr, ek_f, er_f, *_ = raten(woerter, lebt, nur_frei=True)
        r_kf, r_rf, s_frei = freier_index(fk, fr, ek_f, er_f)

        mehrdeutig = sum(1 for x, y in zip(lebt, lebt_s) if x != y) // 2
        n_komp_wort = sum(1 for t in komp if dbio.WORT_RE.search(t))
        n_gesamt = sum(1 for w in woerter if w["ist_wort"])

        con.execute(
            "UPDATE kompressat SET n_kern=?, n_red=?, n_gesamt=?, "
            "redundanzanteil=?, r_kern=?, r_red=?, s_index=?, ln_s=?, "
            "r_red_null=?, r_kern_frei=?, r_red_frei=?, n_kern_frei=?, "
            "n_red_frei=?, s_frei=?, ln_s_frei=?, "
            "r_kern_spiegel=?, r_red_spiegel=?, s_spiegel=?, "
            "n_mehrdeutig=?, anteil_mehrdeutig=?, "
            "s_intervall_offen=?, basis_ist_teilfolge=?, "
            "n_basis_unzugeordnet=?, n_komp_unzugeordnet=? "
            "WHERE kompressat_id=?",
            (n_k, n_r, n_gesamt, (n_r / n_gesamt) if n_gesamt else None,
             r_k, r_r, s, math.log(s) if s else None,
             1 if (n_r and r_r == 0) else 0,
             r_kf, r_rf, fk, fr, s_frei,
             math.log(s_frei) if s_frei else None,
             r_ks, r_rs, s_sp, mehrdeutig,
             (mehrdeutig / n_komp_wort) if n_komp_wort else None,
             1 if (s is not None and s_sp is not None
                   and abs(s - s_sp) > 1e-9) else 0,
             teilfolge, 0, unzugeordnet, a["kompressat_id"]))

        # Fuer rho = 0 ist das Kompressat der Prompt selbst. Jede Position ist
        # erhalten, jede Erhaltungsrate ist 1. Diese Zeilen traegt die
        # Datenbank nicht: sie sind aus rho_ziel = 0 ableitbar, machten aber
        # rund vierzig Prozent von wort_erhalt aus. Die Kennzahlen der
        # Bedingung stehen weiterhin in kompressat.
        if a["rho_ziel"] == 0:
            n += 1
            continue

        con.execute("DELETE FROM abschnitt_erhalt WHERE kompressat_id = ?",
                    (a["kompressat_id"],))
        con.executemany(
            "INSERT INTO abschnitt_erhalt (kompressat_id, abschnitt, herkunft, "
            "n_woerter, n_erhalten, erhaltungsrate) VALUES (?,?,?,?,?,?)",
            [(a["kompressat_id"], k[0], k[1], v[0], v[1],
              v[1] / v[0] if v[0] else None)
             for k, v in sorted(abschnittsraten(woerter, lebt).items())])

        if not args.ohne_wortebene:
            con.execute("DELETE FROM wort_erhalt WHERE kompressat_id = ?",
                        (a["kompressat_id"],))
            con.executemany(
                "INSERT INTO wort_erhalt (kompressat_id, wort_id, erhalten, "
                "erhalten_spiegel) VALUES (?,?,?,?)",
                [(a["kompressat_id"], w["wort_id"], int(l), int(ls))
                 for w, l, ls in zip(woerter, lebt, lebt_s)])

        n += 1
        if n % 200 == 0:
            con.commit()
            print("  %d / %d" % (n, len(aufgaben)), file=sys.stderr)
    con.commit()
    dbio.lauf_beenden(con, lauf_id, n)

    print("%d Kompressate ausgewertet" % n)
    print()
    print("%-14s %6s %6s %9s %9s %9s %9s %8s %8s" %
          ("variante", "rho", "n", "R_kern_f", "R_red_f", "S_frei", "S_roh",
           "R_red0", "offen"))
    for z in con.execute(
            "SELECT variante, rho_ziel, n, r_kern_frei_mittel, "
            "r_red_frei_mittel, s_frei_geometrisch, s_geometrisch, "
            "n_r_red_null, n_intervall_offen FROM v_selektivitaet "
            "ORDER BY CASE variante WHEN 'filler' THEN 1 WHEN 'semantic' "
            "THEN 2 WHEN 'instruction' THEN 3 WHEN 'demonstration' THEN 4 "
            "ELSE 5 END, rho_ziel"):
        f = lambda x: ("%.3f" % x) if x is not None else "-"
        print("%-14s %6.2f %6d %9s %9s %9s %9s %8d %8d" % (
            z["variante"], z["rho_ziel"], z["n"], f(z["r_kern_frei_mittel"]),
            f(z["r_red_frei_mittel"]), f(z["s_frei_geometrisch"]),
            f(z["s_geometrisch"]), z["n_r_red_null"], z["n_intervall_offen"]))
    print()
    print("S_frei ist die Pruefgroesse: nur Woerter, ueber die der Kompressor "
          "entscheiden konnte,\nmit Stetigkeitskorrektur. S_roh entspricht "
          "eq:selektivitaet und ist deskriptiv.")
    con.close()


if __name__ == "__main__":
    main()
