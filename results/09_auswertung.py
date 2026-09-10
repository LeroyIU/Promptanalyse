#!/usr/bin/env python3
"""
Auswertung fuer den Ergebnisteil (sec:Ergebnisse).

Rechnet ohne Fremdpakete, allein aus promptanalyse.db. Alle Groessen mit
95-Prozent-Konfidenzintervall.

Die Beobachtungen sind innerhalb einer Aufgabe nicht unabhaengig, weil alle
Bedingungen einer Aufgabe auf denselben Kerninhalt zurueckgehen. Der
Variantenvergleich wird deshalb durchgaengig innerhalb der Aufgabe gepaart
gerechnet: Fuer jede Aufgabe wird die Differenz zweier Varianten gebildet und
ueber die 300 Aufgaben gemittelt. Das entspricht dem zufaelligen
Achsenabschnitt je Aufgabe aus subsec:auswertung und ist ohne Modellanpassung
exakt.

Bloecke:
  1 Ratenkonformitaet          V3, eq:istrate
  2 Kalibrierung               V2, V13, V14, eq:kalibrierung
  3 Auszeichnung               V12, Kontrollgroessen
  4 Selektivitaet              H1, eq:selektivitaetfrei
  5 Kontraste                  H2, gerichtet, Holm-korrigiert
  6 Abschnittsverteilung
  7 Genauigkeit                H3
  8 Kompressionstoleranz       eq:toleranz
  9 Fehlerklassen
 10 Effizienz
 11 Hoechste Stufe        Verwertbarkeit von rho = 0,75 in Teilstudie B

Aufruf:
    python3 09_auswertung.py
"""

import argparse
import math
from collections import defaultdict

import dbio

VARIANTEN = ("filler", "semantic", "instruction", "demonstration")
STUFEN = (0.25, 0.50, 0.75)
PRUEFSTUFEN = (0.25, 0.50)


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------

def mittel_ki(werte):
    """Mittelwert mit 95-Prozent-Intervall ueber die Normalapproximation."""
    n = len(werte)
    if n < 2:
        return (werte[0] if n else float("nan")), float("nan"), float("nan"), n
    m = sum(werte) / n
    var = sum((x - m) ** 2 for x in werte) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - 1.96 * se, m + 1.96 * se, n


def wilson(k, n):
    """Wilson-Intervall fuer einen Anteil, robuster als die Normalformel."""
    if not n:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.96
    nenner = 1 + z * z / n
    mitte = (p + z * z / (2 * n)) / nenner
    rand = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / nenner
    return p, mitte - rand, mitte + rand


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gepaart(a, b):
    """Gepaarter Vergleich a - b. Einseitiger p-Wert fuer a > b."""
    diff = [x - y for x, y in zip(a, b)]
    m, lo, hi, n = mittel_ki(diff)
    var = sum((d - m) ** 2 for d in diff) / (n - 1)
    se = math.sqrt(var / n)
    z = m / se if se else float("inf") * (1 if m > 0 else -1)
    return m, lo, hi, 1 - normal_cdf(z), n


def holm(paare):
    """Sequenziell ablehnendes Verfahren nach Holm."""
    sortiert = sorted(paare, key=lambda t: t[1])
    m = len(sortiert)
    ergebnis, vorher = {}, 0.0
    for i, (name, p) in enumerate(sortiert):
        korr = max(vorher, min(1.0, (m - i) * p))
        ergebnis[name] = korr
        vorher = korr
    return ergebnis


def zeile(label, m, lo, hi, extra=""):
    print("  %-30s %8.3f  [%7.3f, %7.3f]%s" % (label, m, lo, hi, extra))


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    args = p.parse_args()
    con = dbio.verbinde(args.db, readonly=True)

    # ---- 1 Ratenkonformitaet ---------------------------------------------
    print("\n" + "=" * 72)
    print("1  RATENKONFORMITAET  (V3, eq:istrate)")
    print("=" * 72)
    for stufe in STUFEN:
        print("\n rho = %.2f" % stufe)
        for v in ("basis",) + VARIANTEN:
            werte = [r["rho_ist_token"] for r in con.execute(
                "SELECT rho_ist_token FROM v_bedingung "
                "WHERE variante=? AND rho_ziel=?", (v, stufe))]
            m, lo, hi, n = mittel_ki(werte)
            zeile(v, m, lo, hi, "   delta = %+.3f" % (m - stufe))

    # ---- 2 Kalibrierung ---------------------------------------------------
    print("\n" + "=" * 72)
    print("2  KALIBRIERUNG  (V2, V13, V14, eq:kalibrierung)")
    print("=" * 72)
    for r in con.execute("SELECT * FROM v_kalibrierung"):
        print("\n %s" % r["variante"])
        print("   Quote L_red/L_basis   %.3f  [min %.3f, max %.3f]"
              % (r["quote_token_mittel"], r["quote_token_min"],
                 r["quote_token_max"]))
        print("   Anteil im Band        %.3f" % r["anteil_im_band"])
        print("   Einheiten             %.2f" % r["n_einheiten_mittel"])
        print("   lex. Ueberlappung     %.3f" % r["lex_overlap_mittel"])
        print("   Antwortleckage        %.4f" % r["anteil_antwortleckage"])

    # ---- 3 Auszeichnung ---------------------------------------------------
    print("\n" + "=" * 72)
    print("3  KONTROLLGROESSEN DER AUSZEICHNUNG  (V12)")
    print("=" * 72)
    r = con.execute(
        "SELECT COUNT(*) n, SUM(s_intervall_offen) offen, "
        "SUM(CASE WHEN anteil_mehrdeutig>0 THEN 1 ELSE 0 END) mehrdeutig, "
        "AVG(anteil_mehrdeutig) anteil, "
        "SUM(CASE WHEN basis_ist_teilfolge=0 THEN 1 ELSE 0 END) keine_tf, "
        "SUM(n_komp_unzugeordnet) unzug "
        "FROM v_bedingung WHERE rho_ziel>0").fetchone()
    print("   Bedingungen                        %d" % r["n"])
    print("   mit mehrdeutigem Wort              %d  (%.2f %%)"
          % (r["mehrdeutig"], 100 * r["mehrdeutig"] / r["n"]))
    print("   mittlerer Anteil mehrdeutig        %.4f %%" % (100 * r["anteil"]))
    print("   Aufloesungsintervall offen         %d  (%.2f %%)"
          % (r["offen"], 100 * r["offen"] / r["n"]))
    print("   Basis nicht Teilfolge              %d" % r["keine_tf"])
    print("   Kompressatwoerter unzugeordnet     %d" % r["unzug"])

    # ---- 4 Selektivitaet --------------------------------------------------
    print("\n" + "=" * 72)
    print("4  SELEKTIVITAET  (H1, eq:selektivitaetfrei)")
    print("=" * 72)
    ln_werte = {}
    for stufe in STUFEN:
        print("\n rho = %.2f      S~ geometrisch mit 95-Prozent-Intervall" % stufe)
        for v in VARIANTEN:
            rows = con.execute(
                "SELECT frage_id, ln_s_frei, r_red_null FROM v_bedingung "
                "WHERE variante=? AND rho_ziel=? AND n_red>0 ORDER BY frage_id",
                (v, stufe)).fetchall()
            werte = [r["ln_s_frei"] for r in rows]
            ln_werte[(v, stufe)] = {r["frage_id"]: r["ln_s_frei"] for r in rows}
            m, lo, hi, n = mittel_ki(werte)
            n_null = sum(r["r_red_null"] or 0 for r in rows)
            marke = "  H1 bestaetigt" if lo > 0 else (
                "  gegen H1" if hi < 0 else "  n.s.")
            zeile(v, math.exp(m), math.exp(lo), math.exp(hi),
                  "  n=%d  R_red=0: %d%s" % (n, n_null, marke))

    # ---- 5 Kontraste ------------------------------------------------------
    print("\n" + "=" * 72)
    print("5  GEPLANTE KONTRASTE  (H2, einseitig, Holm-korrigiert)")
    print("=" * 72)
    print(" postulierte Rangfolge: filler > semantic > instruction > demonstration")
    kontraste = [("filler", "semantic"), ("semantic", "instruction"),
                 ("instruction", "demonstration")]
    for stufe in PRUEFSTUFEN:
        print("\n rho = %.2f" % stufe)
        roh, ergebnisse = [], {}
        for hoch, tief in kontraste:
            gem = sorted(set(ln_werte[(hoch, stufe)]) & set(ln_werte[(tief, stufe)]))
            a = [ln_werte[(hoch, stufe)][f] for f in gem]
            b = [ln_werte[(tief, stufe)][f] for f in gem]
            m, lo, hi, pw, n = gepaart(a, b)
            name = "%s > %s" % (hoch, tief)
            ergebnisse[name] = (m, lo, hi, n)
            roh.append((name, pw))
        korr = holm(roh)
        for name, _ in roh:
            m, lo, hi, n = ergebnisse[name]
            urteil = "bestaetigt" if korr[name] < 0.05 and m > 0 else (
                "GEGENLAEUFIG" if m < 0 else "nicht bestaetigt")
            print("  %-28s d(ln S~) = %+6.3f  [%+6.3f, %+6.3f]  "
                  "p_holm = %.4f  %s" % (name, m, lo, hi, korr[name], urteil))

        # Tatsaechliche Rangfolge
        rang = sorted(VARIANTEN, key=lambda v: -sum(
            ln_werte[(v, stufe)].values()) / len(ln_werte[(v, stufe)]))
        print("  beobachtete Rangfolge:      %s" % " > ".join(rang))

    # ---- 6 Abschnittsverteilung ------------------------------------------
    print("\n" + "=" * 72)
    print("6  ERHALTUNGSRATEN JE PROMPTABSCHNITT  (Kerninhalt)")
    print("=" * 72)
    for stufe in PRUEFSTUFEN:
        print("\n rho = %.2f" % stufe)
        print("  %-16s %10s %10s %10s" % ("Variante", "Instrukt.", "Demos", "Testfrage"))
        for v in ("basis",) + VARIANTEN:
            raten = {}
            for r in con.execute(
                    "SELECT ae.abschnitt, AVG(ae.erhaltungsrate) rate "
                    "FROM abschnitt_erhalt ae "
                    "JOIN kompressat k ON k.kompressat_id=ae.kompressat_id "
                    "JOIN prompt p ON p.prompt_id=k.prompt_id "
                    "WHERE ae.herkunft='kern' AND p.variante=? AND k.rho_ziel=? "
                    "GROUP BY ae.abschnitt", (v, stufe)):
                raten[r["abschnitt"]] = r["rate"]
            print("  %-16s %10.3f %10.3f %10.3f"
                  % (v, raten.get("instruktion", float("nan")),
                     raten.get("demonstration", float("nan")),
                     raten.get("testfrage", float("nan"))))

    # ---- 7 Genauigkeit ----------------------------------------------------
    print("\n" + "=" * 72)
    print("7  GENAUIGKEIT  (H3)")
    print("=" * 72)
    genau = {}
    for v in ("basis",) + VARIANTEN:
        for stufe in (0.0,) + STUFEN:
            rows = con.execute(
                "SELECT frage_id, korrekt FROM v_studie_b "
                "WHERE variante=? AND rho_ziel=? AND wiederholung=1 "
                "ORDER BY frage_id", (v, stufe)).fetchall()
            genau[(v, stufe)] = {r["frage_id"]: r["korrekt"] for r in rows}
    for stufe in (0.0,) + STUFEN:
        print("\n rho = %.2f" % stufe)
        for v in ("basis",) + VARIANTEN:
            d = genau[(v, stufe)]
            k, n = sum(d.values()), len(d)
            pr, lo, hi = wilson(k, n)
            zeile(v, pr, lo, hi, "   %d/%d" % (k, n))

    print("\n Gepaarte Differenz zur Variante basis, je Stufe")
    for stufe in STUFEN:
        print("\n rho = %.2f" % stufe)
        for v in VARIANTEN:
            gem = sorted(set(genau[(v, stufe)]) & set(genau[("basis", stufe)]))
            a = [genau[(v, stufe)][f] for f in gem]
            b = [genau[("basis", stufe)][f] for f in gem]
            m, lo, hi, pw, n = gepaart(a, b)
            zeile("%s - basis" % v, m, lo, hi)

    # ---- 8 Kompressionstoleranz ------------------------------------------
    print("\n" + "=" * 72)
    print("8  KOMPRESSIONSTOLERANZ  (eq:toleranz, delta = 0,05)")
    print("=" * 72)
    for v in ("basis",) + VARIANTEN:
        ref = sum(genau[(v, 0.0)].values()) / len(genau[(v, 0.0)])
        tau = 0.0
        for stufe in STUFEN:
            d = genau[(v, stufe)]
            if sum(d.values()) / len(d) >= ref - 0.05:
                tau = stufe
        print("  %-16s A(0) = %.3f    tau = %.2f" % (v, ref, tau))

    # ---- 9 Fehlerklassen --------------------------------------------------
    print("\n" + "=" * 72)
    print("9  FEHLERKLASSEN  (Anteil an allen Bedingungen je Zelle)")
    print("=" * 72)
    for stufe in (0.0,) + STUFEN:
        print("\n rho = %.2f" % stufe)
        print("  %-16s %10s %10s %10s %8s"
              % ("Variante", "falsch", "format", "leer/abg.", "verweig."))
        for v in ("basis",) + VARIANTEN:
            z = defaultdict(int)
            for r in con.execute(
                    "SELECT fehlerklasse, COUNT(*) n FROM v_studie_b "
                    "WHERE variante=? AND rho_ziel=? AND wiederholung=1 "
                    "AND korrekt=0 GROUP BY fehlerklasse", (v, stufe)):
                z[r["fehlerklasse"]] = r["n"]
            print("  %-16s %10d %10d %10d %8d"
                  % (v, z["falsche_entitaet"], z["formatverstoss"],
                     z["leer_abgeschnitten"], z["verweigerung"]))

    # ---- 10 Effizienz -----------------------------------------------------
    print("\n" + "=" * 72)
    print("10  EFFIZIENZ  (subsubsec:effizienzkennzahlen)")
    print("=" * 72)
    print("  %-16s %5s %9s %10s %10s %11s"
          % ("Variante", "rho", "Eingabe", "t_komp", "t_inf", "netto"))
    for v in ("basis",) + VARIANTEN:
        ref = con.execute(
            "SELECT AVG(dauer_gesamt_s) t FROM v_studie_b "
            "WHERE variante=? AND rho_ziel=0 AND wiederholung=1",
            (v,)).fetchone()["t"]
        for stufe in (0.0,) + STUFEN:
            r = con.execute(
                "SELECT AVG(n_eingabe_token) tok, AVG(dauer_gesamt_s) t_inf, "
                "AVG(COALESCE(dauer_kompression_s,0)) t_komp, "
                "AVG(dauer_pipeline_s) t_pipe FROM v_studie_b "
                "WHERE variante=? AND rho_ziel=? AND wiederholung=1",
                (v, stufe)).fetchone()
            print("  %-16s %5.2f %9.1f %10.3f %10.3f %+11.3f"
                  % (v, stufe, r["tok"], r["t_komp"], r["t_inf"],
                     ref - r["t_pipe"]))
    print()

    # ---- 11 Hoechste Stufe: woran haengt die Genauigkeit bei rho = 0,75? ----
    print("\n" + "=" * 72)
    print("11  HOECHSTE REDUKTIONSSTUFE  (Verwertbarkeit von rho = 0,75)")
    print("=" * 72)
    print("\n Die Genauigkeit haengt fast vollstaendig daran, ob die Testfrage")
    print(" die Kompression ueberstanden hat.\n")

    TF = ("SELECT ae.kompressat_id, p.variante, ae.erhaltungsrate r_tf "
          "FROM abschnitt_erhalt ae "
          "JOIN kompressat k ON k.kompressat_id = ae.kompressat_id "
          "JOIN prompt p ON p.prompt_id = k.prompt_id "
          "WHERE ae.abschnitt = 'testfrage' AND ae.herkunft = 'kern' "
          "AND k.rho_ziel = 0.75")

    gesamt = con.execute(
        "SELECT CASE WHEN t.r_tf >= 0.5 THEN 1 ELSE 0 END s, "
        "COUNT(*) n, SUM(i.korrekt) k FROM (%s) t "
        "JOIN inferenz i ON i.kompressat_id = t.kompressat_id "
        "AND i.wiederholung = 1 GROUP BY s" % TF).fetchall()
    for r in gesamt:
        pr, lo, hi = wilson(r["k"], r["n"])
        zeile("Testfrage %s 50 %% erhalten" % (">=" if r["s"] else "<"),
              pr, lo, hi, "   %d/%d" % (r["k"], r["n"]))

    print("\n Anteil der Bedingungen mit erhaltener Testfrage, je Variante")
    anteile = {}
    for r in con.execute(
            "SELECT variante, COUNT(*) n, "
            "SUM(CASE WHEN r_tf >= 0.5 THEN 1 ELSE 0 END) k "
            "FROM (%s) GROUP BY variante" % TF):
        anteile[r["variante"]] = r["k"] / r["n"]
        pr, lo, hi = wilson(r["k"], r["n"])
        zeile(r["variante"], pr, lo, hi, "   %d/%d" % (r["k"], r["n"]))

    print("\n Genauigkeit innerhalb der Schicht mit erhaltener Testfrage")
    print(" (deskriptive Zerlegung: die Schicht ist selbst variantenabhaengig)")
    for r in con.execute(
            "SELECT t.variante, COUNT(*) n, SUM(i.korrekt) k FROM (%s) t "
            "JOIN inferenz i ON i.kompressat_id = t.kompressat_id "
            "AND i.wiederholung = 1 WHERE t.r_tf >= 0.5 "
            "GROUP BY t.variante ORDER BY 3 DESC" % TF):
        if r["n"] < 10:
            print("  %-30s n = %d, zu wenige Faelle" % (r["variante"], r["n"]))
            continue
        pr, lo, hi = wilson(r["k"], r["n"])
        zeile(r["variante"], pr, lo, hi, "   %d/%d" % (r["k"], r["n"]))

    print("\n Markenanteil der eingefuegten Redundanz gegen Erhalt der Testfrage")
    for r in con.execute(
            "SELECT p.variante, "
            "SUM(CASE WHEN w.form IN ('Q','A') THEN 1 ELSE 0 END)*1.0/COUNT(*) m "
            "FROM wort w JOIN prompt p ON p.prompt_id = w.prompt_id "
            "WHERE w.ist_wort = 1 AND w.herkunft = 'redundanz' "
            "GROUP BY p.variante ORDER BY m"):
        print("  %-16s Marken %5.1f %%   Testfrage erhalten %5.1f %%"
              % (r["variante"], 100 * r["m"],
                 100 * anteile.get(r["variante"], float("nan"))))
    print()



if __name__ == "__main__":
    main()
