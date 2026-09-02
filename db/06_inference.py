#!/usr/bin/env python3
"""
Schritt 6: Teilstudie B, Inferenz und Antwortkorrektheit (subsec:inferenzsetup).

Uebergibt jede Bedingung der Teilstichprobe B an das Zielmodell, bewertet die
Ausgabe als Exact Match ueber die Aliasliste (subsubsec:korrektheit), ordnet
Fehler einer von vier Klassen zu und protokolliert die Effizienzkennzahlen
(subsubsec:effizienzkennzahlen). Alles landet in der Tabelle inferenz.

Bei 30 Aufgaben x 5 Varianten x 4 Stufen ergeben sich 600 Inferenzen.

Determinierung: Temperatur 0, kein Sampling, fixierter Zufallsstartwert. Mit
--wiederholung 2 laesst sich die in subsec:inferenzsetup vorgesehene
Wiederholungsmessung auf einer Teilmenge fahren; die Zeilen stehen dann
nebeneinander und lassen sich unmittelbar vergleichen.

Aufruf:
    python3 06_inference.py
    python3 06_inference.py --limit 20 --wiederholung 2   Determinismuspruefung
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio   # noqa: E402

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Zwei Ausfuehrungswege fuer dasselbe Modell.
#
#   "ollama"        llama.cpp ueber den lokalen Ollama-Server. Auf Apple
#                   Silicon der einzig praktikable Weg, da bitsandbytes dort
#                   nicht laeuft. Quantisierung Q4_K_M, also ebenfalls 4 Bit.
#   "transformers"  transformers mit bitsandbytes in nf4, auf CUDA.
#
# Die Wahl wird je Lauf protokolliert und gehoert in subsec:inferenzsetup,
# denn Quantisierungsverfahren und Laufzeitumgebung sind nicht neutral
# gegenueber der Genauigkeit.
BACKEND = "ollama"

MODELL = "llama3.1:8b"                 # Ollama
MODELL_HF = "meta-llama/Llama-3.1-8B-Instruct"
QUANTISIERUNG = {"ollama": "Q4_K_M", "transformers": "4bit-nf4"}
TEMPERATUR = 0.0
TOP_P = 1.0
SEED = 42
MAX_NEW_TOKENS = 32
STOP_STRINGS = ["\n"]

# Regelbasierte Fehlertypologie. Die Zuordnung wird nach subsubsec:korrektheit
# an fuenfzig Faellen manuell geprueft; das Ergebnis gehoert in die Spalte
# fehlerklasse_manuell.
VERWEIGERUNG_RE = re.compile(
    r"\b(i (do not|don't) know|cannot|can't|unable|no information|not "
    r"(sure|specified|provided|enough)|sorry|as an ai)\b", re.I)
# Ab dieser Wortzahl gilt eine Ausgabe als Formatverstoss, da die Instruktion
# ausschliesslich den Entitaetsnamen verlangt.
MAX_WORTE_FORMATTREU = 6


# ---------------------------------------------------------------------------
# Bewertung
# ---------------------------------------------------------------------------

def normalisiere(text):
    """Kleinschreibung, keine Satzzeichen, keine englischen Artikel."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def bewerte(ausgabe, aliasformen):
    """Exact Match als Teilzeichenkette ueber die Aliasliste.

    Rueckgabe: (korrekt, getroffener Alias oder None, normalisierte Ausgabe).
    """
    norm = normalisiere(ausgabe)
    gepolstert = " %s " % norm
    for a in aliasformen:
        an = normalisiere(a)
        if an and (" %s " % an) in gepolstert:
            return 1, a, norm
    return 0, None, norm


def fehlerklasse(ausgabe, norm, abgeschnitten):
    """Eine von vier Klassen fuer jede als falsch gewertete Ausgabe."""
    if not norm:
        return "leer_abgeschnitten"
    if abgeschnitten and len(norm.split()) <= 1:
        return "leer_abgeschnitten"
    if VERWEIGERUNG_RE.search(ausgabe or ""):
        return "verweigerung"
    if len(norm.split()) > MAX_WORTE_FORMATTREU:
        return "formatverstoss"
    return "falsche_entitaet"


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

_modell = None
_tokenizer = None


def lade_modell(name=MODELL):
    global _modell, _tokenizer
    if _modell is not None:
        return _modell, _tokenizer
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    print("Lade %s ..." % name, file=sys.stderr)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _modell = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, device_map="auto")
    _modell.eval()
    return _modell, _tokenizer


def erzeuge_ollama(prompt, modell, url=None):
    """Deterministische Dekodierung ueber den lokalen Ollama-Server.

    raw=True unterdrueckt die Chatvorlage. Der Prompt ist ein
    Vervollstaendigungsprompt nach subsubsec:basisprompt und darf nicht
    zusaetzlich in ein Rollenformat gehuellt werden, weil das die
    Promptstruktur veraendern und den Vergleich der Bedingungen entwerten
    wuerde. Die Zeiten stammen aus der Antwort des Servers und sind dort in
    Nanosekunden angegeben.
    """
    import json as _json
    import urllib.request
    url = (url or dbio.OLLAMA_URL).rstrip("/")
    daten = _json.dumps({
        "model": modell, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": TEMPERATUR, "top_p": TOP_P, "seed": SEED,
                    "num_predict": MAX_NEW_TOKENS, "stop": STOP_STRINGS},
    }).encode("utf-8")
    req = urllib.request.Request(url + "/api/generate", data=daten,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as fh:
        erg = _json.loads(fh.read())
    t_gesamt = time.perf_counter() - t0
    n_out = erg.get("eval_count") or 0
    prefill = (erg.get("prompt_eval_duration") or 0) / 1e9
    dekod = (erg.get("eval_duration") or 0) / 1e9
    return {"ausgabe": (erg.get("response") or "").strip(),
            "n_in": erg.get("prompt_eval_count"), "n_out": n_out,
            "t_prefill": prefill, "t_dekodierung": dekod,
            "t_gesamt": t_gesamt,
            "abgeschnitten": 1 if n_out >= MAX_NEW_TOKENS else 0}


def erzeuge(prompt):
    """Deterministische Dekodierung. Rueckgabe: dict mit Ausgabe und Zeiten."""
    import torch
    from transformers import set_seed
    modell, tok = lade_modell()
    set_seed(SEED)

    eingabe = tok(prompt, return_tensors="pt").to(modell.device)
    n_in = int(eingabe["input_ids"].shape[-1])

    t0 = time.perf_counter()
    with torch.no_grad():
        # Vorverarbeitung getrennt messen: ein Durchlauf ohne Erzeugung
        # entspricht dem Prefill.
        modell(**eingabe)
        t_prefill = time.perf_counter() - t0
        aus = modell.generate(**eingabe, max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False, temperature=None, top_p=None,
                              stop_strings=STOP_STRINGS, tokenizer=tok,
                              pad_token_id=tok.eos_token_id)
    t_gesamt = time.perf_counter() - t0

    neu = aus[0][n_in:]
    text = tok.decode(neu, skip_special_tokens=True)
    n_out = int(neu.shape[-1])
    return {"ausgabe": text.strip(), "n_in": n_in, "n_out": n_out,
            "t_prefill": t_prefill, "t_dekodierung": t_gesamt - t_prefill,
            "t_gesamt": t_gesamt,
            "abgeschnitten": 1 if n_out >= MAX_NEW_TOKENS else 0}


# ---------------------------------------------------------------------------

def pruefe_determinismus(con):
    """Vergleicht Wiederholung 1 und 2 derselben Bedingung (V8).

    subsec:inferenzsetup sichert zu, dass dieselbe Eingabe bei Temperatur 0
    und fixiertem Zufallsstartwert dieselbe Ausgabe liefert. Die Zusicherung
    wird hier belegt statt behauptet: erzeugt werden die Wiederholungen mit
    --wiederholung 2 auf einer Teilmenge, verglichen wird hier.
    """
    zeilen = con.execute(
        "SELECT a.kompressat_id, a.ausgabe o1, b.ausgabe o2, "
        "a.korrekt k1, b.korrekt k2, p.variante, k.rho_ziel "
        "FROM inferenz a JOIN inferenz b "
        "  ON b.kompressat_id = a.kompressat_id AND b.wiederholung = 2 "
        "JOIN kompressat k ON k.kompressat_id = a.kompressat_id "
        "JOIN prompt p ON p.prompt_id = k.prompt_id "
        "WHERE a.wiederholung = 1").fetchall()
    if not zeilen:
        print("Keine Wiederholungsmessung vorhanden. Zuerst:\n"
              "  python3 06_inference.py --wiederholung 2 --limit 20")
        return
    gleich = sum(1 for z in zeilen if z["o1"] == z["o2"])
    gleich_k = sum(1 for z in zeilen if z["k1"] == z["k2"])
    print("Determinismuspruefung (V8) ueber %d Bedingungen" % len(zeilen))
    print("  identische Ausgabe:      %d von %d" % (gleich, len(zeilen)))
    print("  identische Bewertung:    %d von %d" % (gleich_k, len(zeilen)))
    for z in zeilen:
        if z["o1"] != z["o2"]:
            print("  Abweichung %s rho=%.2f:\n    1: %r\n    2: %r"
                  % (z["variante"], z["rho_ziel"], z["o1"], z["o2"]))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--wiederholung", type=int, default=1,
                   help="Laufnummer, >1 fuer die Determinismuspruefung")
    p.add_argument("--varianten", nargs="*", default=None)
    p.add_argument("--stufen", nargs="*", type=float, default=None)
    p.add_argument("--neu", action="store_true")
    p.add_argument("--nur-bewerten", action="store_true",
                   help="kein Modell laden, nur vorhandene Ausgaben neu "
                        "bewerten und klassifizieren")
    p.add_argument("--modell", default=None,
                   help="Modellname; Vorgabe je nach Backend")
    p.add_argument("--backend", choices=("ollama", "transformers"),
                   default=BACKEND)
    p.add_argument("--hardware", default=None)
    p.add_argument("--alle-aufgaben", action="store_true",
                   help="Teilstudie B auf alle Aufgaben der Grundstichprobe "
                        "ausdehnen statt auf die Teilstichprobe. Die "
                        "Kompressate liegen aus Teilstudie A ohnehin fuer "
                        "alle Aufgaben vor, es kommt allein Inferenzzeit "
                        "hinzu; das Skript arbeitet nur offene Bedingungen ab, "
                        "die Erweiterung setzt also auf einem fertigen Lauf "
                        "der Teilstichprobe auf.")
    p.add_argument("--pruefe-determinismus", action="store_true",
                   help="vergleicht Wiederholung 1 und 2 und berichtet die "
                        "Uebereinstimmung; laedt kein Modell")
    args = p.parse_args()
    if args.modell is None:
        args.modell = MODELL if args.backend == "ollama" else MODELL_HF

    con = dbio.verbinde(args.db)

    if args.pruefe_determinismus:
        pruefe_determinismus(con)
        return

    if args.nur_bewerten:
        n = 0
        for z in con.execute(
                "SELECT i.inferenz_id, i.ausgabe, i.abgeschnitten, f.antworten "
                "FROM inferenz i JOIN kompressat k USING(kompressat_id) "
                "JOIN prompt p ON p.prompt_id = k.prompt_id "
                "JOIN frage f ON f.frage_id = p.frage_id").fetchall():
            korrekt, alias, norm = bewerte(z["ausgabe"],
                                           json.loads(z["antworten"]))
            kl = None if korrekt else fehlerklasse(z["ausgabe"], norm,
                                                   z["abgeschnitten"])
            con.execute("UPDATE inferenz SET korrekt=?, getroffener_alias=?, "
                        "ausgabe_normalisiert=?, fehlerklasse=? "
                        "WHERE inferenz_id=?",
                        (korrekt, alias, norm, kl, z["inferenz_id"]))
            n += 1
        con.commit()
        print("%d Ausgaben neu bewertet" % n)
        return

    lauf_id = dbio.lauf_beginnen(con, "inferenz", __file__, konfiguration={
        "modell": args.modell, "backend": args.backend,
        "quantisierung": QUANTISIERUNG[args.backend],
        "temperatur": TEMPERATUR, "top_p": TOP_P, "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "stop_strings": STOP_STRINGS,
        "wiederholung": args.wiederholung, "hardware": args.hardware,
        "alle_aufgaben": args.alle_aufgaben,
    })

    sql = ("SELECT k.kompressat_id, k.text, f.antworten, p.variante, "
           "k.rho_ziel, f.frage_id FROM kompressat k "
           "JOIN prompt p ON p.prompt_id = k.prompt_id "
           "JOIN frage f ON f.frage_id = p.frage_id WHERE 1=1")
    params = []
    if not args.alle_aufgaben:
        sql += " AND f.in_teilstudie_b = 1"
    if args.varianten:
        sql += " AND p.variante IN (%s)" % ",".join("?" * len(args.varianten))
        params += list(args.varianten)
    if args.stufen:
        sql += " AND k.rho_ziel IN (%s)" % ",".join("?" * len(args.stufen))
        params += list(args.stufen)
    if not args.neu:
        sql += (" AND NOT EXISTS (SELECT 1 FROM inferenz i WHERE "
                "i.kompressat_id = k.kompressat_id AND i.wiederholung = ?)")
        params.append(args.wiederholung)
    sql += " ORDER BY f.frage_id, p.variante, k.rho_ziel"
    if args.limit:
        sql += " LIMIT %d" % args.limit
    offen = con.execute(sql, params).fetchall()
    print("%d Inferenzen offen" % len(offen), file=sys.stderr)

    laufzeit = None
    if args.backend == "ollama":
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(
                    dbio.OLLAMA_URL + "/api/version", timeout=10) as fh:
                laufzeit = "ollama %s" % _json.loads(fh.read()).get("version")
        except Exception:                      # noqa: BLE001
            laufzeit = "ollama"
    else:
        try:
            import torch
            import transformers
            laufzeit = "torch %s, transformers %s" % (torch.__version__,
                                                      transformers.__version__)
        except ImportError:
            pass

    n = 0
    t0 = time.time()
    for i, z in enumerate(offen, start=1):
        erg = (erzeuge_ollama(z["text"], args.modell)
               if args.backend == "ollama" else erzeuge(z["text"]))
        korrekt, alias, norm = bewerte(erg["ausgabe"],
                                       json.loads(z["antworten"]))
        kl = None if korrekt else fehlerklasse(erg["ausgabe"], norm,
                                               erg["abgeschnitten"])
        con.execute(
            "INSERT OR REPLACE INTO inferenz (kompressat_id, wiederholung, "
            "ausgabe, ausgabe_normalisiert, korrekt, getroffener_alias, "
            "fehlerklasse, abgeschnitten, n_eingabe_token, n_ausgabe_token, "
            "dauer_prefill_s, dauer_dekodierung_s, dauer_gesamt_s, modell, "
            "quantisierung, temperatur, top_p, seed, max_new_tokens, "
            "laufzeitumgebung, hardware, erzeugt, lauf_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (z["kompressat_id"], args.wiederholung, erg["ausgabe"], norm,
             korrekt, alias, kl, erg["abgeschnitten"], erg["n_in"],
             erg["n_out"], erg["t_prefill"], erg["t_dekodierung"],
             erg["t_gesamt"], args.modell, QUANTISIERUNG[args.backend],
             TEMPERATUR, TOP_P,
             SEED, MAX_NEW_TOKENS, laufzeit, args.hardware, dbio.jetzt(),
             lauf_id))
        n += 1
        if i % 25 == 0:
            con.commit()
            print("  %d / %d  (%.0f s)" % (i, len(offen), time.time() - t0),
                  file=sys.stderr)
    con.commit()
    dbio.lauf_beenden(con, lauf_id, n)

    print("%d Inferenzen, %.0f s" % (n, time.time() - t0))
    print()
    print("%-14s %6s %6s %10s %12s %10s" %
          ("variante", "rho", "n", "genauigkeit", "n_eingabe", "dauer_s"))
    for z in con.execute("SELECT * FROM v_genauigkeit ORDER BY variante, rho_ziel"):
        print("%-14s %6.2f %6d %10.3f %12.1f %10.2f" % (
            z["variante"], z["rho_ziel"], z["n"], z["genauigkeit"],
            z["n_eingabe_token_mittel"] or 0, z["dauer_inferenz_mittel"] or 0))
    print()
    print("Fehlerklassen:")
    for z in con.execute("SELECT fehlerklasse, COUNT(*) n FROM inferenz "
                         "WHERE korrekt = 0 GROUP BY fehlerklasse ORDER BY n DESC"):
        print("   %-22s %5d" % (z["fehlerklasse"], z["n"]))
    con.close()


if __name__ == "__main__":
    main()
