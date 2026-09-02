#!/usr/bin/env python3
"""
Schreibschicht fuer Prompts, Segmente und Wortebene.

Wird von 02_build_prompts.py und 03_build_redundancy.py gemeinsam benutzt,
damit Basisprompt und Redundanzvarianten in exakt derselben Weise abgelegt
werden.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbio            # noqa: E402
import promptbau as pb  # noqa: E402

# Toleranzband der Kalibrierung nach eq:kalibrierung.
#
# Das Band ist nicht frei gewaehlt. Die groesste Einfuegeeinheit ist eine
# vollstaendige Demonstration und misst rund 17 Prozent der Basispromptlaenge.
# Eine Teilmengenauswahl kann einen Zielwert deshalb hoechstens bis auf die
# halbe Einheitengroesse treffen; ein Band unterhalb von 0,075 waere fuer die
# Stufe demonstration konstruktiv nicht erreichbar und wuerde zum Ausschluss
# von Aufgaben zwingen, der die Schichtung der Stichprobe zerstoert.
KALIBRIERUNG_ZIEL = 0.50
KALIBRIERUNG_BAND = 0.075


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def material_laden(con, frage_id):
    """Alles, was zum Bau der fuenf Varianten einer Aufgabe noetig ist."""
    f = dbio.eine_zeile(con, "SELECT * FROM frage WHERE frage_id = ?",
                        (frage_id,))
    if f is None:
        return None

    demos, demos_red = [], []
    for d in con.execute("SELECT * FROM demonstration WHERE frage_id = ? "
                         "ORDER BY slot", (frage_id,)):
        eintrag = {"slot": d["slot"], "demo_id": d["demo_id"],
                   "relation": d["demo_relation"], "frage": d["demo_frage"],
                   "antwort": d["demo_antwort"]}
        (demos if d["rolle"] == "base" else demos_red).append(eintrag)

    return {
        "frage_id": frage_id,
        "frage": f["fragetext"],
        "relation": f["relation"],
        "antworten": json.loads(f["antworten"]),
        "demos": demos,
        "demos_red": demos_red,
        "para_frage": paraphrase_waehlen(con, frage_id),
        "para_demos": {i: paraphrase_waehlen(con, d["demo_id"])
                       for i, d in enumerate(demos, start=1)},
    }


def paraphrase_waehlen(con, quelle_id):
    """Deterministische Auswahl einer Paraphrase ueber die Frage-ID.

    Rotiert ueber die zulaessigen template_id, damit nicht durchgaengig
    dieselbe Schablone verwendet wird und die Stufe semantic nicht die
    Eigenheiten einer einzelnen Schablone abbildet. Die Auswahl haengt nur von
    der Frage-ID ab, nicht von der Kalibrierung: welche Paraphrase verwendet
    wird, steht damit vor der Laengenanpassung fest, und die Kalibrierung
    entscheidet allein darueber, an welchen Stellen ueberhaupt eine Paraphrase
    eingefuegt wird.
    """
    zeilen = con.execute(
        "SELECT template_id, text FROM paraphrase WHERE quelle_id = ? "
        "AND verwendbar = 1 ORDER BY template_id", (quelle_id,)).fetchall()
    if not zeilen:
        return None
    try:
        schluessel = int(quelle_id)
    except (TypeError, ValueError):
        schluessel = sum(ord(c) for c in str(quelle_id))
    return zeilen[schluessel % len(zeilen)]["text"]


# ---------------------------------------------------------------------------
# Kalibrierung
# ---------------------------------------------------------------------------

def kalibrierung(n_modell_token, n_modell_token_basis):
    """L_red / L_basis nach eq:kalibrierung, in Modelltoken.

    L_red ist die Zahl der Token, um die die Variante den Basisprompt
    verlaengert. Das ist zugleich der Tokenaufwand der Einfuegung und damit die
    Groesse, auf die sich die Aussage der Arbeit bezieht.
    """
    if not n_modell_token_basis or n_modell_token is None:
        return None
    return (n_modell_token - n_modell_token_basis) / n_modell_token_basis


def im_band(quote):
    if quote is None:
        return None
    return int(abs(quote - KALIBRIERUNG_ZIEL) <= KALIBRIERUNG_BAND)


# ---------------------------------------------------------------------------
# Schreiben
# ---------------------------------------------------------------------------

def schreibe_prompt(con, frage_id, variante, text, segmente, lauf_id,
                    antworten, basis=None, tokenizer=None, studie="A",
                    mit_wortebene=True, neu=False, bau=None):
    """Legt Prompt, Segmente und Wortebene an. Rueckgabe: (prompt_id, geaendert).

    basis ist ein dict mit n_ws_token und n_modell_token des Basisprompts
    derselben Aufgabe, oder None fuer den Basisprompt selbst.

    Ist bereits ein Prompt mit demselben Text hinterlegt, bleibt er unberuehrt
    und der Schritt meldet "unveraendert". Das ist nicht nur schneller: das
    Loeschen eines Prompts nimmt ueber die Fremdschluessel auch seine
    Kompressate mit, und ein wiederholter Aufruf von 02 oder 03 wuerde sonst
    stundenlange Kompressionslaeufe vernichten. Mit neu=True wird der Prompt
    ausdruecklich verworfen und mit ihm alles, was darauf aufbaut.
    """
    if tokenizer is None:
        raise ValueError(
            "Kein Tokenizer uebergeben. eq:kalibrierung und eq:istrate sind in "
            "Modelltoken definiert; ohne Tokenizer waere die Kalibrierung nicht "
            "pruefbar. Aufruf mit --tokenizer <name>.")

    sha = dbio.sha256_text(text)
    vorhanden = dbio.eine_zeile(
        con, "SELECT prompt_id, sha256 FROM prompt WHERE frage_id = ? "
             "AND variante = ? AND studie = ?", (frage_id, variante, studie))
    if vorhanden is not None and vorhanden["sha256"] == sha and not neu:
        return vorhanden["prompt_id"], False

    n_ws = dbio.ws_token(text)
    n_mt = dbio.modell_token(text, tokenizer)
    kern_ws, red_ws = pb.ws_token_nach_herkunft(text, segmente)
    kern_mt = sum(dbio.modell_token(pb.segmenttext(text, s), tokenizer)
                  for s in segmente if s["herkunft"] == pb.KERN)
    red_mt = sum(dbio.modell_token(pb.segmenttext(text, s), tokenizer)
                 for s in segmente if s["herkunft"] == pb.RED)

    basis_ws = basis["n_ws_token"] if basis else n_ws
    basis_mt = basis["n_modell_token"] if basis else n_mt

    kal_ws = ((n_ws - basis_ws) / basis_ws) if basis_ws else None
    kal_mt = kalibrierung(n_mt, basis_mt)
    band = None if variante == "basis" else im_band(kal_mt)
    abweichung = (None if (variante == "basis" or kal_mt is None)
                  else abs(kal_mt - KALIBRIERUNG_ZIEL))

    leck, leck_abschnitt = pb.antwortleckage(text, segmente, antworten)
    einheiten = pb.bau_einheiten(variante, bau)

    con.execute("DELETE FROM prompt WHERE frage_id = ? AND variante = ? "
                "AND studie = ?", (frage_id, variante, studie))
    cur = con.execute(
        "INSERT INTO prompt (frage_id, variante, studie, text, sha256, "
        "n_zeichen, n_ws_token, n_modell_token, n_ws_token_kern, "
        "n_ws_token_red, n_modell_token_kern, n_modell_token_red, n_demos, "
        "n_ws_token_basis, n_modell_token_basis, kalibrierung_ws, "
        "kalibrierung_token, kalibrierung_im_band, kalibrierung_abweichung, "
        "bau_parameter, n_einheiten, lex_overlap_red_kern, "
        "antwortleckage, leckage_abschnitt, erzeugt, lauf_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (frage_id, variante, studie, text, sha,
         len(text), n_ws, n_mt, kern_ws, red_ws, kern_mt, red_mt,
         sum(1 for s in segmente if s["label"].endswith("_antwort")),
         basis_ws, basis_mt, kal_ws, kal_mt, band, abweichung,
         json.dumps(bau, sort_keys=True) if bau else None, len(einheiten),
         pb.lexikalische_ueberlappung(text, segmente),
         leck, leck_abschnitt, dbio.jetzt(), lauf_id))
    prompt_id = cur.lastrowid

    segment_ids = _schreibe_segmente(con, prompt_id, text, segmente)
    if mit_wortebene:
        _schreibe_woerter(con, prompt_id, text, segmente, segment_ids)
    return prompt_id, True


def _schreibe_segmente(con, prompt_id, text, segmente):
    ids = []
    for s in segmente:
        stueck = pb.segmenttext(text, s)
        cur = con.execute(
            "INSERT INTO segment (prompt_id, ord, label, herkunft, abschnitt, "
            "slot, start_zeichen, ende_zeichen, n_ws_token) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (prompt_id, s["ord"], s["label"], s["herkunft"], s["abschnitt"],
             s["slot"], s["start_zeichen"], s["ende_zeichen"],
             len(stueck.split())))
        ids.append(cur.lastrowid)
    return ids


def _schreibe_woerter(con, prompt_id, text, segmente, segment_ids):
    """Tokenfolge des Gesamtprompts mit Herkunft und Abschnitt.

    Die Zuordnung erfolgt ueber die Zeichenposition des Tokens und ist damit
    exakt. Ein spaeterer Abgleich mit dem Kompressat muss nur noch klaeren,
    welche dieser Positionen erhalten sind.
    """
    grenzen = [(s["start_zeichen"], s["ende_zeichen"], i)
               for i, s in enumerate(segmente)]
    zeilen = []
    treffer = list(dbio.TOKEN_RE.finditer(text))
    n = len(treffer)
    j = 0
    for pos, m in enumerate(treffer):
        while j < len(grenzen) - 1 and m.start() >= grenzen[j][1]:
            j += 1
        idx = grenzen[j][2] if grenzen[j][0] <= m.start() < grenzen[j][1] else None
        seg = segmente[idx] if idx is not None else None
        form = m.group(0)
        zeilen.append((
            prompt_id, pos, (pos / (n - 1)) if n > 1 else 0.0,
            form, form.casefold() if dbio.CASEFOLD else form,
            1 if dbio.WORT_RE.search(form) else 0,
            1 if dbio.ZIFFER_RE.search(form) else 0,
            seg["herkunft"] if seg else pb.KERN,
            seg["abschnitt"] if seg else pb.A_TESTFRAGE,
            segment_ids[idx] if idx is not None else None,
        ))
    con.executemany(
        "INSERT INTO wort (prompt_id, pos, rel_pos, form, schluessel, "
        "ist_wort, ist_ziffer, herkunft, abschnitt, segment_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", zeilen)


def basis_laengen(con, frage_id, studie="A"):
    z = dbio.eine_zeile(
        con, "SELECT n_ws_token, n_modell_token FROM prompt WHERE frage_id = ? "
             "AND variante = 'basis' AND studie = ?", (frage_id, studie))
    if z is None:
        return None
    return {"n_ws_token": z["n_ws_token"], "n_modell_token": z["n_modell_token"]}
