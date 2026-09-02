#!/usr/bin/env python3
"""
Gemeinsame Datenbankschicht der Promptanalyse.

Kapselt Verbindung, Schemaanlage, Laufprotokoll und die Tokenisierung, damit
alle Pipelineschritte dieselben Festlegungen verwenden. Keine Verarbeitung,
nur Infrastruktur.
"""

import hashlib
import json
import platform
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PFAD = BASE_DIR / "promptanalyse.db"
SCHEMA_PFAD = BASE_DIR / "schema.sql"

ENCODING = "utf-8"

# Kanonische Variantennamen. Reihenfolge ist die Rangfolge aus H2.
VARIANTEN = ("basis", "filler", "semantic", "instruction", "demonstration")

# Reduktionsstufen nach eq:stufen.
STUFEN = (0.0, 0.25, 0.50, 0.75)

KERN = "kern"
RED = "redundanz"

# Promptabschnitte fuer die Aufschluesselung der gestrichenen Woerter.
ABSCHNITTE = ("instruktion", "demonstration", "testfrage", "filler")

# Ein Token ist eine Wortkette oder ein einzelnes Satzzeichen. Apostrophe
# trennen, weil LLMLingua-2 das Genitiv-s haeufig einzeln entfernt.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORT_RE = re.compile(r"\w", re.UNICODE)
ZIFFER_RE = re.compile(r"\d", re.UNICODE)

CASEFOLD = True

# Geschuetzte Zeichen der Kompressorkonfiguration (subsubsec:konfiguration).
# Hier und nicht in 04_compress.py, weil 05_selectivity.py dieselbe Liste
# braucht: der Selektivitaetsindex wird ueber die freien Woerter gerechnet,
# also ueber die, bei denen das Verfahren ueberhaupt eine Wahl hatte.
FORCE_TOKENS = ["\n", "Q", "A", ".", ",", "?", "!", ":", ";", "-", "(", ")",
                "%", "\u20ac"]
RESERVE_DIGITS = True

# Wortformen, die durch den Schutz der Streichung entzogen sind. Satzzeichen
# stehen zwar ebenfalls in FORCE_TOKENS, zaehlen aber ohnehin nicht als Wort.
SCHUTZ_FORMEN = frozenset(t for t in FORCE_TOKENS if WORT_RE.search(t))


def ist_geschuetzt(form, ist_ziffer):
    """Konnte das Verfahren ueber dieses Wort ueberhaupt entscheiden?"""
    return form in SCHUTZ_FORMEN or (RESERVE_DIGITS and ist_ziffer)


# ---------------------------------------------------------------------------
# Verbindung
# ---------------------------------------------------------------------------

def verbinde(pfad=None, readonly=False):
    """Verbindung mit Fremdschluesselpruefung und Zeilen als sqlite3.Row."""
    pfad = Path(pfad) if pfad else DB_PFAD
    if readonly:
        con = sqlite3.connect("file:%s?mode=ro" % pfad, uri=True)
    else:
        con = sqlite3.connect(pfad)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    # TRUNCATE statt des Vorgabemodus DELETE: der Journaldateiname bleibt
    # bestehen, statt bei jeder Transaktion geloescht zu werden. Das ist auf
    # Netzlaufwerken und in synchronisierten Ordnern wie OneDrive robuster.
    # WAL scheidet aus demselben Grund aus, es legt zwei Nebendateien an.
    if not readonly:
        con.execute("PRAGMA journal_mode = TRUNCATE")
        con.execute("PRAGMA synchronous = FULL")
    _mathe_ergaenzen(con)
    return con


def _mathe_ergaenzen(con):
    """exp, ln und sqrt nachruesten, falls SQLite ohne Mathefunktionen gebaut ist."""
    import math
    for name, n, fn in (("exp", 1, math.exp), ("ln", 1, math.log),
                        ("log", 1, math.log10), ("sqrt", 1, math.sqrt),
                        ("power", 2, lambda a, b: a ** b)):
        try:
            con.execute("SELECT %s(%s)" % (name, ",".join(["1"] * n)))
        except sqlite3.OperationalError:
            con.create_function(name, n, _null_sicher(fn))


def _null_sicher(fn):
    def wrapper(*args):
        if any(a is None for a in args):
            return None
        try:
            return fn(*args)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    return wrapper


def schema_anlegen(con, schema_pfad=None):
    """Schema idempotent anlegen. Views werden dabei stets neu gesetzt."""
    pfad = Path(schema_pfad) if schema_pfad else SCHEMA_PFAD
    con.executescript(pfad.read_text(encoding=ENCODING))
    con.commit()


# ---------------------------------------------------------------------------
# Laufprotokoll
# ---------------------------------------------------------------------------

def lauf_beginnen(con, schritt, skript, konfiguration=None, bemerkung=None):
    """Legt eine Zeile in lauf an und liefert die lauf_id."""
    skript_pfad = Path(skript)
    sha = (sha256_datei(skript_pfad) if skript_pfad.is_file() else None)
    cur = con.execute(
        "INSERT INTO lauf (schritt, gestartet, skript, skript_sha256, "
        "konfiguration, git_commit, python_version, plattform, bemerkung) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (schritt, jetzt(), skript_pfad.name, sha,
         json.dumps(konfiguration, ensure_ascii=False, default=str)
         if konfiguration is not None else None,
         git_commit(skript_pfad.parent), sys.version.split()[0],
         platform.platform(), bemerkung))
    con.commit()
    return cur.lastrowid


def lauf_beenden(con, lauf_id, n_zeilen=None, bemerkung=None):
    con.execute("UPDATE lauf SET beendet = ?, n_zeilen = ?, "
                "bemerkung = COALESCE(?, bemerkung) WHERE lauf_id = ?",
                (jetzt(), n_zeilen, bemerkung, lauf_id))
    con.commit()


def jetzt():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(cwd):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_text(text):
    return hashlib.sha256(text.encode(ENCODING)).hexdigest()


def sha256_datei(pfad):
    h = hashlib.sha256()
    with open(pfad, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tokenisierung
# ---------------------------------------------------------------------------

def tokenisiere(text):
    """Zerlegt Text in Token. Rueckgabe: (formen, schluessel, ist_wort)."""
    formen = TOKEN_RE.findall(text)
    schluessel = [t.casefold() if CASEFOLD else t for t in formen]
    ist_wort = [bool(WORT_RE.search(t)) for t in formen]
    return formen, schluessel, ist_wort


def ws_token(text):
    """Whitespace-Woerter. Diese Zahl steuert den Ratenparameter von LLMLingua-2."""
    return len(text.split())


_tokenizer = None
_tokenizer_name = None


# Ollama als Tokenizerquelle.
#
# eq:kalibrierung und eq:istrate sind in Token des Zielmodells definiert. Der
# offizielle Tokenizer von Llama-3.1 liegt hinter einer Zugangsbeschraenkung.
# Laeuft das Zielmodell ohnehin unter Ollama, ist dessen Tokenizer derselbe,
# und die Zahl der Eingabetoken laesst sich ueber prompt_eval_count auslesen.
# Das vermeidet eine zweite Bezugsquelle und damit die Moeglichkeit, dass
# Kalibrierung und Inferenz auf verschiedenen Tokenisierungen beruhen.

OLLAMA_URL = "http://localhost:11434"


class OllamaTokenizer:
    """Zaehlt Token ueber prompt_eval_count des Ollama-Servers.

    llama.cpp stellt dem Prompt ein Satzanfangstoken voran. Der Aufschlag wird
    einmal an der leeren Zeichenkette gemessen und abgezogen, damit die Zahl
    die Token des Textes selbst angibt. Ergebnisse werden zwischengespeichert,
    da die Kalibrierung dieselben Bausteine vielfach vermisst.
    """

    def __init__(self, modell, url=None, cache_max=200000):
        self.modell = modell
        self.url = (url or OLLAMA_URL).rstrip("/")
        self._cache = {}
        self._cache_max = cache_max
        self.schnell = self._pruefe_tokenize()
        self.offset = 0 if self.schnell else self._roh("")

    def _pruefe_tokenize(self):
        """Manche Ollama-Fassungen bieten /api/tokenize an. Das ist um
        Groessenordnungen schneller als ein Prompt-Durchlauf und wird bevorzugt.
        """
        try:
            return isinstance(self._tokenize("Test"), int)
        except Exception:                      # noqa: BLE001
            return False

    def _tokenize(self, text):
        import json as _json
        import urllib.request
        daten = _json.dumps({"model": self.modell, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/api/tokenize", data=daten,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as fh:
            return len(_json.loads(fh.read())["tokens"])

    def _roh(self, text):
        import json as _json
        import urllib.request
        daten = _json.dumps({
            "model": self.modell, "prompt": text, "raw": True,
            "stream": False, "options": {"num_predict": 1, "temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/api/generate", data=daten,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            return _json.loads(fh.read())["prompt_eval_count"]

    def encode(self, text, add_special_tokens=False):
        if text in self._cache:
            n = self._cache[text]
        else:
            n = (self._tokenize(text) if self.schnell
                 else max(0, self._roh(text) - self.offset))
            if len(self._cache) < self._cache_max:
                self._cache[text] = n
        return list(range(n))


def modell_tokenizer(name=None):
    """Tokenizer des Zielmodells. Liefert None, wenn keiner ladbar ist.

    Die Kalibrierung nach eq:kalibrierung und die Ratenermittlung nach
    eq:istrate sind in Token des Zielmodells definiert. Der Tokenizer ist
    deshalb keine Kuer, sondern Voraussetzung; 02 und 03 brechen ohne ihn ab.

    Gesucht wird in dieser Reihenfolge:
      0. "ollama:<modell>" fragt den lokalen Ollama-Server,
      1. ein lokaler Pfad (Ordner mit tokenizer.json oder die Datei selbst),
      2. transformers.AutoTokenizer,
      3. tokenizers.Tokenizer als schlanke Rueckfallebene.
    Die tatsaechlich verwendete Quelle wird in lauf.konfiguration protokolliert.
    """
    global _tokenizer, _tokenizer_name
    name = name or "meta-llama/Llama-3.1-8B-Instruct"
    if _tokenizer is not None and _tokenizer_name == name:
        return _tokenizer

    if name.startswith("ollama:"):
        try:
            _tokenizer = OllamaTokenizer(name.split(":", 1)[1])
            _tokenizer_name = name
            return _tokenizer
        except Exception as e:                 # noqa: BLE001
            print("Ollama-Tokenizer nicht erreichbar (%s). Laeuft der Server "
                  "und ist das Modell geladen?" % e, file=sys.stderr)
            return None

    pfad = Path(name)
    if pfad.exists():
        try:
            from tokenizers import Tokenizer
            datei = pfad / "tokenizer.json" if pfad.is_dir() else pfad
            _tokenizer = _Huelle(Tokenizer.from_file(str(datei)))
            _tokenizer_name = name
            return _tokenizer
        except Exception as e:                 # noqa: BLE001
            print("Hinweis: lokaler Tokenizer %s nicht ladbar (%s)."
                  % (name, e), file=sys.stderr)

    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _tokenizer_name = name
        return _tokenizer
    except Exception as e:                     # noqa: BLE001
        erste = e

    try:
        from tokenizers import Tokenizer
        _tokenizer = _Huelle(Tokenizer.from_pretrained(name))
        _tokenizer_name = name
        return _tokenizer
    except Exception as e:                     # noqa: BLE001
        print("Tokenizer %s nicht ladbar.\n  transformers: %s\n  tokenizers:  %s"
              % (name, erste, e), file=sys.stderr)
        _tokenizer = None
        _tokenizer_name = None
    return _tokenizer


class _Huelle:
    """Gibt einem tokenizers.Tokenizer die encode-Schnittstelle von transformers."""

    def __init__(self, tok):
        self._tok = tok

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids


def modell_token(text, tokenizer=None):
    """Tokenzahl im Tokenizer des Zielmodells. None, wenn keiner uebergeben ist.

    Bewusst kein selbsttaetiges Nachladen: die Kalibrierung soll erkennbar
    entweder in Modelltoken oder gar nicht gemessen werden.
    """
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Einbettung von Teilfolgen
# ---------------------------------------------------------------------------

def einbetten_links(klein, gross):
    """Frueheste Einbettung von klein als Teilfolge von gross.

    Prinzip der fruehesten noch nicht belegten Position aus
    subsubsec:auszeichnung. Liefert die Positionen in gross oder None.
    """
    stellen = []
    j = 0
    n = len(gross)
    for t in klein:
        while j < n and gross[j] != t:
            j += 1
        if j == n:
            return None
        stellen.append(j)
        j += 1
    return stellen


def einbetten_rechts(klein, gross):
    """Spaeteste Einbettung, Gegenprobe zur Auszeichnung."""
    n = len(gross)
    stellen = einbetten_links(klein[::-1], gross[::-1])
    if stellen is None:
        return None
    return [n - 1 - i for i in reversed(stellen)]


def einbetten_block(klein, gross):
    """Rueckfall auf Blockausrichtung, falls keine Teilfolge vorliegt."""
    import difflib
    sm = difflib.SequenceMatcher(None, klein, gross, autojunk=False)
    stellen = []
    for i, j, n in sm.get_matching_blocks():
        stellen.extend(range(j, j + n))
    return stellen


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------

def eine_zeile(con, sql, params=()):
    cur = con.execute(sql, params)
    return cur.fetchone()


def skalar(con, sql, params=()):
    zeile = eine_zeile(con, sql, params)
    return None if zeile is None else zeile[0]


def tabelle_leeren(con, *namen):
    for n in namen:
        con.execute("DELETE FROM %s" % n)
    con.commit()
