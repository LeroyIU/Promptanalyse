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
    """Zaehlt Token ueber den Ollama-Server. Nur Rueckfallebene.

    Bevorzugt wird /api/tokenize. Fehlt der Endpunkt, bleibt nur
    prompt_eval_count eines Durchlaufs, und das ist heikel: llama.cpp haelt
    den zuletzt ausgewerteten Prompt vor und wertet bei einem Prompt mit
    gleichem Anfang nur noch den neuen Teil aus. prompt_eval_count zaehlt dann
    nicht die Token des Textes, sondern die neu ausgewerteten. Da sich die
    Prompts dieser Arbeit einen langen gemeinsamen Anfang teilen, waere die
    Kalibrierung damit still falsch. Der Selbsttest unten erkennt den Fall und
    verweigert den Dienst, statt unbrauchbare Zahlen zu liefern.

    Empfohlen ist deshalb der Tokenizer des Modells ueber transformers, siehe
    modell_tokenizer.
    """

    # Zwei Texte mit sauberer Wortgrenze. Ihre Token addieren sich, was den
    # konstanten Aufschlag des Satzanfangstokens messbar macht.
    PROBE_A = "The harbour warden checked the printed tide table."
    PROBE_B = " Small boats timed their departure by the morning water."

    def __init__(self, modell, url=None, cache_max=200000):
        self.modell = modell
        self.url = (url or OLLAMA_URL).rstrip("/")
        self._cache = {}
        self._cache_max = cache_max
        self.schnell = self._pruefe_tokenize()
        self.offset = 0 if self.schnell else self._miss_offset()
        self._pruefe_zwischenspeicher()

    # -- Endpunkte ----------------------------------------------------------

    def _post(self, pfad, nutzlast):
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            self.url + pfad, data=_json.dumps(nutzlast).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            return _json.loads(fh.read())

    def _pruefe_tokenize(self):
        try:
            return isinstance(self._tokenize(self.PROBE_A), int)
        except Exception:                      # noqa: BLE001
            return False

    def _tokenize(self, text):
        return len(self._post("/api/tokenize",
                              {"model": self.modell, "text": text})["tokens"])

    def _roh(self, text):
        erg = self._post("/api/generate", {
            "model": self.modell, "prompt": text, "raw": True, "stream": False,
            "options": {"num_predict": 1, "temperature": 0}})
        n = erg.get("prompt_eval_count")
        if n is None:
            raise RuntimeError(
                "Ollama liefert kein prompt_eval_count fuer diesen Prompt. "
                "Ohne diese Zahl laesst sich nicht tokenisieren.")
        return n

    # -- Kalibrierung des Zaehlers ------------------------------------------

    def _miss_offset(self):
        """Konstanter Aufschlag je Aufruf, etwa das Satzanfangstoken.

        Gemessen ueber zwei Texte, deren Token sich addieren:
        n(A) + n(B) - n(AB) ist der Aufschlag. Der leere Prompt scheidet als
        Messpunkt aus, weil Ollama fuer ihn gar nichts auswertet und kein
        prompt_eval_count zurueckgibt.
        """
        a = self._roh(self.PROBE_A)
        b = self._roh(self.PROBE_B)
        ab = self._roh(self.PROBE_A + self.PROBE_B)
        offset = a + b - ab
        if not 0 <= offset <= 4:
            raise RuntimeError(
                "Der Tokenzaehler von Ollama verhaelt sich nicht additiv "
                "(gemessener Aufschlag %d). Bitte den Tokenizer des Modells "
                "ueber transformers verwenden." % offset)
        return offset

    def _pruefe_zwischenspeicher(self):
        """Wird derselbe Prompt zweimal gleich gezaehlt?

        Wertet llama.cpp einen bereits gesehenen Anfang nicht erneut aus, faellt
        die zweite Zahl kleiner aus. Dann ist prompt_eval_count als Tokenzaehler
        unbrauchbar, und zwar auf eine Weise, die sonst niemandem auffiele.
        """
        if self.schnell:
            return
        text = self.PROBE_A + self.PROBE_B
        erste, zweite = self._roh(text), self._roh(text)
        if erste != zweite:
            raise RuntimeError(
                "Ollama zaehlt denselben Prompt zweimal verschieden (%d und "
                "%d). Der Server wertet zwischengespeicherte Promptanfaenge "
                "nicht erneut aus, sodass prompt_eval_count nicht die Token "
                "des Textes angibt. Bitte den Tokenizer des Modells ueber "
                "transformers verwenden, etwa\n"
                "  --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct"
                % (erste, zweite))

    # -- Schnittstelle ------------------------------------------------------

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
            print("Ollama-Tokenizer nicht verwendbar:\n  %s" % e,
                  file=sys.stderr)
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
        _melde(name, _tokenizer)
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


def _melde(name, tok):
    """Quelle und Vokabulargroesse ausgeben.

    Llama-3.1 hat 128256 Token. Weicht die Zahl ab, ist ein anderes Modell
    geladen als angenommen, und die Kalibrierung bezoege sich auf eine andere
    Tokenisierung als die Inferenz. Das faellt sonst nirgends auf.
    """
    groesse = None
    for zugriff in (lambda: len(tok), lambda: tok.vocab_size,
                    lambda: tok.get_vocab_size()):
        try:
            groesse = zugriff()
            break
        except Exception:                      # noqa: BLE001
            continue
    print("Tokenizer: %s (Vokabular %s)" % (name, groesse), file=sys.stderr)
    if groesse is not None and groesse != 128256:
        print("  Achtung: erwartet waren 128256 Token fuer Llama-3.1.",
              file=sys.stderr)


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
