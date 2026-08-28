#
#//  LLMLingua2.py
#/#/  
#//
#//  Created by Leroy Osagie on 27.08.26.
#//


#!/usr/bin/env python3
"""
LLMLingua-2 Batch-Kompressor.

    python lingua.py 0.5
    python lingua.py 0.8 0.5 0.33
    python lingua.py            # nutzt RATES aus der Konfiguration

Liest alle passenden Dateien aus INPUT_DIR, komprimiert sie mit jeder
angegebenen Rate und schreibt sie als <name>_<rate><endung> nach OUTPUT_DIR.
    protokoll.txt  ->  protokoll_0.8.txt, protokoll_0.5.txt, protokoll_0.33.txt

Das Modell wird einmal geladen und für alle Dateien und Raten wiederverwendet.
"""

import sys
from pathlib import Path

# ----------------------------------------------------------------------
# KONFIGURATION
# ----------------------------------------------------------------------

INPUT_DIR = Path("./tests/input")
OUTPUT_DIR = Path("./tests/output")

# Raten, falls keine auf der Kommandozeile übergeben werden
RATES = [0.8, 0.5, 0.33]

PATTERN = "*.txt"       # z.B. "*.md", "**/*.txt" für Unterordner
OVERWRITE = False       # False = vorhandene Ausgaben überspringen

#MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"   # größer, besser

DEVICE = "mps"          # "cpu" | "mps" (Apple Silicon) | "cuda"

# Tokens, die nie gestrichen werden
FORCE_TOKENS = ["\n", ".", ",", "?", "!", ":", ";", "-", "(", ")", "%", "€"]

RESERVE_DIGITS = True   # Zahlen schützen
DROP_CONSECUTIVE = True # doppelte Satzzeichen entfernen
CHUNK_END_TOKENS = [".", "\n"]

QUESTION = None         # z.B. "Wie hoch war der Umsatz?" -> fragebezogene Kompression

# ----------------------------------------------------------------------

_compressor = None


def get_compressor():
    """Lädt das Modell beim ersten Aufruf, danach aus dem Cache."""
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor

        print(f"Lade {MODEL} auf {DEVICE} ...", file=sys.stderr)
        _compressor = PromptCompressor(
            model_name=MODEL, use_llmlingua2=True, device_map=DEVICE
        )
    return _compressor


def compress(text: str, rate: float) -> dict:
    kwargs = {
        "rate": rate,
        "force_tokens": FORCE_TOKENS,
        "force_reserve_digit": RESERVE_DIGITS,
        "drop_consecutive": DROP_CONSECUTIVE,
        "chunk_end_tokens": CHUNK_END_TOKENS,
    }
    if QUESTION:
        kwargs["question"] = QUESTION
    return get_compressor().compress_prompt(text, **kwargs)


def output_path(src: Path, rate: float) -> Path:
    """protokoll.txt + 0.5 -> OUTPUT_DIR/protokoll_0.5.txt"""
    rel = src.relative_to(INPUT_DIR)
    name = f"{rel.stem}_{rate:g}{rel.suffix}"
    return OUTPUT_DIR / rel.parent / name


def parse_rates(args: list[str]) -> list[float]:
    """Kommandozeile schlägt Konfiguration; '0.8,0.5' und '0.8 0.5' gehen beide."""
    if not args:
        rates = list(RATES)
    else:
        rates = []
        for a in args:
            for part in a.replace(",", " ").split():
                try:
                    rates.append(float(part))
                except ValueError:
                    sys.exit(f"Keine gültige Rate: {part!r}")

    if not rates:
        sys.exit("Keine Raten angegeben und RATES ist leer.")

    for r in rates:
        if not 0 < r <= 1:
            sys.exit(f"Rate {r:g} liegt nicht zwischen 0 und 1.")

    # Duplikate raus, absteigend: erst schonend, dann aggressiv
    return sorted(set(rates), reverse=True)


def main():
    rates = parse_rates(sys.argv[1:])

    if not INPUT_DIR.is_dir():
        sys.exit(f"Eingabeordner fehlt: {INPUT_DIR.resolve()}")

    files = sorted(f for f in INPUT_DIR.glob(PATTERN) if f.is_file())
    if not files:
        sys.exit(f"Keine Dateien für '{PATTERN}' in {INPUT_DIR.resolve()}")

    print(
        f"{len(files)} Datei(en) x {len(rates)} Rate(n): "
        f"{', '.join(f'{r:g}' for r in rates)}",
        file=sys.stderr,
    )

    ok = skipped = failed = 0

    for src in files:
        text = src.read_text(encoding="utf-8")
        if not text.strip():
            print(f"skip   {src.name} (leer)", file=sys.stderr)
            skipped += len(rates)
            continue

        for rate in rates:
            dst = output_path(src, rate)

            if dst.exists() and not OVERWRITE:
                print(f"skip   {dst.name} (existiert)", file=sys.stderr)
                skipped += 1
                continue

            try:
                r = compress(text, rate)
            except Exception as e:
                print(f"FEHLER {src.name} @ {rate:g}: {e}", file=sys.stderr)
                failed += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(r["compressed_prompt"], encoding="utf-8")

            ist = r["compressed_tokens"] / r["origin_tokens"]
            print(
                f"ok     {src.name} -> {dst.name}   "
                f"{r['origin_tokens']} -> {r['compressed_tokens']} Tokens "
                f"(ist {ist:.2f})",
                file=sys.stderr,
            )
            ok += 1

    print(
        f"\nFertig: {ok} komprimiert, {skipped} übersprungen, {failed} fehlgeschlagen",
        file=sys.stderr,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()