"""Shared record loading: local CSV/JSON/JSONL files or Hugging Face hub."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_records(source: str | Path, split: str = "test") -> list[dict[str, Any]]:
    """Load rows from a local file or a Hugging Face dataset id.

    A ``source`` that exists on disk is parsed by extension (.csv, .tsv,
    .json, .jsonl/.ndjson). Anything else is treated as a Hugging Face hub id
    and loaded via the optional ``datasets`` dependency.
    """
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(path, encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f, delimiter=delimiter))
        if suffix in (".jsonl", ".ndjson"):
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a JSON array of records")
            return data
        raise ValueError(f"{path}: unsupported file format '{suffix}'")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"'{source}' is not a local file, and loading it from the "
            "Hugging Face hub requires the 'datasets' package. "
            "Install it with: pip install redundanzgenerator[hf]"
        ) from exc
    return [dict(row) for row in load_dataset(str(source), split=split)]
