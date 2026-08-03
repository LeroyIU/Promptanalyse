#!/usr/bin/env python3
"""Extract the 8k+ bucket of LongBench-E TriviaQA into a standalone JSONL.

LongBench-E re-samples each LongBench task into three roughly-balanced,
pre-annotated context-length buckets (0-4k / 4-8k / 8k+ words, ~100 samples
each) so that accuracy can be compared *within* a length bucket instead of
being confounded by it. For a few-shot QA prompt-compression study, the 8k+
bucket is the interesting regime: it is long enough that compression has
something to remove, and short enough to still fit common context windows.

Data source: the ``THUDM/LongBench`` dataset on the Hugging Face Hub ships
as a single ``data.zip`` plus a loading script (``LongBench.py``). Current
``datasets`` releases (>=4) no longer execute loading scripts at all (not
even with ``trust_remote_code=True`` - see the script's own docstring), so
this tool downloads ``data.zip`` via ``huggingface_hub`` and reads
``data/triviaqa_e.jsonl`` out of it directly, exactly as the loading script
would have.

Note on the config name: the LongBench-E task for TriviaQA is called
``triviaqa_e`` (no underscore in "triviaqa"), not ``trivia_qa_e``.

Each row already carries a ``length`` field (word count of the rendered
few-shot context) that LongBench used to build the three buckets; buckets
are cleanly separated at 4000/8000, so it is used directly as the bucket
key instead of re-tokenizing.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent

REPO_ID = "THUDM/LongBench"
CONFIG_NAME = "triviaqa_e"
SOURCE_MEMBER = f"data/{CONFIG_NAME}.jsonl"
LENGTH_THRESHOLD = 8000

OUTPUT_FIELDS = ("context", "input", "answers", "length", "all_classes", "_id")


def _load_source_rows() -> list[dict]:
    archive = hf_hub_download(REPO_ID, "data.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as zf:
        with zf.open(SOURCE_MEMBER) as f:
            return [json.loads(line) for line in f]


def _filter_8k_plus(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["length"] >= LENGTH_THRESHOLD]


def _validate(rows: list[dict]) -> None:
    ids = [row["_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate _id in 8k+ subset"
    empty_answers = [row["_id"] for row in rows if not row.get("answers")]
    assert not empty_answers, f"empty answers for: {empty_answers}"


def _print_stats(rows: list[dict]) -> None:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        token_counts = [len(enc.encode(row["context"])) for row in rows]
    except ImportError:
        print("(tiktoken not installed - skipping token-count stats)")
        token_counts = None

    word_counts = [len(row["context"].split()) for row in rows]

    import pandas as pd

    stats = {"length_field": pd.Series([row["length"] for row in rows]), "words": pd.Series(word_counts)}
    if token_counts is not None:
        stats["tokens"] = pd.Series(token_counts)
    df = pd.DataFrame(stats)

    print(f"\nSamples: {len(rows)}")
    print(df.agg(["min", "max", "mean"]).round(1).to_string())

    example = rows[0]
    print("\nExample sample (context truncated to 500 chars):")
    print(f"  _id: {example['_id']}")
    print(f"  input: {example['input']!r}")
    print(f"  answers: {example['answers']}")
    print(f"  length: {example['length']}")
    print(f"  context[:500]: {example['context'][:500]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "triviaqa_longbench_e_8k_plus.jsonl",
        help="output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()

    rows = _load_source_rows()
    subset = _filter_8k_plus(rows)
    _validate(subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in subset:
            record = {field: row[field] for field in OUTPUT_FIELDS}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(subset)} samples to {args.output}")
    _print_stats(subset)


if __name__ == "__main__":
    main()
