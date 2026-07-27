#!/usr/bin/env python3
"""Fetch the MuSiQue data into this directory.

The dataset is not committed to the repository (the official archive is a few
hundred MB), so it is downloaded on demand. Two routes are supported:

``--via gdown`` (default)
    The official release archive from the authors' Google Drive, exactly as
    ``download_data.sh`` in StonyBrookNLP/musique does. Authoritative, and the
    only route that also yields ``dev_test_singlehop_questions_v1.0.json``
    (the leakage-avoidance id list).

``--via hf``
    A Hugging Face mirror, written back out as official-format JSONL. Use this
    when Google Drive is unreachable; verify the mirror you point at.

Only the ``dev`` splits are kept by default: ``test`` ships without answers or
supporting labels, and ``train`` is not needed for a prompt-compression study.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view
DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
DEFAULT_HF_ID = "dgslibisey/MuSiQue"

DEV_FILES = ("musique_ans_v1.0_dev.jsonl", "musique_full_v1.0_dev.jsonl")
EXTRA_FILES = ("dev_test_singlehop_questions_v1.0.json",)


def _fetch_via_gdown(keep_all: bool) -> list[Path]:
    try:
        import gdown  # noqa: F401
    except ImportError:
        sys.exit("gdown is required for --via gdown; install it with: pip install gdown")

    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "musique_v1.0.zip"
        subprocess.run(
            [sys.executable, "-m", "gdown", "--id", DRIVE_ID, "--output", str(archive)],
            check=True,
        )
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)

        wanted = None if keep_all else set(DEV_FILES) | set(EXTRA_FILES)
        for path in sorted(Path(tmp).rglob("*")):
            if not path.is_file() or path.suffix not in (".jsonl", ".json"):
                continue
            if wanted is not None and path.name not in wanted:
                continue
            target = HERE / path.name
            shutil.copyfile(path, target)
            written.append(target)
    return written


def _fetch_via_hf(dataset_id: str) -> list[Path]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "The 'datasets' package is required for --via hf; "
            "install it with: pip install datasets"
        )

    written: list[Path] = []
    for split, filename in (("validation", "musique_ans_v1.0_dev.jsonl"),):
        rows = load_dataset(dataset_id, split=split)
        target = HERE / filename
        with open(target, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--via", choices=("gdown", "hf"), default="gdown")
    parser.add_argument(
        "--hf-id",
        default=DEFAULT_HF_ID,
        help=f"Hugging Face dataset id for --via hf (default: {DEFAULT_HF_ID})",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="keep the train/test splits too (gdown route only; they are gitignored)",
    )
    args = parser.parse_args()

    written = (
        _fetch_via_gdown(args.keep_all)
        if args.via == "gdown"
        else _fetch_via_hf(args.hf_id)
    )
    if not written:
        sys.exit("No files were written — check the download output above.")
    for path in written:
        print(f"{path.relative_to(HERE.parent.parent)}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
