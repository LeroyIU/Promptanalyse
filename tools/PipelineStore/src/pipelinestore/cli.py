"""Small CLI: (re)build the tidy manifest for an experiment from its tree.

    pipelinestore build-manifest --root experiments --experiment demo-experiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .store import PipelineStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipelinestore")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-manifest", help="rebuild manifest.jsonl and manifest.csv from the tree"
    )
    build.add_argument("--root", default="experiments", help="directory holding experiments")
    build.add_argument("--experiment", required=True, help="experiment id (subdirectory name)")

    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        store = PipelineStore(Path(args.root), args.experiment)
        jsonl_path, csv_path = store.build_manifest()
        rows = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
        print(f"wrote {rows} rows -> {csv_path}")
        print(f"           -> {jsonl_path}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
