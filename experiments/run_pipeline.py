"""Real end-to-end pipeline run on MuSiQue.

Builds context prompts from the MuSiQue dev set, injects the three redundancy
types with the real ``redundanzgenerator`` (plus a redundancy-free baseline),
compresses each variant at four rates, scores it, and writes the full stage tree
plus ``manifest.{jsonl,csv}`` under ``experiments/<experiment-id>/``.

Prerequisites (both editable, from the repo root)::

    pip install -e tools/PipelineStore
    pip install -e tools/Redundanzgenerator
    python datasets/musique/fetch_musique.py

Example::

    python experiments/run_pipeline.py --experiment run-2026-07 \
        --n-prompts 20 --rates 0.2 0.4 0.6 0.8

Without ``--query-ids`` the first ``--n-prompts`` records of the dataset are
used, so a run needs no prior knowledge of the ids.

Compression and inference use the offline baselines from
``pipelinestore.redundancy_pipeline`` (deterministic truncation +
gold-containment scoring). Swap in a real compressor / the Claude API by calling
``run_pipeline(..., compress=..., infer=...)`` from your own script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelinestore import PipelineStore
from pipelinestore.redundancy_pipeline import RedundancyCounts, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MUSIQUE = REPO_ROOT / "datasets" / "musique" / "musique_ans_v1.0_dev.jsonl"


def _first_ids(source: str, n: int) -> list[str]:
    from redundanzgenerator import MuSiQueLoader

    return [str(row["id"]) for row in MuSiQueLoader(source).records[:n]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the redundancy/compression pipeline.")
    parser.add_argument("--experiment", default="musique-demo", help="experiment id")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent),
                        help="directory that holds experiments (default: experiments/)")
    parser.add_argument("--musique", default=str(DEFAULT_MUSIQUE))
    parser.add_argument("--query-ids", nargs="+", default=None,
                        help="MuSiQue ids (default: the first --n-prompts records)")
    parser.add_argument("--n-prompts", type=int, default=3,
                        help="how many records to use when --query-ids is omitted")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--n-demos", type=int, default=0,
                        help="demonstrations in the base prompt (default: 0, zero-shot)")
    parser.add_argument("--no-distractors", action="store_true",
                        help="build prompts from the supporting passages only")
    parser.add_argument("--n-passages", type=int, default=3)
    parser.add_argument("--passage-mode", choices=("duplicate", "restate"),
                        default="duplicate")
    parser.add_argument("--n-demonstrations", type=int, default=3)
    parser.add_argument("--n-instructions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--description",
                        default="Redundanz-/Kompressions-Pipeline auf MuSiQue.",
                        help="goes into experiment.json")
    args = parser.parse_args(argv)

    if not Path(args.musique).exists():
        parser.error(
            f"MuSiQue data not found at {args.musique}. "
            "Fetch it with: python datasets/musique/fetch_musique.py"
        )
    query_ids = args.query_ids or _first_ids(args.musique, args.n_prompts)

    store = PipelineStore(args.root, args.experiment)
    counts = RedundancyCounts(
        passages=args.n_passages,
        passage_mode=args.passage_mode,
        demonstrations=args.n_demonstrations,
        instructions=args.n_instructions,
    )
    _, csv_path = run_pipeline(
        store,
        musique_source=args.musique,
        query_ids=query_ids,
        compression_rates=args.rates,
        counts=counts,
        n_demos=args.n_demos,
        include_distractors=not args.no_distractors,
        seed=args.seed,
        description=args.description,
    )
    n_compressates = len(query_ids) * 4 * len(args.rates)
    print(f"{len(query_ids)} raw prompts x 4 variants x {len(args.rates)} rates "
          f"= {n_compressates} compressates")
    print(f"manifest -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
