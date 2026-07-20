"""Real end-to-end pipeline run on the local PopQA data.

Builds raw prompts from ``datasets/popQA/test.tsv``, injects the three redundancy
types with the real ``redundanzgenerator`` (plus a redundancy-free baseline),
compresses each variant at four rates, scores it, and writes the full stage tree
plus ``manifest.{jsonl,csv}`` under ``experiments/<experiment-id>/``.

Prerequisites (both editable, from the repo root)::

    pip install -e tools/PipelineStore
    pip install -e tools/Redundanzgenerator

Example::

    python experiments/run_pipeline.py --experiment run-2026-07 \
        --query-ids 4222362 4725190 4382392 --rates 0.2 0.4 0.6 0.8

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
DEFAULT_POPQA = REPO_ROOT / "datasets" / "popQA" / "test.tsv"
DEFAULT_POPQA_TP = REPO_ROOT / "datasets" / "popQA" / "popQA_template_paraphrases.csv"

# Three ids present in both PopQA and the paraphrase table (category: occupation).
DEFAULT_QUERY_IDS = ["4222362", "4725190", "4382392"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the redundancy/compression pipeline.")
    parser.add_argument("--experiment", default="popqa-occupation-demo", help="experiment id")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent),
                        help="directory that holds experiments (default: experiments/)")
    parser.add_argument("--popqa", default=str(DEFAULT_POPQA))
    parser.add_argument("--popqa-tp", default=str(DEFAULT_POPQA_TP))
    parser.add_argument("--query-ids", nargs="+", default=DEFAULT_QUERY_IDS)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--n-demos", type=int, default=4)
    parser.add_argument("--n-lexical", type=int, default=3)
    parser.add_argument("--n-demonstrations", type=int, default=3)
    parser.add_argument("--n-instructions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    store = PipelineStore(args.root, args.experiment)
    counts = RedundancyCounts(
        lexical=args.n_lexical,
        demonstrations=args.n_demonstrations,
        instructions=args.n_instructions,
    )
    _, csv_path = run_pipeline(
        store,
        popqa_source=args.popqa,
        popqa_tp_source=args.popqa_tp,
        query_ids=args.query_ids,
        compression_rates=args.rates,
        counts=counts,
        n_demos=args.n_demos,
        seed=args.seed,
    )
    n_compressates = len(args.query_ids) * 4 * len(args.rates)
    print(f"{len(args.query_ids)} raw prompts x 4 variants x {len(args.rates)} rates "
          f"= {n_compressates} compressates")
    print(f"manifest -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
