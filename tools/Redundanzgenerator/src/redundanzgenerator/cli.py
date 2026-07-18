"""Command line interface: redundanzgen generate | build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data.popqa import PopQALoader
from .generator import RedundancyGenerator
from .models import Demonstration, FewShotPrompt, RedundancyConfig
from .render import render_prompt


def _write_output(
    prompt: FewShotPrompt,
    report: list[dict[str, Any]],
    output: str | None,
    text_output: str | None,
) -> None:
    payload = {"prompt": prompt.to_dict(), "redundancy_report": report}
    rendered = render_prompt(prompt)
    if output:
        Path(output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if text_output:
        Path(text_output).write_text(rendered + "\n", encoding="utf-8")
    if not output and not text_output:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        print()


def _cmd_generate(args: argparse.Namespace) -> None:
    prompt = FewShotPrompt.from_json(args.input)
    config = RedundancyConfig(
        n_paraphrases=args.lexical,
        paraphrase_demonstrations=args.paraphrase_demos,
        n_demonstrations=args.demos,
        n_instructions=args.instructions,
        instruction_position=args.instruction_position,
        seed=args.seed,
    )
    generator = RedundancyGenerator.from_config(
        config,
        popqa=args.popqa,
        popqa_tp=args.popqa_tp,
        use_llm=args.llm,
        llm_model=args.llm_model,
    )
    result, report = generator.generate(prompt)
    _write_output(result, report, args.output, args.text)


def _cmd_build(args: argparse.Namespace) -> None:
    popqa = PopQALoader(args.popqa)
    if args.query_id is not None:
        query_row = popqa.by_id(args.query_id)
        if query_row is None:
            raise SystemExit(f"No PopQA record with id {args.query_id}")
    else:
        query_row = popqa.by_question(args.query)
        if query_row is None:
            raise SystemExit(f"No PopQA record with question {args.query!r}")

    import random

    rng = random.Random(args.seed)
    demo_rows = popqa.by_category(
        str(query_row.get("prop", "")), exclude_questions={str(query_row["question"])}
    )
    rng.shuffle(demo_rows)
    demos = [
        Demonstration(
            question=str(row["question"]),
            answer=popqa.answer_of(row),
            meta={"id": row.get("id"), "prop": row.get("prop")},
        )
        for row in demo_rows[: args.n_demos]
    ]
    prompt = FewShotPrompt(
        instructions=[args.instruction],
        demonstrations=demos,
        query=str(query_row["question"]),
        meta={"id": query_row.get("id"), "prop": query_row.get("prop")},
    )
    _write_output(prompt, [], args.output, args.text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="redundanzgen",
        description=(
            "Inject controlled redundancy (paraphrases, demonstrations, "
            "instructions) into few-shot prompts, based on PopQA/PopQA-TP."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Add redundancy to an existing prompt JSON")
    gen.add_argument("--input", required=True, help="Few-shot prompt JSON file")
    gen.add_argument(
        "--lexical", type=int, default=0, metavar="N",
        help="Paraphrases from PopQA-TP added to the query (default: 0)",
    )
    gen.add_argument(
        "--paraphrase-demos", action="store_true",
        help="Also paraphrase every demonstration question",
    )
    gen.add_argument(
        "--demos", type=int, default=0, metavar="N",
        help="Redundant same-category demonstrations from PopQA (default: 0)",
    )
    gen.add_argument(
        "--instructions", type=int, default=0, metavar="N",
        help="Redundant instruction restatements (default: 0)",
    )
    gen.add_argument(
        "--instruction-position", choices=("start", "end", "both"), default="start",
        help="Where redundant instructions are placed (default: start)",
    )
    gen.add_argument(
        "--popqa", help="PopQA source: local CSV/JSON file or HF dataset id"
    )
    gen.add_argument(
        "--popqa-tp", help="PopQA-TP source: local CSV/JSON file or HF dataset id"
    )
    gen.add_argument(
        "--llm", action="store_true",
        help="Rephrase instructions via the Anthropic API instead of templates",
    )
    gen.add_argument("--llm-model", default="claude-sonnet-5")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--output", help="Write result JSON here (default: stdout)")
    gen.add_argument("--text", help="Also write the rendered prompt text here")
    gen.set_defaults(func=_cmd_generate)

    build = sub.add_parser("build", help="Build a base few-shot prompt from PopQA")
    build.add_argument("--popqa", required=True)
    query_group = build.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-id", help="PopQA id of the query question")
    query_group.add_argument("--query", help="Exact question text of the query")
    build.add_argument("--n-demos", type=int, default=4)
    build.add_argument(
        "--instruction",
        default="Answer the following question with a short factual answer.",
    )
    build.add_argument("--seed", type=int, default=None)
    build.add_argument("--output")
    build.add_argument("--text")
    build.set_defaults(func=_cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
