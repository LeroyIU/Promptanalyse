"""Command line interface: redundanzgen generate | build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data.musique import DEFAULT_INSTRUCTION, MuSiQueLoader
from .generator import RedundancyGenerator
from .models import FewShotPrompt, RedundancyConfig
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
        n_passages=args.passages,
        passage_mode=args.passage_mode,
        passage_target=args.passage_target,
        passage_position=args.passage_position,
        n_demonstrations=args.demos,
        demonstration_context=args.demo_context,
        n_instructions=args.instructions,
        instruction_position=args.instruction_position,
        seed=args.seed,
    )
    generator = RedundancyGenerator.from_config(
        config,
        musique=args.musique,
        use_llm=args.llm,
        llm_model=args.llm_model,
    )
    result, report = generator.generate(prompt)
    _write_output(result, report, args.output, args.text)


def _cmd_build(args: argparse.Namespace) -> None:
    musique = MuSiQueLoader(args.musique)
    try:
        prompt = musique.build_prompt(
            args.query_id,
            instruction=args.instruction,
            n_demos=args.n_demos,
            include_distractors=not args.no_distractors,
            demo_context=args.demo_context,
            seed=args.seed,
        )
    except KeyError:
        raise SystemExit(f"No MuSiQue record with id {args.query_id!r}") from None
    _write_output(prompt, [], args.output, args.text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="redundanzgen",
        description=(
            "Inject controlled redundancy (context passages, demonstrations, "
            "instructions) into context-based prompts built from MuSiQue."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Add redundancy to an existing prompt JSON")
    gen.add_argument("--input", required=True, help="Prompt JSON file")
    gen.add_argument(
        "--passages", type=int, default=0, metavar="N",
        help="Redundant copies of context passages (default: 0)",
    )
    gen.add_argument(
        "--passage-mode", choices=("duplicate", "restate"), default="duplicate",
        help="Verbatim copy, or wrapped in a restating template (default: duplicate)",
    )
    gen.add_argument(
        "--passage-target", choices=("supporting", "any"), default="supporting",
        help="Which passages get duplicated (default: supporting, i.e. gold first)",
    )
    gen.add_argument(
        "--passage-position", choices=("interleave", "append"), default="interleave",
        help="Where the copies go (default: interleave)",
    )
    gen.add_argument(
        "--demos", type=int, default=0, metavar="N",
        help="Redundant demonstrations with the same hop count (default: 0)",
    )
    gen.add_argument(
        "--demo-context", action="store_true",
        help="Give redundant demonstrations their own passages",
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
        "--musique", help="MuSiQue source: local JSONL/JSON file or HF dataset id"
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

    build = sub.add_parser(
        "build", help="Build a context-based prompt from MuSiQue (20 passages per question)"
    )
    build.add_argument(
        "--musique", required=True,
        help="MuSiQue source: local JSONL/JSON file or HF dataset id",
    )
    build.add_argument(
        "--query-id", required=True, help="MuSiQue id, e.g. 2hop__128801_205185"
    )
    build.add_argument(
        "--n-demos", type=int, default=0,
        help="Demonstrations with the same hop count (default: 0, i.e. zero-shot)",
    )
    build.add_argument(
        "--demo-context", action="store_true",
        help="Give each demonstration its own passages (multiplies prompt length)",
    )
    build.add_argument(
        "--no-distractors", action="store_true",
        help="Keep only the supporting passages (oracle context)",
    )
    build.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    build.add_argument("--seed", type=int, default=None)
    build.add_argument("--output")
    build.add_argument("--text")
    build.set_defaults(func=_cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
