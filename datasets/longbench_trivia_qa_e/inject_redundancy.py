#!/usr/bin/env python3
"""Inject controlled redundancy into the LongBench-E TriviaQA 8k+ subset.

Takes the output of ``extract_triviaqa_8k.py`` (one few-shot QA prompt per
row: ``context`` = concatenated few-shot demonstrations, ``input`` = the
final passage+question awaiting an answer, ``answers`` = gold answers) and
produces five JSONL variants for prompt-compression experiments:

- ``triviaqa_baseline.jsonl``               - unmodified passthrough
- ``triviaqa_redundancy_lexical.jsonl``      - paraphrase sentences inserted
  next to selected passage sentences
- ``triviaqa_redundancy_demonstration.jsonl``- whole few-shot demonstrations
  duplicated (verbatim or, with ``--near-duplicate``, paraphrased) and
  reinserted at a randomized, non-adjacent position
- ``triviaqa_redundancy_instruction.jsonl``  - the task instruction
  (LongBench's own ``triviaqa`` template) repeated at several positions,
  verbatim and/or paraphrased
- ``triviaqa_redundancy_combined.jsonl``     - all three stacked

Every non-baseline row carries ``redundancy_spans`` - per category, the
exact character span in the (new) ``context`` string that was injected,
plus the source text it is redundant with - so a compressor's output can
later be checked against exactly what should be removable.

Paraphrasing is done via the Anthropic API (model configurable, default
temperature 0.3 for consistency): ``ANTHROPIC_API_KEY`` is read from the
environment, never hardcoded. Variants that don't need paraphrasing
(demonstration duplication in verbatim mode, instruction repetition in
verbatim mode) run without any API key; variants that do (lexical,
combined, near-duplicate demonstrations, paraphrased instructions) are
skipped with a clear warning if no key is configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).resolve().parent

# LongBench's own prompt template for `triviaqa`/`triviaqa_e`
# (LongBench/config/dataset2prompt.json in THUDM/LongBench), applied as
# `{instruction}\n\n{context}\n\n{input}` at evaluation time. Neither
# `context` nor `input` contains it, so it is the "standard template" this
# tool falls back to for instruction-redundancy injection.
TRIVIAQA_INSTRUCTION = (
    "Answer the question based on the given passage. Only give me the answer "
    "and do not output any other words. The following are some examples."
)

PASSAGE_MARK = "Passage:\n"
QUESTION_MARK = "\nQuestion:\n"
ANSWER_MARK = "\nAnswer:\n"

DEMO_SPLIT_RE = re.compile(r"(?=^Passage:\n)", re.MULTILINE)

CONDITIONS = ("lexical", "demonstration", "instruction", "combined")


# --------------------------------------------------------------------------
# Parsing: split a LongBench-E `context` string into its few-shot demos
# --------------------------------------------------------------------------


@dataclass
class Demo:
    """One few-shot demonstration, still carrying its exact source text.

    ``passage_start``/``passage_end`` locate the passage substring within
    ``raw`` so later steps can splice new text into the passage only,
    without disturbing the "Question:"/"Answer:" tail.
    """

    raw: str
    passage: str
    question: str
    answer: str
    passage_start: int
    passage_end: int


def _parse_demo_block(raw: str) -> Demo:
    if not raw.startswith(PASSAGE_MARK):
        raise ValueError(f"demo block does not start with {PASSAGE_MARK!r}: {raw[:50]!r}")
    q_idx = raw.index(QUESTION_MARK)
    a_idx = raw.index(ANSWER_MARK, q_idx)
    passage_start = len(PASSAGE_MARK)
    passage_end = q_idx
    return Demo(
        raw=raw,
        passage=raw[passage_start:passage_end],
        question=raw[q_idx + len(QUESTION_MARK) : a_idx],
        answer=raw[a_idx + len(ANSWER_MARK) :],
        passage_start=passage_start,
        passage_end=passage_end,
    )


def parse_context_demos(context: str) -> list[Demo]:
    blocks = [b for b in DEMO_SPLIT_RE.split(context) if b.strip()]
    if "".join(blocks) != context:
        raise ValueError("splitting into demo blocks did not reconstruct the original context")
    return [_parse_demo_block(b) for b in blocks]


# --------------------------------------------------------------------------
# Sentence segmentation (nltk punkt, falling back to a regex splitter)
# --------------------------------------------------------------------------

_sentence_backend: str | None = None


def _nltk_sentence_spans(text: str) -> list[tuple[int, int]] | None:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import PunktSentenceTokenizer

        return list(PunktSentenceTokenizer().span_tokenize(text))
    except Exception:
        return None


def _regex_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences in ``text``."""
    global _sentence_backend
    if _sentence_backend is None:
        _sentence_backend = "nltk" if _nltk_sentence_spans(text) is not None else "regex"
        if _sentence_backend == "regex":
            print("(nltk punkt data unavailable - falling back to a regex sentence splitter)")
    if _sentence_backend == "nltk":
        spans = _nltk_sentence_spans(text)
        if spans is not None:
            return spans
    return _regex_sentence_spans(text)


# --------------------------------------------------------------------------
# Paraphrasing
# --------------------------------------------------------------------------


class Paraphraser(Protocol):
    def paraphrase(self, text: str) -> str: ...


class AnthropicParaphraser:
    """Paraphrases via the Anthropic Messages API, with a disk cache.

    The cache is keyed by a hash of the input text so the same sentence is
    never paraphrased twice across variants/reruns - the lexical and
    combined variants otherwise call this on largely overlapping text.
    """

    def __init__(self, model: str, temperature: float = 0.3, cache_path: Path | None = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def paraphrase(self, text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max(64, int(len(text.split()) * 2.5)),
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Paraphrasiere den folgenden Satz, erhalte die Bedeutung exakt, "
                        "variiere Wortwahl und Satzbau. Gib NUR die Paraphrase zurück, "
                        "ohne Anführungszeichen oder Zusatzkommentar.\n\n" + text
                    ),
                }
            ],
        )
        out = "".join(block.text for block in response.content if block.type == "text").strip()
        self._cache[key] = out
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        return out


def build_paraphraser(backend: str, model: str, cache_path: Path) -> Paraphraser | None:
    if backend == "none":
        return None
    if backend == "anthropic":
        try:
            return AnthropicParaphraser(model=model, cache_path=cache_path)
        except RuntimeError as exc:
            print(
                f"WARNING: paraphraser unavailable ({exc}). Lexical redundancy, the "
                "combined variant, --near-duplicate demonstrations, and paraphrased "
                "instructions all need it and will be skipped/downgraded. Set "
                "ANTHROPIC_API_KEY to enable them."
            )
            return None
    raise ValueError(f"unknown --paraphrase-backend: {backend}")


# --------------------------------------------------------------------------
# Generic offset-tracking insertion helper
# --------------------------------------------------------------------------


def _apply_insertions(base: str, insertions: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Insert each (offset_in_base, text) into base; return new text plus the
    (start, end) span of each inserted text in the *new* string, in the same
    order as ``insertions`` was given (not sorted order)."""
    order = sorted(range(len(insertions)), key=lambda i: insertions[i][0])
    pieces: list[str] = []
    spans: list[tuple[int, int] | None] = [None] * len(insertions)
    cursor = 0
    out_len = 0
    for idx in order:
        offset, text = insertions[idx]
        pieces.append(base[cursor:offset])
        out_len += offset - cursor
        start = out_len
        pieces.append(text)
        out_len += len(text)
        spans[idx] = (start, out_len)
        cursor = offset
    pieces.append(base[cursor:])
    return "".join(pieces), spans  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Redundancy module: lexical (paraphrase sentences)
# --------------------------------------------------------------------------


def inject_lexical(
    demos: list[Demo],
    ratio: float,
    include_answer_sentences: bool,
    paraphraser: Paraphraser,
    rng: random.Random,
) -> tuple[list[str], list[dict]]:
    """Returns (new demo raw texts in original order, local lexical spans).

    Local spans carry ``demo_index`` plus start/end *within that demo's new
    raw text*; the caller offsets them into the final context.
    """
    candidates: list[tuple[int, int, int, str]] = []
    total_sentences = 0
    for di, d in enumerate(demos):
        for s, e in split_sentences(d.passage):
            total_sentences += 1
            sentence = d.passage[s:e]
            contains_answer = bool(d.answer.strip()) and d.answer.strip().lower() in sentence.lower()
            if not include_answer_sentences and contains_answer:
                continue
            candidates.append((di, s, e, sentence))

    new_raws = [d.raw for d in demos]
    if total_sentences == 0 or not candidates:
        return new_raws, []

    k = min(max(1, round(ratio * total_sentences)), len(candidates))
    chosen = rng.sample(candidates, k)

    by_demo: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for di, s, e, sentence in chosen:
        by_demo[di].append((s, e, sentence))

    spans: list[dict] = []
    for di, d in enumerate(demos):
        items = by_demo.get(di)
        if not items:
            continue
        insertions = [(d.passage_start + e, " " + paraphraser.paraphrase(sentence)) for s, e, sentence in items]
        new_raw, local_spans = _apply_insertions(d.raw, insertions)
        new_raws[di] = new_raw
        for (s, e, sentence), (ls, le) in zip(items, local_spans):
            spans.append(
                {
                    "demo_index": di,
                    "start": ls + 1,  # skip the separating space
                    "end": le,
                    "type": "lexical_paraphrase",
                    "original_text": sentence,
                    "inserted_text": new_raw[ls + 1 : le],
                }
            )
    return new_raws, spans


# --------------------------------------------------------------------------
# Redundancy module: demonstration duplication
# --------------------------------------------------------------------------


def _paraphrase_whole_demo(raw: str, paraphraser: Paraphraser) -> str:
    """Rewrite every passage sentence of a demo block; keep Q/A verbatim."""
    demo = _parse_demo_block(raw)
    pieces = []
    cursor = 0
    for s, e in split_sentences(demo.passage):
        pieces.append(demo.passage[cursor:s])
        pieces.append(paraphraser.paraphrase(demo.passage[s:e]))
        cursor = e
    pieces.append(demo.passage[cursor:])
    new_passage = "".join(pieces)
    return PASSAGE_MARK + new_passage + QUESTION_MARK + demo.question + ANSWER_MARK + demo.answer


def inject_demonstration(
    demo_raws: list[str],
    ratio: float,
    near_duplicate: bool,
    paraphraser: Paraphraser | None,
    rng: random.Random,
) -> list[dict]:
    """Returns a list of duplicate-insertion ops: {source, gap, text, type, original_text}."""
    n = len(demo_raws)
    k = min(max(1, round(ratio * n)), n)
    sources = rng.sample(range(n), k)

    duplicates = []
    for src in sources:
        if near_duplicate:
            assert paraphraser is not None, "near_duplicate requires a paraphraser"
            dup_text = _paraphrase_whole_demo(demo_raws[src], paraphraser)
            dtype = "demonstration_near_duplicate"
        else:
            dup_text = demo_raws[src]
            dtype = "demonstration_duplicate"

        forbidden = {src, src + 1}
        allowed_gaps = [g for g in range(n + 1) if g not in forbidden] or list(range(n + 1))
        gap = rng.choice(allowed_gaps)
        duplicates.append(
            {"source": src, "gap": gap, "text": dup_text, "type": dtype, "original_text": demo_raws[src]}
        )
    return duplicates


# --------------------------------------------------------------------------
# Redundancy module: instruction repetition
# --------------------------------------------------------------------------


def instruction_gap_positions(n_demos: int, repeats: int) -> list[int]:
    """Gap indices (0..n_demos) to insert instruction copies at.

    Gap 0 is "before the first demo", gap n_demos is "before the final
    question" (i.e. the end of the context); intermediate gaps are "between
    demos", spread evenly.
    """
    if repeats <= 0:
        return []
    if repeats == 1:
        return [0]
    gaps = [0, n_demos]
    extra_needed = repeats - 2
    if extra_needed > 0:
        candidates = list(range(1, n_demos)) if n_demos > 1 else [0]
        for i in range(extra_needed):
            idx = max(0, min(len(candidates) - 1, round((i + 1) * len(candidates) / (extra_needed + 1)) - 1))
            gaps.append(candidates[idx])
    return sorted(gaps)


def build_instruction_texts(
    gaps: list[int], mode: str, paraphraser: Paraphraser | None, rng: random.Random
) -> list[dict]:
    """Returns per-gap instruction insertions: {gap, text, type, original_text}."""
    out = []
    for i, gap in enumerate(gaps):
        use_paraphrase = mode == "paraphrased" or (mode == "both" and i % 2 == 1)
        if use_paraphrase:
            assert paraphraser is not None, "paraphrased instructions require a paraphraser"
            text = paraphraser.paraphrase(TRIVIAQA_INSTRUCTION)
            itype = "instruction_paraphrased"
        else:
            text = TRIVIAQA_INSTRUCTION
            itype = "instruction_verbatim"
        out.append({"gap": gap, "text": text, "type": itype, "original_text": TRIVIAQA_INSTRUCTION})
    return out


# --------------------------------------------------------------------------
# Assembly: stitch demo blocks + duplicates + instructions into one context
# --------------------------------------------------------------------------


def assemble_context(
    demo_raws: list[str],
    lexical_spans: list[dict],
    duplicates: list[dict],
    instructions: list[dict],
) -> tuple[str, dict[str, list[dict]]]:
    n = len(demo_raws)
    by_gap_instr: dict[int, list[dict]] = defaultdict(list)
    for item in instructions:
        by_gap_instr[item["gap"]].append(item)
    by_gap_dup: dict[int, list[dict]] = defaultdict(list)
    for item in duplicates:
        by_gap_dup[item["gap"]].append(item)
    lexical_by_demo: dict[int, list[dict]] = defaultdict(list)
    for sp in lexical_spans:
        lexical_by_demo[sp["demo_index"]].append(sp)

    pieces: list[str] = []
    cursor = 0
    spans: dict[str, list[dict]] = {"lexical": [], "demonstration": [], "instruction": []}

    def emit(text: str) -> int:
        nonlocal cursor
        start = cursor
        pieces.append(text)
        cursor += len(text)
        return start

    for gap in range(n + 1):
        for item in by_gap_instr.get(gap, []):
            start = emit(item["text"] + "\n\n")
            spans["instruction"].append(
                {
                    "start": start,
                    "end": start + len(item["text"]),
                    "type": item["type"],
                    "original_text": item["original_text"],
                    "inserted_text": item["text"],
                }
            )
        for item in by_gap_dup.get(gap, []):
            start = emit(item["text"])
            spans["demonstration"].append(
                {
                    "start": start,
                    "end": start + len(item["text"]),
                    "type": item["type"],
                    "original_text": item["original_text"],
                    "inserted_text": item["text"],
                    "duplicated_from_demo_index": item["source"],
                }
            )
        if gap < n:
            demo_start = emit(demo_raws[gap])
            for sp in lexical_by_demo.get(gap, []):
                spans["lexical"].append(
                    {
                        "start": demo_start + sp["start"],
                        "end": demo_start + sp["end"],
                        "type": sp["type"],
                        "original_text": sp["original_text"],
                        "inserted_text": sp["inserted_text"],
                        "demo_index": gap,
                    }
                )
    return "".join(pieces), spans


# --------------------------------------------------------------------------
# Per-sample, per-condition driver
# --------------------------------------------------------------------------


@dataclass
class RedundancyConfig:
    lexical_ratio: float = 0.15
    include_answer_sentences: bool = False
    demo_duplicate_ratio: float = 0.2
    near_duplicate: bool = False
    instruction_repeats: int = 2
    instruction_mode: str = "both"  # verbatim | paraphrased | both


def build_variant_record(
    row: dict, condition: str, cfg: RedundancyConfig, paraphraser: Paraphraser | None, enc, rng: random.Random
) -> dict:
    demos = parse_context_demos(row["context"])
    n = len(demos)
    demo_raws = [d.raw for d in demos]
    lexical_spans: list[dict] = []
    duplicates: list[dict] = []
    instructions: list[dict] = []

    if condition in ("lexical", "combined"):
        assert paraphraser is not None
        demo_raws, lexical_spans = inject_lexical(
            demos, cfg.lexical_ratio, cfg.include_answer_sentences, paraphraser, rng
        )
    if condition in ("demonstration", "combined"):
        duplicates = inject_demonstration(demo_raws, cfg.demo_duplicate_ratio, cfg.near_duplicate, paraphraser, rng)
    if condition in ("instruction", "combined"):
        gaps = instruction_gap_positions(n, cfg.instruction_repeats)
        instructions = build_instruction_texts(gaps, cfg.instruction_mode, paraphraser, rng)

    new_context, spans = assemble_context(demo_raws, lexical_spans, duplicates, instructions)

    original_length = len(enc.encode(row["context"]))
    augmented_length = len(enc.encode(new_context))

    return {
        "_id": row["_id"],
        "context": new_context,
        "input": row["input"],
        "answers": row["answers"],
        "length": row["length"],
        "all_classes": row.get("all_classes"),
        "condition": condition,
        "redundancy_spans": spans,
        "original_length": original_length,
        "augmented_length": augmented_length,
        "redundancy_ratio": (augmented_length - original_length) / augmented_length if augmented_length else 0.0,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_variant(records: list[dict], original_by_id: dict[str, dict], sample_size: int = 8) -> list[str]:
    rng = random.Random(0)
    sample = rng.sample(records, min(sample_size, len(records)))
    problems = []
    for rec in sample:
        orig = original_by_id[rec["_id"]]
        if rec["input"] != orig["input"]:
            problems.append(f"{rec['_id']}: input field was modified")
        if not rec["answers"] or rec["answers"] != orig["answers"]:
            problems.append(f"{rec['_id']}: answers field is empty or was modified")
        ctx = rec["context"]
        for category, spans in rec["redundancy_spans"].items():
            for sp in spans:
                if ctx[sp["start"] : sp["end"]] != sp["inserted_text"]:
                    problems.append(f"{rec['_id']}: {category} span offset does not match inserted_text")
    return problems


def print_variant_stats(records: list[dict], condition: str) -> None:
    n = len(records)
    if n == 0:
        print(f"[{condition}] 0 samples (skipped)")
        return
    increases = [
        (r["augmented_length"] - r["original_length"]) / r["original_length"] * 100
        for r in records
        if r["original_length"]
    ]
    span_counts: Counter = Counter()
    for r in records:
        for category, spans in r["redundancy_spans"].items():
            span_counts[category] += len(spans)
    avg_increase = sum(increases) / len(increases) if increases else 0.0
    print(
        f"[{condition}] n={n}  avg_length_increase={avg_increase:.1f}%  "
        f"spans_per_category={dict(span_counts)}  "
        f"spans_per_sample={sum(span_counts.values()) / n:.1f}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=HERE / "triviaqa_longbench_e_8k_plus.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--lexical-ratio", type=float, default=0.15)
    parser.add_argument("--include-answer-sentences", action="store_true")
    parser.add_argument("--demo-duplicate-ratio", type=float, default=0.2)
    parser.add_argument("--near-duplicate", action="store_true")
    parser.add_argument("--instruction-repeats", type=int, default=2)
    parser.add_argument(
        "--instruction-mode", choices=("verbatim", "paraphrased", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paraphrase-backend", choices=("anthropic", "none"), default="anthropic")
    parser.add_argument("--paraphrase-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--paraphrase-cache", type=Path, default=HERE / ".paraphrase_cache.json")
    args = parser.parse_args()

    cfg = RedundancyConfig(
        lexical_ratio=args.lexical_ratio,
        include_answer_sentences=args.include_answer_sentences,
        demo_duplicate_ratio=args.demo_duplicate_ratio,
        near_duplicate=args.near_duplicate,
        instruction_repeats=args.instruction_repeats,
        instruction_mode=args.instruction_mode,
    )

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    original_by_id = {row["_id"]: row for row in rows}

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.output_dir / "triviaqa_baseline.jsonl"
    with open(baseline_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[baseline] wrote {len(rows)} samples to {baseline_path}")

    paraphraser = build_paraphraser(args.paraphrase_backend, args.paraphrase_model, args.paraphrase_cache)

    needs_paraphraser = {
        "lexical": True,
        "demonstration": cfg.near_duplicate,
        "instruction": cfg.instruction_mode != "verbatim",
        "combined": True,
    }

    for condition in args.conditions:
        effective_cfg = cfg
        if paraphraser is None and needs_paraphraser[condition]:
            if condition == "instruction":
                print(
                    "WARNING: no paraphraser configured - downgrading --instruction-mode "
                    f"{cfg.instruction_mode!r} to 'verbatim' for the instruction variant."
                )
                effective_cfg = RedundancyConfig(**{**cfg.__dict__, "instruction_mode": "verbatim"})
            else:
                print(f"[{condition}] skipped: no paraphraser configured (see warning above)")
                continue

        records = []
        for row in rows:
            rng = random.Random(f"{args.seed}:{condition}:{row['_id']}")
            records.append(build_variant_record(row, condition, effective_cfg, paraphraser, enc, rng))

        out_path = args.output_dir / f"triviaqa_redundancy_{condition}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        problems = validate_variant(records, original_by_id)
        if problems:
            raise AssertionError(f"[{condition}] validation failed:\n" + "\n".join(problems))

        print(f"[{condition}] wrote {len(records)} samples to {out_path} (validation OK)")
        print_variant_stats(records, condition)


if __name__ == "__main__":
    main()
