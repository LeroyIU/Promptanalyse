"""Loader for MuSiQue (StonyBrookNLP/musique or a local file).

MuSiQue is the data basis of this project. Every question ships **20 Wikipedia
paragraphs** it is to be answered from, each flagged ``is_supporting`` -- 2-4
of them carry the reasoning chain, the rest are distractors. Questions are
composed from single-hop questions (2-4 hops) and filtered so that no hop can
be skipped, which is exactly the property that makes them sensitive to
compression: drop one supporting paragraph and the answer becomes underivable
rather than merely harder.

Rows are indexed by hop count -- the dataset's own measure of comparable
difficulty, and thus the category to draw comparable demonstrations from.

Official record format (``musique_ans_v1.0_dev.jsonl``)::

    {"id": "2hop__128801_205185",
     "question": "...",
     "question_decomposition": [
         {"id": ..., "question": "...", "answer": "...",
          "paragraph_support_idx": 3}, ...],
     "paragraphs": [
         {"idx": 0, "title": "...", "paragraph_text": "...",
          "is_supporting": false}, ...],
     "answer": "...", "answer_aliases": [...], "answerable": true}
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any

from ..models import ContextPassage, Demonstration, FewShotPrompt
from ._records import load_records

_HOP_PREFIX = re.compile(r"^(\d+)hop")

DEFAULT_INSTRUCTION = (
    "Answer the question using only the passages provided. "
    "Respond with a short factual answer."
)


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def parse_json_field(value: Any) -> list[Any]:
    """Parse a list-valued field that may arrive as a JSON-ish string.

    JSONL rows already carry real lists; CSV exports of the same data carry
    their string repr. Both are accepted so the loader does not depend on how
    the dataset was materialised.
    """
    if isinstance(value, list):
        return list(value)
    if value in (None, ""):
        return []
    text = str(value)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return list(parsed)
    return []


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    return str(value).strip().lower() in ("true", "1", "yes")


class MuSiQueLoader:
    """Index MuSiQue records for prompt construction.

    ``answerable_only`` drops the unanswerable half of MuSiQue-Full; it has no
    effect on MuSiQue-Ans, where every record is answerable by construction.
    """

    def __init__(
        self,
        source: str | Path,
        split: str = "validation",
        answerable_only: bool = False,
    ) -> None:
        records = load_records(source, split=split)
        if answerable_only:
            records = [r for r in records if _as_bool(r.get("answerable")) is not False]
        self.records = records
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_question: dict[str, dict[str, Any]] = {}
        self._by_hops: dict[int, list[dict[str, Any]]] = {}
        for row in self.records:
            if row.get("id") not in (None, ""):
                self._by_id[str(row["id"])] = row
            question = row.get("question")
            if question:
                self._by_question.setdefault(_normalize(question), row)
            hops = self.n_hops_of(row)
            if hops is not None:
                self._by_hops.setdefault(hops, []).append(row)

    @property
    def hop_counts(self) -> list[int]:
        """The hop counts present in the data, e.g. ``[2, 3, 4]``."""
        return sorted(self._by_hops)

    def by_id(self, record_id: Any) -> dict[str, Any] | None:
        return self._by_id.get(str(record_id))

    def by_question(self, question: str) -> dict[str, Any] | None:
        return self._by_question.get(_normalize(question))

    def by_hops(
        self, hops: int, exclude_questions: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """All rows with ``hops`` reasoning steps, minus excluded questions."""
        rows = self._by_hops.get(hops, [])
        if not exclude_questions:
            return list(rows)
        excluded = {_normalize(q) for q in exclude_questions}
        return [r for r in rows if _normalize(r.get("question", "")) not in excluded]

    # --- record-level accessors ------------------------------------------
    @staticmethod
    def n_hops_of(row: dict[str, Any]) -> int | None:
        """Number of reasoning steps, from the decomposition or the id prefix.

        MuSiQue ids encode the composition shape (``2hop__``, ``3hop1__``,
        ``4hop3__``); the decomposition is authoritative where present.
        """
        decomposition = parse_json_field(row.get("question_decomposition"))
        if decomposition:
            return len(decomposition)
        match = _HOP_PREFIX.match(str(row.get("id", "")))
        return int(match.group(1)) if match else None

    @staticmethod
    def paragraphs_of(row: dict[str, Any]) -> list[dict[str, Any]]:
        """The row's paragraphs, in their original order."""
        return [p for p in parse_json_field(row.get("paragraphs")) if isinstance(p, dict)]

    @classmethod
    def supporting_paragraphs(cls, row: dict[str, Any]) -> list[dict[str, Any]]:
        """The gold paragraphs carrying the reasoning chain."""
        return [p for p in cls.paragraphs_of(row) if _as_bool(p.get("is_supporting"))]

    @classmethod
    def distractor_paragraphs(cls, row: dict[str, Any]) -> list[dict[str, Any]]:
        """The paragraphs that are topically related but carry no evidence."""
        return [
            p for p in cls.paragraphs_of(row) if not _as_bool(p.get("is_supporting"))
        ]

    @staticmethod
    def answer_of(row: dict[str, Any]) -> str:
        """Canonical answer of a MuSiQue row."""
        answer = row.get("answer")
        return "" if answer in (None, "") else str(answer)

    @classmethod
    def answers_of(cls, row: dict[str, Any]) -> list[str]:
        """Canonical answer plus its aliases, canonical first."""
        answers = [cls.answer_of(row)] if cls.answer_of(row) else []
        for alias in parse_json_field(row.get("answer_aliases")):
            text = str(alias)
            if text and text not in answers:
                answers.append(text)
        return answers

    # --- prompt construction ---------------------------------------------
    @classmethod
    def context_of(
        cls,
        row: dict[str, Any],
        include_distractors: bool = True,
    ) -> list[ContextPassage]:
        """The row's paragraphs as context passages, gold labels preserved.

        ``include_distractors=False`` yields the gold-only context, i.e. the
        oracle condition every compressor is implicitly aiming at.
        """
        paragraphs = (
            cls.paragraphs_of(row) if include_distractors
            else cls.supporting_paragraphs(row)
        )
        return [
            ContextPassage(
                text=str(p.get("paragraph_text", "")),
                title=str(p.get("title", "")),
                is_supporting=_as_bool(p.get("is_supporting")),
                meta={"idx": p.get("idx")},
            )
            for p in paragraphs
        ]

    def build_prompt(
        self,
        record_id: Any,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        n_demos: int = 0,
        include_distractors: bool = True,
        demo_context: bool = False,
        seed: int | None = None,
    ) -> FewShotPrompt:
        """Build a context-based prompt for one MuSiQue record.

        Defaults to zero-shot, which is the standard reading-comprehension
        setting and keeps the context the dominant part of the prompt. Any
        demonstrations are drawn from the same hop count; ``demo_context``
        decides whether they bring their own 20 paragraphs along -- off by
        default, since that multiplies prompt length several times over.
        """
        row = self.by_id(record_id)
        if row is None:
            raise KeyError(f"no MuSiQue record with id {record_id!r}")

        demonstrations: list[Demonstration] = []
        if n_demos > 0:
            hops = self.n_hops_of(row)
            candidates = (
                self.by_hops(hops, exclude_questions={str(row["question"])})
                if hops is not None
                else [r for r in self.records if r is not row]
            )
            random.Random(seed).shuffle(candidates)
            demonstrations = [
                Demonstration(
                    question=str(demo_row["question"]),
                    answer=self.answer_of(demo_row),
                    context=(
                        self.context_of(demo_row, include_distractors)
                        if demo_context
                        else []
                    ),
                    meta={"id": demo_row.get("id"), "n_hops": self.n_hops_of(demo_row)},
                )
                for demo_row in candidates[:n_demos]
            ]

        return FewShotPrompt(
            instructions=[instruction] if instruction else [],
            demonstrations=demonstrations,
            query=str(row["question"]),
            context=self.context_of(row, include_distractors),
            meta={
                "id": row.get("id"),
                "n_hops": self.n_hops_of(row),
                "n_supporting": len(self.supporting_paragraphs(row)),
                "n_distractors": len(self.distractor_paragraphs(row)),
            },
        )
