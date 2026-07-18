"""Loader for the PopQA dataset (akariasai/PopQA or a local file).

PopQA rows carry entity-centric factual questions with a relationship type
in ``prop`` (occupation, place of birth, genre, father, country, producer,
director, capital of, screenwriter, composer, color, religion, sport,
author, mother, capital). The loader indexes rows by ``prop`` so that
demonstration redundancy can sample from the same category as the query.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ._records import load_records


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def parse_possible_answers(value: Any) -> list[str]:
    """PopQA stores possible_answers as a JSON-ish string list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None or value == "":
        return []
    text = str(value)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            continue
    return [text]


class PopQALoader:
    def __init__(self, source: str | Path, split: str = "test") -> None:
        self.records = load_records(source, split=split)
        self._by_prop: dict[str, list[dict[str, Any]]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_question: dict[str, dict[str, Any]] = {}
        for row in self.records:
            prop = str(row.get("prop", "")).strip()
            self._by_prop.setdefault(prop, []).append(row)
            if row.get("id") not in (None, ""):
                self._by_id[str(row["id"])] = row
            question = row.get("question")
            if question:
                self._by_question.setdefault(_normalize(question), row)

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_prop)

    def by_id(self, record_id: Any) -> dict[str, Any] | None:
        return self._by_id.get(str(record_id))

    def by_question(self, question: str) -> dict[str, Any] | None:
        return self._by_question.get(_normalize(question))

    def by_category(
        self, prop: str, exclude_questions: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """All rows of a category, minus rows whose question is excluded."""
        rows = self._by_prop.get(prop, [])
        if not exclude_questions:
            return list(rows)
        excluded = {_normalize(q) for q in exclude_questions}
        return [r for r in rows if _normalize(r.get("question", "")) not in excluded]

    @staticmethod
    def answer_of(row: dict[str, Any]) -> str:
        """Canonical answer of a PopQA row (``obj``, falling back to the
        first possible answer)."""
        obj = row.get("obj")
        if obj not in (None, ""):
            return str(obj)
        answers = parse_possible_answers(row.get("possible_answers"))
        return answers[0] if answers else ""
