"""Loader for PopQA-TP (ibm-research/popqa-tp or a local file).

PopQA-TP extends PopQA with 3-10 manually created, strictly
semantically-equivalent paraphrases per original question, preserving the
original metadata (Rabinovich et al. 2023). Rows that share an ``id``
belong to the same original question; the loader groups them so that, for
any question, the remaining phrasings of its group can be used as
redundant paraphrases. Grouping falls back to normalized question text if
no id column is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._records import load_records

_ID_COLUMNS = ("id", "orig_id", "original_id", "popqa_id", "question_id")
_QUESTION_COLUMNS = ("question", "paraphrase", "paraphrased_question")


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


class PopQATPLoader:
    def __init__(self, source: str | Path, split: str = "test") -> None:
        self.records = load_records(source, split=split)
        self._groups: dict[str, list[str]] = {}
        self._group_by_question: dict[str, str] = {}
        for row in self.records:
            question = self._question_of(row)
            if not question:
                continue
            key = self._group_key(row, question)
            group = self._groups.setdefault(key, [])
            if _normalize(question) not in {_normalize(q) for q in group}:
                group.append(question)
            self._group_by_question.setdefault(_normalize(question), key)

    @staticmethod
    def _question_of(row: dict[str, Any]) -> str:
        for col in _QUESTION_COLUMNS:
            value = row.get(col)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    @staticmethod
    def _group_key(row: dict[str, Any], question: str) -> str:
        for col in _ID_COLUMNS:
            value = row.get(col)
            if value not in (None, ""):
                return f"{col}:{value}"
        return f"q:{_normalize(question)}"

    def paraphrases_for(self, question: str, record_id: Any = None) -> list[str]:
        """Semantically equivalent alternative phrasings of ``question``.

        Looks the question's paraphrase group up by PopQA id first (exact),
        then by the question text itself. The question's own phrasing is
        never included in the result.
        """
        key = None
        if record_id not in (None, ""):
            for col in _ID_COLUMNS:
                candidate = f"{col}:{record_id}"
                if candidate in self._groups:
                    key = candidate
                    break
        if key is None:
            key = self._group_by_question.get(_normalize(question))
        if key is None:
            return []
        norm = _normalize(question)
        return [q for q in self._groups[key] if _normalize(q) != norm]
