"""Demonstration redundancy: extra same-category examples from PopQA."""

from __future__ import annotations

import random
import warnings
from typing import Any

from ..data.popqa import PopQALoader
from ..models import Demonstration, FewShotPrompt
from .base import RedundancyStrategy


class DemonstrationRedundancy(RedundancyStrategy):
    """Appends ``n`` additional demonstrations drawn from PopQA.

    Candidates share the query's relationship category (``prop``); if the
    query's category is unknown or exhausted, the remaining slots are filled
    from other categories. The query itself and questions already present in
    the prompt are never used. ``position`` is "append" (after the existing
    demonstrations) or "interleave" (shuffled in between them).
    """

    name = "demonstrations"

    def __init__(
        self, popqa: PopQALoader, n: int = 1, position: str = "append"
    ) -> None:
        if position not in ("append", "interleave"):
            raise ValueError(f"position must be 'append' or 'interleave', got {position!r}")
        self.popqa = popqa
        self.n = n
        self.position = position

    def _query_category(self, prompt: FewShotPrompt) -> str | None:
        prop = prompt.meta.get("prop")
        if prop:
            return str(prop)
        row = self.popqa.by_question(prompt.query)
        if row is None and prompt.meta.get("id") is not None:
            row = self.popqa.by_id(prompt.meta["id"])
        return str(row["prop"]) if row and row.get("prop") else None

    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        used_questions = {prompt.query, *(d.question for d in prompt.demonstrations)}
        category = self._query_category(prompt)

        candidates: list[dict[str, Any]] = []
        if category is not None:
            candidates = self.popqa.by_category(category, exclude_questions=used_questions)
        else:
            warnings.warn(
                f"Could not determine PopQA category for query {prompt.query!r}; "
                "sampling redundant demonstrations from all categories.",
                stacklevel=2,
            )

        rng.shuffle(candidates)
        picked = candidates[: self.n]
        if len(picked) < self.n:
            fallback = [
                row
                for cat in self.popqa.categories
                if cat != category
                for row in self.popqa.by_category(cat, exclude_questions=used_questions)
            ]
            rng.shuffle(fallback)
            picked.extend(fallback[: self.n - len(picked)])

        new_demos = [
            Demonstration(
                question=str(row["question"]),
                answer=self.popqa.answer_of(row),
                meta={
                    "id": row.get("id"),
                    "prop": row.get("prop"),
                    "source": "popqa-redundant",
                },
            )
            for row in picked
        ]

        if self.position == "append":
            prompt.demonstrations.extend(new_demos)
        else:
            for demo in new_demos:
                index = rng.randint(0, len(prompt.demonstrations))
                prompt.demonstrations.insert(index, demo)

        return {
            "strategy": self.name,
            "category": category,
            "inserted": [d.to_dict() for d in new_demos],
        }
