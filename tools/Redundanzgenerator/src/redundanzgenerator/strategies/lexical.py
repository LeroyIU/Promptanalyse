"""Lexical redundancy: add PopQA-TP paraphrases of questions."""

from __future__ import annotations

import random
import warnings
from typing import Any

from ..data.popqa_tp import PopQATPLoader
from ..models import FewShotPrompt
from .base import RedundancyStrategy


class ParaphraseRedundancy(RedundancyStrategy):
    """Adds up to ``n`` semantically equivalent paraphrases from PopQA-TP to
    the query (and, if ``include_demonstrations``, to each demonstration
    question). Questions without a PopQA-TP paraphrase group are left
    unchanged with a warning."""

    name = "lexical"

    def __init__(
        self,
        paraphrases: PopQATPLoader,
        n: int = 1,
        include_demonstrations: bool = False,
    ) -> None:
        self.paraphrases = paraphrases
        self.n = n
        self.include_demonstrations = include_demonstrations

    def _sample(
        self, question: str, record_id: Any, existing: list[str], rng: random.Random
    ) -> list[str]:
        pool = [
            p
            for p in self.paraphrases.paraphrases_for(question, record_id)
            if p not in existing
        ]
        if not pool:
            return []
        return rng.sample(pool, min(self.n, len(pool)))

    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        report: dict[str, Any] = {"strategy": self.name, "inserted": []}

        added = self._sample(
            prompt.query, prompt.meta.get("id"), prompt.query_paraphrases, rng
        )
        if added:
            prompt.query_paraphrases.extend(added)
            report["inserted"].append({"target": "query", "paraphrases": added})
        else:
            warnings.warn(
                f"No PopQA-TP paraphrases found for query: {prompt.query!r}",
                stacklevel=2,
            )

        if self.include_demonstrations:
            for i, demo in enumerate(prompt.demonstrations):
                added = self._sample(
                    demo.question, demo.meta.get("id"), demo.paraphrases, rng
                )
                if added:
                    demo.paraphrases.extend(added)
                    report["inserted"].append(
                        {"target": f"demonstration[{i}]", "paraphrases": added}
                    )
                else:
                    warnings.warn(
                        "No PopQA-TP paraphrases found for demonstration: "
                        f"{demo.question!r}",
                        stacklevel=2,
                    )
        return report
