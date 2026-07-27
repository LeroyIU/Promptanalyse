"""Demonstration redundancy: extra examples with the same hop count."""

from __future__ import annotations

import random
import warnings
from typing import Any

from ..data.musique import MuSiQueLoader
from ..models import Demonstration, FewShotPrompt
from .base import RedundancyStrategy


class DemonstrationRedundancy(RedundancyStrategy):
    """Appends ``n`` additional demonstrations drawn from MuSiQue.

    Candidates share the query's hop count, which is the dataset's own measure
    of comparable difficulty; if it is unknown or exhausted, the remaining
    slots are filled from the other hop counts. The query itself and questions
    already present in the prompt are never used.

    ``with_context`` decides whether the extra demonstrations bring their own
    passages along. It is off by default: one demonstration with full context
    roughly doubles the prompt, which would confound "more demonstrations"
    with "much more context".

    ``position`` is "append" (after the existing demonstrations) or
    "interleave" (shuffled in between them).
    """

    name = "demonstrations"

    def __init__(
        self,
        musique: MuSiQueLoader,
        n: int = 1,
        position: str = "append",
        with_context: bool = False,
    ) -> None:
        if position not in ("append", "interleave"):
            raise ValueError(f"position must be 'append' or 'interleave', got {position!r}")
        self.musique = musique
        self.n = n
        self.position = position
        self.with_context = with_context

    def _query_hops(self, prompt: FewShotPrompt) -> int | None:
        hops = prompt.meta.get("n_hops")
        if hops not in (None, ""):
            return int(hops)
        row = self.musique.by_question(prompt.query)
        if row is None and prompt.meta.get("id") is not None:
            row = self.musique.by_id(prompt.meta["id"])
        return self.musique.n_hops_of(row) if row else None

    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        used_questions = {prompt.query, *(d.question for d in prompt.demonstrations)}
        hops = self._query_hops(prompt)

        candidates: list[dict[str, Any]] = []
        if hops is not None:
            candidates = self.musique.by_hops(hops, exclude_questions=used_questions)
        else:
            warnings.warn(
                f"Could not determine the hop count for query {prompt.query!r}; "
                "sampling redundant demonstrations from all hop counts.",
                stacklevel=2,
            )

        rng.shuffle(candidates)
        picked = candidates[: self.n]
        if len(picked) < self.n:
            fallback = [
                row
                for other in self.musique.hop_counts
                if other != hops
                for row in self.musique.by_hops(other, exclude_questions=used_questions)
            ]
            rng.shuffle(fallback)
            picked.extend(fallback[: self.n - len(picked)])

        new_demos = [
            Demonstration(
                question=str(row["question"]),
                answer=self.musique.answer_of(row),
                context=self.musique.context_of(row) if self.with_context else [],
                meta={
                    "id": row.get("id"),
                    "n_hops": self.musique.n_hops_of(row),
                    "source": "musique-redundant",
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
            "n_hops": hops,
            "with_context": self.with_context,
            "inserted": [d.to_dict() for d in new_demos],
        }
