"""Passage redundancy: the same evidence stated twice in the context.

This is the context-level counterpart to instruction redundancy. With a
context-based dataset the context is by far the largest part of the prompt and
therefore the part a compressor spends most of its budget on -- so it is the
part where redundancy has to be injected for the experiment to be about
anything.

Redundant copies default to the *supporting* passages: duplicating a distractor
adds tokens but no information a compressor could meaningfully preserve or
drop, whereas duplicating gold evidence poses the real question -- does the
compressor recognise the second copy as redundant, or does it spend budget on
both and lose something else instead?
"""

from __future__ import annotations

import copy
import json
import random
from importlib import resources
from typing import Any

from ..models import ContextPassage, FewShotPrompt
from .base import RedundancyStrategy


def _load_passage_wrappers() -> list[str]:
    text = (
        resources.files("redundanzgenerator.data")
        .joinpath("instruction_templates.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)["passage_wrappers"]


class PassageRedundancy(RedundancyStrategy):
    """Inserts ``n`` redundant copies of context passages.

    ``mode``:
        - ``"duplicate"`` (default) -- verbatim second copy. The purest form of
          lexical redundancy, fully deterministic, and realistic: retrievers
          routinely return near-duplicate passages.
        - ``"restate"`` -- the passage wrapped in a restating template
          ("In other words: ..."), so the duplication is not detectable by
          string equality alone.

    ``target``: ``"supporting"`` prefers gold passages (falling back to the
    rest once they are exhausted), ``"any"`` samples from the whole context.

    ``position``: ``"interleave"`` (default) scatters the copies through the
    context, ``"append"`` puts them at the end. Interleaving is the default
    because a copy adjacent to its original is trivially detectable and would
    understate what compression has to cope with.

    Every inserted copy is marked in ``meta`` with the ``idx`` of the passage
    it came from, so redundant material can be told apart from original
    material after compression.
    """

    name = "passages"

    def __init__(
        self,
        n: int = 1,
        mode: str = "duplicate",
        target: str = "supporting",
        position: str = "interleave",
    ) -> None:
        if mode not in ("duplicate", "restate"):
            raise ValueError(f"mode must be 'duplicate' or 'restate', got {mode!r}")
        if target not in ("supporting", "any"):
            raise ValueError(f"target must be 'supporting' or 'any', got {target!r}")
        if position not in ("interleave", "append"):
            raise ValueError(
                f"position must be 'interleave' or 'append', got {position!r}"
            )
        self.n = n
        self.mode = mode
        self.target = target
        self.position = position
        self._wrappers = _load_passage_wrappers()

    def _candidates(self, prompt: FewShotPrompt, rng: random.Random) -> list[ContextPassage]:
        """Passages eligible for duplication, in the order they will be used."""
        originals = [p for p in prompt.context if "redundant_copy_of" not in p.meta]
        if self.target == "any":
            pool = list(originals)
            rng.shuffle(pool)
            return pool
        supporting = [p for p in originals if p.is_supporting]
        rest = [p for p in originals if not p.is_supporting]
        rng.shuffle(supporting)
        rng.shuffle(rest)
        return supporting + rest

    def _copy(self, passage: ContextPassage, rng: random.Random) -> ContextPassage:
        duplicate = copy.deepcopy(passage)
        if self.mode == "restate":
            duplicate.text = rng.choice(self._wrappers).format(passage=passage.text)
        duplicate.meta = dict(passage.meta)
        duplicate.meta["redundant_copy_of"] = passage.meta.get("idx")
        duplicate.meta["redundancy_mode"] = self.mode
        return duplicate

    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        if not prompt.context:
            return {"strategy": self.name, "inserted": [], "skipped": "no context"}

        candidates = self._candidates(prompt, rng)
        # More copies than passages: cycle, so n is always honoured.
        picked = [candidates[i % len(candidates)] for i in range(self.n)]
        copies = [self._copy(passage, rng) for passage in picked]

        for duplicate in copies:
            if self.position == "append":
                prompt.context.append(duplicate)
            else:
                prompt.context.insert(rng.randint(0, len(prompt.context)), duplicate)

        return {
            "strategy": self.name,
            "mode": self.mode,
            "target": self.target,
            "inserted": [c.to_dict() for c in copies],
        }
