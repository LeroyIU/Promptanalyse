"""Instruction redundancy: repeated or rephrased instructions."""

from __future__ import annotations

import json
import random
from importlib import resources
from typing import Any

from ..models import FewShotPrompt
from .base import RedundancyStrategy


def _load_wrappers() -> list[str]:
    text = (
        resources.files("redundanzgenerator.data")
        .joinpath("instruction_templates.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)["wrappers"]


class InstructionRedundancy(RedundancyStrategy):
    """Adds ``n`` redundant statements of the prompt's first instruction.

    Rule-based mode (default) restates the original instruction inside
    varying wrapper templates ("Remember: ...", "Once again: ...").
    With ``use_llm=True`` genuine rephrasings are generated once per prompt
    via the Anthropic API (optional ``anthropic`` dependency, key from
    ``ANTHROPIC_API_KEY``).

    ``position``: "start" places extras after the original instruction,
    "end" places them after the query, "both" alternates between the two.
    """

    name = "instructions"

    def __init__(
        self,
        n: int = 1,
        position: str = "start",
        use_llm: bool = False,
        llm_model: str = "claude-sonnet-5",
    ) -> None:
        if position not in ("start", "end", "both"):
            raise ValueError(f"position must be 'start', 'end' or 'both', got {position!r}")
        self.n = n
        self.position = position
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._wrappers = _load_wrappers()

    def _rule_based(self, instruction: str, rng: random.Random) -> list[str]:
        wrappers = [w for w in self._wrappers if w != "{instruction}"]
        rng.shuffle(wrappers)
        out = []
        for i in range(self.n):
            if i < len(wrappers):
                out.append(wrappers[i].format(instruction=instruction))
            else:
                out.append(instruction)  # more repetitions than wrappers: verbatim
        return out

    def _llm_based(self, instruction: str) -> list[str]:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM-based instruction rephrasing requires the 'anthropic' "
                "package. Install it with: pip install redundanzgenerator[llm]"
            ) from exc
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Rephrase the following instruction in {self.n} "
                        "different ways. Keep the meaning identical. Reply "
                        "with a JSON array of strings and nothing else.\n\n"
                        f"Instruction: {instruction}"
                    ),
                }
            ],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        rephrasings = json.loads(text)
        if not isinstance(rephrasings, list):
            raise ValueError(f"Expected a JSON array from the LLM, got: {text!r}")
        return [str(r) for r in rephrasings[: self.n]]

    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        if not prompt.instructions:
            return {"strategy": self.name, "inserted": [], "skipped": "no instruction"}
        original = prompt.instructions[0]
        extras = (
            self._llm_based(original) if self.use_llm else self._rule_based(original, rng)
        )

        placements: list[str] = []
        for i, extra in enumerate(extras):
            if self.position == "start" or (self.position == "both" and i % 2 == 0):
                prompt.instructions.append(extra)
                placements.append("start")
            else:
                prompt.trailing_instructions.append(extra)
                placements.append("end")

        return {
            "strategy": self.name,
            "mode": "llm" if self.use_llm else "rule-based",
            "inserted": [
                {"text": text, "position": pos} for text, pos in zip(extras, placements)
            ],
        }
