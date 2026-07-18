"""Orchestrates redundancy strategies over a few-shot prompt."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from .data.popqa import PopQALoader
from .data.popqa_tp import PopQATPLoader
from .models import FewShotPrompt, RedundancyConfig
from .strategies import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    ParaphraseRedundancy,
    RedundancyStrategy,
)


class RedundancyGenerator:
    """Applies a fixed sequence of redundancy strategies to prompts.

    Either pass ready-made ``strategies``, or use :meth:`from_config` to
    assemble them from a :class:`RedundancyConfig` plus data sources. The
    input prompt is never mutated; ``generate`` returns an enriched copy and
    a report of everything that was inserted.
    """

    def __init__(
        self, strategies: list[RedundancyStrategy], seed: int | None = None
    ) -> None:
        self.strategies = strategies
        self.seed = seed

    @classmethod
    def from_config(
        cls,
        config: RedundancyConfig,
        popqa: PopQALoader | str | Path | None = None,
        popqa_tp: PopQATPLoader | str | Path | None = None,
        use_llm: bool = False,
        llm_model: str = "claude-sonnet-5",
    ) -> "RedundancyGenerator":
        strategies: list[RedundancyStrategy] = []
        if config.n_paraphrases > 0:
            if popqa_tp is None:
                raise ValueError("Paraphrase redundancy requires a PopQA-TP source")
            if not isinstance(popqa_tp, PopQATPLoader):
                popqa_tp = PopQATPLoader(popqa_tp)
            strategies.append(
                ParaphraseRedundancy(
                    popqa_tp,
                    n=config.n_paraphrases,
                    include_demonstrations=config.paraphrase_demonstrations,
                )
            )
        if config.n_demonstrations > 0:
            if popqa is None:
                raise ValueError("Demonstration redundancy requires a PopQA source")
            if not isinstance(popqa, PopQALoader):
                popqa = PopQALoader(popqa)
            strategies.append(DemonstrationRedundancy(popqa, n=config.n_demonstrations))
        if config.n_instructions > 0:
            strategies.append(
                InstructionRedundancy(
                    n=config.n_instructions,
                    position=config.instruction_position,
                    use_llm=use_llm,
                    llm_model=llm_model,
                )
            )
        return cls(strategies, seed=config.seed)

    def generate(
        self, prompt: FewShotPrompt
    ) -> tuple[FewShotPrompt, list[dict[str, Any]]]:
        rng = random.Random(self.seed)
        result = copy.deepcopy(prompt)
        report = [strategy.apply(result, rng) for strategy in self.strategies]
        return result, report
