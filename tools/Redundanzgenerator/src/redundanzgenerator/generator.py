"""Orchestrates redundancy strategies over a few-shot prompt."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from .data.musique import MuSiQueLoader
from .models import FewShotPrompt, RedundancyConfig
from .strategies import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    PassageRedundancy,
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
        musique: MuSiQueLoader | str | Path | None = None,
        use_llm: bool = False,
        llm_model: str = "claude-sonnet-5",
    ) -> "RedundancyGenerator":
        strategies: list[RedundancyStrategy] = []
        if config.n_passages > 0:
            strategies.append(
                PassageRedundancy(
                    n=config.n_passages,
                    mode=config.passage_mode,
                    target=config.passage_target,
                    position=config.passage_position,
                )
            )
        if config.n_demonstrations > 0:
            if musique is None:
                raise ValueError("Demonstration redundancy requires a MuSiQue source")
            if not isinstance(musique, MuSiQueLoader):
                musique = MuSiQueLoader(musique)
            strategies.append(
                DemonstrationRedundancy(
                    musique,
                    n=config.n_demonstrations,
                    with_context=config.demonstration_context,
                )
            )
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
