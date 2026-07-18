"""Redundanzgenerator: controlled redundancy for few-shot prompts."""

from .data.popqa import PopQALoader
from .data.popqa_tp import PopQATPLoader
from .generator import RedundancyGenerator
from .models import Demonstration, FewShotPrompt, RedundancyConfig
from .render import render_prompt
from .strategies import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    ParaphraseRedundancy,
    RedundancyStrategy,
)

__all__ = [
    "Demonstration",
    "FewShotPrompt",
    "RedundancyConfig",
    "RedundancyGenerator",
    "PopQALoader",
    "PopQATPLoader",
    "RedundancyStrategy",
    "ParaphraseRedundancy",
    "DemonstrationRedundancy",
    "InstructionRedundancy",
    "render_prompt",
]

__version__ = "0.1.0"
