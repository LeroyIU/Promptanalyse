"""Redundanzgenerator: controlled redundancy for few-shot prompts."""

from .data.musique import MuSiQueLoader
from .data.popqa import PopQALoader
from .data.popqa_tp import PopQATPLoader
from .generator import RedundancyGenerator
from .models import ContextPassage, Demonstration, FewShotPrompt, RedundancyConfig
from .render import render_context, render_prompt
from .strategies import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    ParaphraseRedundancy,
    RedundancyStrategy,
)

__all__ = [
    "ContextPassage",
    "Demonstration",
    "FewShotPrompt",
    "RedundancyConfig",
    "RedundancyGenerator",
    "MuSiQueLoader",
    "PopQALoader",
    "PopQATPLoader",
    "RedundancyStrategy",
    "ParaphraseRedundancy",
    "DemonstrationRedundancy",
    "InstructionRedundancy",
    "render_context",
    "render_prompt",
]

__version__ = "0.1.0"
