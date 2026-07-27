"""Redundanzgenerator: controlled redundancy for context-based prompts."""

from .data.musique import MuSiQueLoader
from .generator import RedundancyGenerator
from .models import ContextPassage, Demonstration, FewShotPrompt, RedundancyConfig
from .render import render_context, render_prompt
from .strategies import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    PassageRedundancy,
    RedundancyStrategy,
)

__all__ = [
    "ContextPassage",
    "Demonstration",
    "FewShotPrompt",
    "RedundancyConfig",
    "RedundancyGenerator",
    "MuSiQueLoader",
    "RedundancyStrategy",
    "PassageRedundancy",
    "DemonstrationRedundancy",
    "InstructionRedundancy",
    "render_context",
    "render_prompt",
]

__version__ = "0.2.0"
