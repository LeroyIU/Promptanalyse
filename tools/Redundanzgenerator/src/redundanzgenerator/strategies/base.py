"""Strategy interface for injecting one type of redundancy into a prompt."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from ..models import FewShotPrompt


class RedundancyStrategy(ABC):
    """Mutates a FewShotPrompt in place and reports what it inserted.

    ``apply`` returns a report dict (strategy name plus inserted items) that
    the generator collects, so experiments can trace exactly which redundancy
    was injected where.
    """

    name: str

    @abstractmethod
    def apply(self, prompt: FewShotPrompt, rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError
