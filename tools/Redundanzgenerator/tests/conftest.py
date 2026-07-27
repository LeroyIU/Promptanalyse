from pathlib import Path

import pytest

from redundanzgenerator import FewShotPrompt, MuSiQueLoader

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def musique() -> MuSiQueLoader:
    """A small pool: five 2-hop, two 3-hop and one 4-hop record."""
    return MuSiQueLoader(FIXTURES / "musique_pool.jsonl")


@pytest.fixture
def prompt() -> FewShotPrompt:
    """A 2-hop prompt with 4 context passages, 2 of them supporting."""
    return FewShotPrompt.from_json(FIXTURES / "prompt_example.json")
