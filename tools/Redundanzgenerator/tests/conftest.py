from pathlib import Path

import pytest

from redundanzgenerator import FewShotPrompt, PopQALoader, PopQATPLoader

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def popqa() -> PopQALoader:
    return PopQALoader(FIXTURES / "popqa_sample.csv")


@pytest.fixture
def popqa_tp() -> PopQATPLoader:
    return PopQATPLoader(FIXTURES / "popqa_tp_sample.csv")


@pytest.fixture
def prompt() -> FewShotPrompt:
    return FewShotPrompt.from_json(FIXTURES / "prompt_example.json")
