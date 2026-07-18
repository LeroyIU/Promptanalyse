import random

import pytest

from redundanzgenerator import (
    DemonstrationRedundancy,
    InstructionRedundancy,
    ParaphraseRedundancy,
)


def test_paraphrase_redundancy_on_query(popqa_tp, prompt):
    strategy = ParaphraseRedundancy(popqa_tp, n=2)
    report = strategy.apply(prompt, random.Random(42))
    assert len(prompt.query_paraphrases) == 2
    assert prompt.query not in prompt.query_paraphrases
    assert report["inserted"][0]["target"] == "query"


def test_paraphrase_redundancy_includes_demonstrations(popqa_tp, prompt):
    strategy = ParaphraseRedundancy(popqa_tp, n=1, include_demonstrations=True)
    with pytest.warns(UserWarning):  # Canada demo has no paraphrases in fixture
        strategy.apply(prompt, random.Random(0))
    japan_demo = prompt.demonstrations[0]
    assert len(japan_demo.paraphrases) == 1
    canada_demo = prompt.demonstrations[1]
    assert canada_demo.paraphrases == []


def test_paraphrase_redundancy_warns_without_match(popqa_tp, prompt):
    prompt.query = "Unknown question?"
    prompt.meta = {}
    with pytest.warns(UserWarning):
        ParaphraseRedundancy(popqa_tp, n=2).apply(prompt, random.Random(0))
    assert prompt.query_paraphrases == []


def test_paraphrase_redundancy_capped_at_pool_size(popqa_tp, prompt):
    ParaphraseRedundancy(popqa_tp, n=99).apply(prompt, random.Random(0))
    assert len(prompt.query_paraphrases) == 4  # fixture has 4 alternations


def test_demonstration_redundancy_same_category(popqa, prompt):
    strategy = DemonstrationRedundancy(popqa, n=3)
    report = strategy.apply(prompt, random.Random(42))
    assert len(prompt.demonstrations) == 5
    new_demos = prompt.demonstrations[2:]
    assert all(d.meta["prop"] == "capital" for d in new_demos)
    assert report["category"] == "capital"
    questions = [d.question for d in prompt.demonstrations]
    assert prompt.query not in questions
    assert len(set(questions)) == len(questions)


def test_demonstration_redundancy_falls_back_to_other_categories(popqa, prompt):
    # only 3 unused capital rows exist; ask for 6
    strategy = DemonstrationRedundancy(popqa, n=6)
    strategy.apply(prompt, random.Random(1))
    new_demos = prompt.demonstrations[2:]
    assert len(new_demos) == 6
    assert sum(d.meta["prop"] == "capital" for d in new_demos) == 3
    assert all(d.answer for d in new_demos)


def test_demonstration_redundancy_unknown_category_warns(popqa, prompt):
    prompt.query = "Unknown question?"
    prompt.meta = {}
    with pytest.warns(UserWarning):
        DemonstrationRedundancy(popqa, n=2).apply(prompt, random.Random(0))
    assert len(prompt.demonstrations) == 4


def test_demonstration_redundancy_interleave(popqa, prompt):
    strategy = DemonstrationRedundancy(popqa, n=2, position="interleave")
    strategy.apply(prompt, random.Random(3))
    assert len(prompt.demonstrations) == 4


def test_instruction_redundancy_rule_based(prompt):
    strategy = InstructionRedundancy(n=2)
    report = strategy.apply(prompt, random.Random(42))
    assert len(prompt.instructions) == 3
    original = prompt.instructions[0]
    for extra in prompt.instructions[1:]:
        assert original in extra
    assert report["mode"] == "rule-based"


def test_instruction_redundancy_positions(prompt):
    strategy = InstructionRedundancy(n=3, position="both")
    strategy.apply(prompt, random.Random(0))
    assert len(prompt.instructions) == 3  # original + 2 at start
    assert len(prompt.trailing_instructions) == 1


def test_instruction_redundancy_end_position(prompt):
    InstructionRedundancy(n=2, position="end").apply(prompt, random.Random(0))
    assert len(prompt.instructions) == 1
    assert len(prompt.trailing_instructions) == 2


def test_instruction_redundancy_more_than_templates(prompt):
    InstructionRedundancy(n=12).apply(prompt, random.Random(0))
    assert len(prompt.instructions) == 13
    assert prompt.instructions.count(prompt.instructions[0]) >= 4


def test_instruction_redundancy_no_instruction():
    from redundanzgenerator import FewShotPrompt

    empty = FewShotPrompt(instructions=[], demonstrations=[], query="Q?")
    report = InstructionRedundancy(n=2).apply(empty, random.Random(0))
    assert report["skipped"] == "no instruction"
    assert empty.instructions == []
