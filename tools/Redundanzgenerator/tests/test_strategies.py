import random

import pytest

from redundanzgenerator import (
    DemonstrationRedundancy,
    FewShotPrompt,
    InstructionRedundancy,
    PassageRedundancy,
)

from tests.conftest import FIXTURES


def _copies(prompt) -> list:
    return [p for p in prompt.context if "redundant_copy_of" in p.meta]


def test_passage_redundancy_duplicates_supporting_passages(prompt):
    report = PassageRedundancy(n=2).apply(prompt, random.Random(42))
    assert len(prompt.context) == 6
    copies = _copies(prompt)
    assert len(copies) == 2
    assert all(c.is_supporting for c in copies), "gold passages are targeted first"
    assert {c.meta["redundant_copy_of"] for c in copies} == {0, 2}
    assert report["strategy"] == "passages"
    assert report["mode"] == "duplicate"


def test_duplicate_mode_copies_verbatim(prompt):
    PassageRedundancy(n=1).apply(prompt, random.Random(0))
    copy = _copies(prompt)[0]
    original = next(
        p for p in prompt.context if p.meta.get("idx") == copy.meta["redundant_copy_of"]
    )
    assert copy.text == original.text


def test_restate_mode_rewords_the_copy(prompt):
    PassageRedundancy(n=1, mode="restate").apply(prompt, random.Random(0))
    copy = _copies(prompt)[0]
    original = next(
        p for p in prompt.context if p.meta.get("idx") == copy.meta["redundant_copy_of"]
    )
    assert copy.text != original.text
    assert original.text in copy.text
    assert copy.meta["redundancy_mode"] == "restate"


def test_passage_redundancy_never_duplicates_a_duplicate(prompt):
    """Copies are excluded from the candidate pool, so originals are cycled."""
    PassageRedundancy(n=4).apply(prompt, random.Random(7))
    copies = _copies(prompt)
    assert len(copies) == 4
    assert all(c.meta["redundant_copy_of"] in (0, 1, 2, 3) for c in copies)


def test_passage_redundancy_target_any_reaches_distractors(prompt):
    PassageRedundancy(n=4, target="any").apply(prompt, random.Random(1))
    assert any(not c.is_supporting for c in _copies(prompt))


def test_passage_redundancy_append_keeps_originals_first(prompt):
    PassageRedundancy(n=2, position="append").apply(prompt, random.Random(0))
    assert all("redundant_copy_of" not in p.meta for p in prompt.context[:4])
    assert all("redundant_copy_of" in p.meta for p in prompt.context[4:])


def test_passage_redundancy_without_context():
    empty = FewShotPrompt(instructions=["Do it."], demonstrations=[], query="Q?")
    report = PassageRedundancy(n=2).apply(empty, random.Random(0))
    assert report["skipped"] == "no context"
    assert empty.context == []


def test_passage_redundancy_rejects_bad_arguments():
    with pytest.raises(ValueError, match="mode"):
        PassageRedundancy(mode="paraphrase")
    with pytest.raises(ValueError, match="target"):
        PassageRedundancy(target="gold")
    with pytest.raises(ValueError, match="position"):
        PassageRedundancy(position="start")


def test_demonstration_redundancy_same_hop_count(musique, prompt):
    report = DemonstrationRedundancy(musique, n=3).apply(prompt, random.Random(42))
    assert len(prompt.demonstrations) == 4
    new_demos = prompt.demonstrations[1:]
    assert all(d.meta["n_hops"] == 2 for d in new_demos)
    assert report["n_hops"] == 2
    questions = [d.question for d in prompt.demonstrations]
    assert prompt.query not in questions
    assert len(set(questions)) == len(questions)


def test_demonstration_redundancy_falls_back_to_other_hop_counts(musique, prompt):
    # 4 unused 2-hop rows exist (the query itself and the existing demo are out)
    DemonstrationRedundancy(musique, n=6).apply(prompt, random.Random(1))
    new_demos = prompt.demonstrations[1:]
    assert len(new_demos) == 6
    assert sum(d.meta["n_hops"] == 2 for d in new_demos) == 3
    assert all(d.answer for d in new_demos)


def test_demonstration_redundancy_context_is_opt_in(musique, prompt):
    DemonstrationRedundancy(musique, n=1).apply(prompt, random.Random(0))
    assert prompt.demonstrations[-1].context == []

    fresh = FewShotPrompt.from_json(FIXTURES / "prompt_example.json")
    DemonstrationRedundancy(musique, n=1, with_context=True).apply(
        fresh, random.Random(0)
    )
    assert fresh.demonstrations[-1].context != []


def test_demonstration_redundancy_unknown_hop_count_warns(musique, prompt):
    prompt.query = "Unknown question?"
    prompt.meta = {}
    with pytest.warns(UserWarning):
        DemonstrationRedundancy(musique, n=2).apply(prompt, random.Random(0))
    assert len(prompt.demonstrations) == 3


def test_demonstration_redundancy_interleave(musique, prompt):
    DemonstrationRedundancy(musique, n=2, position="interleave").apply(
        prompt, random.Random(3)
    )
    assert len(prompt.demonstrations) == 3


def test_instruction_redundancy_rule_based(prompt):
    report = InstructionRedundancy(n=2).apply(prompt, random.Random(42))
    assert len(prompt.instructions) == 3
    original = prompt.instructions[0]
    for extra in prompt.instructions[1:]:
        assert original in extra
    assert report["mode"] == "rule-based"


def test_instruction_redundancy_positions(prompt):
    InstructionRedundancy(n=3, position="both").apply(prompt, random.Random(0))
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
    empty = FewShotPrompt(instructions=[], demonstrations=[], query="Q?")
    report = InstructionRedundancy(n=2).apply(empty, random.Random(0))
    assert report["skipped"] == "no instruction"
    assert empty.instructions == []
