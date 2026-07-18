import pytest

from redundanzgenerator import RedundancyConfig, RedundancyGenerator, render_prompt


def _make_generator(popqa, popqa_tp, **kwargs) -> RedundancyGenerator:
    defaults = dict(
        n_paraphrases=2, n_demonstrations=2, n_instructions=1, seed=42
    )
    defaults.update(kwargs)
    return RedundancyGenerator.from_config(
        RedundancyConfig(**defaults), popqa=popqa, popqa_tp=popqa_tp
    )


def test_generate_applies_all_strategies(popqa, popqa_tp, prompt):
    result, report = _make_generator(popqa, popqa_tp).generate(prompt)
    assert len(result.query_paraphrases) == 2
    assert len(result.demonstrations) == 4
    assert len(result.instructions) == 2
    assert [r["strategy"] for r in report] == [
        "lexical",
        "demonstrations",
        "instructions",
    ]


def test_generate_does_not_mutate_input(popqa, popqa_tp, prompt):
    _make_generator(popqa, popqa_tp).generate(prompt)
    assert prompt.query_paraphrases == []
    assert len(prompt.demonstrations) == 2
    assert len(prompt.instructions) == 1


def test_generate_is_deterministic(popqa, popqa_tp, prompt):
    first, _ = _make_generator(popqa, popqa_tp).generate(prompt)
    second, _ = _make_generator(popqa, popqa_tp).generate(prompt)
    assert first.to_dict() == second.to_dict()


def test_generate_seed_changes_output(popqa, popqa_tp, prompt):
    first, _ = _make_generator(popqa, popqa_tp, seed=1).generate(prompt)
    second, _ = _make_generator(popqa, popqa_tp, seed=2).generate(prompt)
    assert first.to_dict() != second.to_dict()


def test_from_config_accepts_paths(prompt):
    from tests.conftest import FIXTURES

    generator = RedundancyGenerator.from_config(
        RedundancyConfig(n_paraphrases=1, n_demonstrations=1, seed=0),
        popqa=FIXTURES / "popqa_sample.csv",
        popqa_tp=FIXTURES / "popqa_tp_sample.csv",
    )
    result, _ = generator.generate(prompt)
    assert len(result.query_paraphrases) == 1


def test_missing_sources_raise(prompt):
    with pytest.raises(ValueError, match="PopQA-TP"):
        RedundancyGenerator.from_config(RedundancyConfig(n_paraphrases=1))
    with pytest.raises(ValueError, match="PopQA"):
        RedundancyGenerator.from_config(RedundancyConfig(n_demonstrations=1))


def test_render_prompt(popqa, popqa_tp, prompt):
    result, _ = _make_generator(popqa, popqa_tp).generate(prompt)
    text = render_prompt(result)
    assert text.startswith(result.instructions[0])
    assert text.count("Q: ") == len(result.demonstrations) + 3  # query + 2 paraphrases
    assert text.rstrip().endswith("A:")
    assert "Q: What is the capital of France?" in text
