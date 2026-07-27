import pytest

from redundanzgenerator import RedundancyConfig, RedundancyGenerator, render_prompt

from tests.conftest import FIXTURES


def _make_generator(musique, **kwargs) -> RedundancyGenerator:
    defaults = dict(n_passages=2, n_demonstrations=2, n_instructions=1, seed=42)
    defaults.update(kwargs)
    return RedundancyGenerator.from_config(
        RedundancyConfig(**defaults), musique=musique
    )


def test_generate_applies_all_strategies(musique, prompt):
    result, report = _make_generator(musique).generate(prompt)
    assert len(result.context) == 6
    assert len(result.demonstrations) == 3
    assert len(result.instructions) == 2
    assert [r["strategy"] for r in report] == [
        "passages",
        "demonstrations",
        "instructions",
    ]


def test_generate_does_not_mutate_input(musique, prompt):
    _make_generator(musique).generate(prompt)
    assert len(prompt.context) == 4
    assert len(prompt.demonstrations) == 1
    assert len(prompt.instructions) == 1


def test_generate_is_deterministic(musique, prompt):
    first, _ = _make_generator(musique).generate(prompt)
    second, _ = _make_generator(musique).generate(prompt)
    assert first.to_dict() == second.to_dict()


def test_generate_seed_changes_output(musique, prompt):
    first, _ = _make_generator(musique, seed=1).generate(prompt)
    second, _ = _make_generator(musique, seed=2).generate(prompt)
    assert first.to_dict() != second.to_dict()


def test_from_config_accepts_paths(prompt):
    generator = RedundancyGenerator.from_config(
        RedundancyConfig(n_passages=1, n_demonstrations=1, seed=0),
        musique=FIXTURES / "musique_pool.jsonl",
    )
    result, _ = generator.generate(prompt)
    assert len(result.context) == 5
    assert len(result.demonstrations) == 2


def test_passage_redundancy_needs_no_data_source(prompt):
    """Passage redundancy works off the prompt itself -- no dataset required."""
    generator = RedundancyGenerator.from_config(RedundancyConfig(n_passages=1, seed=0))
    result, _ = generator.generate(prompt)
    assert len(result.context) == 5


def test_missing_source_raises():
    with pytest.raises(ValueError, match="MuSiQue"):
        RedundancyGenerator.from_config(RedundancyConfig(n_demonstrations=1))


def test_render_prompt(musique, prompt):
    result, _ = _make_generator(musique).generate(prompt)
    text = render_prompt(result)
    assert text.startswith(result.instructions[0])
    assert text.count("Q: ") == len(result.demonstrations) + 1
    assert text.rstrip().endswith("A:")
    assert "Q: In which country is the city where the Eiffel Tower stands?" in text
    assert text.count("Context:") == 1, "only the query carries context by default"
