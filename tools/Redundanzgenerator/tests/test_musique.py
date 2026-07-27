import json

import pytest

from redundanzgenerator import FewShotPrompt, MuSiQueLoader, render_prompt
from redundanzgenerator.data.musique import parse_json_field

from tests.conftest import FIXTURES


def test_musique_loads_jsonl(musique):
    """MuSiQue ships as JSONL, so the loader must read line-delimited JSON."""
    row = musique.by_id("2hop__101_201")
    assert row is not None
    assert musique.answer_of(row) == "France"
    assert musique.by_question(
        "in which country is the city where the eiffel tower stands?"
    ) is not None


def test_hop_counts_are_the_sampling_category(musique):
    assert musique.hop_counts == [2, 3, 4]
    assert len(musique.by_hops(2)) == 5
    assert len(musique.by_hops(3)) == 2
    assert len(musique.by_hops(4)) == 1


def test_n_hops_from_decomposition_and_from_id():
    """The decomposition wins; the id prefix is the fallback."""
    assert MuSiQueLoader.n_hops_of({"id": "4hop3__1_2_3_4"}) == 4
    assert MuSiQueLoader.n_hops_of(
        {"id": "2hop__1_2", "question_decomposition": [{}, {}, {}]}
    ) == 3
    assert MuSiQueLoader.n_hops_of({"id": "unknown"}) is None


def test_by_hops_excludes_questions(musique):
    rows = musique.by_hops(
        2,
        exclude_questions={"Who wrote the novel that the film Blade Runner is based on?"},
    )
    assert "2hop__102_202" not in {r["id"] for r in rows}
    assert len(rows) == 4


def test_gold_and_distractor_paragraphs_are_separable(musique):
    row = musique.by_id("2hop__101_201")
    assert [p["title"] for p in MuSiQueLoader.supporting_paragraphs(row)] == [
        "Eiffel Tower",
        "Paris",
    ]
    assert len(MuSiQueLoader.distractor_paragraphs(row)) == 2
    assert len(MuSiQueLoader.paragraphs_of(row)) == 4


def test_answers_include_aliases(musique):
    row = musique.by_id("2hop__106_206")
    assert musique.answers_of(row) == ["Swedish krona", "krona", "SEK"]


def test_context_keeps_gold_labels_and_order(musique):
    row = musique.by_id("2hop__101_201")
    context = musique.context_of(row)
    assert [c.title for c in context][:3] == ["Eiffel Tower", "Gustave Eiffel", "Paris"]
    assert [c.is_supporting for c in context] == [True, False, True, False]
    assert context[0].meta["idx"] == 0


def test_context_without_distractors_is_the_oracle_condition(musique):
    row = musique.by_id("3hop1__103_203_303")
    context = musique.context_of(row, include_distractors=False)
    assert len(context) == 3
    assert all(c.is_supporting for c in context)


def test_build_prompt_is_zero_shot_with_context(musique):
    prompt = musique.build_prompt("2hop__101_201")
    assert prompt.demonstrations == []
    assert len(prompt.context) == 4
    assert prompt.query.startswith("In which country")
    assert prompt.meta == {
        "id": "2hop__101_201",
        "n_hops": 2,
        "n_supporting": 2,
        "n_distractors": 2,
    }


def test_build_prompt_demos_share_the_hop_count(musique):
    prompt = musique.build_prompt("3hop1__103_203_303", n_demos=1, seed=42)
    demo = prompt.demonstrations[0]
    assert demo.meta["id"] == "3hop1__107_207_307"
    assert demo.meta["n_hops"] == 3
    assert demo.context == [], "demonstrations stay context-free by default"


def test_build_prompt_demo_context_is_opt_in(musique):
    prompt = musique.build_prompt(
        "3hop1__103_203_303", n_demos=1, demo_context=True, seed=42
    )
    assert len(prompt.demonstrations[0].context) == 4


def test_build_prompt_unknown_id(musique):
    with pytest.raises(KeyError):
        musique.build_prompt("2hop__does_not_exist")


def test_rendered_prompt_puts_context_above_the_question(musique):
    text = render_prompt(musique.build_prompt("2hop__101_201"))
    assert "Context:" in text
    assert "[1] Eiffel Tower" in text
    assert "[3] Paris" in text
    assert text.index("[1] Eiffel Tower") < text.index("Q: In which country")
    assert text.rstrip().endswith("A:")


def test_rendered_demo_context_precedes_its_own_question(musique):
    text = render_prompt(
        musique.build_prompt("3hop1__103_203_303", n_demos=1, demo_context=True, seed=42)
    )
    demo_context = text.index("[1] Ford Model T")
    demo_question = text.index("Q: Who founded the company")
    assert demo_context < demo_question < text.index("Q: What is the capital")


def test_prompt_with_context_round_trips(musique):
    prompt = musique.build_prompt(
        "3hop1__103_203_303", n_demos=1, demo_context=True, seed=42
    )
    restored = FewShotPrompt.from_dict(prompt.to_dict())
    assert restored.to_dict() == prompt.to_dict()
    assert restored.context[0].is_supporting is True
    assert restored.demonstrations[0].context[0].title == "Ford Model T"


def test_parse_json_field_accepts_lists_and_their_string_repr():
    assert parse_json_field(["a"]) == ["a"]
    assert parse_json_field('["a", "b"]') == ["a", "b"]
    assert parse_json_field("['a', 'b']") == ["a", "b"]
    assert parse_json_field("") == []
    assert parse_json_field("not a list") == []


def test_answerable_only_filters_musique_full(tmp_path):
    """MuSiQue-Full mixes in unanswerable records; MuSiQue-Ans is unaffected."""
    source = tmp_path / "musique_full.jsonl"
    source.write_text(
        FIXTURES.joinpath("musique_pool.jsonl").read_text(encoding="utf-8")
        + json.dumps(
            {
                "id": "2hop__109_209",
                "question": "Unanswerable by construction?",
                "paragraphs": [],
                "answer": None,
                "answerable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(MuSiQueLoader(source).records) == 9
    assert len(MuSiQueLoader(source, answerable_only=True).records) == 8
