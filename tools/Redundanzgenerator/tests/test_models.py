from redundanzgenerator import ContextPassage, Demonstration, FewShotPrompt


def test_from_dict_accepts_single_instruction_key():
    prompt = FewShotPrompt.from_dict(
        {"instruction": "Do it.", "query": "Q?", "demonstrations": []}
    )
    assert prompt.instructions == ["Do it."]


def test_roundtrip(prompt):
    data = prompt.to_dict()
    again = FewShotPrompt.from_dict(data)
    assert again.to_dict() == data
    assert again.query == "In which country is the city where the Eiffel Tower stands?"
    assert len(again.demonstrations) == 1
    assert again.meta["n_hops"] == 2


def test_context_roundtrip(prompt):
    again = FewShotPrompt.from_dict(prompt.to_dict())
    assert len(again.context) == 4
    assert again.context[0].title == "Eiffel Tower"
    assert again.context[0].is_supporting is True
    assert again.context[1].is_supporting is False
    assert again.context[0].meta["idx"] == 0


def test_context_passage_roundtrip():
    passage = ContextPassage(
        text="Paris is the capital of France.",
        title="Paris",
        is_supporting=True,
        meta={"idx": 2},
    )
    assert ContextPassage.from_dict(passage.to_dict()) == passage


def test_unlabelled_context_passage_omits_is_supporting():
    """An unlabelled passage must not serialise as 'not supporting'."""
    passage = ContextPassage(text="Some retrieved text.")
    assert passage.to_dict() == {"text": "Some retrieved text."}
    assert ContextPassage.from_dict(passage.to_dict()).is_supporting is None


def test_demonstration_roundtrip():
    demo = Demonstration(
        question="Q?",
        answer="A",
        context=[ContextPassage(text="Evidence.", title="T", is_supporting=True)],
        meta={"id": "2hop__1_2"},
    )
    assert Demonstration.from_dict(demo.to_dict()) == demo


def test_context_free_prompt_omits_context_key():
    """Closed-book prompts stay byte-identical to the pre-context format."""
    prompt = FewShotPrompt(instructions=["Do it."], demonstrations=[], query="Q?")
    assert "context" not in prompt.to_dict()
