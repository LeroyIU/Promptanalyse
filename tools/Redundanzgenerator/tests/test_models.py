from redundanzgenerator import Demonstration, FewShotPrompt


def test_from_dict_accepts_single_instruction_key():
    prompt = FewShotPrompt.from_dict(
        {"instruction": "Do it.", "query": "Q?", "demonstrations": []}
    )
    assert prompt.instructions == ["Do it."]


def test_roundtrip(prompt):
    data = prompt.to_dict()
    again = FewShotPrompt.from_dict(data)
    assert again.to_dict() == data
    assert again.query == "What is the capital of France?"
    assert len(again.demonstrations) == 2
    assert again.meta["prop"] == "capital"


def test_demonstration_roundtrip():
    demo = Demonstration(
        question="Q?", answer="A", paraphrases=["Q2?"], meta={"id": 1}
    )
    assert Demonstration.from_dict(demo.to_dict()) == demo
