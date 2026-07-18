from redundanzgenerator.data.popqa import parse_possible_answers


def test_popqa_categories(popqa):
    assert popqa.categories == [
        "author",
        "capital",
        "director",
        "occupation",
        "place of birth",
    ]


def test_popqa_lookup(popqa):
    row = popqa.by_id(101)
    assert row is not None and row["subj"] == "France"
    assert popqa.by_question("what is the capital of  france?") is not None
    assert popqa.answer_of(row) == "Paris"


def test_popqa_category_exclusion(popqa):
    rows = popqa.by_category(
        "capital", exclude_questions={"What is the capital of France?"}
    )
    questions = {r["question"] for r in rows}
    assert "What is the capital of France?" not in questions
    assert len(rows) == 5


def test_parse_possible_answers():
    assert parse_possible_answers('["Paris"]') == ["Paris"]
    assert parse_possible_answers("['Paris', 'Lutetia']") == ["Paris", "Lutetia"]
    assert parse_possible_answers(["Tokyo"]) == ["Tokyo"]
    assert parse_possible_answers("") == []
    assert parse_possible_answers("Paris") == ["Paris"]


def test_popqa_tp_paraphrases_by_id(popqa_tp):
    paraphrases = popqa_tp.paraphrases_for(
        "What is the capital of France?", record_id=101
    )
    assert len(paraphrases) == 4
    assert "What is the capital of France?" not in paraphrases
    assert "Who wrote Pride and Prejudice?" not in paraphrases


def test_popqa_tp_paraphrases_by_question_text(popqa_tp):
    paraphrases = popqa_tp.paraphrases_for("Who wrote Pride and Prejudice?")
    assert "Who is the author of Pride and Prejudice?" in paraphrases
    assert "Who wrote Pride and Prejudice?" not in paraphrases


def test_popqa_tp_unknown_question(popqa_tp):
    assert popqa_tp.paraphrases_for("Completely unknown question?") == []
