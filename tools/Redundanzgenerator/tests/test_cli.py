import json

import pytest

from redundanzgenerator.cli import main

from tests.conftest import FIXTURES


def test_cli_generate(tmp_path):
    out_json = tmp_path / "out.json"
    out_text = tmp_path / "out.txt"
    main(
        [
            "generate",
            "--input", str(FIXTURES / "prompt_example.json"),
            "--lexical", "2",
            "--demos", "2",
            "--instructions", "1",
            "--popqa", str(FIXTURES / "popqa_sample.csv"),
            "--popqa-tp", str(FIXTURES / "popqa_tp_sample.csv"),
            "--seed", "42",
            "--output", str(out_json),
            "--text", str(out_text),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    prompt = payload["prompt"]
    assert len(prompt["query_paraphrases"]) == 2
    assert len(prompt["demonstrations"]) == 4
    assert len(prompt["instructions"]) == 2
    assert len(payload["redundancy_report"]) == 3
    text = out_text.read_text(encoding="utf-8")
    assert "Q: What is the capital of France?" in text


def test_cli_generate_stdout(capsys):
    main(
        [
            "generate",
            "--input", str(FIXTURES / "prompt_example.json"),
            "--instructions", "1",
            "--seed", "0",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["prompt"]["instructions"]) == 2


def test_cli_build(tmp_path):
    out_json = tmp_path / "built.json"
    main(
        [
            "build",
            "--popqa", str(FIXTURES / "popqa_sample.csv"),
            "--query-id", "301",
            "--n-demos", "3",
            "--seed", "7",
            "--output", str(out_json),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    prompt = payload["prompt"]
    assert prompt["query"] == "Who is the author of Pride and Prejudice?"
    assert len(prompt["demonstrations"]) == 3
    assert all(d["meta"]["prop"] == "author" for d in prompt["demonstrations"])
    assert prompt["query"] not in [d["question"] for d in prompt["demonstrations"]]


def test_cli_build_context(tmp_path):
    out_json = tmp_path / "out.json"
    out_text = tmp_path / "out.txt"
    main(
        [
            "build-context",
            "--musique", str(FIXTURES / "musique_sample.jsonl"),
            "--query-id", "2hop__101_201",
            "--output", str(out_json),
            "--text", str(out_text),
        ]
    )
    prompt = json.loads(out_json.read_text(encoding="utf-8"))["prompt"]
    assert len(prompt["context"]) == 5
    assert prompt["context"][0]["is_supporting"] is True
    assert prompt["meta"]["n_hops"] == 2
    assert "[1] Eiffel Tower" in out_text.read_text(encoding="utf-8")


def test_cli_build_context_oracle(tmp_path):
    out_json = tmp_path / "out.json"
    main(
        [
            "build-context",
            "--musique", str(FIXTURES / "musique_sample.jsonl"),
            "--query-id", "3hop1__103_203_303",
            "--no-distractors",
            "--output", str(out_json),
        ]
    )
    prompt = json.loads(out_json.read_text(encoding="utf-8"))["prompt"]
    assert [c["is_supporting"] for c in prompt["context"]] == [True, True, True]


def test_cli_build_context_unknown_id():
    with pytest.raises(SystemExit):
        main(
            [
                "build-context",
                "--musique", str(FIXTURES / "musique_sample.jsonl"),
                "--query-id", "9hop__nope",
            ]
        )
