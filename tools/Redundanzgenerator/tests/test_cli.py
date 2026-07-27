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
            "--passages", "2",
            "--demos", "2",
            "--instructions", "1",
            "--musique", str(FIXTURES / "musique_pool.jsonl"),
            "--seed", "42",
            "--output", str(out_json),
            "--text", str(out_text),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    prompt = payload["prompt"]
    assert len(prompt["context"]) == 6
    assert len(prompt["demonstrations"]) == 3
    assert len(prompt["instructions"]) == 2
    assert len(payload["redundancy_report"]) == 3
    text = out_text.read_text(encoding="utf-8")
    assert "Q: In which country is the city where the Eiffel Tower stands?" in text


def test_cli_generate_passage_modes(tmp_path):
    out_json = tmp_path / "out.json"
    main(
        [
            "generate",
            "--input", str(FIXTURES / "prompt_example.json"),
            "--passages", "1",
            "--passage-mode", "restate",
            "--passage-position", "append",
            "--seed", "0",
            "--output", str(out_json),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    copy = payload["prompt"]["context"][-1]
    assert copy["meta"]["redundancy_mode"] == "restate"
    assert payload["redundancy_report"][0]["mode"] == "restate"


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
    out_json = tmp_path / "out.json"
    out_text = tmp_path / "out.txt"
    main(
        [
            "build",
            "--musique", str(FIXTURES / "musique_pool.jsonl"),
            "--query-id", "2hop__101_201",
            "--output", str(out_json),
            "--text", str(out_text),
        ]
    )
    prompt = json.loads(out_json.read_text(encoding="utf-8"))["prompt"]
    assert len(prompt["context"]) == 4
    assert prompt["context"][0]["is_supporting"] is True
    assert prompt["meta"]["n_hops"] == 2
    assert "[1] Eiffel Tower" in out_text.read_text(encoding="utf-8")


def test_cli_build_oracle_context(tmp_path):
    out_json = tmp_path / "out.json"
    main(
        [
            "build",
            "--musique", str(FIXTURES / "musique_pool.jsonl"),
            "--query-id", "3hop1__103_203_303",
            "--no-distractors",
            "--output", str(out_json),
        ]
    )
    prompt = json.loads(out_json.read_text(encoding="utf-8"))["prompt"]
    assert [c["is_supporting"] for c in prompt["context"]] == [True, True, True]


def test_cli_build_unknown_id():
    with pytest.raises(SystemExit):
        main(
            [
                "build",
                "--musique", str(FIXTURES / "musique_pool.jsonl"),
                "--query-id", "9hop__nope",
            ]
        )
