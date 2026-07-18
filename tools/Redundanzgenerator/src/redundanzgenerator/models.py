"""Core data model for few-shot prompts and redundancy configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Demonstration:
    """A single few-shot demonstration (question/answer pair).

    ``paraphrases`` holds additional, semantically equivalent phrasings of the
    question that are rendered alongside it when paraphrase redundancy is
    applied.
    """

    question: str
    answer: str
    paraphrases: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Demonstration":
        return cls(
            question=data["question"],
            answer=data["answer"],
            paraphrases=list(data.get("paraphrases", [])),
            meta=dict(data.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"question": self.question, "answer": self.answer}
        if self.paraphrases:
            out["paraphrases"] = list(self.paraphrases)
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


@dataclass
class FewShotPrompt:
    """A structured few-shot prompt.

    ``instructions`` is a list so that instruction redundancy (repeated or
    rephrased instructions) can be represented explicitly; a plain prompt has
    exactly one entry. ``query_paraphrases`` holds redundant phrasings of the
    final query.
    """

    instructions: list[str]
    demonstrations: list[Demonstration]
    query: str
    query_paraphrases: list[str] = field(default_factory=list)
    trailing_instructions: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FewShotPrompt":
        if "instructions" in data:
            instructions = list(data["instructions"])
        elif "instruction" in data:
            instructions = [data["instruction"]]
        else:
            instructions = []
        return cls(
            instructions=instructions,
            demonstrations=[
                Demonstration.from_dict(d) for d in data.get("demonstrations", [])
            ],
            query=data["query"],
            query_paraphrases=list(data.get("query_paraphrases", [])),
            trailing_instructions=list(data.get("trailing_instructions", [])),
            meta=dict(data.get("meta", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FewShotPrompt":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "instructions": list(self.instructions),
            "demonstrations": [d.to_dict() for d in self.demonstrations],
            "query": self.query,
        }
        if self.query_paraphrases:
            out["query_paraphrases"] = list(self.query_paraphrases)
        if self.trailing_instructions:
            out["trailing_instructions"] = list(self.trailing_instructions)
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


@dataclass
class RedundancyConfig:
    """How much redundancy of each type to inject.

    - ``n_paraphrases``: paraphrases (from PopQA-TP) added to the query.
    - ``paraphrase_demonstrations``: also paraphrase each demonstration question.
    - ``n_demonstrations``: extra demonstrations drawn from PopQA, preferring
      the same relationship category (``prop``) as the query.
    - ``n_instructions``: extra instruction rephrasings (rule-based templates
      or LLM-generated).
    - ``instruction_position``: where extra instructions go ("start", "end",
      or "both", alternating).
    """

    n_paraphrases: int = 0
    paraphrase_demonstrations: bool = False
    n_demonstrations: int = 0
    n_instructions: int = 0
    instruction_position: str = "start"
    seed: int | None = None
