"""Render a FewShotPrompt to plain prompt text."""

from __future__ import annotations

from .models import FewShotPrompt


def render_prompt(
    prompt: FewShotPrompt,
    question_prefix: str = "Q: ",
    answer_prefix: str = "A: ",
) -> str:
    """Q:/A:-style rendering.

    Paraphrases are rendered as additional question lines directly after the
    question they duplicate, so lexical redundancy appears as the same
    question asked several times in different words. The final answer prefix
    is left empty for the model to complete.
    """
    blocks: list[str] = []
    if prompt.instructions:
        blocks.append("\n".join(prompt.instructions))

    for demo in prompt.demonstrations:
        lines = [f"{question_prefix}{demo.question}"]
        lines += [f"{question_prefix}{p}" for p in demo.paraphrases]
        lines.append(f"{answer_prefix}{demo.answer}")
        blocks.append("\n".join(lines))

    query_lines = [f"{question_prefix}{prompt.query}"]
    query_lines += [f"{question_prefix}{p}" for p in prompt.query_paraphrases]
    if prompt.trailing_instructions:
        query_lines.append("\n".join(prompt.trailing_instructions))
    query_lines.append(answer_prefix.rstrip())
    blocks.append("\n".join(query_lines))

    return "\n\n".join(blocks)
