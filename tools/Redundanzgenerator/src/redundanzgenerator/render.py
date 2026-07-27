"""Render a FewShotPrompt to plain prompt text."""

from __future__ import annotations

from .models import ContextPassage, FewShotPrompt


def render_context(
    passages: list[ContextPassage],
    header: str = "Context:",
    numbered: bool = True,
) -> str:
    """Render context passages as a numbered, titled block.

    Numbering is kept because it is what makes a compressed context auditable:
    a passage that disappears is identifiable by its index, so the surviving
    context can be matched back against the gold ``is_supporting`` labels.
    """
    lines: list[str] = [header] if header else []
    for i, passage in enumerate(passages, start=1):
        label = f"[{i}] " if numbered else ""
        if passage.title:
            lines.append(f"{label}{passage.title}")
            lines.append(passage.text)
        else:
            lines.append(f"{label}{passage.text}")
    return "\n".join(lines)


def render_prompt(
    prompt: FewShotPrompt,
    question_prefix: str = "Q: ",
    answer_prefix: str = "A: ",
    context_header: str = "Context:",
) -> str:
    """Q:/A:-style rendering.

    Paraphrases are rendered as additional question lines directly after the
    question they duplicate, so lexical redundancy appears as the same
    question asked several times in different words. Context passages are
    rendered as their own block directly above the question they belong to.
    The final answer prefix is left empty for the model to complete.
    """
    blocks: list[str] = []
    if prompt.instructions:
        blocks.append("\n".join(prompt.instructions))

    for demo in prompt.demonstrations:
        lines: list[str] = []
        if demo.context:
            lines.append(render_context(demo.context, header=context_header))
            lines.append("")
        lines.append(f"{question_prefix}{demo.question}")
        lines += [f"{question_prefix}{p}" for p in demo.paraphrases]
        lines.append(f"{answer_prefix}{demo.answer}")
        blocks.append("\n".join(lines))

    query_lines: list[str] = []
    if prompt.context:
        query_lines.append(render_context(prompt.context, header=context_header))
        query_lines.append("")
    query_lines.append(f"{question_prefix}{prompt.query}")
    query_lines += [f"{question_prefix}{p}" for p in prompt.query_paraphrases]
    if prompt.trailing_instructions:
        query_lines.append("\n".join(prompt.trailing_instructions))
    query_lines.append(answer_prefix.rstrip())
    blocks.append("\n".join(query_lines))

    return "\n\n".join(blocks)
