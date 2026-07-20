"""End-to-end driver: Redundanzgenerator -> PipelineStore.

Wires the real ``redundanzgenerator`` (stages 1-2) into the storage layer and
runs the full fan-out per raw prompt::

    baseline + 3 redundancy types  x  4 compression rates  =  16 compressates

``redundanzgenerator`` is imported lazily so the core storage package stays
dependency-free; install it via the ``redundancy`` extra to use this module.

The two stages that are *not* prompt construction -- compression and inference
-- are injected as callables so real methods (e.g. LLMLingua, the Claude API)
drop straight in. Honest, offline defaults are provided
(:func:`head_ratio_compressor`, :func:`gold_containment_inference`) so a full
run works with no models and no network; both are explicitly named as baselines,
not neural methods.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from . import ids
from .models import Compressate, InferenceResult, RawPrompt, RedundantVariant
from .store import PipelineStore, TokenCounter, whitespace_tokens

BASELINE = "baseline"
REDUNDANCY_TYPES = (BASELINE, "lexical", "demonstrations", "instructions")

DEFAULT_INSTRUCTION = "Answer the following question with a short factual answer."


# --- injectable stage interfaces -----------------------------------------
class Compressor(Protocol):
    """Compress ``text`` down to (about) ``target_ratio`` of its tokens."""

    def __call__(self, text: str, target_ratio: float) -> str: ...


@dataclass
class InferenceOutcome:
    """What an inference backend reports for one compressate."""

    output: str
    is_correct: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Inference(Protocol):
    """Run a model on ``text`` and score it against ``gold_answers``."""

    def __call__(self, text: str, gold_answers: list[str]) -> InferenceOutcome: ...


# --- offline default backends --------------------------------------------
def head_ratio_compressor(text: str, target_ratio: float) -> str:
    """Deterministic truncation baseline: keep the first ``ratio`` of tokens.

    A reproducible, model-free stand-in. Replace with the real compressor in a
    production run -- the storage calls do not change.
    """

    tokens = text.split()
    keep = max(1, round(len(tokens) * target_ratio))
    return " ".join(tokens[:keep])


def gold_containment_inference(
    model_name: str = "offline-gold-containment",
) -> Inference:
    """Offline scoring proxy: correct iff a gold answer survives in the text.

    Not a real model -- it measures whether compression kept the answer-bearing
    span, which is a useful sanity signal on its own. Swap in a Claude-API
    backend with the same signature for real inference.
    """

    def _infer(text: str, gold_answers: list[str]) -> InferenceOutcome:
        haystack = text.lower()
        hit = next((g for g in gold_answers if g and g.lower() in haystack), None)
        return InferenceOutcome(
            output=hit or "",
            is_correct=hit is not None,
            metrics={"gold_retained": float(hit is not None)},
        )

    _infer.__name__ = model_name.replace("-", "_")
    return _infer


# --- configuration --------------------------------------------------------
@dataclass
class RedundancyCounts:
    """How much of each redundancy type to inject (per variant)."""

    lexical: int = 3          # query paraphrases from PopQA-TP
    demonstrations: int = 3   # extra same-category demonstrations from PopQA
    instructions: int = 2     # restated instructions
    instruction_position: str = "both"


def build_raw_prompt(
    popqa: Any,
    query_id: str,
    *,
    n_demos: int,
    instruction: str,
    seed: int,
) -> Any:
    """Build the base few-shot prompt for one query id (same logic as the CLI)."""

    from redundanzgenerator import Demonstration, FewShotPrompt

    query_row = popqa.by_id(query_id)
    if query_row is None:
        raise KeyError(f"no PopQA record with id {query_id!r}")
    rng = random.Random(seed)
    demo_rows = popqa.by_category(
        str(query_row.get("prop", "")),
        exclude_questions={str(query_row["question"])},
    )
    rng.shuffle(demo_rows)
    demonstrations = [
        Demonstration(
            question=str(row["question"]),
            answer=popqa.answer_of(row),
            meta={"id": row.get("id"), "prop": row.get("prop")},
        )
        for row in demo_rows[:n_demos]
    ]
    return FewShotPrompt(
        instructions=[instruction],
        demonstrations=demonstrations,
        query=str(query_row["question"]),
        meta={"id": query_row.get("id"), "prop": query_row.get("prop")},
    )


def _variant_config(redundancy_type: str, counts: RedundancyCounts, seed: int) -> Any:
    """RedundancyConfig that activates exactly one redundancy type (or none)."""

    from redundanzgenerator import RedundancyConfig

    kwargs: dict[str, Any] = {"seed": seed}
    if redundancy_type == "lexical":
        kwargs["n_paraphrases"] = counts.lexical
    elif redundancy_type == "demonstrations":
        kwargs["n_demonstrations"] = counts.demonstrations
    elif redundancy_type == "instructions":
        kwargs["n_instructions"] = counts.instructions
        kwargs["instruction_position"] = counts.instruction_position
    return RedundancyConfig(**kwargs)


def run_pipeline(
    store: PipelineStore,
    *,
    popqa_source: str | Path,
    popqa_tp_source: str | Path,
    query_ids: list[str],
    compression_rates: list[float],
    compress: Compressor = head_ratio_compressor,
    infer: Inference | None = None,
    counts: RedundancyCounts | None = None,
    n_demos: int = 4,
    instruction: str = DEFAULT_INSTRUCTION,
    seed: int = 42,
    inference_model: str = "offline-gold-containment",
    token_counter: TokenCounter = whitespace_tokens,
) -> tuple[Path, Path]:
    """Run the full pipeline for every query id and (re)build the manifest.

    For each query: build the raw prompt, derive the 4 variants (baseline + the
    3 redundancy types via the real generator), compress each at every rate, and
    -- if ``infer`` is given -- score it. Returns the manifest paths.
    """

    from redundanzgenerator import PopQALoader, PopQATPLoader, render_prompt

    counts = counts or RedundancyCounts()
    if infer is None:
        infer = gold_containment_inference(inference_model)
    popqa = PopQALoader(popqa_source)
    popqa_tp = PopQATPLoader(popqa_tp_source)

    store.store_experiment(
        {
            "description": "Redundanz-/Kompressions-Pipeline auf PopQA.",
            "popqa_source": str(popqa_source),
            "popqa_tp_source": str(popqa_tp_source),
            "query_ids": query_ids,
            "redundancy_types": list(REDUNDANCY_TYPES),
            "redundancy_counts": vars(counts),
            "compression_rates": compression_rates,
            "compressor": getattr(compress, "__name__", str(compress)),
            "inference_model": inference_model,
            "n_demos": n_demos,
            "instruction": instruction,
            "seed": seed,
        }
    )

    for query_id in query_ids:
        base = build_raw_prompt(
            popqa, query_id, n_demos=n_demos, instruction=instruction, seed=seed
        )
        gold = _gold_answers(popqa, query_id)
        pid = ids.prompt_id(query_id)
        raw_text = render_prompt(base)
        raw = RawPrompt(
            prompt_id=pid,
            text=raw_text,
            n_tokens=token_counter(raw_text),
            source="popqa",
            source_id=str(query_id),
            source_prop=str(base.meta.get("prop") or ""),
            structured=base.to_dict(),
            meta={"gold_answers": gold},
        )
        store.store_raw(raw)

        for rtype in REDUNDANCY_TYPES:
            variant_text, report = _make_variant(
                rtype, base, popqa, popqa_tp, counts, seed, render_prompt
            )
            variant = RedundantVariant(
                prompt_id=pid,
                redundancy_type=rtype,
                variant_id=ids.variant_id(pid, rtype),
                text=variant_text,
                n_tokens=token_counter(variant_text),
                redundancy_report=report,
                generator="redundanzgenerator@0.1.0",
                seed=seed,
            )
            store.store_redundant(variant)

            for rate in compression_rates:
                ctext = compress(variant_text, rate)
                comp = Compressate(
                    prompt_id=pid,
                    redundancy_type=rtype,
                    target_ratio=rate,
                    compressate_id=ids.compressate_id(pid, rtype, rate),
                    text=ctext,
                    n_tokens=token_counter(ctext),
                    n_tokens_source=variant.n_tokens,
                    compressor=getattr(compress, "__name__", str(compress)),
                    seed=seed,
                )
                store.store_compressate(comp)

                outcome = infer(ctext, gold)
                store.store_inference(
                    InferenceResult(
                        compressate_id=comp.compressate_id,
                        model=inference_model,
                        output=outcome.output,
                        gold_answer=gold[0] if gold else None,
                        is_correct=outcome.is_correct,
                        metrics=outcome.metrics,
                        seed=seed,
                    )
                )

    return store.build_manifest()


def _make_variant(
    rtype: str,
    base: Any,
    popqa: Any,
    popqa_tp: Any,
    counts: RedundancyCounts,
    seed: int,
    render_prompt: Callable[[Any], str],
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(rendered_text, report)`` for one variant condition."""

    if rtype == BASELINE:
        return render_prompt(base), []

    from redundanzgenerator import RedundancyGenerator

    generator = RedundancyGenerator.from_config(
        _variant_config(rtype, counts, seed), popqa=popqa, popqa_tp=popqa_tp
    )
    variant, report = generator.generate(base)
    return render_prompt(variant), report


def _gold_answers(popqa: Any, query_id: str) -> list[str]:
    from redundanzgenerator.data.popqa import parse_possible_answers

    row = popqa.by_id(query_id) or {}
    answers = parse_possible_answers(row.get("possible_answers"))
    canonical = popqa.answer_of(row)
    if canonical and canonical not in answers:
        answers.insert(0, canonical)
    return answers
