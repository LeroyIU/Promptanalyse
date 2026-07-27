"""End-to-end driver: Redundanzgenerator -> PipelineStore.

Wires the real ``redundanzgenerator`` (stages 1-2) into the storage layer and
runs the full fan-out per raw prompt::

    baseline + 3 redundancy types  x  4 compression rates  =  16 compressates

Prompts are built from MuSiQue, so every prompt carries its context: ~20
Wikipedia passages, of which 2-4 are labelled as supporting the answer. The
three redundancy types target the three parts such a prompt has -- its context
passages, its demonstrations, its instructions.

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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from . import ids
from .models import Compressate, InferenceResult, RawPrompt, RedundantVariant
from .store import PipelineStore, TokenCounter, whitespace_tokens

BASELINE = "baseline"
REDUNDANCY_TYPES = (BASELINE, "passages", "demonstrations", "instructions")

DEFAULT_INSTRUCTION = (
    "Answer the question using only the passages provided. "
    "Respond with a short factual answer."
)


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


# --- evidence retention ---------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def evidence_retention(prompt: Any, compressed_text: str) -> dict[str, float]:
    """How much of the gold evidence a compressed prompt still contains.

    This is what the MuSiQue labels buy. Whether the answer survived is a
    coarse signal; the share of *supporting* passage tokens that survived says
    whether a compressor preserved the right material. The distractor share is
    reported next to it -- a compressor that keeps evidence and noise at the
    same rate is not selecting, only shortening.

    Measured as bag-of-token containment, because compression deletes tokens
    inside passages rather than dropping passages whole.
    """

    kept = set(_tokens(compressed_text))

    def share(passages: list[Any]) -> float | None:
        wanted = [t for p in passages for t in _tokens(p.text)]
        if not wanted:
            return None
        return round(sum(t in kept for t in wanted) / len(wanted), 4)

    supporting_share = share([p for p in prompt.context if p.is_supporting])
    distractor_share = share([p for p in prompt.context if p.is_supporting is False])

    metrics: dict[str, float] = {}
    if supporting_share is not None:
        metrics["supporting_retained"] = supporting_share
    if distractor_share is not None:
        metrics["distractor_retained"] = distractor_share
    if supporting_share is not None and distractor_share is not None:
        # > 0 means the compressor favoured evidence over noise.
        metrics["evidence_selectivity"] = round(supporting_share - distractor_share, 4)
    return metrics


# --- configuration --------------------------------------------------------
@dataclass
class RedundancyCounts:
    """How much of each redundancy type to inject (per variant)."""

    passages: int = 3          # redundant copies of context passages
    demonstrations: int = 3    # extra demonstrations with the same hop count
    instructions: int = 2      # restated instructions
    passage_mode: str = "duplicate"
    passage_target: str = "supporting"
    passage_position: str = "interleave"
    demonstration_context: bool = False
    instruction_position: str = "both"


def build_raw_prompt(
    musique: Any,
    query_id: str,
    *,
    n_demos: int,
    instruction: str,
    seed: int,
    include_distractors: bool = True,
    demo_context: bool = False,
) -> Any:
    """Build the base context prompt for one MuSiQue id (same logic as the CLI)."""

    return musique.build_prompt(
        query_id,
        instruction=instruction,
        n_demos=n_demos,
        include_distractors=include_distractors,
        demo_context=demo_context,
        seed=seed,
    )


def _variant_config(redundancy_type: str, counts: RedundancyCounts, seed: int) -> Any:
    """RedundancyConfig that activates exactly one redundancy type (or none)."""

    from redundanzgenerator import RedundancyConfig

    kwargs: dict[str, Any] = {"seed": seed}
    if redundancy_type == "passages":
        kwargs["n_passages"] = counts.passages
        kwargs["passage_mode"] = counts.passage_mode
        kwargs["passage_target"] = counts.passage_target
        kwargs["passage_position"] = counts.passage_position
    elif redundancy_type == "demonstrations":
        kwargs["n_demonstrations"] = counts.demonstrations
        kwargs["demonstration_context"] = counts.demonstration_context
    elif redundancy_type == "instructions":
        kwargs["n_instructions"] = counts.instructions
        kwargs["instruction_position"] = counts.instruction_position
    return RedundancyConfig(**kwargs)


def run_pipeline(
    store: PipelineStore,
    *,
    musique_source: str | Path,
    query_ids: list[str],
    compression_rates: list[float],
    compress: Compressor = head_ratio_compressor,
    infer: Inference | None = None,
    counts: RedundancyCounts | None = None,
    n_demos: int = 0,
    include_distractors: bool = True,
    demo_context: bool = False,
    instruction: str = DEFAULT_INSTRUCTION,
    seed: int = 42,
    inference_model: str = "offline-gold-containment",
    token_counter: TokenCounter = whitespace_tokens,
    description: str = "Redundanz-/Kompressions-Pipeline auf MuSiQue.",
) -> tuple[Path, Path]:
    """Run the full pipeline for every query id and (re)build the manifest.

    For each query: build the raw context prompt, derive the 4 variants
    (baseline + the 3 redundancy types via the real generator), compress each at
    every rate, and score it. Returns the manifest paths.
    """

    from redundanzgenerator import MuSiQueLoader, render_prompt

    counts = counts or RedundancyCounts()
    if infer is None:
        infer = gold_containment_inference(inference_model)
    musique = MuSiQueLoader(musique_source)

    store.store_experiment(
        {
            "description": description,
            "musique_source": str(musique_source),
            "query_ids": query_ids,
            "redundancy_types": list(REDUNDANCY_TYPES),
            "redundancy_counts": vars(counts),
            "compression_rates": compression_rates,
            "compressor": getattr(compress, "__name__", str(compress)),
            "inference_model": inference_model,
            "n_demos": n_demos,
            "include_distractors": include_distractors,
            "demo_context": demo_context,
            "instruction": instruction,
            "seed": seed,
        }
    )

    for query_id in query_ids:
        base = build_raw_prompt(
            musique,
            query_id,
            n_demos=n_demos,
            instruction=instruction,
            seed=seed,
            include_distractors=include_distractors,
            demo_context=demo_context,
        )
        gold = musique.answers_of(musique.by_id(query_id) or {})
        pid = ids.prompt_id(query_id)
        raw_text = render_prompt(base)
        raw = RawPrompt(
            prompt_id=pid,
            text=raw_text,
            n_tokens=token_counter(raw_text),
            source="musique",
            source_id=str(query_id),
            source_category=str(base.meta.get("n_hops") or ""),
            structured=base.to_dict(),
            meta={
                "gold_answers": gold,
                "n_supporting": base.meta.get("n_supporting"),
                "n_distractors": base.meta.get("n_distractors"),
            },
        )
        store.store_raw(raw)

        for rtype in REDUNDANCY_TYPES:
            variant, variant_text, report = _make_variant(
                rtype, base, musique, counts, seed, render_prompt
            )
            variant_tokens = token_counter(variant_text)
            store.store_redundant(
                RedundantVariant(
                    prompt_id=pid,
                    redundancy_type=rtype,
                    variant_id=ids.variant_id(pid, rtype),
                    text=variant_text,
                    n_tokens=variant_tokens,
                    redundancy_report=report,
                    generator="redundanzgenerator@0.2.0",
                    seed=seed,
                )
            )

            for rate in compression_rates:
                ctext = compress(variant_text, rate)
                comp = Compressate(
                    prompt_id=pid,
                    redundancy_type=rtype,
                    target_ratio=rate,
                    compressate_id=ids.compressate_id(pid, rtype, rate),
                    text=ctext,
                    n_tokens=token_counter(ctext),
                    n_tokens_source=variant_tokens,
                    compressor=getattr(compress, "__name__", str(compress)),
                    seed=seed,
                )
                store.store_compressate(comp)

                outcome = infer(ctext, gold)
                metrics = dict(outcome.metrics)
                metrics.update(evidence_retention(variant, ctext))
                store.store_inference(
                    InferenceResult(
                        compressate_id=comp.compressate_id,
                        model=inference_model,
                        output=outcome.output,
                        gold_answer=gold[0] if gold else None,
                        is_correct=outcome.is_correct,
                        metrics=metrics,
                        seed=seed,
                    )
                )

    return store.build_manifest()


def _make_variant(
    rtype: str,
    base: Any,
    musique: Any,
    counts: RedundancyCounts,
    seed: int,
    render_prompt: Callable[[Any], str],
) -> tuple[Any, str, list[dict[str, Any]]]:
    """Return ``(prompt, rendered_text, report)`` for one variant condition."""

    if rtype == BASELINE:
        return base, render_prompt(base), []

    from redundanzgenerator import RedundancyGenerator

    generator = RedundancyGenerator.from_config(
        _variant_config(rtype, counts, seed), musique=musique
    )
    variant, report = generator.generate(base)
    return variant, render_prompt(variant), report
