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
(:func:`query_preserving_compressor`, :func:`gold_containment_inference`) so a
full run works with no models and no network; both are explicitly named as
baselines, not neural methods.
"""

from __future__ import annotations

import re
from bisect import bisect_left
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
    """Naive truncation: keep the first ``ratio`` of tokens, whatever they are.

    Kept for reference, but **not** the default: on a context prompt the
    question sits at the very end, so at low rates this throws the question
    away and every condition scores zero for the same uninformative reason.
    Use :func:`query_preserving_compressor` unless you specifically want the
    unguarded baseline.
    """

    tokens = text.split()
    keep = max(1, round(len(tokens) * target_ratio))
    return " ".join(tokens[:keep])


QUERY_MARKER = "\n\nQ: "


def query_preserving_compressor(text: str, target_ratio: float) -> str:
    """Truncation baseline that spends its budget on the context only.

    Real prompt compressors (LLMLingua and successors) compress the context and
    leave the question standing -- a prompt without its question is not a
    shorter prompt, it is a different task. This baseline does the same: the
    final query block survives whole, and the token budget is applied to
    everything before it.

    The split relies on the rendering of ``redundanzgenerator.render_prompt``
    (the query block starts at the last ``\\n\\nQ: ``). Text without that marker
    falls back to plain truncation. When the query alone already exceeds the
    budget the query still wins, so the achieved ratio can exceed the target --
    that is visible in ``achieved_ratio`` rather than silently hidden.
    """

    head, sep, tail = text.rpartition(QUERY_MARKER)
    if not sep:
        return head_ratio_compressor(text, target_ratio)

    query_block = sep + tail
    budget = round(len(text.split()) * target_ratio) - len(query_block.split())
    if budget <= 0:
        return query_block.lstrip("\n")
    return " ".join(head.split()[:budget]) + query_block


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


def _positions(tokens: list[str]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, token in enumerate(tokens):
        index.setdefault(token, []).append(i)
    return index


def _match_in_order(
    passage_tokens: list[str], index: dict[str, list[int]], cursor: int
) -> tuple[float, int]:
    """Share of a passage that survives *as an ordered subsequence*.

    Order matters. Set membership would ask "does this word appear anywhere in
    the compressed prompt", which every function word answers yes to -- on a
    20-passage context that reports ~95 % retention at a 20 % keep rate, i.e.
    it measures vocabulary overlap, not survival. Matching in order, each token
    consumed at most once, tracks what compression actually did: it deletes
    tokens and preserves the order of the rest.
    """

    matched = 0
    for token in passage_tokens:
        places = index.get(token)
        if not places:
            continue
        i = bisect_left(places, cursor)
        if i < len(places):
            matched += 1
            cursor = places[i] + 1
    return matched / len(passage_tokens), cursor


def evidence_retention(prompt: Any, compressed_text: str) -> dict[str, float]:
    """How much of the gold evidence a compressed prompt still contains.

    This is what the MuSiQue labels buy. Whether the answer survived is a
    coarse signal; the share of *supporting* passage material that survived
    says whether a compressor preserved the right thing. The distractor share
    sits next to it -- a compressor that keeps evidence and noise at the same
    rate is not selecting, only shortening.

    Redundant copies are folded back onto their original: each piece of
    evidence counts once, scored by its best-surviving copy. Otherwise the
    ``passages`` condition would be measured against a larger denominator than
    the other three and the numbers would not be comparable across conditions.

    Assumes the compressed text is a subsequence of the prompt (true for
    truncation and for token-dropping compressors such as LLMLingua). An
    abstractive compressor that rewrites text would be understated here.
    """

    index = _positions(_tokens(compressed_text))
    cursor = 0
    supporting: dict[Any, tuple[float, int]] = {}
    distractors: dict[Any, tuple[float, int]] = {}

    for position, passage in enumerate(prompt.context):
        tokens = _tokens(passage.text)
        if not tokens:
            continue
        share, cursor = _match_in_order(tokens, index, cursor)
        if passage.is_supporting is None:
            continue
        bucket = supporting if passage.is_supporting else distractors
        key = passage.meta.get("redundant_copy_of", passage.meta.get("idx", position))
        best = bucket.get(key)
        if best is None or share > best[0]:
            bucket[key] = (share, len(tokens))

    def weighted(bucket: dict[Any, tuple[float, int]]) -> float | None:
        if not bucket:
            return None
        total = sum(n for _, n in bucket.values())
        return round(sum(s * n for s, n in bucket.values()) / total, 4)

    supporting_share = weighted(supporting)
    distractor_share = weighted(distractors)

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
    compress: Compressor = query_preserving_compressor,
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
