"""End-to-end integration test: Redundanzgenerator -> PipelineStore.

Skipped automatically where the sibling ``redundanzgenerator`` package is not
installed, so the core storage suite stays independent of it.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("redundanzgenerator")

from pipelinestore import PipelineStore  # noqa: E402
from pipelinestore.redundancy_pipeline import (  # noqa: E402
    RedundancyCounts,
    evidence_retention,
    head_ratio_compressor,
    query_preserving_compressor,
    run_pipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"
RATES = [0.2, 0.4, 0.6, 0.8]


def _run(tmp_path, query_ids, **kwargs):
    store = PipelineStore(tmp_path, "it")
    return store, run_pipeline(
        store,
        musique_source=FIXTURES / "musique.jsonl",
        query_ids=query_ids,
        compression_rates=RATES,
        counts=RedundancyCounts(passages=2, demonstrations=2, instructions=1),
        seed=42,
        **kwargs,
    )


def test_single_prompt_yields_16_compressates(tmp_path):
    store, _ = _run(tmp_path, ["2hop__101_201"])
    rows = store.build_rows()
    assert len(rows) == 16  # 4 variants x 4 rates
    variants = {r.redundancy_type for r in rows}
    assert variants == {"baseline", "passages", "demonstrations", "instructions"}


def test_baseline_adds_no_redundancy_but_others_do(tmp_path):
    store, _ = _run(tmp_path, ["2hop__101_201"])
    by_type = {}
    for r in store.build_rows():
        by_type.setdefault(r.redundancy_type, r)
    assert by_type["baseline"].redundancy_added_tokens == 0
    assert by_type["passages"].redundancy_added_tokens > 0
    assert by_type["demonstrations"].redundancy_added_tokens > 0
    assert by_type["instructions"].redundancy_added_tokens > 0


def test_raw_prompt_carries_context_and_provenance(tmp_path):
    _run(tmp_path, ["2hop__101_201"])
    raw = json.loads(
        (tmp_path / "it" / "01_raw" / "p-2hop-101-201.json").read_text(encoding="utf-8")
    )
    assert raw["source"] == "musique"
    assert raw["source_category"] == "2"
    assert len(raw["structured"]["context"]) == 4
    assert raw["meta"]["n_supporting"] == 2
    assert raw["meta"]["gold_answers"][0] == "France"
    assert "Context:" in raw["text"]


def test_real_reports_are_stored(tmp_path):
    _run(tmp_path, ["2hop__101_201"])
    variant = json.loads(
        (tmp_path / "it" / "02_redundant" / "p-2hop-101-201" / "passages.json").read_text(
            encoding="utf-8"
        )
    )
    strategies = [entry["strategy"] for entry in variant["redundancy_report"]]
    assert strategies == ["passages"]
    assert len(variant["redundancy_report"][0]["inserted"]) == 2


def test_manifest_has_one_row_per_compressate(tmp_path):
    _, (jsonl, _) = _run(tmp_path, ["2hop__101_201", "3hop1__103_203_303"])
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32  # 2 prompts x 16


def test_evidence_metrics_reach_the_manifest(tmp_path):
    _, (_, csv_path) = _run(tmp_path, ["2hop__101_201"])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "metric_supporting_retained",
        "metric_distractor_retained",
        "metric_evidence_selectivity",
    ):
        assert column in header


def test_higher_rates_retain_more_evidence(tmp_path):
    store, _ = _run(tmp_path, ["2hop__101_201"])
    baseline = {
        r.target_ratio: r.metrics["supporting_retained"]
        for r in store.build_rows()
        if r.redundancy_type == "baseline"
    }
    assert baseline[0.2] <= baseline[0.4] <= baseline[0.6] <= baseline[0.8]


def test_evidence_retention_counts_supporting_and_distractor_separately():
    from redundanzgenerator import ContextPassage, FewShotPrompt

    prompt = FewShotPrompt(
        instructions=[],
        demonstrations=[],
        query="Q?",
        context=[
            ContextPassage(text="alpha beta", is_supporting=True),
            ContextPassage(text="gamma delta", is_supporting=False),
        ],
    )
    metrics = evidence_retention(prompt, "alpha beta gamma")
    assert metrics["supporting_retained"] == 1.0
    assert metrics["distractor_retained"] == 0.5
    assert metrics["evidence_selectivity"] == 0.5


def test_evidence_retention_without_labels_reports_nothing():
    from redundanzgenerator import ContextPassage, FewShotPrompt

    prompt = FewShotPrompt(
        instructions=[],
        demonstrations=[],
        query="Q?",
        context=[ContextPassage(text="unlabelled passage")],
    )
    assert evidence_retention(prompt, "unlabelled") == {}


def test_head_ratio_compressor_hits_target():
    text = " ".join(str(i) for i in range(100))
    assert len(head_ratio_compressor(text, 0.2).split()) == 20
    assert len(head_ratio_compressor(text, 0.5).split()) == 50


def _prompt_text(context_tokens: int) -> str:
    context = " ".join(f"w{i}" for i in range(context_tokens))
    return f"Instruction here.\n\nContext:\n[1] Title\n{context}\n\nQ: The question?\nA:"


def test_query_preserving_compressor_keeps_the_question():
    """The naive baseline cuts the question off; this one must not."""
    text = _prompt_text(500)
    assert "Q: The question?" not in head_ratio_compressor(text, 0.2)
    compressed = query_preserving_compressor(text, 0.2)
    assert compressed.endswith("Q: The question?\nA:")
    assert "Instruction here." in compressed


def test_query_preserving_compressor_hits_target_overall():
    text = _prompt_text(500)
    total = len(text.split())
    for rate in (0.2, 0.5, 0.8):
        kept = len(query_preserving_compressor(text, rate).split())
        assert abs(kept - round(total * rate)) <= 1


def test_query_preserving_compressor_keeps_query_even_over_budget():
    """A prompt is not a prompt without its question, whatever the budget."""
    text = _prompt_text(4)
    compressed = query_preserving_compressor(text, 0.05)
    assert compressed.startswith("Q: The question?")


def test_query_preserving_compressor_falls_back_without_marker():
    text = " ".join(str(i) for i in range(100))
    assert len(query_preserving_compressor(text, 0.3).split()) == 30


def test_retention_is_order_aware_not_vocabulary_overlap():
    """Set membership would score a truncated context as almost fully retained."""
    from redundanzgenerator import ContextPassage, FewShotPrompt

    # Same six words in both passages, different order: a set-based measure
    # cannot tell them apart, an order-aware one can.
    prompt = FewShotPrompt(
        instructions=[],
        demonstrations=[],
        query="Q?",
        context=[
            ContextPassage(text="alpha beta gamma", is_supporting=True, meta={"idx": 0}),
            ContextPassage(text="gamma beta alpha", is_supporting=False, meta={"idx": 1}),
        ],
    )
    metrics = evidence_retention(prompt, "alpha beta gamma")
    assert metrics["supporting_retained"] == 1.0
    assert metrics["distractor_retained"] < 1.0


def test_retention_falls_when_evidence_is_cut():
    from redundanzgenerator import ContextPassage, FewShotPrompt

    prompt = FewShotPrompt(
        instructions=[],
        demonstrations=[],
        query="Q?",
        context=[ContextPassage(text="one two three four", is_supporting=True, meta={"idx": 0})],
    )
    assert evidence_retention(prompt, "one two three four")["supporting_retained"] == 1.0
    assert evidence_retention(prompt, "one two")["supporting_retained"] == 0.5
    assert evidence_retention(prompt, "")["supporting_retained"] == 0.0


def test_redundant_copies_are_folded_onto_their_original():
    """Each piece of evidence counts once, so conditions stay comparable."""
    from redundanzgenerator import ContextPassage, FewShotPrompt

    original = ContextPassage(text="one two three four", is_supporting=True, meta={"idx": 0})
    copy = ContextPassage(
        text="one two three four",
        is_supporting=True,
        meta={"idx": 0, "redundant_copy_of": 0},
    )
    with_copy = FewShotPrompt(
        instructions=[], demonstrations=[], query="Q?", context=[original, copy]
    )
    without = FewShotPrompt(
        instructions=[], demonstrations=[], query="Q?", context=[original]
    )
    # The surviving copy carries the evidence, so the score matches the
    # single-passage prompt instead of being halved by the larger denominator.
    text = "one two three four"
    assert (
        evidence_retention(with_copy, text)["supporting_retained"]
        == evidence_retention(without, text)["supporting_retained"]
        == 1.0
    )
