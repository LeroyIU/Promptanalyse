"""End-to-end integration test: Redundanzgenerator -> PipelineStore.

Skipped automatically where the sibling ``redundanzgenerator`` package is not
installed, so the core storage suite stays independent of it.
"""

from pathlib import Path

import pytest

pytest.importorskip("redundanzgenerator")

from pipelinestore import PipelineStore  # noqa: E402
from pipelinestore.redundancy_pipeline import (  # noqa: E402
    RedundancyCounts,
    head_ratio_compressor,
    run_pipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"
RATES = [0.2, 0.4, 0.6, 0.8]


def _run(tmp_path, query_ids):
    store = PipelineStore(tmp_path, "it")
    return store, run_pipeline(
        store,
        popqa_source=FIXTURES / "popqa.csv",
        popqa_tp_source=FIXTURES / "popqa_tp.csv",
        query_ids=query_ids,
        compression_rates=RATES,
        counts=RedundancyCounts(lexical=2, demonstrations=2, instructions=1),
        n_demos=2,
        seed=42,
    )


def test_single_prompt_yields_16_compressates(tmp_path):
    store, (jsonl, _) = _run(tmp_path, ["101"])
    rows = store.build_rows()
    assert len(rows) == 16  # 4 variants x 4 rates
    variants = {r.redundancy_type for r in rows}
    assert variants == {"baseline", "lexical", "demonstrations", "instructions"}


def test_baseline_adds_no_redundancy_but_others_do(tmp_path):
    store, _ = _run(tmp_path, ["101"])
    by_type = {}
    for r in store.build_rows():
        by_type.setdefault(r.redundancy_type, r)
    assert by_type["baseline"].redundancy_added_tokens == 0
    assert by_type["lexical"].redundancy_added_tokens > 0
    assert by_type["demonstrations"].redundancy_added_tokens > 0
    assert by_type["instructions"].redundancy_added_tokens > 0


def test_real_reports_are_stored(tmp_path):
    _run(tmp_path, ["101"])
    import json

    lex = json.loads(
        (tmp_path / "it" / "02_redundant" / "p-101" / "lexical.json").read_text()
    )
    strategies = [entry["strategy"] for entry in lex["redundancy_report"]]
    assert strategies == ["lexical"]


def test_manifest_has_one_row_per_compressate(tmp_path):
    store, (jsonl, csv_path) = _run(tmp_path, ["101", "102"])
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32  # 2 prompts x 16


def test_head_ratio_compressor_hits_target():
    text = " ".join(str(i) for i in range(100))
    assert len(head_ratio_compressor(text, 0.2).split()) == 20
    assert len(head_ratio_compressor(text, 0.5).split()) == 50
