import csv
import json

from pipelinestore import (
    Compressate,
    InferenceResult,
    PipelineStore,
    RawPrompt,
    RedundantVariant,
    ids,
    whitespace_tokens,
)

REDUNDANCY_TYPES = ["baseline", "lexical", "demonstrations", "instructions"]
RATES = [0.2, 0.4, 0.6, 0.8]


def _populate(store: PipelineStore) -> None:
    raw = RawPrompt(
        prompt_id=ids.prompt_id(101),
        text="What is the capital of France ?",
        n_tokens=whitespace_tokens("What is the capital of France ?"),
        source="popqa",
        source_id="101",
        source_prop="capital",
    )
    store.store_raw(raw)
    for rtype in REDUNDANCY_TYPES:
        vtext = raw.text + " " + " ".join(["redundant"] * 5)
        store.store_redundant(
            RedundantVariant(
                prompt_id=raw.prompt_id,
                redundancy_type=rtype,
                variant_id=ids.variant_id(raw.prompt_id, rtype),
                text=vtext,
                n_tokens=whitespace_tokens(vtext),
            )
        )
        for rate in RATES:
            keep = max(1, round(whitespace_tokens(vtext) * rate))
            ctext = " ".join(vtext.split()[:keep])
            comp = Compressate(
                prompt_id=raw.prompt_id,
                redundancy_type=rtype,
                target_ratio=rate,
                compressate_id=ids.compressate_id(raw.prompt_id, rtype, rate),
                text=ctext,
                n_tokens=whitespace_tokens(ctext),
                n_tokens_source=whitespace_tokens(vtext),
            )
            store.store_compressate(comp)
            store.store_inference(
                InferenceResult(
                    compressate_id=comp.compressate_id,
                    model="demo-model",
                    output="Paris",
                    gold_answer="Paris",
                    is_correct=True,
                    metrics={"f1": 1.0, "latency_ms": 12.0},
                )
            )


def test_fanout_is_16_rows_per_prompt(tmp_path):
    store = PipelineStore(tmp_path, "exp")
    _populate(store)
    rows = store.build_rows()
    assert len(rows) == len(REDUNDANCY_TYPES) * len(RATES) == 16


def test_manifest_csv_has_stable_header_and_metric_columns(tmp_path):
    store = PipelineStore(tmp_path, "exp")
    _populate(store)
    _, csv_path = store.build_manifest()
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    assert header[0] == "experiment_id"
    assert "metric_f1" in header and "metric_latency_ms" in header
    assert len(rows) == 16
    # achieved ratio is populated from token counts
    assert all(r["achieved_ratio"] for r in rows)


def test_manifest_jsonl_roundtrips(tmp_path):
    store = PipelineStore(tmp_path, "exp")
    _populate(store)
    jsonl_path, _ = store.build_manifest()
    lines = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 16
    sample = lines[0]
    assert sample["compressate_id"].count(ids.SEP) == 2
    assert sample["n_tokens_raw"] > 0


def test_paths_are_deterministic(tmp_path):
    store = PipelineStore(tmp_path, "exp")
    p1 = store.compressed_path("p-101", "lexical", 0.4)
    p2 = store.compressed_path("p-101", "lexical", 0.4)
    assert p1 == p2
    assert p1.name == "cr040.json"
