"""Materialise a small, fully worked example of the pipeline storage layout.

Run from the repo root (after ``pip install -e tools/PipelineStore``)::

    python experiments/make_demo_experiment.py

It writes ``experiments/demo-experiment/`` with the full stage tree and a rebuilt
manifest. The "compression" here is a trivial token-truncation stand-in so the
example runs offline with no models -- swap in the real redundancy generator and
compressor in your actual pipeline; the storage calls stay identical.
"""

from __future__ import annotations

from pathlib import Path

from pipelinestore import (
    Compressate,
    InferenceResult,
    PipelineStore,
    RawPrompt,
    RedundantVariant,
    ids,
    whitespace_tokens,
)

# 4 variant conditions x 4 compression rates = 16 compressates per raw prompt.
# "baseline" is the control: the raw prompt with NO added redundancy, compressed
# at the same rates. The other three are the redundancy types of the generator.
REDUNDANCY_TYPES = ["baseline", "lexical", "demonstrations", "instructions"]
COMPRESSION_RATES = [0.2, 0.4, 0.6, 0.8]

RAW_PROMPTS = [
    {"id": 101, "prop": "capital", "text": "Answer shortly. Q: What is the capital of France? A:", "gold": "Paris"},
    {"id": 205, "prop": "author", "text": "Answer shortly. Q: Who wrote Hamlet? A:", "gold": "Shakespeare"},
    {"id": 377, "prop": "capital", "text": "Answer shortly. Q: What is the capital of Japan? A:", "gold": "Tokyo"},
]

FILLER = {
    "baseline": "",  # control: no redundancy added, variant == raw prompt
    "lexical": "Rephrased: state the answer. Again: give the answer.",
    "demonstrations": "Q: Capital of Egypt? A: Cairo. Q: Capital of Italy? A: Rome.",
    "instructions": "Answer shortly. Please answer shortly. Reply with a short factual answer.",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    store = PipelineStore(root, "demo-experiment")
    store.store_experiment(
        {
            "description": "Worked example of the redundancy/compression storage layout.",
            "redundancy_types": REDUNDANCY_TYPES,
            "compression_rates": COMPRESSION_RATES,
            "inference_model": "demo-echo",
            "token_counter": "whitespace (placeholder -- replace with a real tokenizer)",
            "seed": 42,
        }
    )

    for spec in RAW_PROMPTS:
        pid = ids.prompt_id(spec["id"])
        raw = RawPrompt(
            prompt_id=pid,
            text=spec["text"],
            n_tokens=whitespace_tokens(spec["text"]),
            source="popqa",
            source_id=str(spec["id"]),
            source_prop=spec["prop"],
        )
        store.store_raw(raw)

        for rtype in REDUNDANCY_TYPES:
            filler = FILLER[rtype]
            vtext = raw.text + ("  " + filler if filler else "")
            variant = RedundantVariant(
                prompt_id=pid,
                redundancy_type=rtype,
                variant_id=ids.variant_id(pid, rtype),
                text=vtext,
                n_tokens=whitespace_tokens(vtext),
                generator="redundanzgenerator@0.1.0 (placeholder)",
                seed=42,
            )
            store.store_redundant(variant)

            for rate in COMPRESSION_RATES:
                keep = max(1, round(variant.n_tokens * rate))
                ctext = " ".join(vtext.split()[:keep])
                comp = Compressate(
                    prompt_id=pid,
                    redundancy_type=rtype,
                    target_ratio=rate,
                    compressate_id=ids.compressate_id(pid, rtype, rate),
                    text=ctext,
                    n_tokens=whitespace_tokens(ctext),
                    n_tokens_source=variant.n_tokens,
                    compressor="truncate@demo (placeholder)",
                    seed=42,
                )
                store.store_compressate(comp)

                # Placeholder "inference": correct if the gold token survived.
                correct = spec["gold"].lower() in ctext.lower()
                store.store_inference(
                    InferenceResult(
                        compressate_id=comp.compressate_id,
                        model="demo-echo",
                        output=spec["gold"] if correct else "(unknown)",
                        gold_answer=spec["gold"],
                        is_correct=correct,
                        metrics={"exact_match": float(correct), "latency_ms": 10.0 + keep},
                        seed=42,
                    )
                )

    jsonl_path, csv_path = store.build_manifest()
    n = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
    print(f"demo-experiment: {len(RAW_PROMPTS)} raw prompts -> {n} compressates")
    print(f"manifest -> {csv_path.relative_to(root.parent)}")


if __name__ == "__main__":
    main()
