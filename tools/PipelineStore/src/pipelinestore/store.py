"""The on-disk store: stage-partitioned JSON artifacts + a tidy manifest.

Directory layout of one experiment::

    <root>/<experiment_id>/
        experiment.json          # experiment-level config (model, seeds, enums)
        01_raw/<prompt_id>.json
        02_redundant/<prompt_id>/<redundancy_type>.json
        03_compressed/<prompt_id>/<redundancy_type>/<rate_label>.json
        04_inference/<prompt_id>/<redundancy_type>/<rate_label>.json
        manifest.jsonl           # lossless, one JSON record per compressate
        manifest.csv             # flat, analysis-ready projection of the same

The tree is written incrementally (each ``store_*`` call writes one file); the
manifest is *rebuilt* from the tree with :meth:`build_manifest`, so it is always
a faithful, reproducible index of what is actually on disk rather than a log
that can drift out of sync.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import ids
from .models import (
    TIDY_COLUMNS,
    Compressate,
    InferenceResult,
    RawPrompt,
    RedundantVariant,
    TidyRow,
)

RAW_DIR = "01_raw"
REDUNDANT_DIR = "02_redundant"
COMPRESSED_DIR = "03_compressed"
INFERENCE_DIR = "04_inference"

TokenCounter = Callable[[str], int]


def whitespace_tokens(text: str) -> int:
    """Dependency-free default token count (whitespace words).

    Good enough for wiring up the pipeline and for relative comparisons. For
    absolute token budgets, pass a real tokenizer (e.g. ``tiktoken``) as the
    ``token_counter`` argument to the ``store_*`` helpers instead.
    """

    return len(text.split())


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass
class PipelineStore:
    """Read/write access to one experiment's artifact tree and manifest."""

    root: Path
    experiment_id: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- paths -------------------------------------------------------------
    @property
    def base(self) -> Path:
        return self.root / self.experiment_id

    def raw_path(self, prompt_id: str) -> Path:
        return self.base / RAW_DIR / f"{prompt_id}.json"

    def redundant_path(self, prompt_id: str, redundancy_type: str) -> Path:
        return self.base / REDUNDANT_DIR / prompt_id / f"{ids.slug(redundancy_type)}.json"

    def compressed_path(self, prompt_id: str, redundancy_type: str, target_ratio: float) -> Path:
        return (
            self.base
            / COMPRESSED_DIR
            / prompt_id
            / ids.slug(redundancy_type)
            / f"{ids.rate_label(target_ratio)}.json"
        )

    def inference_path(self, prompt_id: str, redundancy_type: str, target_ratio: float) -> Path:
        return (
            self.base
            / INFERENCE_DIR
            / prompt_id
            / ids.slug(redundancy_type)
            / f"{ids.rate_label(target_ratio)}.json"
        )

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.base))

    # -- writing -----------------------------------------------------------
    def store_experiment(self, config: dict) -> Path:
        payload = {"experiment_id": self.experiment_id, **config}
        return _write_json(self.base / "experiment.json", payload)

    def store_raw(self, raw: RawPrompt) -> Path:
        return _write_json(self.raw_path(raw.prompt_id), raw.to_dict())

    def store_redundant(self, variant: RedundantVariant) -> Path:
        return _write_json(
            self.redundant_path(variant.prompt_id, variant.redundancy_type),
            variant.to_dict(),
        )

    def store_compressate(self, comp: Compressate) -> Path:
        return _write_json(
            self.compressed_path(comp.prompt_id, comp.redundancy_type, comp.target_ratio),
            comp.to_dict(),
        )

    def store_inference(self, result: InferenceResult) -> Path:
        prompt, rtype, rate = ids.parse_compressate_id(result.compressate_id)
        # rate label -> ratio for path building
        target_ratio = int(rate[2:]) / 100
        return _write_json(
            self.inference_path(prompt, rtype, target_ratio),
            result.to_dict(),
        )

    # -- reading -----------------------------------------------------------
    def iter_raw(self) -> Iterable[RawPrompt]:
        for path in sorted((self.base / RAW_DIR).glob("*.json")):
            yield RawPrompt.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _load(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- manifest ----------------------------------------------------------
    def build_rows(self) -> list[TidyRow]:
        """Join the four stages into one :class:`TidyRow` per compressate."""

        rows: list[TidyRow] = []
        for raw in self.iter_raw():
            raw_p = self.raw_path(raw.prompt_id)
            for rdir in sorted((self.base / COMPRESSED_DIR / raw.prompt_id).glob("*")):
                if not rdir.is_dir():
                    continue
                redundancy_type = rdir.name
                variant = self._load(self.redundant_path(raw.prompt_id, redundancy_type))
                var_tokens = variant.get("n_tokens") if variant else None
                for cpath in sorted(rdir.glob("*.json")):
                    comp = Compressate.from_dict(self._load(cpath))
                    infer = self._load(
                        self.inference_path(
                            raw.prompt_id, redundancy_type, comp.target_ratio
                        )
                    )
                    rows.append(
                        _join_row(
                            self.experiment_id,
                            raw,
                            var_tokens,
                            comp,
                            infer,
                            raw_path=self._rel(raw_p),
                            redundant_path=self._rel(
                                self.redundant_path(raw.prompt_id, redundancy_type)
                            ),
                            compressed_path=self._rel(cpath),
                            inference_path=(
                                self._rel(
                                    self.inference_path(
                                        raw.prompt_id, redundancy_type, comp.target_ratio
                                    )
                                )
                                if infer
                                else None
                            ),
                        )
                    )
        return rows

    def build_manifest(self) -> tuple[Path, Path]:
        """Rebuild ``manifest.jsonl`` and ``manifest.csv`` from the tree."""

        rows = self.build_rows()
        flat = [r.to_flat_dict() for r in rows]

        jsonl_path = self.base / "manifest.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in flat:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        # union of columns: fixed schema first, then any metric_* extras (sorted)
        extra = sorted({k for row in flat for k in row} - set(TIDY_COLUMNS))
        columns = TIDY_COLUMNS + extra
        csv_path = self.base / "manifest.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in flat:
                writer.writerow(row)
        return jsonl_path, csv_path


def _join_row(
    experiment_id: str,
    raw: RawPrompt,
    var_tokens: int | None,
    comp: Compressate,
    infer: dict | None,
    *,
    raw_path: str,
    redundant_path: str,
    compressed_path: str,
    inference_path: str | None,
) -> TidyRow:
    added = (var_tokens - raw.n_tokens) if var_tokens is not None else None
    removed = (var_tokens - comp.n_tokens) if var_tokens is not None else None
    return TidyRow(
        experiment_id=experiment_id,
        compressate_id=comp.compressate_id,
        prompt_id=raw.prompt_id,
        redundancy_type=comp.redundancy_type,
        rate_label=comp.rate_label,
        target_ratio=comp.target_ratio,
        source=raw.source,
        source_id=raw.source_id,
        source_prop=raw.source_prop,
        achieved_ratio=comp.achieved_ratio,
        n_tokens_raw=raw.n_tokens,
        n_tokens_redundant=var_tokens,
        n_tokens_compressed=comp.n_tokens,
        redundancy_added_tokens=added,
        compression_removed_tokens=removed,
        model=(infer or {}).get("model"),
        output=(infer or {}).get("output"),
        gold_answer=(infer or {}).get("gold_answer"),
        is_correct=(infer or {}).get("is_correct"),
        seed=comp.seed,
        metrics=(infer or {}).get("metrics", {}) or {},
        raw_path=raw_path,
        redundant_path=redundant_path,
        compressed_path=compressed_path,
        inference_path=inference_path,
    )
