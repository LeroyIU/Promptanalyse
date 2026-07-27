"""Record types for each pipeline stage plus the flat row used for statistics.

Design in one sentence: **heavy text lives in per-stage JSON records, light
measurements are projected into one tidy row per compressate**. The record
classes below are the on-disk artifacts (one file each); :class:`TidyRow` is the
denormalised, analysis-ready projection that becomes one line in the manifest.

All classes are plain dataclasses with ``to_dict`` / ``from_dict`` so they
round-trip through JSON without extra dependencies, mirroring the style of the
``redundanzgenerator`` package in this repo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from . import ids


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values so records stay compact and diff-friendly."""

    return {k: v for k, v in data.items() if v is not None}


@dataclass
class RawPrompt:
    """Stage 01 -- an original, uncompressed prompt as it enters the pipeline.

    ``text`` is the fully rendered prompt string; ``structured`` optionally
    keeps the structured form (e.g. a ``redundanzgenerator`` ``FewShotPrompt``
    dict) so the raw material is reproducible, not just its rendering.
    """

    prompt_id: str
    text: str
    n_tokens: int
    source: str = ""          # dataset / provenance, e.g. "musique"
    source_id: str | None = None      # e.g. MuSiQue id
    source_category: str | None = None    # dataset-side stratum, e.g. hop count
    structured: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawPrompt":
        return cls(**{k: data.get(k) for k in _field_names(cls) if k in data})


@dataclass
class RedundantVariant:
    """Stage 02 -- one raw prompt after injecting exactly one redundancy type."""

    prompt_id: str
    redundancy_type: str
    variant_id: str
    text: str
    n_tokens: int
    redundancy_report: Any | None = None   # what the generator inserted, verbatim
    generator: str = ""                    # tool + version that produced this
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedundantVariant":
        return cls(**{k: data.get(k) for k in _field_names(cls) if k in data})


@dataclass
class Compressate:
    """Stage 03 -- one redundant variant compressed at one target ratio.

    ``target_ratio`` is the requested fraction of tokens to keep (0..1);
    ``achieved_ratio`` is what actually happened, computed from token counts.
    """

    prompt_id: str
    redundancy_type: str
    target_ratio: float
    compressate_id: str
    text: str
    n_tokens: int
    n_tokens_source: int          # tokens of the redundant variant it came from
    compressor: str = ""          # method + version, e.g. "llmlingua-2@0.2"
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rate_label(self) -> str:
        return ids.rate_label(self.target_ratio)

    @property
    def achieved_ratio(self) -> float | None:
        if not self.n_tokens_source:
            return None
        return round(self.n_tokens / self.n_tokens_source, 4)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Compressate":
        return cls(**{k: data.get(k) for k in _field_names(cls) if k in data})


@dataclass
class InferenceResult:
    """Stage 04 -- the model output and scoring for one compressate."""

    compressate_id: str
    model: str
    output: str
    gold_answer: str | None = None
    is_correct: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)   # e.g. f1, latency_ms
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceResult":
        return cls(**{k: data.get(k) for k in _field_names(cls) if k in data})


# Columns of the tidy manifest, in a stable order. Keeping this list explicit
# (rather than deriving it) guarantees a fixed CSV header across runs.
TIDY_COLUMNS: list[str] = [
    "experiment_id",
    "compressate_id",
    "prompt_id",
    "source",
    "source_id",
    "source_category",
    "redundancy_type",
    "rate_label",
    "target_ratio",
    "achieved_ratio",
    "n_tokens_raw",
    "n_tokens_redundant",
    "n_tokens_compressed",
    "redundancy_added_tokens",     # redundant - raw
    "compression_removed_tokens",  # redundant - compressed
    "model",
    "output",
    "gold_answer",
    "is_correct",
    "seed",
    "raw_path",
    "redundant_path",
    "compressed_path",
    "inference_path",
]


@dataclass
class TidyRow:
    """One row of the analysis table: one compressate, optionally scored.

    This is the denormalised join of all four stages. ``metrics`` (arbitrary
    numeric scores from stage 04) are flattened into their own columns when the
    row is serialised, so e.g. ``metrics={"f1": 0.8}`` becomes a ``metric_f1``
    column. That keeps the core schema fixed while still letting each experiment
    add its own measures.
    """

    experiment_id: str
    compressate_id: str
    prompt_id: str
    redundancy_type: str
    rate_label: str
    target_ratio: float
    source: str = ""
    source_id: str | None = None
    source_category: str | None = None
    achieved_ratio: float | None = None
    n_tokens_raw: int | None = None
    n_tokens_redundant: int | None = None
    n_tokens_compressed: int | None = None
    redundancy_added_tokens: int | None = None
    compression_removed_tokens: int | None = None
    model: str | None = None
    output: str | None = None
    gold_answer: str | None = None
    is_correct: bool | None = None
    seed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    raw_path: str | None = None
    redundant_path: str | None = None
    compressed_path: str | None = None
    inference_path: str | None = None

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict: fixed columns first, then ``metric_*`` extras."""

        row = {col: getattr(self, col, None) for col in TIDY_COLUMNS}
        for key, value in self.metrics.items():
            row[f"metric_{_column_key(key)}"] = value
        return row


def _field_names(cls: type) -> list[str]:
    return [f for f in cls.__dataclass_fields__]  # type: ignore[attr-defined]


def _column_key(key: str) -> str:
    """Normalise a metric name into a pandas/R friendly column suffix.

    Lower-cased, non-alphanumerics collapsed to single underscores (kept, not
    hyphenated) so columns are valid identifiers: ``"Latency (ms)"`` ->
    ``"latency_ms"``.
    """

    import re

    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_") or "na"
