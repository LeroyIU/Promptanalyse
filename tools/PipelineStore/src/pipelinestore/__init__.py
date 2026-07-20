"""Structured storage for the redundancy/compression experiment pipeline.

Stages: raw prompt -> redundancy type -> compression rate -> inference result.
Fan-out: 1 raw prompt = 4 redundancy types x 4 compression rates = 16 compressates.

Two-layer storage model:

* **Artifacts** (variable-length prompt texts) -> per-stage JSON files in a
  directory tree (see :class:`PipelineStore`).
* **Measurements** (ids, token counts, scores) -> one tidy row per compressate,
  emitted as ``manifest.jsonl`` (lossless) and ``manifest.csv`` (analysis-ready).
"""

from . import ids

# The redundancy_pipeline integration is intentionally NOT imported here: it
# depends on the sibling `redundanzgenerator` package. Import it explicitly
# (``from pipelinestore.redundancy_pipeline import run_pipeline``) when needed.
from .models import (
    Compressate,
    InferenceResult,
    RawPrompt,
    RedundantVariant,
    TidyRow,
    TIDY_COLUMNS,
)
from .store import PipelineStore, whitespace_tokens

__all__ = [
    "ids",
    "Compressate",
    "InferenceResult",
    "RawPrompt",
    "RedundantVariant",
    "TidyRow",
    "TIDY_COLUMNS",
    "PipelineStore",
    "whitespace_tokens",
]
