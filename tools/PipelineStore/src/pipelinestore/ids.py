"""Deterministic identifier scheme that links artifacts to manifest rows.

The pipeline has four stages and a fixed fan-out per raw prompt::

    raw prompt  ->  redundancy type  ->  compression rate  ->  inference

    1 raw prompt = 4 redundancy types x 4 compression rates = 16 compressates

Every unit of analysis (one compressate, later scored by one inference run) is
addressed by a single, stable id assembled from three parts::

    <prompt_id>__<redundancy_type>__<rate_label>

The same three parts also define where the artifact lives on disk, so an id can
always be turned back into a file path and vice versa. Keeping the id purely a
function of its coordinates (not of content or wall-clock time) makes reruns
idempotent: re-storing the same coordinate overwrites in place instead of
piling up duplicates.
"""

from __future__ import annotations

import re

SEP = "__"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(value: str | int) -> str:
    """Normalise an arbitrary label into a filesystem/column safe token."""

    text = str(value).strip().lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text or "na"


def prompt_id(raw_id: str | int) -> str:
    """Id of a raw prompt, e.g. a MuSiQue id or a running counter."""

    return f"p-{slug(raw_id)}"


def rate_label(target_ratio: float) -> str:
    """Compact, sortable label for a compression rate.

    ``0.2`` -> ``"cr020"`` (keep 20 % of the tokens). Fixed width keeps the
    labels lexically sortable, which is convenient in file listings and axes.
    """

    pct = round(float(target_ratio) * 100)
    if not 0 <= pct <= 100:
        raise ValueError(f"compression ratio out of range: {target_ratio!r}")
    return f"cr{pct:03d}"


def variant_id(prompt: str, redundancy_type: str) -> str:
    """Id of a redundant variant (stage 02): one raw prompt + one redundancy type."""

    return f"{prompt}{SEP}{slug(redundancy_type)}"


def compressate_id(prompt: str, redundancy_type: str, target_ratio: float) -> str:
    """Id of a single compressate / observation (stages 03 and 04).

    This is *the* unit of statistical analysis. There are exactly 16 of these
    per raw prompt.
    """

    return f"{variant_id(prompt, redundancy_type)}{SEP}{rate_label(target_ratio)}"


def parse_compressate_id(value: str) -> tuple[str, str, str]:
    """Split a compressate id back into ``(prompt_id, redundancy_type, rate_label)``."""

    parts = value.split(SEP)
    if len(parts) != 3:
        raise ValueError(f"not a compressate id: {value!r}")
    return parts[0], parts[1], parts[2]
