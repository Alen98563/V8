"""
features/cross_section.py — Task 3: 横截面百分位排名 (Polars, Rust-backed)
======================================================================

Computes全市场横截面 composite percentile rank (``cs_composite``) and feature
alignment across instruments. Polars' engine is Rust-driven, so the heavy group
ops stay off the Python GIL.

For the ETH 5m MVP there is a single instrument, so cross-section degenerates to
a rolling percentile against its own history; the API is written for the
multi-asset Phase 2 fan-out (BTC/ETH/SOL/... cross-section) without changes.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from common.logging_setup import get_logger

_log = get_logger("features.cross_section")


def _pl():
    import polars as pl  # imported lazily so import-time stays cheap

    return pl


def cs_percentile_rank(
    frame: Any,
    value_cols: Sequence[str],
    group_col: str = "ts_ms",
) -> Any:
    """Add ``<col>_pct`` columns: cross-sectional percentile rank within each
    timestamp group.

    frame: polars DataFrame with columns [inst_id, ts_ms, *value_cols]
    Returns a new DataFrame with appended percentile columns in [0,1].
    """
    pl = _pl()
    exprs = []
    for c in value_cols:
        # rank within group, normalised to [0,1]
        r = (pl.col(c).rank(method="average").over(group_col) - 1) / (
            (pl.count().over(group_col) - 1).clip(lower_bound=1)
        )
        exprs.append(r.alias(f"{c}_pct"))
    return frame.with_columns(exprs)


def cs_composite(
    frame: Any,
    value_cols: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    group_col: str = "ts_ms",
) -> Any:
    """Compute a composite cross-section score = weighted mean of percentile
    ranks, appended as ``cs_composite`` in [0,1]."""
    pl = _pl()
    ranked = cs_percentile_rank(frame, value_cols, group_col=group_col)
    if weights is None:
        weights = [1.0 / len(value_cols)] * len(value_cols)
    if len(weights) != len(value_cols):
        raise ValueError("weights length must match value_cols")
    expr = None
    for w, c in zip(weights, value_cols):
        term = pl.col(f"{c}_pct") * float(w)
        expr = term if expr is None else (expr + term)
    return ranked.with_columns(expr.alias("cs_composite"))


def rolling_self_percentile(values: Sequence[float], window: int = 288) -> float:
    """Single-instrument fallback: percentile of the latest value within its own
    rolling window (288 = 24h of 5m bars). Returns [0,1].
    """
    vals = list(values)[-window:]
    if not vals:
        return 0.5
    last = vals[-1]
    below = sum(1 for v in vals if v < last)
    return below / max(len(vals) - 1, 1)


def align_features(frames: dict[str, Any], on: str = "ts_ms") -> Any:
    """Outer-join per-instrument frames on the timestamp grid and forward-fill,
    producing an aligned panel for cross-section computation."""
    pl = _pl()
    if not frames:
        return pl.DataFrame()
    merged = None
    for inst, fr in frames.items():
        f = fr.with_columns(pl.lit(inst).alias("inst_id"))
        merged = f if merged is None else pl.concat([merged, f], how="vertical_relaxed")
    return merged.sort([on, "inst_id"])


__all__ = [
    "cs_percentile_rank",
    "cs_composite",
    "rolling_self_percentile",
    "align_features",
]
