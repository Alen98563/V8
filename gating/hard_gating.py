"""
gating/hard_gating.py — Task 4: G1–G5 分层风控硬门控
==================================================

Layered, fail-closed gates applied to every candidate signal before it may
become a NEW open. A signal must pass ALL active gates.

    G1  Liquidity regime  — spread/depth sanity
    G2  Volatility regime  — realized-vol band (no dead / no chaos markets)
    G3  Cross-section      — composite percentile floor (Phase 2)
    G4  MetaLabeler        — LightGBM proba floor (Phase 3)
    G5  Time gate          — OKX funding-settlement blackout
                             "结算前 30 分钟内严禁一切新开仓"

Each gate returns (passed, reason). The门控 is hard: the first failing gate
short-circuits and the decision is logged with trace_id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from common.logging_setup import get_logger, get_trace

_log = get_logger("gating.hard_gating")

# OKX perpetual funding settles every 8h at 00:00 / 08:00 / 16:00 UTC
_FUNDING_HOURS_UTC = (0, 8, 16)


@dataclass
class GateResult:
    passed: bool
    gate: str
    reason: str


@dataclass
class GateContext:
    """Everything the gates need to judge a candidate open."""

    spread_bps: float = 0.0
    bid_depth_10: float = 0.0
    ask_depth_10: float = 0.0
    realized_vol: float = 0.0
    cs_composite: float = 1.0          # default pass for single-asset MVP
    meta_proba: float = 1.0            # default pass until Phase 3
    confidence: float = 1.0
    uncertainty: float = 0.0
    is_open_intent: bool = True        # only NEW opens hit G5
    now_ms: Optional[int] = None       # injectable for tests
    next_funding_ms: Optional[int] = None  # from OKX funding snapshot if known


class HardGating:
    def __init__(self, cfg: dict) -> None:
        g = cfg or {}
        self.min_depth_10 = float(g.get("min_depth_10", 5.0))
        self.max_spread_bps = float(g.get("max_spread_bps", 8.0))
        self.max_realized_vol = float(g.get("max_realized_vol", 0.04))
        self.min_realized_vol = float(g.get("min_realized_vol", 0.00002))
        self.cs_min_pct = float(g.get("cs_min_pct", 0.55))
        self.meta_min_proba = float(g.get("meta_min_proba", 0.50))
        self.funding_blackout_min = int(g.get("funding_blackout_min", 30))
        self.min_confidence = float(g.get("min_confidence", 0.55))
        self.max_uncertainty = float(g.get("max_uncertainty", 0.05))

    # ----- individual gates --------------------------------------------------
    def g1_liquidity(self, ctx: GateContext) -> GateResult:
        if ctx.spread_bps > self.max_spread_bps:
            return GateResult(False, "G1", f"spread {ctx.spread_bps:.2f}bps > {self.max_spread_bps}")
        if min(ctx.bid_depth_10, ctx.ask_depth_10) < self.min_depth_10:
            return GateResult(
                False, "G1",
                f"depth {min(ctx.bid_depth_10, ctx.ask_depth_10):.2f} < {self.min_depth_10}",
            )
        return GateResult(True, "G1", "ok")

    def g2_regime(self, ctx: GateContext) -> GateResult:
        if ctx.realized_vol > self.max_realized_vol:
            return GateResult(False, "G2", f"vol {ctx.realized_vol:.5f} > {self.max_realized_vol}")
        if ctx.realized_vol < self.min_realized_vol:
            return GateResult(False, "G2", f"vol {ctx.realized_vol:.5f} < {self.min_realized_vol} (dead)")
        return GateResult(True, "G2", "ok")

    def g3_cross_section(self, ctx: GateContext) -> GateResult:
        if ctx.cs_composite < self.cs_min_pct:
            return GateResult(False, "G3", f"cs {ctx.cs_composite:.3f} < {self.cs_min_pct}")
        return GateResult(True, "G3", "ok")

    def g4_metalabeler(self, ctx: GateContext) -> GateResult:
        if ctx.meta_proba < self.meta_min_proba:
            return GateResult(False, "G4", f"meta {ctx.meta_proba:.3f} < {self.meta_min_proba}")
        if ctx.confidence < self.min_confidence:
            return GateResult(False, "G4", f"conf {ctx.confidence:.3f} < {self.min_confidence}")
        if ctx.uncertainty > self.max_uncertainty:
            return GateResult(False, "G4", f"sigma {ctx.uncertainty:.4f} > {self.max_uncertainty}")
        return GateResult(True, "G4", "ok")

    def g5_time(self, ctx: GateContext) -> GateResult:
        if not ctx.is_open_intent:
            return GateResult(True, "G5", "not an open")
        now_ms = ctx.now_ms if ctx.now_ms is not None else int(time.time() * 1000)
        mins_to_funding = self._minutes_to_funding(now_ms, ctx.next_funding_ms)
        if mins_to_funding <= self.funding_blackout_min:
            return GateResult(
                False, "G5",
                f"funding blackout: {mins_to_funding:.1f}min <= {self.funding_blackout_min}min",
            )
        return GateResult(True, "G5", "ok")

    # ----- composite ---------------------------------------------------------
    def evaluate(self, ctx: GateContext) -> GateResult:
        """Run all gates in order; first failure short-circuits (fail-closed)."""
        for gate in (
            self.g1_liquidity,
            self.g2_regime,
            self.g3_cross_section,
            self.g4_metalabeler,
            self.g5_time,
        ):
            res = gate(ctx)
            if not res.passed:
                _log.info(
                    "gate_blocked",
                    extra={"gate": res.gate, "reason": res.reason, "trace_id": get_trace()},
                )
                return res
        return GateResult(True, "ALL", "ok")

    # ----- helpers -----------------------------------------------------------
    def _minutes_to_funding(self, now_ms: int, next_funding_ms: Optional[int]) -> float:
        # Prefer the authoritative settlement time from OKX funding snapshot.
        if next_funding_ms and next_funding_ms > now_ms:
            return (next_funding_ms - now_ms) / 60000.0
        # Otherwise derive the next 8h boundary (00/08/16 UTC) from the clock.
        import datetime as dt

        now = dt.datetime.utcfromtimestamp(now_ms / 1000.0)
        # Build today's + tomorrow's settlement instants, pick the soonest future one.
        candidates: list[dt.datetime] = []
        for day_off in (0, 1):
            base = (now + dt.timedelta(days=day_off)).replace(
                minute=0, second=0, microsecond=0
            )
            for h in _FUNDING_HOURS_UTC:
                candidates.append(base.replace(hour=h))
        future = [c for c in candidates if c > now]
        nxt = min(future)
        return (nxt - now).total_seconds() / 60.0


__all__ = ["HardGating", "GateContext", "GateResult"]
