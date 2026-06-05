"""gating/hard_gating.py — Task 4: G1–G6 分层风控硬门控"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
from common.logging_setup import get_logger, get_trace

_log = get_logger("gating.hard_gating")
_FUNDING_HOURS_UTC = (0, 8, 16)

@dataclass
class GateResult:
    passed: bool
    gate: str
    reason: str

@dataclass
class GateContext:
    spread_bps: float = 0.0
    bid_depth_10: float = 0.0
    ask_depth_10: float = 0.0
    realized_vol: float = 0.0
    cs_composite: float = 1.0
    meta_proba: float = 1.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    is_open_intent: bool = True
    current_position: float = 0.0
    now_ms: Optional[int] = None
    next_funding_ms: Optional[int] = None

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
        self.max_position_abs = float(g.get("max_position_abs", 10.0))
        self.min_confidence = float(g.get("min_confidence", 0.55))
        self.max_uncertainty = float(g.get("max_uncertainty", 0.05))

    def g1_liquidity(self, ctx):
        if ctx.spread_bps > self.max_spread_bps:
            return GateResult(False, "G1", f"spread {ctx.spread_bps:.2f}bps > {self.max_spread_bps}")
        if min(ctx.bid_depth_10, ctx.ask_depth_10) < self.min_depth_10:
            return GateResult(False, "G1", f"depth {min(ctx.bid_depth_10, ctx.ask_depth_10):.2f} < {self.min_depth_10}")
        return GateResult(True, "G1", "ok")

    def g2_volatility(self, ctx):
        if ctx.realized_vol > self.max_realized_vol:
            return GateResult(False, "G2", f"vol {ctx.realized_vol:.4f} > {self.max_realized_vol}")
        if ctx.realized_vol < self.min_realized_vol:
            return GateResult(False, "G2", f"vol {ctx.realized_vol:.6f} < {self.min_realized_vol}")
        return GateResult(True, "G2", "ok")

    def g3_cross_section(self, ctx):
        if ctx.cs_composite < self.cs_min_pct:
            return GateResult(False, "G3", f"cs {ctx.cs_composite:.2f} < {self.cs_min_pct}")
        return GateResult(True, "G3", "ok")

    def g4_meta(self, ctx):
        if ctx.confidence < self.min_confidence:
            return GateResult(False, "G4", f"confidence {ctx.confidence:.3f} < {self.min_confidence}")
        if ctx.uncertainty > self.max_uncertainty:
            return GateResult(False, "G4", f"uncertainty {ctx.uncertainty:.4f} > {self.max_uncertainty}")
        return GateResult(True, "G4", "ok")

    def g5_funding_blackout(self, ctx):
        if not ctx.is_open_intent:
            return GateResult(True, "G5", "not an open")
        now = ctx.now_ms or int(time.time() * 1000)
        nf = ctx.next_funding_ms or 0
        mins_to_funding = (nf - now) / 60000.0 if nf > 0 else 999.0
        if mins_to_funding <= self.funding_blackout_min:
            return GateResult(False, "G5", f"funding blackout: {mins_to_funding:.1f}min <= {self.funding_blackout_min}min")
        return GateResult(True, "G5", "ok")

    def g6_position_limit(self, ctx):
        pos = abs(ctx.current_position)
        if pos > 0 and pos >= self.max_position_abs:
            return GateResult(False, "G6", f"position {pos:.2f} >= {self.max_position_abs}")
        return GateResult(True, "G6", "ok")

    def evaluate(self, ctx):
        for gate_fn in [self.g1_liquidity, self.g2_volatility, self.g3_cross_section,
                        self.g4_meta, self.g5_funding_blackout, self.g6_position_limit]:
            result = gate_fn(ctx)
            if not result.passed:
                return result
        return GateResult(True, "ALL", "ok")