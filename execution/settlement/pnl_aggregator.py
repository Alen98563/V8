"""
execution/settlement/pnl_aggregator.py �?Task 6: 实时对账 + P&L 结算
================================================================

Tracks fills, reconstructs realized/unrealized P&L, slippage error, and a
rolling Sharpe trend. Implements the "�?50 笔成交触发一次在�?Temperature
Scaling" scheduler hook: when ``fill_count % 50 == 0`` it fires a callback so the
orchestrator can recalibrate AlphaCast confidence.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

from common.logging_setup import get_logger

_log = get_logger("settlement.pnl")


@dataclass
class Fill:
    trace_id: str
    cl_ord_id: str
    side: str            # "buy" | "sell"
    fill_px: float
    fill_sz: float
    fee: float
    intended_px: float   # for slippage measurement
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class PnlSnapshot:
    realized_pnl: float
    unrealized_pnl: float
    position: float
    avg_entry: float
    fees_paid: float
    fill_count: int
    slippage_bps_mean: float
    sharpe: float
    last_px: float


class PnlAggregator:
    """Single-instrument net-position P&L engine (NET pos mode for BTC-SWAP)."""

    def __init__(
        self,
        on_recalibrate: Optional[Callable[[int], None]] = None,
        recalibrate_every: int = 50,
        sharpe_window: int = 200,
    ) -> None:
        self.position = 0.0          # +long / -short, base ccy
        self.avg_entry = 0.0
        self.realized = 0.0
        self.fees = 0.0
        self.fill_count = 0
        self.last_px = 0.0
        self._slippage_bps: Deque[float] = deque(maxlen=1000)
        self._equity_curve: Deque[float] = deque(maxlen=sharpe_window)
        self._on_recalibrate = on_recalibrate
        self._recal_every = recalibrate_every

    def on_fill(self, fill: Fill) -> PnlSnapshot:
        signed = fill.fill_sz if fill.side == "buy" else -fill.fill_sz
        self.fees += fill.fee
        self.last_px = fill.fill_px

        # slippage vs intended (signed against trade direction, in bps)
        if fill.intended_px > 0:
            raw = (fill.fill_px - fill.intended_px) / fill.intended_px * 1e4
            slip = raw if fill.side == "buy" else -raw  # positive = adverse
            self._slippage_bps.append(slip)

        prev_pos = self.position
        new_pos = prev_pos + signed

        if prev_pos == 0 or (prev_pos > 0) == (signed > 0):
            # opening or increasing �?weighted avg entry
            if new_pos != 0:
                self.avg_entry = (
                    abs(prev_pos) * self.avg_entry + abs(signed) * fill.fill_px
                ) / (abs(prev_pos) + abs(signed))
        else:
            # reducing or flipping �?realize against avg_entry
            closed = min(abs(signed), abs(prev_pos))
            direction = 1.0 if prev_pos > 0 else -1.0
            self.realized += direction * (fill.fill_px - self.avg_entry) * closed
            if (prev_pos > 0) != (new_pos > 0) and new_pos != 0:
                # flipped �?new entry at fill px
                self.avg_entry = fill.fill_px
            elif new_pos == 0:
                self.avg_entry = 0.0

        self.position = new_pos
        self.fill_count += 1
        self._equity_curve.append(self.realized - self.fees)

        # scheduler hook: every N fills �?online temperature scaling
        if self._on_recalibrate and self.fill_count % self._recal_every == 0:
            try:
                self._on_recalibrate(self.fill_count)
                _log.info("recalibrate_triggered", extra={"fill_count": self.fill_count})
            except Exception as exc:  # never let recalibration crash settlement
                _log.warning("recalibrate_failed", extra={"err": str(exc)})

        snap = self.snapshot()
        _log.info(
            "fill_settled",
            extra={
                "trace_id": fill.trace_id,
                "cl_ord_id": fill.cl_ord_id,
                "side": fill.side,
                "px": fill.fill_px,
                "sz": fill.fill_sz,
                "realized": round(self.realized, 4),
                "position": round(self.position, 6),
            },
        )
        return snap

    def mark(self, last_px: float) -> None:
        self.last_px = last_px

    def unrealized(self) -> float:
        if self.position == 0:
            return 0.0
        return (self.last_px - self.avg_entry) * self.position

    def _sharpe(self) -> float:
        eq = list(self._equity_curve)
        if len(eq) < 3:
            return 0.0
        rets = [eq[i] - eq[i - 1] for i in range(1, len(eq))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        sd = math.sqrt(var)
        if sd == 0:
            return 0.0
        # annualised-ish for 5m bars: sqrt(288*365) per-bar→annual scaler
        return (mean / sd) * math.sqrt(288 * 365)

    def snapshot(self) -> PnlSnapshot:
        slips = list(self._slippage_bps)
        return PnlSnapshot(
            realized_pnl=round(self.realized, 6),
            unrealized_pnl=round(self.unrealized(), 6),
            position=round(self.position, 6),
            avg_entry=round(self.avg_entry, 4),
            fees_paid=round(self.fees, 6),
            fill_count=self.fill_count,
            slippage_bps_mean=round(sum(slips) / len(slips), 4) if slips else 0.0,
            sharpe=round(self._sharpe(), 4),
            last_px=self.last_px,
        )


__all__ = ["PnlAggregator", "Fill", "PnlSnapshot"]
