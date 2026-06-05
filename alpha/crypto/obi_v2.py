"""
alpha/crypto/obi_v2.py —Task 4: 经验 OBI/OFI 信号引擎
====================================================

Computes Order-Book-Imbalance (OBI) and Order-Flow-Imbalance (OFI) from the
microstructure snapshot stream and emits an ``AlphaSignal`` (proto-shaped dict)
with a globally-injected ``trace_id``.

OBI  = (bid_depth - ask_depth) / (bid_depth + ask_depth)        in [-1, 1]
OFI  = signed best-level size delta accumulated over the window  (Cont et al.)

The raw signal is the tanh-squashed blend of instantaneous OBI and normalised
OFI; confidence scales with book depth and |signal| consistency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from common.logging_setup import get_logger, get_pulse, get_trace

_log = get_logger("alpha.obi_v2")


@dataclass
class _BookState:
    bid_px: float = 0.0
    bid_sz: float = 0.0
    ask_px: float = 0.0
    ask_sz: float = 0.0


@dataclass
class AlphaSignal:
    """Mirror of schemas/alpha_signal.proto AlphaSignal (subset for MVP)."""

    trace_id: str
    inst_id: str
    ts_ms: int
    pulse_id: int
    alpha_name: str = "obi_v2"
    raw_signal: float = 0.0
    confidence: float = 0.0
    obi: float = 0.0
    ofi: float = 0.0
    g1_pass: bool = False
    g2_pass: bool = False
    g3_pass: bool = False
    g4_pass: bool = False
    passed: bool = False
    feature_snapshot: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ObiV2Engine:
    """Stateful OBI/OFI engine. Feed it snapshots; it returns an AlphaSignal."""

    def __init__(self, inst_id: str = "BTC-USDT-SWAP", ofi_window: int = 60) -> None:
        self.inst_id = inst_id
        self._prev: Optional[_BookState] = None
        self._ofi_hist: Deque[float] = deque(maxlen=ofi_window)
        self._sig_hist: Deque[float] = deque(maxlen=ofi_window)

    def _ofi_increment(self, cur: _BookState, prev: _BookState) -> float:
        """Cont-Kukanov-Stoikov OFI for the best level."""
        e = 0.0
        # bid side
        if cur.bid_px > prev.bid_px:
            e += cur.bid_sz
        elif cur.bid_px < prev.bid_px:
            e -= prev.bid_sz
        else:
            e += cur.bid_sz - prev.bid_sz
        # ask side
        if cur.ask_px < prev.ask_px:
            e -= cur.ask_sz
        elif cur.ask_px > prev.ask_px:
            e += prev.ask_sz
        else:
            e -= cur.ask_sz - prev.ask_sz
        return e

    def on_snapshot(self, snap: dict) -> AlphaSignal:
        bid = float(snap.get("bid1", 0.0))
        ask = float(snap.get("ask1", 0.0))
        bid_sz = float(snap.get("bid1_sz", 0.0))
        ask_sz = float(snap.get("ask1_sz", 0.0))
        ts = int(snap.get("ts_ms", 0))

        cur = _BookState(bid, bid_sz, ask, ask_sz)

        # OBI instantaneous
        denom = bid_sz + ask_sz
        obi = (bid_sz - ask_sz) / denom if denom > 0 else 0.0

        # OFI accumulation
        ofi_inc = self._ofi_increment(cur, self._prev) if self._prev else 0.0
        self._ofi_hist.append(ofi_inc)
        self._prev = cur
        ofi_sum = sum(self._ofi_hist)
        # normalise OFI by rolling abs scale
        scale = sum(abs(x) for x in self._ofi_hist) or 1.0
        ofi_norm = ofi_sum / scale  # in [-1, 1]

        # blended raw signal
        raw = math.tanh(1.5 * obi + 1.0 * ofi_norm)
        self._sig_hist.append(raw)

        # confidence: depth-weighted + signal persistence
        depth_conf = min(denom / 20.0, 1.0)  # saturates at 20 base units
        if len(self._sig_hist) >= 5:
            recent = list(self._sig_hist)[-5:]
            same_sign = sum(1 for s in recent if (s > 0) == (raw > 0)) / 5.0
        else:
            same_sign = 0.5
        confidence = max(0.0, min(1.0, 0.5 * depth_conf + 0.5 * same_sign))

        sig = AlphaSignal(
            trace_id=get_trace(),
            inst_id=self.inst_id,
            ts_ms=ts,
            pulse_id=get_pulse(),
            raw_signal=round(raw, 6),
            confidence=round(confidence, 4),
            obi=round(obi, 6),
            ofi=round(ofi_norm, 6),
        )
        return sig


__all__ = ["ObiV2Engine", "AlphaSignal"]
