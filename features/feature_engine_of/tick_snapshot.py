#!/usr/bin/env python3
"""
tick_snapshot.py — V8 tick DB → OF MarketSnapshot 适配层
────────────────────────────────────────────────────────
将 V8 SQLite tick 数据转换为 OF FeatureEngine v3.1 兼容的
MarketSnapshot / MarketStateBuffer 接口。

V8 tick 字段:
  ts_ms, inst_id, last_px, bid1, ask1, bid1_sz, ask1_sz,
  vol, turnover, funding_rate, open_interest, high_24h, low_24h, ...

OF MarketSnapshot 必需字段:
  timestamp, market_id, mid_price, best_bid, best_ask,
  spread, spread_pct, bid_depth, ask_depth, obi,
  trade_count, buy_volume, sell_volume, net_flow,
  funding_rate, open_interest
"""

from __future__ import annotations

import sqlite3
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# MarketSnapshot (V8-compatible)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class MarketSnapshot:
    """One immutable snapshot — compatible with OF FeatureEngine v3.1"""
    timestamp: float       # unix epoch seconds
    market_id: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_pct: float
    bid_depth: float       # single-level proxy from V8
    ask_depth: float       # single-level proxy from V8
    obi: float             # order book imbalance
    trade_count: int
    buy_volume: float      # inferred from vol delta + price delta
    sell_volume: float
    net_flow: float
    yes_price: float = 0.0
    no_price: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    resolve_time: float = 0.0

    @classmethod
    def from_tick_row(
        cls,
        market_id: str,
        row: dict,
        prev_row: Optional[dict] = None,
    ) -> "MarketSnapshot":
        """
        Factory: build snapshot from a single V8 tick DB row.

        Args:
            market_id: instrument ID (e.g. "BTC-USDT-SWAP")
            row: current tick row dict (keys: ts_ms, last_px, bid1, ask1, ...)
            prev_row: previous tick row dict (for vol delta computation)
        """
        ts_ms = row["ts"]
        last_px = float(row.get("last", 0))
        bid1 = float(row.get("bid", 0))
        ask1 = float(row.get("ask", 0))
        bid1_sz = float(row.get("bid_sz", 0))
        ask1_sz = float(row.get("ask_sz", 0))

        # Use pre-computed mid_px and spread when available
        mid = float(row.get("mid_px", 0)) or (bid1 + ask1) / 2.0 if bid1 > 0 and ask1 > 0 else last_px
        spread = float(row.get("spread", 0)) or (ask1 - bid1 if ask1 > 0 else 0.0)
        spread_pct = spread / mid if mid > 0 else 0.0

        total_depth = bid1_sz + ask1_sz
        obi = (bid1_sz - ask1_sz) / total_depth if total_depth > 0 else 0.0

        # V8 tick DB has no per-tick volume; flow features use zero (cross-sectional fallback)
        trade_count = 1
        buy_volume = 0.0
        sell_volume = 0.0

        return cls(
            timestamp=float(ts_ms) / 1000.0,
            market_id=market_id,
            best_bid=bid1,
            best_ask=ask1,
            mid_price=mid,
            spread=spread,
            spread_pct=spread_pct,
            bid_depth=bid1_sz,
            ask_depth=ask1_sz,
            obi=obi,
            trade_count=trade_count,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_flow=0.0,
            funding_rate=float(row.get("funding_rate", 0)),
            open_interest=float(row.get("open_interest", 0)),
        )


# ─────────────────────────────────────────────
# MarketStateBuffer (V8 tick DB backed)
# ─────────────────────────────────────────────

class MarketStateBuffer:
    """
    Circular per-market time-series buffer — compatible with OF FeatureEngine.
    Data is loaded upfront from V8 tick DB, then FeatureEngine reads it.

    Parameters
    ----------
    market_id    : unique instrument ID
    max_snapshots: hard cap on stored snapshots (default 7200 ≈ 2h at 1s)
    """

    def __init__(self, market_id: str, max_snapshots: int = 7_200):
        self.market_id = market_id
        self._buf: deque[MarketSnapshot] = deque(maxlen=max_snapshots)

    def load_from_db(
        self, db_path: str, start_s: Optional[float] = None, end_s: Optional[float] = None
    ) -> int:
        """Load ticks from V8 SQLite DB into buffer."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM ticks WHERE 1=1"
        params = []

        if start_s is not None:
            query += " AND ts >= ?"
            params.append(int(start_s * 1000))
        if end_s is not None:
            query += " AND ts <= ?"
            params.append(int(end_s * 1000))

        query += " ORDER BY ts ASC"

        cursor = conn.execute(query, params)
        prev_row = None
        count = 0

        for row in cursor:
            row_dict = dict(row)
            snap = MarketSnapshot.from_tick_row(self.market_id, row_dict, prev_row)
            self._buf.append(snap)
            prev_row = row_dict
            count += 1

        conn.close()
        return count

    def push(self, snapshot: MarketSnapshot) -> None:
        self._buf.append(snapshot)

    def __len__(self) -> int:
        return len(self._buf)

    def get_latest(self) -> Optional[MarketSnapshot]:
        return self._buf[-1] if self._buf else None

    def get_window(self, seconds: float) -> List[MarketSnapshot]:
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp - seconds
        result = []
        for snap in reversed(self._buf):
            if snap.timestamp < cutoff:
                break
            result.append(snap)
        return list(reversed(result))

    # ── typed series ───────────────────────────

    def price_series(self, seconds: float) -> np.ndarray:
        return np.array([s.mid_price for s in self.get_window(seconds)])

    def obi_series(self, seconds: float) -> np.ndarray:
        return np.array([s.obi for s in self.get_window(seconds)])

    def spread_series(self, seconds: float) -> np.ndarray:
        return np.array([s.spread for s in self.get_window(seconds)])

    def net_flow_series(self, seconds: float) -> np.ndarray:
        return np.array([s.net_flow for s in self.get_window(seconds)])

    def depth_series(self, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        window = self.get_window(seconds)
        return (
            np.array([s.bid_depth for s in window]),
            np.array([s.ask_depth for s in window]),
        )

    def oi_series(self, seconds: float) -> np.ndarray:
        return np.array([s.open_interest for s in self.get_window(seconds)])

    def cvd(self, seconds: float) -> float:
        flows = self.net_flow_series(seconds)
        return float(flows.sum()) if len(flows) else 0.0

    def realized_vol(self, seconds: float) -> float:
        prices = self.price_series(seconds)
        if len(prices) < 2:
            return 0.0
        log_rets = np.diff(np.log(np.maximum(prices, 1e-10)))
        return float(log_rets.std())


# ─────────────────────────────────────────────
# BufferRegistry
# ─────────────────────────────────────────────

class BufferRegistry:
    """Multi-market buffer registry for cross-sectional features."""

    def __init__(self, max_snapshots: int = 7_200):
        self._max = max_snapshots
        self._buffers: Dict[str, MarketStateBuffer] = {}

    def add(self, buffer: MarketStateBuffer) -> None:
        self._buffers[buffer.market_id] = buffer

    def active_markets(self) -> List[str]:
        return list(self._buffers.keys())

    def all_buffers(self) -> Dict[str, MarketStateBuffer]:
        return dict(self._buffers)
