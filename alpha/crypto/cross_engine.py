# ═══════════════════════════════════════════════════════════════════════════════
# alpha/crypto/cross_engine.py — CrossSectionEngine (独立进程, Redis IPC)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reads market snapshots from Redis keys ``v8:snapshot:{inst_id}`` (published
# by each Orchestrator after pushing to its local SHM), computes cross-section
# features every tick, and writes the result to ``v8:cross_section:latest``.
#
# Flat float32 layout (contract shared with cross_reader.py):
#   Per market:   [obi, ofi_norm]                           ← 2 floats
#   Per pair:     [delta_obi, ofi_ratio, lead_lag_a→b,
#                  lead_lag_b→a, obi_a, obi_b]              ← 6 floats
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import redis as _redis_lib

from common.logging_setup import get_logger, get_trace, new_trace_id

_log = get_logger("alpha.cross_section")

_PER_MARKET = 2
_PER_PAIR = 6
_REDIS_URL = os.getenv("V8_REDIS_URL", "redis://127.0.0.1:6379/0")
_SNAPSHOT_TTL = 30        # seconds until stale snapshot is evicted
_CROSS_TTL = 10            # seconds until cross-section expires


@dataclass
class _MarketWindow:
    inst_id: str
    obi: Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    ofi: Deque[float] = field(default_factory=lambda: deque(maxlen=128))


class CrossSectionEngine:
    """Independent process: reads market snapshots from Redis, computes
    cross-section features, publishes to Redis.

    Usage:
        python -m alpha.crypto.cross_engine
            --inst-ids BTC-USDT-SWAP,ETH-USDT-SWAP
            --tick-hz 10
    """

    def __init__(
        self,
        inst_ids: List[str],
        tick_hz: float = 10.0,
        lead_lag_window: int = 64,
        corr_lag: int = 5,
        redis_url: str = _REDIS_URL,
    ) -> None:
        if len(inst_ids) < 2:
            raise ValueError("cross_section needs at least 2 instruments")

        self.inst_ids = [iid.strip() for iid in inst_ids]
        self.tick_hz = tick_hz
        self._interval_s = 1.0 / tick_hz
        self._lead_lag_window = lead_lag_window
        self._corr_lag = corr_lag

        # ── Redis client ────────────────────────────────────────────────────
        # ── shared Redis connection pool ────────────────────────────
        _pool = _redis_lib.ConnectionPool.from_url(
            redis_url, max_connections=5,
            health_check_interval=30, retry_on_timeout=True,
        )
        self._redis = _redis_lib.Redis(connection_pool=_pool)

        # ── per-market rolling windows ──────────────────────────────────────
        self._windows: Dict[str, _MarketWindow] = {
            iid: _MarketWindow(inst_id=iid) for iid in self.inst_ids
        }

        # ── OFI tracking (stateful, per-market) ─────────────────────────────
        self._prev_snaps: Dict[str, dict] = {}
        self._ofi_acc: Dict[str, float] = {}

        # ── state ───────────────────────────────────────────────────────────
        self._running = False
        self._ticks = 0
        self._started_ms = 0

        _log.info(
            "cross_section_init",
            extra={
                "inst_ids": self.inst_ids,
                "tick_hz": tick_hz,
                "redis_url": redis_url.rsplit("@", 1)[-1] if "@" in redis_url else redis_url,
            },
        )

    # ── OBI ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_obi(snap: dict) -> float:
        bid_sz = float(snap.get("bid1_sz", 0.0))
        ask_sz = float(snap.get("ask1_sz", 0.0))
        denom = bid_sz + ask_sz
        return (bid_sz - ask_sz) / denom if denom > 0 else 0.0

    # ── OFI increment (Cont-Kukanov-Stoikov) ────────────────────────────────
    @staticmethod
    def _ofi_inc(cur: dict, prev: Optional[dict]) -> float:
        if prev is None:
            return 0.0
        e = 0.0
        c_bid, c_ask = float(cur.get("bid1", 0)), float(cur.get("ask1", 0))
        c_bsz, c_asz = float(cur.get("bid1_sz", 0)), float(cur.get("ask1_sz", 0))
        p_bid, p_ask = float(prev.get("bid1", 0)), float(prev.get("ask1", 0))
        p_bsz, p_asz = float(prev.get("bid1_sz", 0)), float(prev.get("ask1_sz", 0))
        if c_bid > p_bid:
            e += c_bsz
        elif c_bid < p_bid:
            e -= p_bsz
        else:
            e += c_bsz - p_bsz
        if c_ask < p_ask:
            e -= c_asz
        elif c_ask > p_ask:
            e += p_asz
        else:
            e -= c_asz - p_asz
        return e

    # ── Pearson ─────────────────────────────────────────────────────────────
    @staticmethod
    def _pearson(a: Deque[float], b: Deque[float]) -> float:
        n = min(len(a), len(b))
        if n < 8:
            return 0.0
        a_vals = list(a)[-n:]
        b_vals = list(b)[-n:]
        ma = sum(a_vals) / n
        mb = sum(b_vals) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a_vals, b_vals))
        da = math.sqrt(sum((x - ma) ** 2 for x in a_vals)) + 1e-10
        db = math.sqrt(sum((y - mb) ** 2 for y in b_vals)) + 1e-10
        return num / (da * db)

    # ── Pull snapshot from Redis ────────────────────────────────────────────
    def _fetch_snapshot(self, iid: str) -> Optional[dict]:
        raw = self._redis.get(f"v8:snapshot:{iid}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ── One tick ────────────────────────────────────────────────────────────
    def _tick(self) -> Optional[List[float]]:
        ts_ms = int(time.time() * 1000)
        new_trace_id()

        # 1. Fetch latest snapshots from all markets
        snaps: Dict[str, Optional[dict]] = {}
        for iid in self.inst_ids:
            s = self._fetch_snapshot(iid)
            if s is None:
                return None  # skip tick if any market is missing
            snaps[iid] = s

        # 2. Compute per-market OBI/OFI
        features_flat: List[float] = []
        for iid in self.inst_ids:
            snap = snaps[iid]
            win = self._windows[iid]

            cur_parsed = {
                "bid1": float(snap.get("bid1", 0.0)),
                "bid1_sz": float(snap.get("bid1_sz", 0.0)),
                "ask1": float(snap.get("ask1", 0.0)),
                "ask1_sz": float(snap.get("ask1_sz", 0.0)),
            }

            obi = self._compute_obi(snap)
            win.obi.append(obi)

            prev = self._prev_snaps.get(iid)
            ofi_inc = self._ofi_inc(cur_parsed, prev)
            self._prev_snaps[iid] = cur_parsed

            alpha = 0.9
            old = self._ofi_acc.get(iid, 0.0)
            ofi_norm = alpha * old + (1.0 - alpha) * ofi_inc
            self._ofi_acc[iid] = ofi_norm
            win.ofi.append(ofi_norm)

            features_flat.append(obi)
            features_flat.append(ofi_norm)

        # 3. Compute pairwise cross-section
        n = len(self.inst_ids)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                wa = self._windows[self.inst_ids[i]]
                wb = self._windows[self.inst_ids[j]]

                oa = wa.obi[-1] if wa.obi else 0.0
                ob = wb.obi[-1] if wb.obi else 0.0
                fa = wa.ofi[-1] if wa.ofi else 0.0
                fb = wb.ofi[-1] if wb.ofi else 0.0

                delta = oa - ob
                ratio = fa / (abs(fb) + 1e-8)

                lag_a = list(wa.obi)[:-self._corr_lag] if len(wa.obi) > self._corr_lag else list(wa.obi)
                lag_b = list(wb.obi)[:-self._corr_lag] if len(wb.obi) > self._corr_lag else list(wb.obi)

                ll_ab = self._pearson(deque(lag_a, maxlen=self._lead_lag_window), wb.obi)
                ll_ba = self._pearson(deque(lag_b, maxlen=self._lead_lag_window), wa.obi)

                features_flat.extend([delta, ratio, ll_ab, ll_ba, oa, ob])

        # 4. Publish to Redis
        payload = json.dumps({
            "inst_ids": self.inst_ids,
            "ts_ms": ts_ms,
            "ticks": self._ticks,
            "features": features_flat,
        })
        self._redis.setex("v8:cross_section:latest", _CROSS_TTL, payload)

        return features_flat

    # ── Main loop ───────────────────────────────────────────────────────────
    def run(self) -> None:
        self._running = True
        self._started_ms = int(time.time() * 1000)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        _log.info("cross_section_loop_start", extra={"tick_hz": self.tick_hz})

        while self._running:
            loop_start = time.perf_counter()
            features = self._tick()
            if features is not None:
                self._ticks += 1
                if self._ticks % 30 == 0:
                    _log.info(
                        "cross_section_tick",
                        extra={"ticks": self._ticks, "feat_dim": len(features)},
                    )

            elapsed = time.perf_counter() - loop_start
            sleep_for = max(0.0, self._interval_s - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

        _log.info("cross_section_loop_stop", extra={"ticks": self._ticks})

    def _handle_signal(self, signum: int, _frame) -> None:
        _log.info("signal_received", extra={"signal": signum})
        self._running = False


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="V8 Cross-Section Engine (Redis IPC)")
    ap.add_argument(
        "--inst-ids",
        default=os.getenv("V8_CROSS_INST_IDS", "BTC-USDT-SWAP,ETH-USDT-SWAP"),
        help="Comma-separated instrument IDs",
    )
    ap.add_argument(
        "--tick-hz", type=float,
        default=float(os.getenv("V8_CROSS_TICK_HZ", "10")),
    )
    ap.add_argument("--redis-url", default=_REDIS_URL)
    args = ap.parse_args()

    inst_ids = [iid.strip() for iid in args.inst_ids.split(",") if iid.strip()]
    engine = CrossSectionEngine(
        inst_ids=inst_ids,
        tick_hz=args.tick_hz,
        redis_url=args.redis_url,
    )
    engine.run()


if __name__ == "__main__":
    main()