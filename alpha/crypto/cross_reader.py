# ═══════════════════════════════════════════════════════════════════════════════
# alpha/crypto/cross_reader.py — CrossSectionReader (Redis-based)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every Orchestrator instance creates ONE CrossSectionReader that reads
# cross-section features from Redis key ``v8:cross_section:latest``.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import redis as _redis_lib

from common.logging_setup import get_logger

_log = get_logger("alpha.cross_reader")

_PER_MARKET = 2
_PER_PAIR = 6
_REDIS_URL = os.getenv("V8_REDIS_URL", "redis://127.0.0.1:6379/0")


@dataclass
class PairCrossFeatures:
    leader: str
    follower: str
    delta_obi: float
    ofi_ratio: float
    lead_lag: float       # Pearson(A[t-L:t], B[t])
    obi_leader: float
    obi_follower: float


@dataclass
class CrossSectionSnapshot:
    ts_ms: int
    inst_ids: List[str] = field(default_factory=list)
    per_market_obi: Dict[str, float] = field(default_factory=dict)
    per_market_ofi: Dict[str, float] = field(default_factory=dict)
    pairs: List[PairCrossFeatures] = field(default_factory=list)

    def get_pair(self, leader: str, follower: str) -> Optional[PairCrossFeatures]:
        for p in self.pairs:
            if p.leader == leader and p.follower == follower:
                return p
        return None

    def delta_obi(self, a: str, b: str) -> float:
        p = self.get_pair(a, b)
        return p.delta_obi if p else 0.0

    def lead_lag(self, a: str, b: str) -> float:
        p = self.get_pair(a, b)
        return p.lead_lag if p else 0.0


class CrossSectionReader:
    """Orchestrator-side reader: polls Redis for latest cross-section snapshot."""

    def __init__(self, inst_ids: List[str], redis_url: str = _REDIS_URL) -> None:
        self.inst_ids = list(inst_ids)
        self._n_markets = len(self.inst_ids)
        self._n_pairs = self._n_markets * (self._n_markets - 1)
        self._expected_len = self._n_markets * _PER_MARKET + self._n_pairs * _PER_PAIR
        # ── shared Redis connection pool ────────────────────────────
        _pool = _redis_lib.ConnectionPool.from_url(
            redis_url, max_connections=3,
            health_check_interval=30, retry_on_timeout=True,
        )
        self._redis = _redis_lib.Redis(connection_pool=_pool)

        _log.info(
            "cross_reader_open",
            extra={"n_markets": self._n_markets, "expected_len": self._expected_len},
        )

    def latest(self) -> Optional[CrossSectionSnapshot]:
        """Read and parse latest cross-section snapshot from Redis."""
        raw = self._redis.get("v8:cross_section:latest")
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None

        features = data.get("features", None)
        if features is None or not isinstance(features, list) or len(features) < self._expected_len:
            return None

        snap = CrossSectionSnapshot(
            ts_ms=int(data.get("ts_ms", 0)),
            inst_ids=list(self.inst_ids),
        )

        idx = 0
        for iid in self.inst_ids:
            snap.per_market_obi[iid] = float(features[idx])
            snap.per_market_ofi[iid] = float(features[idx + 1])
            idx += _PER_MARKET

        for leader in self.inst_ids:
            for follower in self.inst_ids:
                if leader == follower:
                    continue
                snap.pairs.append(PairCrossFeatures(
                    leader=leader,
                    follower=follower,
                    delta_obi=float(features[idx]),
                    ofi_ratio=float(features[idx + 1]),
                    lead_lag=float(features[idx + 2]),
                    obi_leader=float(features[idx + 4]),
                    obi_follower=float(features[idx + 5]),
                ))
                idx += _PER_PAIR

        return snap


__all__ = ["CrossSectionReader", "CrossSectionSnapshot", "PairCrossFeatures"]