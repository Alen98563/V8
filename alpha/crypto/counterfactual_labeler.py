# ═══════════════════════════════════════════════════════════════════════════════
# alpha/crypto/counterfactual_labeler.py — CFL 反事实标签工厂
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every bar close when a NEW open is signalled, the factory:
#   1) records the decision context  (features, alpha_signal, cross-section)
#   2) waits T_bar bars (default 12 = 1h on 5m)
#   3) computes the "counterfactual regret":
#         R = sign · (P_future − P_entry) / AT R
#   4) labels:
#         +1  :  regret >  θ₂  (信号正确)
#          0  :  |regret| ≤ θ₂ (中性)
#         −1  :  regret < −θ₂ (信号错误 — 反事实优于实际)
#   5) persists to PostgreSQL — table ``v8_cfl_labels`` (JSONB)
#
# Usage in Orchestrator:
#     self.cfl = CounterFactualLabeler(pg_url=...)
#     self.cfl.on_open(ts_ms, inst_id, features, signal, cross_snap)
#     self.cfl.on_tick(ts_ms, inst_id, px)
#     self.cfl.settle_pending(pulse_id)   # called each bar close
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import psycopg2
import psycopg2.extras

from common.logging_setup import get_logger

_log = get_logger("alpha.cfl")

_DEFAULT_PG_URL = os.getenv(
    "V8_CFL_PG_URL",
    "postgresql://jerry@localhost/v8_cfl",
)

# ── label constants ─────────────────────────────────────────────────────────
POSITIVE = +1
NEUTRAL  = 0
NEGATIVE = -1


@dataclass
class _PendingOpen:
    """One NEW open awaiting counterfactual settlement."""
    decision_id: str              # uuid4
    ts_entry_ms: int
    inst_id: str
    entry_px: float
    raw_signal: float
    confidence: float
    cs_composite: float
    features_50d: List[float]
    features_178d: List[float]
    direction: str                # "long" | "short"
    bar_pending: int = 12         # T_bar countdown


@dataclass
class CflLabel:
    decision_id: str
    inst_id: str
    ts_entry_ms: int
    ts_exit_ms: int
    direction: str
    entry_px: float
    exit_px: float
    bar_count: int
    regret: float                 # sign · ΔP / ATR,  [-∞, +∞]
    label: int                    # +1 / 0 / −1
    raw_signal: float
    confidence: float
    cs_composite: float
    features_50d: List[float]
    features_178d: List[float]


@dataclass
class CflConfig:
    pg_url: str = _DEFAULT_PG_URL
    bar_hold: int = 12            # 1h on 5m bars
    threshold_pos: float = 0.5    # θ₂
    threshold_neg: float = 0.5
    table: str = "v8_cfl_labels"


class CounterFactualLabeler:
    """Stores pending NEW opens, settles counterfactual outcome after T_bar bars.

    PostgreSQL schema (auto-created on init):

        CREATE TABLE IF NOT EXISTS v8_cfl_labels (
            decision_id   TEXT PRIMARY KEY,
            inst_id       TEXT NOT NULL,
            ts_entry_ms   BIGINT NOT NULL,
            ts_exit_ms    BIGINT NOT NULL,
            direction     TEXT NOT NULL,
            entry_px      NUMERIC(18,8),
            exit_px       NUMERIC(18,8),
            bar_count     INT NOT NULL,
            regret        NUMERIC(18,8),
            label         SMALLINT NOT NULL,
            raw_signal    NUMERIC(12,8),
            confidence    NUMERIC(12,8),
            cs_composite  NUMERIC(12,8),
            features_50d  JSONB,
            created_at    TIMESTAMPTZ DEFAULT now()
        );
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        merged = {
            "pg_url": cfg.get("pg_url") or os.getenv("V8_CFL_PG_URL", _DEFAULT_PG_URL),
            "bar_hold": int(cfg.get("bar_hold", 12)),
            "threshold_pos": float(cfg.get("threshold_pos", 0.5)),
            "threshold_neg": float(cfg.get("threshold_neg", 0.5)),
        }
        self.cfg = CflConfig(**merged)

        self._pending: Dict[str, _PendingOpen] = {}
        self._px_history: Dict[str, Deque[float]] = {}
        self._atr: Dict[str, float] = {}
        self._total_saved = 0

        # PostgreSQL connection (dedicated, used sparingly — one INSERT per close)
        self._conn = psycopg2.connect(self.cfg.pg_url)
        self._conn.autocommit = True
        self._ensure_schema()

        _log.info("cfl_init", extra={"pg": self.cfg.pg_url.split("@")[-1] if "@" in self.cfg.pg_url else self.cfg.pg_url, "bar_hold": self.cfg.bar_hold})

    # ── Schema ─────────────────────────────────────────────────────────────
    def _ensure_schema(self) -> None:
        # Check if table exists before creating (avoids PG type-namespace clash)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'v8_cfl_labels')"
        )
        exists = cur.fetchone()[0]
        if not exists:
            try:
                cur.execute(
                    """
                    CREATE TABLE v8_cfl_labels (
                        decision_id   TEXT PRIMARY KEY,
                        inst_id       TEXT NOT NULL,
                        ts_entry_ms   BIGINT NOT NULL,
                        ts_exit_ms    BIGINT NOT NULL,
                        direction     TEXT NOT NULL,
                        entry_px      NUMERIC(18,8),
                        exit_px       NUMERIC(18,8),
                        bar_count     INT NOT NULL,
                        regret        NUMERIC(18,8),
                        label         SMALLINT NOT NULL,
                        raw_signal    NUMERIC(12,8),
                        confidence    NUMERIC(12,8),
                        cs_composite  NUMERIC(12,8),
                        features_50d  JSONB,
                        features_178d JSONB,
                        created_at    TIMESTAMPTZ DEFAULT now()
                    )
                    """
                )
                _log.info("cfl_schema_created")
            except Exception as e:
                _log.warning("cfl_schema_skip", extra={"error": str(e)[:120]})
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_cfl_inst_time "
                "ON v8_cfl_labels (inst_id, ts_entry_ms DESC)"
            )
        except Exception:
            pass
        cur.close()
        _log.info("cfl_schema_ok")
        # Phase 2A: add features_178d
        try:
            cur.execute("ALTER TABLE v8_cfl_labels ADD COLUMN IF NOT EXISTS features_178d JSONB")
        except Exception:
            pass

    # ── Registration ───────────────────────────────────────────────────────
    def on_open(
        self,
        ts_ms: int,
        inst_id: str,
        px: float,
        raw_signal: float,
        confidence: float,
        direction: str,
        cs_composite: float = 0.0,
        features_50d: List[float] | None = None,
        features_178d: List[float] | None = None,
    ) -> str:
        """Register a candidate NEW open. Returns decision_id."""
        import uuid

        d_id = uuid.uuid4().hex[:16]
        self._pending[d_id] = _PendingOpen(
            decision_id=d_id,
            ts_entry_ms=ts_ms,
            inst_id=inst_id,
            entry_px=px,
            raw_signal=raw_signal,
            confidence=confidence,
            cs_composite=cs_composite,
            features_50d=features_50d or [],
            features_178d=features_178d or [],
            direction=direction,
            bar_pending=self.cfg.bar_hold,
        )
        _log.info("cfl_pending", extra={"d_id": d_id, "inst": inst_id, "px": px, "dir": direction})
        return d_id

    # ── Per-tick update ────────────────────────────────────────────────────
    def on_tick(self, ts_ms: int, inst_id: str, px: float) -> None:
        """Update rolling ATR for each instrument."""
        if inst_id not in self._px_history:
            from collections import deque
            self._px_history[inst_id] = deque(maxlen=256)

        hist = self._px_history[inst_id]
        if hist:
            prev = hist[-1]
            if prev > 0:
                tr = max(
                    abs(px - prev),
                    0.0,
                )
                old = self._atr.get(inst_id, 0.0)
                alpha = 0.05
                self._atr[inst_id] = alpha * tr + (1.0 - alpha) * old

        hist.append(px)

    # ── Settlement (call on every bar close) ────────────────────────────────
    def settle_pending(self, ts_ms: int, current_px: Dict[str, float]) -> List[CflLabel]:
        """Decrement bar counters for all pending opens. If a counter hits zero,
        compute counterfactual regret and persist the label. Returns list of
        newly-settled labels.
        """
        settled: List[CflLabel] = []
        expired: List[str] = []

        for d_id, po in self._pending.items():
            po.bar_pending -= 1
            if po.bar_pending > 0:
                continue

            # Counter expired — settle now
            exit_px = current_px.get(po.inst_id, po.entry_px)
            delta = exit_px - po.entry_px

            # sign: +1 for long direction, −1 for short
            sign = 1.0 if po.direction == "long" else -1.0
            atr = self._atr.get(po.inst_id, abs(delta) + 0.01)

            # regret = sign · ΔP / AT R   (positive = correct direction, negative = wrong)
            regret = sign * delta / (atr + 1e-8)

            # triangular labeler
            abs_r = abs(regret)
            if regret > self.cfg.threshold_pos:
                label = POSITIVE
            elif regret < -self.cfg.threshold_neg:
                label = NEGATIVE
            else:
                label = NEUTRAL

            cfl = CflLabel(
                decision_id=d_id,
                inst_id=po.inst_id,
                ts_entry_ms=po.ts_entry_ms,
                ts_exit_ms=ts_ms,
                direction=po.direction,
                entry_px=po.entry_px,
                exit_px=exit_px,
                bar_count=self.cfg.bar_hold,
                regret=round(regret, 8),
                label=label,
                raw_signal=round(po.raw_signal, 8),
                confidence=round(po.confidence, 8),
                cs_composite=round(po.cs_composite, 8),
                features_50d=po.features_50d,
                features_178d=po.features_178d,
            )

            settled.append(cfl)
            expired.append(d_id)

            # Persist to PG
            self._save_label(cfl)

        # Remove settled from pending
        for d_id in expired:
            del self._pending[d_id]

        if settled:
            _log.info("cfl_settled", extra={"count": len(settled), "pending": len(self._pending)})

        return settled

    # ── PostgreSQL persistence ──────────────────────────────────────────────
    def _save_label(self, cfl: CflLabel) -> None:
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO v8_cfl_labels
                    (decision_id, inst_id, ts_entry_ms, ts_exit_ms, direction,
                     entry_px, exit_px, bar_count, regret, label,
                     raw_signal, confidence, cs_composite, features_50d, features_178d)
                VALUES
                    (%s, %s, %s, %s, %s,  %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s)
                ON CONFLICT (decision_id) DO NOTHING
                """,
                (
                    cfl.decision_id, cfl.inst_id, cfl.ts_entry_ms, cfl.ts_exit_ms,
                    cfl.direction,
                    str(cfl.entry_px), str(cfl.exit_px), cfl.bar_count,
                    str(cfl.regret), cfl.label,
                    str(cfl.raw_signal), str(cfl.confidence), str(cfl.cs_composite),
                    json.dumps(cfl.features_50d),
                    json.dumps(cfl.features_178d) if cfl.features_178d else None,
                ),
            )
            cur.close()
            self._total_saved += 1
        except Exception as e:
            _log.warning("cfl_pg_write_failed", extra={"error": str(e)})

    # ── Stats ──────────────────────────────────────────────────────────────
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def total_saved(self) -> int:
        return self._total_saved

    def close(self) -> None:
        self._conn.close()
        _log.info("cfl_closed", extra={"saved": self._total_saved})


__all__ = [
    "CounterFactualLabeler", "CflLabel", "CflConfig",
    "POSITIVE", "NEUTRAL", "NEGATIVE",
]