"""V8 Crypto TimescaleDB Batch Writer.

Thread-safe, async batch writer for crypto market data.
Designed to be imported by the N150 Python bridge — enqueue data
from WS callbacks, flush to TimescaleDB on a timer.

Usage:
    from services.crypto_streamer.writer import CryptoTsdbWriter

    writer = CryptoTsdbWriter(
        dsn="postgresql://quant:quant2024@localhost:5433/market_data",
        batch_size=1000,
        flush_interval_ms=100,
    )
    await writer.start()

    # In your WS callback:
    writer.enqueue("tick", dict(ts_ns=..., ticker="BTC-USDT-SWAP", ...))
    writer.enqueue("orderbook", dict(...))

    await writer.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import asyncpg

from .models import (
    TickRecord,
    OrderbookRecord,
    OhlcvRecord,
    FundingRateRecord,
    MarkPriceRecord,
    OpenInterestRecord,
    LiquidationRecord,
    LsRatioRecord,
    RegimeRecord,
    IngestionLogRecord,
    WsHealthRecord,
)

logger = logging.getLogger(__name__)

# ── SQL templates ──────────────────────────────────────────────────────────

_INSERT_SQL: Dict[str, str] = {
    "tick": """
        INSERT INTO crypto.tick (ts, ts_ns, ticker, exchange, trade_id, price, size, side, trade_mode, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (ticker, trade_id, ts) DO NOTHING
    """,
    "orderbook": """
        INSERT INTO crypto.orderbook (ts, ts_ns, ticker, side, level, price, size, count, seq_id, action)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """,
    "ohlcv": """
        INSERT INTO crypto.ohlcv (ts, ticker, bar, open, high, low, close, vol, vol_ccy, vol_ccy_quote, confirm, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (ticker, bar, ts) DO UPDATE SET
            high = GREATEST(crypto.ohlcv.high, EXCLUDED.high),
            low = LEAST(crypto.ohlcv.low, EXCLUDED.low),
            close = EXCLUDED.close,
            vol = crypto.ohlcv.vol + EXCLUDED.vol,
            vol_ccy = crypto.ohlcv.vol_ccy + EXCLUDED.vol_ccy,
            vol_ccy_quote = COALESCE(crypto.ohlcv.vol_ccy_quote, 0) + COALESCE(EXCLUDED.vol_ccy_quote, 0)
    """,
    "funding_rate": """
        INSERT INTO crypto.funding_rate (ts, ticker, funding_rate, next_funding_rate, next_funding_time, method, realized_rate, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (ticker, ts) DO NOTHING
    """,
    "mark_price": """
        INSERT INTO crypto.mark_price (ts, ticker, mark_px, index_px, source)
        VALUES ($1, $2, $3, $4, $5)
    """,
    "open_interest": """
        INSERT INTO crypto.open_interest (ts, ticker, oi, oi_ccy, oi_usd, source)
        VALUES ($1, $2, $3, $4, $5, $6)
    """,
    "liquidation": """
        INSERT INTO crypto.liquidations (ts, ticker, side, bk_px, sz, bk_loss, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    """,
    "ls_ratio": """
        INSERT INTO crypto.ls_ratio (ts, ticker, ratio_type, long_ratio, short_ratio, source)
        VALUES ($1, $2, $3, $4, $5, $6)
    """,
    "regime": """
        INSERT INTO crypto.regime (ts, ticker, regime, regime_score, hurst, vol_regime, vol_percentile, detection_model)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """,
    "ingestion_log": """
        INSERT INTO meta.ingestion_log (ts, source, table_name, ticker, records_in, records_ok, records_err, latency_ms, error_msg)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """,
    "ws_health": """
        INSERT INTO meta.ws_health (ts, exchange, channel, status, last_msg_ts, lag_ms, reconnect_count)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (ts) DO UPDATE SET
            status = EXCLUDED.status,
            last_msg_ts = EXCLUDED.last_msg_ts,
            lag_ms = EXCLUDED.lag_ms,
            reconnect_count = meta.ws_health.reconnect_count + EXCLUDED.reconnect_count
    """,
}

# Table name → (record_class, column_tuple) for executemany
_TABLE_META: Dict[str, Tuple[type, Tuple[str, ...]]] = {
    "tick": (TickRecord, TickRecord.columns()),
    "orderbook": (OrderbookRecord, OrderbookRecord.columns()),
    "ohlcv": (OhlcvRecord, OhlcvRecord.columns()),
    "funding_rate": (FundingRateRecord, FundingRateRecord.columns()),
    "mark_price": (MarkPriceRecord, MarkPriceRecord.columns()),
    "open_interest": (OpenInterestRecord, OpenInterestRecord.columns()),
    "liquidation": (LiquidationRecord, LiquidationRecord.columns()),
    "ls_ratio": (LsRatioRecord, LsRatioRecord.columns()),
    "regime": (RegimeRecord, RegimeRecord.columns()),
}

# ── Writer ─────────────────────────────────────────────────────────────────


@dataclass
class CryptoTsdbWriter:
    """Async batch writer for crypto market data → TimescaleDB."""

    dsn: str
    batch_size: int = 1000
    flush_interval_ms: int = 200
    pool_min: int = 2
    pool_max: int = 10

    # Internal state
    _pool: Optional[asyncpg.Pool] = field(default=None, init=False)
    _buffers: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list), init=False)
    _flusher: Optional[asyncio.Task] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _total_inserted: int = field(default=0, init=False)
    _total_errors: int = field(default=0, init=False)

    async def start(self) -> None:
        """Open connection pool and start background flush loop."""
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.pool_min,
            max_size=self.pool_max,
            command_timeout=30,
        )
        self._running = True
        self._flusher = asyncio.create_task(self._flush_loop())
        logger.info("CryptoTsdbWriter started — pool=%d/%d, batch=%d, interval=%dms",
                     self.pool_min, self.pool_max, self.batch_size, self.flush_interval_ms)

    async def stop(self) -> None:
        """Graceful shutdown: flush remaining, close pool."""
        self._running = False
        if self._flusher:
            self._flusher.cancel()
            try:
                await self._flusher
            except asyncio.CancelledError:
                pass

        await self._flush_all()
        if self._pool:
            await self._pool.close()
        logger.info("CryptoTsdbWriter stopped — inserted=%d, errors=%d",
                     self._total_inserted, self._total_errors)

    # ── Public enqueue API ─────────────────────────────────────────────

    def enqueue(self, table: str, record: Any) -> None:
        """Enqueue a record for batch write.

        Args:
            table: One of 'tick', 'orderbook', 'ohlcv', 'funding_rate',
                   'mark_price', 'open_interest', 'liquidation',
                   'ls_ratio', 'regime'.
            record: A dataclass instance matching the table schema.
        """
        self._buffers[table].append(record)

    def enqueue_dict(self, table: str, **kwargs: Any) -> None:
        """Enqueue using keyword arguments (auto-constructs record)."""
        if table not in _TABLE_META:
            raise ValueError(f"Unknown table: {table}. Valid: {list(_TABLE_META)}")
        record_cls, _ = _TABLE_META[table]
        record = record_cls(**kwargs)
        self._buffers[table].append(record)

    # ── Flush logic ───────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        """Background loop: flush every flush_interval_ms."""
        while self._running:
            await asyncio.sleep(self.flush_interval_ms / 1000)
            try:
                await self._flush_all()
            except Exception:
                logger.exception("Flush loop error")

    async def _flush_all(self) -> None:
        """Flush all non-empty buffers to TimescaleDB."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            # Sort by table for deterministic ordering
            for table in sorted(self._buffers.keys()):
                buf = self._buffers[table]
                if not buf:
                    continue

                # Cap at batch_size per table per flush
                batch = buf[:self.batch_size]
                del buf[:self.batch_size]

                try:
                    await self._flush_table(conn, table, batch)
                except Exception:
                    logger.exception("Flush failed for table=%s", table)
                    self._total_errors += len(batch)

    async def _flush_table(
        self, conn: asyncpg.Connection, table: str, batch: List[Any]
    ) -> None:
        """Flush one table's batch."""
        sql = _INSERT_SQL.get(table)
        if not sql:
            return

        _, cols = _TABLE_META.get(table, (None, ()))
        t0 = time.perf_counter()

        # Build params list from records
        params: List[Tuple] = []
        for rec in batch:
            params.append(tuple(getattr(rec, c) for c in cols))

        # Execute batch
        await conn.executemany(sql, params)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_inserted += len(batch)
        logger.debug("Flush %s: %d rows in %.1fms", table, len(batch), elapsed_ms)

    # ── Health monitoring ────────────────────────────────────────────

    async def log_ingestion(
        self, source: str, table: str, records_in: int,
        records_ok: int = 0, records_err: int = 0,
        latency_ms: float = 0.0, error_msg: str = "",
    ) -> None:
        """Write an ingestion quality log entry."""
        if not self._pool:
            return
        from datetime import datetime, timezone
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT_SQL["ingestion_log"],
                    datetime.now(timezone.utc), source, table,
                    "", records_in, records_ok, records_err,
                    latency_ms, error_msg or None,
                )
        except Exception:
            logger.exception("Ingestion log write failed")

    async def report_ws_health(
        self, exchange: str, channel: str, status: str,
        last_msg_ts: Any = None, lag_ms: float = 0.0,
        reconnect_count: int = 0,
    ) -> None:
        """Report WebSocket connection health."""
        if not self._pool:
            return
        from datetime import datetime, timezone
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT_SQL["ws_health"],
                    datetime.now(timezone.utc), exchange, channel,
                    status, last_msg_ts, lag_ms, reconnect_count,
                )
        except Exception:
            logger.exception("WS health write failed")

    @property
    def stats(self) -> Dict[str, int]:
        """Return current buffer sizes and totals."""
        return {
            **{f"buf_{k}": len(v) for k, v in self._buffers.items()},
            "total_inserted": self._total_inserted,
            "total_errors": self._total_errors,
        }
