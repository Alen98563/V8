"""Standalone entry point for crypto_streamer container.

Reads market data from Redis pub/sub channels and batch-writes to TimescaleDB.
Can also be imported by the N150 Python bridge for in-process use.

Usage:
    # As container:
    python main.py

    # As library (from N150 bridge):
    from services.crypto_streamer import CryptoTsdbWriter, TickRecord, ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as redis

from .writer import CryptoTsdbWriter
from .models import (
    TickRecord, OrderbookRecord, OhlcvRecord,
    FundingRateRecord, MarkPriceRecord, OpenInterestRecord,
    LiquidationRecord, LsRatioRecord, RegimeRecord,
)

logger = logging.getLogger("crypto_streamer")

# Channel → (method_name, record_class)
_CHANNEL_MAP = {
    "crypto:tick": ("enqueue_tick", TickRecord),
    "crypto:orderbook": ("enqueue_orderbook", OrderbookRecord),
    "crypto:ohlcv": ("enqueue_ohlcv", OhlcvRecord),
    "crypto:funding_rate": ("enqueue_funding", FundingRateRecord),
    "crypto:mark_price": ("enqueue_mark_price", MarkPriceRecord),
    "crypto:open_interest": ("enqueue_oi", OpenInterestRecord),
    "crypto:liquidation": ("enqueue_liquidation", LiquidationRecord),
    "crypto:ls_ratio": ("enqueue_ls_ratio", LsRatioRecord),
    "crypto:regime": ("enqueue_regime", RegimeRecord),
}


class CryptoStreamerService:
    """Redis subscriber → TimescaleDB writer bridge."""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://:quant2024@localhost:6379/0")
        self.tsdb_dsn = os.getenv(
            "TSDB_DSN", "postgresql://quant:quant2024@localhost:5433/market_data"
        )
        self.batch_size = int(os.getenv("BATCH_SIZE", "1000"))
        self.flush_ms = int(os.getenv("FLUSH_INTERVAL_MS", "200"))

        self.writer = CryptoTsdbWriter(
            dsn=self.tsdb_dsn,
            batch_size=self.batch_size,
            flush_interval_ms=self.flush_ms,
        )
        self.redis: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        self._running = False

    async def start(self):
        logger.info("Starting CryptoStreamerService")
        await self.writer.start()

        self.redis = redis.from_url(self.redis_url)
        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(*list(_CHANNEL_MAP))
        self._running = True

        logger.info("Subscribed to %d channels", len(_CHANNEL_MAP))

        try:
            while self._running:
                msg = await self.pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if msg and msg["type"] == "message":
                    await self._handle_message(msg["channel"], msg["data"])
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _handle_message(self, channel: bytes, data: bytes):
        ch = channel.decode()
        meta = _CHANNEL_MAP.get(ch)
        if not meta:
            return

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON on %s", ch)
            return

        _, record_cls = meta
        try:
            record = record_cls(**payload)
        except TypeError as e:
            logger.warning("Record construct failed for %s: %s", ch, e)
            return

        table = ch.split(":", 1)[1]
        self.writer.enqueue(table, record)

    async def stop(self):
        self._running = False
        if self.pubsub:
            await self.pubsub.unsubscribe()
        if self.redis:
            await self.redis.close()
        await self.writer.stop()
        logger.info("CryptoStreamerService stopped")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    service = CryptoStreamerService()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))

    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
