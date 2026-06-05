"""
adapters/redis_feed.py — Redis-backed data feed for V8 orchestrator
=====================================================================

Replaces the WebSocket client with a Redis reader. Instead of connecting
to OKX directly (blocked from N150 in GFW), reads v8:snapshot:* keys from
a remote Redis server (London VPS via Tailscale).

Interface matches OkxWsClient / LiveOkxWsClient so the orchestrator needs
zero plumbing changes: ``start()``, ``stop()``, ``latest_snapshot()``.

Usage::

    ws = RedisFeedClient("BTC-USDT-SWAP", redis_url=os.getenv("V8_REDIS_FEED_URL", "redis://localhost:6379/0"))
    ws.start()            # no-op (stateless — Redis handles persistence)
    snap = ws.latest_snapshot()  # JSON string or None
    ws.stop()             # no-op
"""

from __future__ import annotations

import json
import time
from typing import Optional

import redis as _redis_lib


class RedisFeedClient:
    """Synchronous Redis reader that looks like a WebSocket client."""

    def __init__(self, inst_id: str = "BTC-USDT-SWAP", redis_url: str = "") -> None:
        import os as _os
        self.inst_id = inst_id
        self._running = True
        self._key = f"v8:snapshot:{inst_id}"

        if not redis_url:
            redis_url = _os.getenv("V8_REDIS_FEED_URL", "redis://localhost:6379/0")
        self._redis = _redis_lib.Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5, decode_responses=True)
        self._redis.ping()

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def latest_snapshot(self) -> Optional[str]:
        try:
            raw = self._redis.get(self._key)
            return raw if raw else None
        except Exception:
            return None