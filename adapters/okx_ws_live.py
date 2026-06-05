"""
adapters/okx_ws_live.py — Sync-safe OKX Live WebSocket Client
===============================================================

Replaces the synthetic random-walk OkxWsClient with real OKX tickers data.
Runs the async OkxWsAdapter in a background daemon thread and exposes a
sync-safe ``latest_snapshot()`` interface — identical to the engine.py
fallback so the orchestrator needs zero plumbing changes.

Usage::

    ws = LiveOkxWsClient("BTC-USDT-SWAP", demo=True)   # wspap (simulated)
    ws = LiveOkxWsClient("ETH-USDT-SWAP", demo=False)  # ws.okx.com (real)
    ws.start()
    snap = ws.latest_snapshot()  # JSON string or None
    ws.stop()

The snapshot JSON shape matches what the orchestrator's _ingest_tick
expects (last_px, bid1, ask1, spread, inst_id, ts_ms, etc).

OKX public channel: "tickers" — no API key required.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

from adapters.okx_ws import OkxWsAdapter, create_public_ws_adapter
from common.logging_setup import get_logger

_log = get_logger("adapters.okx_ws_live")


class LiveOkxWsClient:
    """Synchronous, thread-safe wrapper around an async OKX WebSocket.

    Spins up a private ``asyncio`` event loop on a daemon thread so
    the main orchestrator can call ``latest_snapshot()`` from its
    synchronous pulse loop without touching asyncio.
    """

    def __init__(self, inst_id: str = "BTC-USDT-SWAP", demo: bool = True) -> None:
        self.inst_id = inst_id
        self.demo = demo
        self._running = False
        self._lock = threading.Lock()
        self._latest_snap: Optional[str] = None   # thread-safe snapshot buffer
        self._ws: Optional[OkxWsAdapter] = None
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0
        self._last_ts = 0

    # ------------------------------------------------------------------
    # Public API — same signature as engine.OkxWsClient
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True, name=f"okxws-{self.inst_id}")
        self._thread.start()
        _log.info("live_ws_start", extra={"inst_id": self.inst_id, "demo": self.demo})

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        _log.info("live_ws_stop_requested", extra={"inst_id": self.inst_id})
        # The daemon thread will terminate when the asyncio loop exits.

    def latest_snapshot(self) -> Optional[str]:
        """Return the most recent OKX ticker as a JSON string, or None."""
        with self._lock:
            return self._latest_snap

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Thread entry — runs a private asyncio event loop
    # ------------------------------------------------------------------
    def _run_async(self) -> None:
        """Entry point for the daemon thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_main_loop())
        except Exception as exc:
            _log.error("live_ws_crash", extra={"inst_id": self.inst_id, "err": str(exc)})
        finally:
            loop.run_until_complete(self._cleanup())
            loop.close()

    # ------------------------------------------------------------------
    # Async main — connect, subscribe, keep alive
    # ------------------------------------------------------------------
    async def _ws_main_loop(self) -> None:
        """Connect to OKX, subscribe tickers, loop until stop() is called."""
        url = "wss://ws.okx.com:8443/ws/v5/public" if not self.demo else "wss://wspap.okx.com:8443/ws/v5/public"
        self._ws = OkxWsAdapter(ws_url=url)
        await self._ws.connect()

        # Subscribe public tickers (no auth required)
        await self._ws.subscribe("tickers", self.inst_id, self._on_tickers)

        _log.info("live_ws_ready", extra={"inst_id": self.inst_id, "url": url})

        # Hold the loop open — WS receive + heartbeat run as asyncio tasks
        while self._running:
            await asyncio.sleep(0.1)

    async def _on_tickers(self, data_list: list) -> None:
        """Transform OKX tickers push → orchestrator snapshot dict."""

        if not data_list:
            return

        d = data_list[0]
        try:
            last_px = float(d.get("last", 0))
            bid1 = float(d.get("bidPx", 0))
            ask1 = float(d.get("askPx", 0))
            spread = round(ask1 - bid1, 6)
            ts_str = str(d.get("ts", "0"))
            ts_ms = int(float(ts_str))

            if last_px <= 0:
                return  # stale / empty tick — skip

            with self._lock:
                self._tick_count += 1
                self._last_ts = ts_ms

            snap = json.dumps(
                {
                    "ts_ms": ts_ms,
                    "inst_id": d.get("instId", self.inst_id),
                    "last_px": last_px,
                    "last_sz": float(d.get("lastSz", 0)),
                    "bid1": bid1,
                    "bid1_sz": float(d.get("bidSz", 0)),
                    "ask1": ask1,
                    "ask1_sz": float(d.get("askSz", 0)),
                    "spread": spread,
                    "tick_count": self._tick_count,
                }
            )
            with self._lock:
                self._latest_snap = snap

        except (ValueError, TypeError, KeyError) as exc:
            _log.warning("live_ws_tick_parse", extra={"err": str(exc)[:80]})
            # Skip malformed ticks — wait for the next push

    async def _cleanup(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.disconnect()
            except Exception:
                pass


__all__ = ["LiveOkxWsClient"]