"""
data/shm_bridge.py �?Task 3: zero-copy SHM接驳 (Qwen glue)
=========================================================

Wraps DeepSeek's ``ShmBridge`` (mmap2 /dev/shm/qts_btc5m). On native backend we
read a DLPack capsule and zero-copy into a torch tensor; ``get_raw_ptr`` gives a
numpy view over the Rust-owned float32 buffer with no copy. On the fallback
backend the same API works against a pure-Python ring buffer so the pipeline and
tests run anywhere.

Contract (v8_core_engine.pyi):
    bridge.get_window(secs) -> DLPack capsule          # torch.from_dlpack(...)
    bridge.get_raw_ptr(secs) -> (int ptr, (rows, cols))# np.frombuffer(...)
    bridge.get_latest() -> Optional[bytes]             # MarketSnapshot JSON
    bridge.push_snapshot(ts_ms, features)
    bridge.stats() -> (n, cap);  bridge.inst_id() -> str
"""

from __future__ import annotations

import ctypes
import json
import time
from typing import Any, Optional, Tuple

from common.engine import ShmBridge, is_native
from common.logging_setup import get_logger

_log = get_logger("data.shm_bridge")


class ShmReader:
    """High-speed reader over the Rust SHM ring buffer.

    Target: single feature-window read < 1µs on the native path (DLPack capsule
    is a pointer hand-off, no element copy).
    """

    def __init__(self, inst_id: str = "BTC-USDT-SWAP") -> None:
        self._bridge = ShmBridge(inst_id)
        self._native = is_native()
        _log.info("shm_bridge_open", extra={"native": self._native, "inst_id": self.inst_id})

    # ----- write side (used by feature pump / tests) -------------------------
    def push(self, ts_ms: int, features: list[float]) -> None:
        self._bridge.push_snapshot(int(ts_ms), [float(x) for x in features])

    # ----- read side ---------------------------------------------------------
    def window_tensor(self, secs: int = 60) -> Any:
        """Return the recent window as a torch tensor (zero-copy on native)."""
        cap = self._bridge.get_window(secs)
        if self._native:
            import torch

            return torch.from_dlpack(cap)
        # fallback: cap is already an ndarray (or list); convert lazily
        try:
            import torch

            import numpy as np

            arr = cap if isinstance(cap, (list,)) is False else np.asarray(cap, dtype="float32")
            return torch.as_tensor(arr)
        except Exception:
            return cap  # numpy ndarray or list-of-lists

    def window_numpy(self, secs: int = 60) -> Any:
        """Zero-copy numpy view via the raw pointer path."""
        import numpy as np

        ptr, (rows, cols) = self._bridge.get_raw_ptr(secs)
        if rows == 0 or cols == 0:
            return np.zeros((0, cols), dtype=np.float32)
        if self._native:
            # Wrap Rust-owned memory: build a ctypes array over the address,
            # then np.frombuffer -> reshape. No element copy.
            n = rows * cols
            buf = (ctypes.c_float * n).from_address(ptr)
            arr = np.frombuffer(buf, dtype=np.float32, count=n).reshape(rows, cols)
            return arr
        # fallback ShmBridge already returns a real ctypes buffer address it owns
        n = rows * cols
        buf = (ctypes.c_float * n).from_address(ptr)
        return np.frombuffer(buf, dtype=np.float32, count=n).reshape(rows, cols)

    def latest_snapshot(self) -> Optional[dict]:
        raw = self._bridge.get_latest()
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def benchmark_read(self, secs: int = 60, iters: int = 10_000) -> float:
        """Return mean per-read latency in microseconds."""
        # warm
        for _ in range(100):
            self._bridge.get_window(secs)
        t0 = time.perf_counter()
        for _ in range(iters):
            self._bridge.get_window(secs)
        return (time.perf_counter() - t0) / iters * 1e6

    @property
    def inst_id(self) -> str:
        return self._bridge.inst_id()

    @property
    def stats(self) -> Tuple[int, int]:
        return self._bridge.stats()


def _native_takes_inst() -> bool:
    """Both native and fallback now accept inst_id. Always True."""
    return True


__all__ = ["ShmReader"]
