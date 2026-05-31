"""
harness/pipeline_v1.py �?Task 6: 无功能回归包装器 (Harness)
=========================================================

A zero-behaviour-change wrapper that threads a global ``trace_id`` through every
stage and emits structured JSON timing logs. Wrapping a function must NOT change
its result (functional no-op); it only adds observability + trace propagation.

Usage:
    h = Harness("btc5m")
    result = await h.stage("alpha", obi_engine.on_snapshot, snap)
    # or decorate:
    @h.wrap("gating")
    def run_gate(ctx): ...
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from common.logging_setup import get_logger, get_trace, new_trace_id, set_pulse, set_trace

_log = get_logger("harness")


class Harness:
    def __init__(self, name: str = "pipeline_v1") -> None:
        self.name = name

    def begin_pulse(self, pulse_id: int, trace_id: Optional[str] = None) -> str:
        tid = trace_id or new_trace_id()
        set_trace(tid)
        set_pulse(pulse_id)
        _log.info("pulse_begin", extra={"pulse_id": pulse_id, "harness": self.name})
        return tid

    @contextmanager
    def span(self, stage: str):
        t0 = time.perf_counter()
        err: Optional[str] = None
        try:
            yield
        except Exception as exc:  # observe then re-raise (no behaviour change)
            err = repr(exc)
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1000
            _log.info(
                "stage_done",
                extra={"stage": stage, "ms": round(dt, 3), "trace_id": get_trace(), "err": err},
            )

    def stage(self, stage: str, fn: Callable, *args, **kwargs) -> Any:
        """Run a sync callable inside a timed span; returns its result unchanged."""
        with self.span(stage):
            return fn(*args, **kwargs)

    async def astage(self, stage: str, coro_fn: Callable, *args, **kwargs) -> Any:
        """Run an async callable inside a timed span; returns its result unchanged."""
        t0 = time.perf_counter()
        err: Optional[str] = None
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            err = repr(exc)
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1000
            _log.info(
                "stage_done",
                extra={"stage": stage, "ms": round(dt, 3), "trace_id": get_trace(), "err": err},
            )

    def wrap(self, stage: str) -> Callable:
        """Decorator form of stage() �?functional no-op wrapper."""

        def _deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def _inner(*a, **k):
                with self.span(stage):
                    return fn(*a, **k)

            return _inner

        return _deco


__all__ = ["Harness"]
