"""
common/logging_setup.py — structured JSON logging + trace_id propagation
========================================================================

Every signal in V8 carries a ``trace_id`` (<=32 chars, OKX-safe). The Harness
injects it; this module makes it flow into every structured log line via a
contextvar, so a single trace can be grepped across WS ingest → features →
alpha → gating → MCTS → execution → settlement.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any, Optional

_trace_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_pulse_ctx: contextvars.ContextVar[int] = contextvars.ContextVar("pulse_id", default=0)


def new_trace_id(prefix: str = "v8") -> str:
    """Generate a trace_id guaranteed <= 32 chars (OKX clOrdId/trace constraint)."""
    tid = f"{prefix}{uuid.uuid4().hex}"
    return tid[:32]


def set_trace(trace_id: str) -> None:
    _trace_ctx.set(trace_id[:32])


def get_trace() -> str:
    return _trace_ctx.get()


def set_pulse(pulse_id: int) -> None:
    _pulse_ctx.set(int(pulse_id))


def get_pulse() -> int:
    return _pulse_ctx.get()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": _trace_ctx.get(),
            "pulse_id": _pulse_ctx.get(),
        }
        # merge any structured extras
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _StructAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]):
        extra = kwargs.pop("extra", {}) or {}
        fields = {k: v for k, v in extra.items()}
        kwargs["extra"] = {"extra_fields": fields}
        return msg, kwargs


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True


def get_logger(name: str) -> _StructAdapter:
    setup_logging()
    return _StructAdapter(logging.getLogger(name), {})


__all__ = [
    "new_trace_id",
    "set_trace",
    "get_trace",
    "set_pulse",
    "get_pulse",
    "setup_logging",
    "get_logger",
]
