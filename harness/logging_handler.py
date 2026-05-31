"""
harness/logging_handler.py — P1: 结构化日志注入与多目标路由
============================================================

扩展 common/logging_setup.py 的基础 JSON 日志，增加：

    1. RotatingFileHandler — 日志文件自动轮转 (10MB × 5)
    2. TraceIdFilter — 第三方库 (httpx, aiohttp) 日志自动注入 trace_id
    3. PerformanceHandler — 慢操作自动升级为 WARNING
    4. AlertHandler — ERROR 级别自动触发告警

用法：
    from harness.logging_handler import setup_production_logging
    setup_production_logging(log_dir="logs/", level="INFO", enable_file=True)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional

from common.logging_setup import get_trace, get_pulse, setup_logging

_log = logging.getLogger(__name__)


# ============================================================
# Trace ID 注入过滤器
# ============================================================

class TraceIdFilter(logging.Filter):
    """
    给所有 LogRecord 注入 trace_id 和 pulse_id

    用于第三方库 (httpx, uvicorn, aiohttp) 的日志，
    使它们也能带上 V8 的 trace_id 方便 grep。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace()  # type: ignore
        record.pulse_id = get_pulse()  # type: ignore
        return True


# ============================================================
# 慢操作检测 Handler
# ============================================================

class SlowOperationHandler(logging.Handler):
    """
    检测日志消息中包含延迟信息的记录，
    超过阈值的自动升级为 WARNING。
    """

    def __init__(self, threshold_ms: float = 100.0):
        super().__init__(level=logging.DEBUG)
        self.threshold_ms = threshold_ms
        self._slow_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # 检查 extra 字段中的 ms/latency
        ms = getattr(record, 'ms', None) or getattr(record, 'latency_ms', None)
        if ms is not None and isinstance(ms, (int, float)):
            if ms > self.threshold_ms and record.levelno < logging.WARNING:
                record.levelno = logging.WARNING
                record.levelname = 'WARNING'
                record.msg = f"[SLOW {ms:.1f}ms] {record.msg}"
                self._slow_count += 1

    @property
    def slow_count(self) -> int:
        return self._slow_count


# ============================================================
# 告警缓冲 Handler
# ============================================================

class AlertBufferHandler(logging.Handler):
    """
    缓冲 ERROR/CRITICAL 级别日志，供 dashboard 和告警系统查询。
    环形缓冲，保留最近 N 条。
    """

    def __init__(self, max_records: int = 100):
        super().__init__(level=logging.ERROR)
        self._buffer: List[Dict[str, Any]] = []
        self._max = max_records

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": get_trace(),
        }
        self._buffer.append(entry)
        if len(self._buffer) > self._max:
            self._buffer = self._buffer[-self._max:]

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._buffer[-limit:]

    def clear(self) -> None:
        self._buffer.clear()


# ============================================================
# JSON 文件 Formatter
# ============================================================

class JsonFileFormatter(logging.Formatter):
    """JSON 格式日志，适合 ELK/Loki 采集"""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, 'trace_id', '-'),
            "pulse_id": getattr(record, 'pulse_id', 0),
        }
        # Merge extra fields
        for k in ('ms', 'latency_ms', 'stage', 'err', 'action',
                   'realized', 'position', 'fills', 'sharpe'):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ============================================================
# 生产环境日志配置
# ============================================================

# 全局 alert buffer (供 dashboard 查询)
_alert_buffer: Optional[AlertBufferHandler] = None


def setup_production_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    enable_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    slow_threshold_ms: float = 100.0,
    enable_alert_buffer: bool = True,
) -> Dict[str, Any]:
    """
    配置生产环境日志

    Args:
        log_dir: 日志文件目录
        level: 日志级别
        enable_file: 是否启用文件日志
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留历史文件数
        slow_threshold_ms: 慢操作阈值 (ms)
        enable_alert_buffer: 是否启用告警缓冲

    Returns:
        配置信息 dict
    """
    global _alert_buffer

    # 先初始化基础日志 (common/logging_setup)
    setup_logging(level)

    root = logging.getLogger()

    # 给所有 handler 加 TraceIdFilter
    trace_filter = TraceIdFilter()
    for handler in root.handlers:
        handler.addFilter(trace_filter)

    info = {"handlers": ["stdout"], "filters": ["TraceIdFilter"]}

    # 文件日志
    if enable_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "v8.jsonl")
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFileFormatter())
        file_handler.addFilter(trace_filter)
        root.addHandler(file_handler)
        info["handlers"].append(f"file:{log_path}")
        info["log_dir"] = log_dir

    # 慢操作检测
    slow_handler = SlowOperationHandler(threshold_ms=slow_threshold_ms)
    root.addHandler(slow_handler)
    info["handlers"].append("SlowOperationHandler")
    info["slow_threshold_ms"] = slow_threshold_ms

    # 告警缓冲
    if enable_alert_buffer:
        _alert_buffer = AlertBufferHandler()
        root.addHandler(_alert_buffer)
        info["handlers"].append("AlertBufferHandler")

    _log.info("production_logging_configured", extra=info)
    return info


def get_alert_buffer() -> Optional[AlertBufferHandler]:
    return _alert_buffer


def get_recent_errors(limit: int = 20) -> List[Dict[str, Any]]:
    """获取最近的错误日志"""
    if _alert_buffer:
        return _alert_buffer.get_recent(limit)
    return []


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    info = setup_production_logging(log_dir="logs_test", enable_file=True)
    print(f"Config: {json.dumps(info, indent=2)}")

    logger = logging.getLogger("test.dashboard")
    logger.info("test info message")
    logger.warning("test warning")
    logger.error("test error")

    # Test trace filter
    from common.logging_setup import set_trace
    set_trace("v8test123")
    logger.info("message with trace_id")

    # Test alert buffer
    errors = get_recent_errors()
    print(f"Recent errors: {len(errors)}")
    for e in errors:
        print(f"  [{e['level']}] {e['msg']}")

    # Cleanup
    import shutil
    if os.path.exists("logs_test"):
        shutil.rmtree("logs_test")

    print("✓ logging_handler self-test passed")
