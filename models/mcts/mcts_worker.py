"""
models/mcts/mcts_worker.py — Task 4d: MCTS Worker 调度器
=========================================================

MCTS Worker 负责：
    1. 管理 MctsPool (Rust) 或 EvThresholdEngine (Python 替身) 的生命周期
    2. 根据 backend 自动选择执行路径
    3. 提供统一的 run() 接口给 orchestrator
    4. 监控 MCTS 性能 (延迟/超时率)

架构：
    Orchestrator._on_bar_close()
        → MctsWorker.run(features_bytes, rollout_fn)
            → [native] MctsPool.run_sync() (Rust, <50ms)
            → [fallback] EvThresholdEngine.run_sync() (Python, <1ms)

接口契约：
    - run() 返回 JSON bytes (与 MctsPool.run_sync 格式一致)
    - 超时保护: native 走 MctsPool timeout_ms, fallback 无超时需求
    - 性能监控: latency_ms / timeout_count / avg_latency
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional

from common.engine import MctsPool, backend
from common.logging_setup import get_logger, get_trace
from models.mcts.ev_threshold import EvThresholdEngine, EvThresholdConfig
from models.mcts.mcts_config import MctsConfig, DEFAULT_MCTS_CONFIG, DEGRADED_MCTS_CONFIG

_log = get_logger("models.mcts.worker")


@dataclass
class MctsWorkerStats:
    """MCTS Worker 性能统计"""
    total_calls: int = 0
    timeout_count: int = 0
    fallback_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    _latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float, timeout: bool = False, fallback: bool = False):
        self.total_calls += 1
        if timeout:
            self.timeout_count += 1
        if fallback:
            self.fallback_count += 1
        self._latencies.append(latency_ms)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        if len(self._latencies) >= 100:
            sorted_lats = sorted(self._latencies)
            idx = int(len(sorted_lats) * 0.99)
            self.p99_latency_ms = sorted_lats[idx]

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "timeout_count": self.timeout_count,
            "fallback_count": self.fallback_count,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "timeout_rate": round(self.timeout_count / max(self.total_calls, 1), 4),
        }


class MctsWorker:
    """
    MCTS Worker 调度器

    用法：
        worker = MctsWorker()
        result_bytes = worker.run(features_bytes, rollout_fn)
        result = json.loads(result_bytes)
        action = result["best_action"]
    """

    def __init__(
        self,
        cfg: Optional[MctsConfig] = None,
        force_fallback: bool = False,
    ) -> None:
        self.cfg = cfg or DEFAULT_MCTS_CONFIG
        self._backend = backend()
        self._force_fallback = force_fallback
        self._use_native = (
            self._backend == "native" and not force_fallback
        )

        # 初始化引擎
        if self._use_native:
            self._pool = MctsPool(
                workers=self.cfg.num_workers,
                timeout_ms=self.cfg.timeout_ms,
            )
            self._ev_engine = None
            _log.info("mcts_worker_native", extra={"workers": self.cfg.num_workers})
        else:
            self._pool = None
            self._ev_engine = EvThresholdEngine()
            _log.info(
                "mcts_worker_fallback",
                extra={"backend": self._backend, "force": force_fallback},
            )

        self.stats = MctsWorkerStats()

    def run(
        self,
        features_bytes: bytes,
        rollout_fn: Callable[[bytes], bytes],
    ) -> bytes:
        """
        执行 MCTS 搜索或 EV 阈值评估

        Args:
            features_bytes: 50d 特征向量 (float32 packed)
            rollout_fn: rollout 函数

        Returns:
            JSON bytes (与 MctsPool.run_sync 格式一致)
        """
        t0 = time.perf_counter()

        try:
            if self._use_native and self._pool is not None:
                result_bytes = self._pool.run_sync(features_bytes, rollout_fn)
                latency = (time.perf_counter() - t0) * 1000
                self.stats.record(latency)

                # 超时检测
                result = json.loads(result_bytes)
                if result.get("best_action") == "hold" and "timeout" in result.get("details", {}).get("reason", ""):
                    self.stats.timeout_count += 1
                    _log.warning("mcts_timeout", extra={"latency_ms": round(latency, 1)})

                return result_bytes
            else:
                # Fallback 路径
                assert self._ev_engine is not None
                result_bytes = self._ev_engine.run_sync(features_bytes, rollout_fn)
                latency = (time.perf_counter() - t0) * 1000
                self.stats.record(latency, fallback=True)
                return result_bytes

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            _log.error(
                "mcts_worker_error",
                extra={"err": str(e), "latency_ms": round(latency, 1), "trace_id": get_trace()},
            )
            # 安全回退: hold
            fallback = {
                "best_action": "hold",
                "expected_value": 0.0,
                "best_position": 0.0,
                "num_simulations": 0,
                "search_depth": 0,
                "method": "error_fallback",
                "details": {"error": str(e), "trace_id": get_trace()},
            }
            self.stats.record(latency, fallback=True)
            return json.dumps(fallback).encode("utf-8")

    def degrade(self) -> None:
        """降级到替身模式 (Rust MCTS 出问题时调用)"""
        if self._use_native:
            _log.warning("mcts_degrading_to_fallback", extra={"trace_id": get_trace()})
            self._use_native = False
            self._force_fallback = True
            self._pool = None
            self._ev_engine = EvThresholdEngine()

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats.as_dict(),
            "backend": "native" if self._use_native else "fallback",
            "config": {
                "workers": self.cfg.num_workers,
                "timeout_ms": self.cfg.timeout_ms,
                "max_depth": self.cfg.max_depth,
                "num_simulations": self.cfg.num_simulations,
            },
        }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import struct

    worker = MctsWorker(force_fallback=True)
    print(f"Backend: {worker.get_stats()['backend']}")

    features = struct.pack(f"<{50}f", *([0.1] * 50))

    def mock_rollout(fb):
        return json.dumps({
            "predicted_return": 0.005,
            "confidence": 0.72,
            "uncertainty": 0.02,
            "market_state": [0.1, 0.02, 0.003, 0.0],
        }).encode("utf-8")

    for i in range(5):
        result_bytes = worker.run(features, mock_rollout)
        result = json.loads(result_bytes)
        print(f"  Run {i+1}: action={result['best_action']}, "
              f"ev={result['expected_value']:.6f}, pos={result['best_position']:.4f}")

    print(f"\nStats: {worker.get_stats()}")
    print("✓ MctsWorker self-test passed")
