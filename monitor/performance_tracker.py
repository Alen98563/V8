"""
monitor/performance_tracker.py — 性能追踪器
=============================================

追踪 V8 系统各阶段的性能指标：
    - 各 pipeline 阶段延迟 (ms)
    - 端到端延迟 (tick → fill)
    - 吞吐量 (ticks/sec, pulses/min)
    - 内存使用
    - CPU 使用率

数据存储在环形缓冲区，支持查询最近 N 分钟统计。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from common.logging_setup import get_logger

_log = get_logger("monitor.performance")


@dataclass
class LatencySample:
    """单次延迟采样"""
    ts_ms: int
    stage: str
    latency_ms: float


@dataclass
class PerformanceConfig:
    """性能追踪配置"""
    ring_buffer_size: int = 10000        # 环形缓冲区大小
    slow_threshold_ms: Dict[str, float] = field(default_factory=lambda: {
        "alpha_obi": 5.0,
        "alpha_funding": 10.0,
        "mcts": 100.0,
        "gating": 2.0,
        "execution": 50.0,
        "settlement": 5.0,
        "feature_assemble": 3.0,
        "recalib": 2.0,
        "total_pulse": 500.0,
    })
    report_interval_sec: float = 60.0    # 性能报告间隔


class PerformanceTracker:
    """
    性能追踪器

    用法：
        tracker = PerformanceTracker()
        # 记录延迟
        tracker.record("mcts", 35.2)
        tracker.record("total_pulse", 120.5)
        # 查询统计
        stats = tracker.get_stats("mcts")
        all_stats = tracker.get_all_stats()
    """

    def __init__(self, cfg: Optional[PerformanceConfig] = None) -> None:
        self.cfg = cfg or PerformanceConfig()
        self._samples: Deque[LatencySample] = deque(maxlen=self.cfg.ring_buffer_size)
        self._counters: Dict[str, int] = {}
        self._start_ms: int = int(time.time() * 1000)
        self._last_report_ms: int = self._start_ms
        self._slow_count: Dict[str, int] = {}

    def record(self, stage: str, latency_ms: float) -> None:
        """记录一次延迟采样"""
        now_ms = int(time.time() * 1000)
        self._samples.append(LatencySample(ts_ms=now_ms, stage=stage, latency_ms=latency_ms))
        self._counters[stage] = self._counters.get(stage, 0) + 1

        # 慢操作检测
        threshold = self.cfg.slow_threshold_ms.get(stage, 1000.0)
        if latency_ms > threshold:
            self._slow_count[stage] = self._slow_count.get(stage, 0) + 1
            if self._slow_count[stage] <= 3:  # 只告警前 3 次
                _log.warning(
                    "slow_stage",
                    extra={
                        "stage": stage,
                        "latency_ms": round(latency_ms, 1),
                        "threshold_ms": threshold,
                    },
                )

    def record_tick(self) -> None:
        """记录一次 tick (用于计算吞吐量)"""
        self._counters["_ticks"] = self._counters.get("_ticks", 0) + 1

    def record_pulse(self) -> None:
        """记录一次 pulse"""
        self._counters["_pulses"] = self._counters.get("_pulses", 0) + 1

    def get_stats(self, stage: str, window_sec: float = 60.0) -> Dict[str, Any]:
        """获取指定阶段的统计"""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_sec * 1000)

        relevant = [
            s for s in self._samples
            if s.stage == stage and s.ts_ms >= cutoff
        ]

        if not relevant:
            return {
                "stage": stage,
                "count": 0,
                "window_sec": window_sec,
            }

        latencies = sorted([s.latency_ms for s in relevant])
        n = len(latencies)

        return {
            "stage": stage,
            "count": n,
            "window_sec": window_sec,
            "avg_ms": round(sum(latencies) / n, 3),
            "min_ms": round(latencies[0], 3),
            "max_ms": round(latencies[-1], 3),
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[int(n * 0.95)], 3) if n >= 20 else round(latencies[-1], 3),
            "p99_ms": round(latencies[int(n * 0.99)], 3) if n >= 100 else round(latencies[-1], 3),
            "slow_count": self._slow_count.get(stage, 0),
        }

    def get_all_stats(self, window_sec: float = 60.0) -> Dict[str, Any]:
        """获取所有阶段的统计"""
        stages = set(s.stage for s in self._samples)
        stats = {stage: self.get_stats(stage, window_sec) for stage in stages}

        # 吞吐量
        elapsed_sec = (int(time.time() * 1000) - self._start_ms) / 1000.0
        stats["_throughput"] = {
            "ticks_per_sec": round(self._counters.get("_ticks", 0) / max(elapsed_sec, 1), 2),
            "pulses_per_min": round(
                self._counters.get("_pulses", 0) / max(elapsed_sec / 60, 1), 2
            ),
            "uptime_sec": round(elapsed_sec, 0),
            "total_ticks": self._counters.get("_ticks", 0),
            "total_pulses": self._counters.get("_pulses", 0),
        }

        return stats

    def maybe_report(self) -> Optional[Dict[str, Any]]:
        """如果超过报告间隔则生成性能报告"""
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_report_ms < self.cfg.report_interval_sec * 1000:
            return None

        report = self.get_all_stats(window_sec=self.cfg.report_interval_sec)
        self._last_report_ms = now_ms

        _log.info(
            "performance_report",
            extra={
                "ticks_per_sec": report["_throughput"]["ticks_per_sec"],
                "pulses_per_min": report["_throughput"]["pulses_per_min"],
                "uptime_sec": report["_throughput"]["uptime_sec"],
            },
        )

        return report

    def reset(self) -> None:
        """重置所有统计"""
        self._samples.clear()
        self._counters.clear()
        self._slow_count.clear()
        self._start_ms = int(time.time() * 1000)
        self._last_report_ms = self._start_ms


if __name__ == "__main__":
    import random
    random.seed(42)

    tracker = PerformanceTracker()

    # 模拟 100 次 pulse
    for i in range(100):
        tracker.record("alpha_obi", random.uniform(0.5, 3.0))
        tracker.record("mcts", random.uniform(10, 80))
        tracker.record("gating", random.uniform(0.1, 1.0))
        tracker.record("execution", random.uniform(5, 30))
        tracker.record("total_pulse", random.uniform(50, 200))
        tracker.record_pulse()
        for _ in range(20):
            tracker.record_tick()

    stats = tracker.get_all_stats()
    print("Performance Stats:")
    for stage, info in stats.items():
        if stage.startswith("_"):
            print(f"  {stage}: {info}")
        else:
            print(f"  {stage}: avg={info.get('avg_ms', 0):.1f}ms, "
                  f"p95={info.get('p95_ms', 0):.1f}ms, "
                  f"count={info.get('count', 0)}")

    print("\n✓ PerformanceTracker self-test passed")
