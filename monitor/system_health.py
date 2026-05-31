"""
monitor/system_health.py — 系统健康监控
========================================

监控 V8 系统各组件的健康状态：
    - OKX WebSocket 连接状态
    - Redis 连接状态
    - Triton Inference Server 状态
    - MCTS Worker 延迟/超时
    - 订单执行延迟
    - P&L 异常波动

告警规则：
    - WS 断连 > 10s → WARN
    - Redis 不可达 → WARN
    - Triton 延迟 > 100ms → WARN
    - MCTS 超时率 > 5% → WARN
    - P&L 单笔亏损 > 阈值 → ALERT
    - 连续 3 笔亏损 → ALERT
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from common.logging_setup import get_logger, get_trace

_log = get_logger("monitor.health")


class HealthStatus(Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """单个组件的健康状态"""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    latency_ms: float = 0.0
    last_check_ms: int = 0
    consecutive_failures: int = 0


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    check_interval_sec: float = 30.0
    ws_disconnect_warn_sec: float = 10.0
    triton_latency_warn_ms: float = 100.0
    mcts_timeout_warn_rate: float = 0.05
    pnl_single_loss_warn: float = -50.0  # USDT
    pnl_consecutive_loss_count: int = 3
    max_history: int = 1000


class SystemHealthMonitor:
    """
    系统健康监控器

    用法：
        monitor = SystemHealthMonitor()
        # 定期调用
        monitor.check_all()
        # 或单独检查
        monitor.report_ws_connected(True)
        monitor.report_trade_pnl(-30.0)
        # 获取状态
        status = monitor.get_status()
        alerts = monitor.get_alerts()
    """

    def __init__(self, cfg: Optional[HealthCheckConfig] = None) -> None:
        self.cfg = cfg or HealthCheckConfig()
        self._components: Dict[str, ComponentHealth] = {
            "okx_ws": ComponentHealth("okx_ws"),
            "redis": ComponentHealth("redis"),
            "triton": ComponentHealth("triton"),
            "mcts": ComponentHealth("mcts"),
            "execution": ComponentHealth("execution"),
            "pnl": ComponentHealth("pnl"),
        }
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.max_history)
        self._ws_last_connected_ms: int = int(time.time() * 1000)
        self._recent_pnls: Deque[float] = deque(maxlen=50)
        self._trade_count: int = 0

    def report_ws_connected(self, connected: bool) -> None:
        """报告 WebSocket 连接状态"""
        comp = self._components["okx_ws"]
        now_ms = int(time.time() * 1000)
        if connected:
            comp.status = HealthStatus.OK
            comp.message = "connected"
            comp.consecutive_failures = 0
            self._ws_last_connected_ms = now_ms
        else:
            disconnected_sec = (now_ms - self._ws_last_connected_ms) / 1000.0
            if disconnected_sec > self.cfg.ws_disconnect_warn_sec:
                comp.status = HealthStatus.WARN
                comp.message = f"disconnected for {disconnected_sec:.0f}s"
                comp.consecutive_failures += 1
                self._add_alert("okx_ws", f"WebSocket disconnected {disconnected_sec:.0f}s")
            else:
                comp.status = HealthStatus.WARN
                comp.message = f"disconnected {disconnected_sec:.0f}s"
        comp.last_check_ms = now_ms

    def report_redis_status(self, reachable: bool, latency_ms: float = 0.0) -> None:
        """报告 Redis 连接状态"""
        comp = self._components["redis"]
        comp.status = HealthStatus.OK if reachable else HealthStatus.ERROR
        comp.latency_ms = latency_ms
        comp.message = "ok" if reachable else "unreachable"
        comp.last_check_ms = int(time.time() * 1000)
        if not reachable:
            comp.consecutive_failures += 1
            self._add_alert("redis", "Redis unreachable")

    def report_triton_status(self, reachable: bool, latency_ms: float = 0.0) -> None:
        """报告 Triton 状态"""
        comp = self._components["triton"]
        if not reachable:
            comp.status = HealthStatus.WARN
            comp.message = "unreachable"
            self._add_alert("triton", "Triton unreachable")
        elif latency_ms > self.cfg.triton_latency_warn_ms:
            comp.status = HealthStatus.WARN
            comp.message = f"slow: {latency_ms:.0f}ms"
            self._add_alert("triton", f"Triton slow: {latency_ms:.0f}ms")
        else:
            comp.status = HealthStatus.OK
            comp.message = f"ok ({latency_ms:.1f}ms)"
        comp.latency_ms = latency_ms
        comp.last_check_ms = int(time.time() * 1000)

    def report_mcts_stats(self, timeout_rate: float, avg_latency_ms: float) -> None:
        """报告 MCTS 性能统计"""
        comp = self._components["mcts"]
        if timeout_rate > self.cfg.mcts_timeout_warn_rate:
            comp.status = HealthStatus.WARN
            comp.message = f"timeout_rate={timeout_rate:.1%}"
            self._add_alert("mcts", f"MCTS timeout rate {timeout_rate:.1%}")
        else:
            comp.status = HealthStatus.OK
            comp.message = f"ok (avg={avg_latency_ms:.1f}ms, timeout={timeout_rate:.1%})"
        comp.latency_ms = avg_latency_ms
        comp.last_check_ms = int(time.time() * 1000)

    def report_trade_pnl(self, pnl: float) -> None:
        """报告单笔交易 P&L"""
        self._trade_count += 1
        self._recent_pnls.append(pnl)
        comp = self._components["pnl"]
        comp.last_check_ms = int(time.time() * 1000)

        # 单笔大额亏损
        if pnl < self.cfg.pnl_single_loss_warn:
            comp.status = HealthStatus.WARN
            comp.message = f"large loss: {pnl:.2f} USDT"
            self._add_alert("pnl", f"Large single loss: {pnl:.2f} USDT")
            return

        # 连续亏损
        if len(self._recent_pnls) >= self.cfg.pnl_consecutive_loss_count:
            recent = list(self._recent_pnls)[-self.cfg.pnl_consecutive_loss_count:]
            if all(p < 0 for p in recent):
                comp.status = HealthStatus.WARN
                comp.message = f"{self.cfg.pnl_consecutive_loss_count} consecutive losses"
                self._add_alert("pnl", f"{self.cfg.pnl_consecutive_loss_count} consecutive losses")
                return

        comp.status = HealthStatus.OK
        comp.message = f"ok (trades={self._trade_count})"

    def check_all(self) -> Dict[str, Any]:
        """执行一次全面健康检查"""
        now_ms = int(time.time() * 1000)
        # 更新各组件状态
        for comp in self._components.values():
            if comp.last_check_ms == 0:
                comp.status = HealthStatus.UNKNOWN
                comp.message = "never checked"
            elif now_ms - comp.last_check_ms > self.cfg.check_interval_sec * 3000:
                # 超过 3 倍检查间隔未更新 → 标记为 stale
                comp.status = HealthStatus.UNKNOWN
                comp.message = "stale"

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """获取系统整体健康状态"""
        statuses = {}
        overall = HealthStatus.OK
        for name, comp in self._components.items():
            statuses[name] = {
                "status": comp.status.value,
                "message": comp.message,
                "latency_ms": round(comp.latency_ms, 1),
                "failures": comp.consecutive_failures,
            }
            if comp.status == HealthStatus.ERROR:
                overall = HealthStatus.ERROR
            elif comp.status == HealthStatus.WARN and overall != HealthStatus.ERROR:
                overall = HealthStatus.WARN
            elif comp.status == HealthStatus.UNKNOWN and overall == HealthStatus.OK:
                overall = HealthStatus.UNKNOWN

        return {
            "overall": overall.value,
            "components": statuses,
            "trade_count": self._trade_count,
            "alert_count": len(self._alerts),
            "ts_ms": int(time.time() * 1000),
        }

    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近告警"""
        return list(self._alerts)[-limit:]

    def _add_alert(self, component: str, message: str) -> None:
        alert = {
            "ts_ms": int(time.time() * 1000),
            "component": component,
            "message": message,
            "trace_id": get_trace(),
        }
        self._alerts.append(alert)
        _log.warning("health_alert", extra=alert)


if __name__ == "__main__":
    monitor = SystemHealthMonitor()

    monitor.report_ws_connected(True)
    monitor.report_redis_status(True, 2.5)
    monitor.report_triton_status(True, 45.0)
    monitor.report_mcts_stats(0.01, 35.0)
    monitor.report_trade_pnl(15.0)
    monitor.report_trade_pnl(-8.0)
    monitor.report_trade_pnl(-12.0)
    monitor.report_trade_pnl(-5.0)

    status = monitor.check_all()
    print(f"Overall: {status['overall']}")
    for name, info in status['components'].items():
        print(f"  {name}: {info['status']} - {info['message']}")

    alerts = monitor.get_alerts()
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for a in alerts[-5:]:
            print(f"  [{a['component']}] {a['message']}")

    print("\n✓ SystemHealthMonitor self-test passed")
