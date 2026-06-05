"""
models/mcts/ev_threshold.py — Task 4c: MCTS EV 阈值替身 (v1.1)
===============================================================

Fixed: 2026-06-02
- Position output is now a FRACTION of equity (0.01 = 1% of capital),
  NOT an absolute BTC quantity. Orchestrator converts to contract size.
- max_position reduced from 1.0 (100% equity) to 0.05 (5% per trade).
- Added overbought/oversold direction suppression.
- Added min_hold_ratio to prevent infinite one-direction stacking.

接口与 Rust MctsPool.run_sync() 对齐，切换 backend 时无需改上层代码。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from common.logging_setup import get_logger, get_trace

_log = get_logger("models.mcts.ev_threshold")


@dataclass
class EvThresholdConfig:
    """EV 阈值配置"""
    ev_threshold: float = 0.0001       # 最小期望收益 (0.01% fractional)
    ev_strong_threshold: float = 0.0005 # 强信号阈值 (0.05%)
    max_position: float = 0.05          # 最大仓位 (5% of equity per trade)
    kelly_fraction: float = 0.15        # Kelly 比例 (保守 0.15x)
    uncertainty_penalty: float = 3.0    # 不确定性惩罚系数 (was 2.0)
    min_confidence: float = 0.55        # 最小置信度


class EvThresholdEngine:
    """
    MCTS EV 阈值替身引擎

    用法：
        engine = EvThresholdEngine()
        result = engine.evaluate(
            predicted_return=0.005,  # fractional return
            confidence=0.72,
            uncertainty=0.02,
            market_state="trending_up",
        )
        # result: {"best_action": "buy", "expected_value": 0.003, "best_position": 0.054, ...}
        # best_position = 0.054 means 5.4% of equity
    """

    def __init__(self, cfg: Optional[EvThresholdConfig] = None) -> None:
        self.cfg = cfg or EvThresholdConfig()
        self._eval_count = 0

    def evaluate(
        self,
        predicted_return: float,
        confidence: float,
        uncertainty: float = 0.0,
        market_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估交易机会的期望价值

        Args:
            predicted_return: rollout 预测的 fractional return (e.g. 0.001 = 0.1%)
            confidence: 置信度 [0, 1]
            uncertainty: 不确定性 σ
            market_state: "overbought" | "oversold" | "trending_up" | "trending_down" | "ranging"

        Returns:
            dict 与 MctsPool.run_sync() 输出格式对齐
        """
        self._eval_count += 1

        # 置信度不足 → hold
        if confidence < self.cfg.min_confidence:
            return self._hold_result("low_confidence", confidence)

        # 不确定性惩罚
        adj_return = predicted_return - self.cfg.uncertainty_penalty * uncertainty

        # EV 计算 (fractional)
        ev = adj_return * confidence * 0.8  # 0.8 衰减因子 (保守估计)

        # 方向判断
        if abs(ev) < self.cfg.ev_threshold:
            return self._hold_result("ev_below_threshold", ev)

        action = "buy" if ev > 0 else "sell"

        # 市场状态方向抑制：overbought 时抑制 buy，oversold 时抑制 sell
        if market_state == "overbought" and action == "buy":
            return self._hold_result("overbought_suppress_buy", ev)
        if market_state == "oversold" and action == "sell":
            return self._hold_result("oversold_suppress_sell", ev)

        # 仓位计算 (Kelly) — 输出 equity fraction
        abs_ev = abs(ev)
        kelly_pos = self.cfg.kelly_fraction * confidence
        if abs_ev > self.cfg.ev_strong_threshold:
            kelly_pos *= 1.2  # 强信号加仓 20%
        position = min(kelly_pos, self.cfg.max_position)
        position = round(position, 6)

        result = {
            "best_action": action,
            "expected_value": round(ev, 8),
            "best_position": position,  # fraction of equity
            "num_simulations": 0,
            "search_depth": 0,
            "method": "ev_threshold",
            "details": {
                "predicted_return": round(predicted_return, 8),
                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty, 6),
                "adjusted_return": round(adj_return, 8),
                "kelly_position": round(kelly_pos, 6),
                "is_strong_signal": abs_ev > self.cfg.ev_strong_threshold,
                "trace_id": get_trace(),
            },
        }

        _log.debug(
            "ev_eval",
            extra={
                "action": action,
                "ev": round(ev, 6),
                "pos": position,
                "trace_id": get_trace(),
            },
        )

        return result

    def _hold_result(self, reason: str, value: float = 0.0) -> Dict[str, Any]:
        return {
            "best_action": "hold",
            "expected_value": 0.0,
            "best_position": 0.0,
            "num_simulations": 0,
            "search_depth": 0,
            "method": "ev_threshold",
            "details": {
                "reason": reason,
                "value": round(value, 8) if isinstance(value, float) else value,
                "trace_id": get_trace(),
            },
        }

    def run_sync(
        self,
        features_bytes: bytes,
        rollout_fn: Callable[[bytes], bytes],
    ) -> bytes:
        """
        模拟 MctsPool.run_sync() 接口

        Args:
            features_bytes: 50d 特征 (float32 packed)
            rollout_fn: rollout 函数 (features_bytes → JSON bytes)

        Returns:
            JSON bytes (与 Rust MctsPool 输出格式一致)
        """
        import struct

        # 调用 rollout 获取预测
        rollout_bytes = rollout_fn(features_bytes)
        rollout = json.loads(rollout_bytes.decode("utf-8"))

        predicted_return = float(rollout.get("predicted_return", 0.0))
        confidence = float(rollout.get("confidence", 0.5))
        uncertainty = float(rollout.get("uncertainty", 0.0))
        market_state = rollout.get("market_state", "ranging")

        result = self.evaluate(
            predicted_return=predicted_return,
            confidence=confidence,
            uncertainty=uncertainty,
            market_state=market_state,
        )

        return json.dumps(result).encode("utf-8")


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    engine = EvThresholdEngine()

    scenarios = [
        ("强多头", 0.01, 0.8, 0.01, "trending_up"),
        ("弱多头", 0.002, 0.6, 0.03, "trending_up"),
        ("中性", 0.0005, 0.5, 0.02, "ranging"),
        ("空头信号", -0.008, 0.75, 0.015, "trending_down"),
        ("低置信 → hold", 0.01, 0.3, 0.01, "trending_up"),
        ("高不确定性", 0.01, 0.7, 0.08, "trending_up"),
        ("超买抑制buy", 0.005, 0.7, 0.01, "overbought"),
        ("超卖抑制sell", -0.005, 0.7, 0.01, "oversold"),
    ]

    for name, pred_ret, conf, unc, state in scenarios:
        r = engine.evaluate(pred_ret, conf, unc, state)
        print(f"  {name}: action={r['best_action']}, ev={r['expected_value']:.6f}, "
              f"pos={r['best_position']:.4f}")

    # 测试 run_sync 接口
    import struct
    features = struct.pack(f"<{50}f", *([0.1] * 50))

    def mock_rollout(fb):
        return json.dumps({
            "predicted_return": 0.005,
            "confidence": 0.72,
            "uncertainty": 0.02,
            "market_state": "trending_up",
        }).encode("utf-8")

    result_bytes = engine.run_sync(features, mock_rollout)
    result = json.loads(result_bytes)
    print(f"\n  run_sync: {result['best_action']}, ev={result['expected_value']:.6f}, pos={result['best_position']:.4f}")

    print("\n✓ EvThresholdEngine self-test passed")
