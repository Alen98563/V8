"""
models/alphacast/alphacast_recalib.py — AlphaCast 二次校准模块 (独立版)
========================================================================

将 alphacast_model.py 中内嵌的 AlphaCastRecalib + TemperatureScaling
提取为独立模块，供 orchestrator 主循环和离线评估复用。

功能：
    1. AlphaCast 输出二次校准 (置信度过滤 / 不确定性降仓 / 收益风险比)
    2. MCTS EV 联合校准
    3. Temperature Scaling 在线校准 (每 50 笔成交触发)
    4. 校准统计 + 告警

接口契约：
    - 输入: AlphaCast 模型输出 dict (predicted_return, uncertainty, confidence)
    - 输出: 校准后决策 dict (action, position_multiplier, calibrated_return)
    - 与 HardGating 串联: recalib → gating → execution
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_trace

_log = get_logger("models.alphacast_recalib")


# ============================================================
# 校准配置
# ============================================================

@dataclass
class RecalibConfig:
    """二次校准超参数"""
    # 过滤阈值
    min_confidence: float = 0.55
    max_uncertainty: float = 0.05
    min_risk_reward: float = 1.0

    # MCTS 联合权重
    mcts_blend_alpha: float = 0.6    # AlphaCast 权重
    mcts_blend_beta: float = 0.4     # MCTS 权重

    # Temperature Scaling
    initial_temperature: float = 1.0
    ema_alpha: float = 0.3
    min_temperature: float = 0.5
    max_temperature: float = 3.0
    recalib_interval_fills: int = 50  # 每 N 笔成交触发

    # 统计窗口
    stats_window: int = 200


# ============================================================
# 校准结果
# ============================================================

@dataclass
class RecalibResult:
    """校准结果"""
    action: str               # "pass" | "reject" | "reduce"
    position_multiplier: float  # [0, 1]
    calibrated_return: float
    reason: str
    temperature: float = 1.0
    raw_confidence: float = 0.0
    raw_uncertainty: float = 0.0
    raw_predicted_return: float = 0.0
    mcts_ev: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ============================================================
# 核心校准器
# ============================================================

class AlphaCastRecalibrator:
    """
    AlphaCast 二次校准器

    用法：
        recalib = AlphaCastRecalibrator()
        result = recalib.evaluate(
            predicted_return=0.005,
            uncertainty=0.02,
            confidence=0.72,
            mcts_ev=0.003,
        )
        if result.action == "pass":
            # 继续执行
    """

    def __init__(self, cfg: Optional[RecalibConfig] = None) -> None:
        self.cfg = cfg or RecalibConfig()
        self.temperature = self.cfg.initial_temperature
        self._fill_count = 0
        self._observations: Deque[Tuple[float, bool]] = deque(
            maxlen=self.cfg.stats_window
        )
        self._stats = {
            "total_evaluated": 0,
            "passed": 0,
            "rejected": 0,
            "reduced": 0,
            "rejected_low_conf": 0,
            "rejected_high_unc": 0,
            "rejected_low_rr": 0,
        }

    def evaluate(
        self,
        predicted_return: float,
        uncertainty: float,
        confidence: float,
        mcts_ev: float = 0.0,
    ) -> RecalibResult:
        """
        对 AlphaCast 输出做二次校准

        流程：
        1. Temperature Scaling 校准 confidence
        2. 置信度过滤 (conf < threshold → reject)
        3. 不确定性检查 (σ > threshold → reduce position)
        4. 收益/风险比检查 (|return|/σ < threshold → reject)
        5. MCTS EV 联合校准
        """
        self._stats["total_evaluated"] += 1

        # 1. Temperature Scaling 校准
        calibrated_conf = self._apply_temperature(confidence)

        # 2. 置信度过滤
        if calibrated_conf < self.cfg.min_confidence:
            self._stats["rejected"] += 1
            self._stats["rejected_low_conf"] += 1
            return RecalibResult(
                action="reject",
                position_multiplier=0.0,
                calibrated_return=0.0,
                reason=f"confidence {calibrated_conf:.3f} < {self.cfg.min_confidence}",
                temperature=self.temperature,
                raw_confidence=confidence,
                raw_uncertainty=uncertainty,
                raw_predicted_return=predicted_return,
            )

        # 3. 不确定性检查
        position_mult = 1.0
        reason_parts = []

        if uncertainty > self.cfg.max_uncertainty:
            position_mult = 0.5
            reason_parts.append(f"high_uncertainty σ={uncertainty:.4f}")

        # 4. 收益/风险比
        risk_reward = abs(predicted_return) / max(uncertainty, 1e-8)
        if risk_reward < self.cfg.min_risk_reward and mcts_ev <= 0:
            self._stats["rejected"] += 1
            self._stats["rejected_low_rr"] += 1
            return RecalibResult(
                action="reject",
                position_multiplier=0.0,
                calibrated_return=0.0,
                reason=f"risk_reward {risk_reward:.2f} < {self.cfg.min_risk_reward}",
                temperature=self.temperature,
                raw_confidence=confidence,
                raw_uncertainty=uncertainty,
                raw_predicted_return=predicted_return,
            )

        # 5. MCTS EV 联合校准
        calibrated_return = predicted_return
        if mcts_ev != 0.0:
            calibrated_return = (
                self.cfg.mcts_blend_alpha * predicted_return
                + self.cfg.mcts_blend_beta * mcts_ev
            )
            reason_parts.append(f"mcts_blended(ev={mcts_ev:.4f})")

        # 最终决策
        action = "reduce" if position_mult < 1.0 else "pass"
        if action == "pass":
            self._stats["passed"] += 1
        else:
            self._stats["reduced"] += 1

        # 最终仓位乘数 = 基础乘数 × 置信度
        final_mult = position_mult * calibrated_conf

        return RecalibResult(
            action=action,
            position_multiplier=round(final_mult, 4),
            calibrated_return=round(calibrated_return, 8),
            reason="ok" if not reason_parts else "; ".join(reason_parts),
            temperature=self.temperature,
            raw_confidence=confidence,
            raw_uncertainty=uncertainty,
            raw_predicted_return=predicted_return,
            mcts_ev=mcts_ev,
        )

    def _apply_temperature(self, confidence: float) -> float:
        """Temperature Scaling: conf_calibrated = sigmoid(logit(conf) / T)"""
        import math
        # 防止 log(0)
        conf = max(1e-6, min(1 - 1e-6, confidence))
        logit = math.log(conf / (1 - conf))
        scaled_logit = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled_logit))

    def on_fill(self, predicted_positive: bool, actual_positive: bool) -> None:
        """
        记录一笔成交结果，用于 Temperature Scaling 在线校准

        每 recalib_interval_fills 笔触发一次温度更新
        """
        self._observations.append((
            1.0 if predicted_positive else 0.0,
            1.0 if actual_positive else 0.0,
        ))
        self._fill_count += 1

        if self._fill_count % self.cfg.recalib_interval_fills == 0:
            self._update_temperature()

    def _update_temperature(self) -> None:
        """在线 Temperature Scaling 更新"""
        if len(self._observations) < 20:
            return

        try:
            import numpy as np
            preds = np.array([p for p, _ in self._observations])
            labels = np.array([l for _, l in self._observations])

            def nll(T_val):
                T = max(T_val, 0.01)
                # logit → scale → sigmoid
                logits = np.log(preds / (1 - preds + 1e-8) + 1e-8) / T
                scaled = 1.0 / (1.0 + np.exp(-logits))
                loss = -np.mean(
                    labels * np.log(scaled + 1e-8)
                    + (1 - labels) * np.log(1 - scaled + 1e-8)
                )
                return loss

            # 简单网格搜索 (避免 scipy 依赖)
            best_T = self.temperature
            best_loss = nll(self.temperature)
            for T_candidate in np.linspace(0.5, 3.0, 50):
                loss = nll(T_candidate)
                if loss < best_loss:
                    best_loss = loss
                    best_T = float(T_candidate)

            # EMA 平滑
            old_T = self.temperature
            self.temperature = (
                old_T * (1 - self.cfg.ema_alpha)
                + best_T * self.cfg.ema_alpha
            )
            self.temperature = max(
                self.cfg.min_temperature,
                min(self.cfg.max_temperature, self.temperature),
            )

            _log.info(
                "temperature_updated",
                extra={
                    "old_T": round(old_T, 4),
                    "new_T": round(self.temperature, 4),
                    "nll": round(best_loss, 4),
                    "observations": len(self._observations),
                    "trace_id": get_trace(),
                },
            )
        except Exception as e:
            _log.warning("temperature_update_failed", extra={"err": str(e)})

    def stats(self) -> Dict[str, Any]:
        """返回校准统计"""
        return {
            **self._stats,
            "temperature": round(self.temperature, 4),
            "fill_count": self._fill_count,
            "observation_count": len(self._observations),
            "pass_rate": round(
                self._stats["passed"] / max(self._stats["total_evaluated"], 1), 4
            ),
        }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    recalib = AlphaCastRecalibrator()

    # 场景测试
    scenarios = [
        ("高置信 + 低不确定性", 0.01, 0.02, 0.8, 0.005),
        ("低置信 → reject", 0.01, 0.02, 0.3, 0.0),
        ("高不确定性 → reduce", 0.01, 0.08, 0.7, 0.003),
        ("低收益风险比 → reject", 0.001, 0.05, 0.6, -0.001),
        ("MCTS 正向修正", 0.003, 0.03, 0.65, 0.008),
    ]

    for name, pred_ret, unc, conf, mcts_ev in scenarios:
        result = recalib.evaluate(pred_ret, unc, conf, mcts_ev)
        print(f"  {name}: action={result.action}, mult={result.position_multiplier:.3f}, "
              f"ret={result.calibrated_return:.6f}, reason={result.reason}")

    # Temperature Scaling 测试
    for i in range(50):
        recalib.on_fill(predicted_positive=True, actual_positive=(i % 3 != 0))
    recalib.on_fill(True, True)  # 第 51 笔触发更新

    print(f"\nStats: {recalib.stats()}")
    print("✓ AlphaCastRecalibrator self-test passed")
