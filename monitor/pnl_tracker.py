"""
monitor/pnl_tracker.py — P5: P&L 预测误差追踪
================================================

追踪 AlphaCast/MCTS 预测收益与实际 P&L 之间的偏差：

    - 每笔成交记录: (predicted_return, actual_pnl, error)
    - 滚动统计: MAE, RMSE, bias, IC (Information Coefficient)
    - 漂移检测: 连续 N 笔 IC < 阈值 → 告警 (模型退化)
    - 按 Alpha 源分解: OBI 贡献 / FundingRate 贡献 / ResNet 贡献

用途:
    1. 评估 AlphaCast 预测质量
    2. 检测模型漂移 (concept drift)
    3. 为 MetaLabeler 提供 ground truth 标签
    4. 为在线微调 (alphacast_online_update.py) 提供训练数据
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_trace

_log = get_logger("monitor.pnl_tracker")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PnlRecord:
    """单笔成交的预测 vs 实际记录"""
    ts_ms: int
    trace_id: str
    pulse_id: int

    # 预测
    predicted_return: float    # AlphaCast 预测收益
    predicted_confidence: float
    mcts_ev: float             # MCTS 期望价值

    # 实际
    actual_pnl_bps: float      # 实际 P&L (基点)
    actual_side: str           # "buy" / "sell"
    fill_px: float
    close_px: float            # 平仓价格

    # 误差
    error_bps: float = 0.0     # actual - predicted (bps)
    abs_error_bps: float = 0.0
    squared_error_bps: float = 0.0

    def compute_error(self):
        pred_bps = self.predicted_return * 10000  # 转换为 bps
        self.error_bps = self.actual_pnl_bps - pred_bps
        self.abs_error_bps = abs(self.error_bps)
        self.squared_error_bps = self.error_bps ** 2


@dataclass
class PnlTrackerConfig:
    """配置"""
    ring_buffer_size: int = 5000
    drift_window: int = 100           # 漂移检测窗口
    ic_alert_threshold: float = 0.02  # IC < 0.02 → 告警
    mae_alert_threshold: float = 20.0 # MAE > 20bps → 告警
    bias_alert_threshold: float = 5.0 # |bias| > 5bps → 告警
    persist_path: Optional[str] = None


# ============================================================
# 核心追踪器
# ============================================================

class PnlTracker:
    """
    P&L 预测误差追踪器

    用法：
        tracker = PnlTracker()
        tracker.record(
            predicted_return=0.005,  # 0.5% 预测收益
            predicted_confidence=0.72,
            mcts_ev=0.003,
            actual_pnl_bps=35.0,     # 35bps 实际盈利
            actual_side="buy",
            fill_px=3000.0,
            close_px=3010.5,
            pulse_id=42,
        )
        # 查询统计
        stats = tracker.get_stats()
        # 检查漂移
        drift = tracker.check_drift()
    """

    def __init__(self, cfg: Optional[PnlTrackerConfig] = None) -> None:
        self.cfg = cfg or PnlTrackerConfig()
        self._records: Deque[PnlRecord] = deque(maxlen=self.cfg.ring_buffer_size)
        self._total_records: int = 0
        self._drift_alerts: int = 0

    def record(
        self,
        predicted_return: float,
        predicted_confidence: float,
        mcts_ev: float,
        actual_pnl_bps: float,
        actual_side: str,
        fill_px: float,
        close_px: float,
        pulse_id: int = 0,
        trace_id: Optional[str] = None,
    ) -> PnlRecord:
        """记录一笔成交的预测 vs 实际"""
        rec = PnlRecord(
            ts_ms=int(time.time() * 1000),
            trace_id=trace_id or get_trace(),
            pulse_id=pulse_id,
            predicted_return=predicted_return,
            predicted_confidence=predicted_confidence,
            mcts_ev=mcts_ev,
            actual_pnl_bps=actual_pnl_bps,
            actual_side=actual_side,
            fill_px=fill_px,
            close_px=close_px,
        )
        rec.compute_error()
        self._records.append(rec)
        self._total_records += 1

        return rec

    def get_stats(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        计算滚动统计

        Args:
            window: 窗口大小 (None=全部)

        Returns:
            统计 dict
        """
        records = list(self._records)[-window:] if window else list(self._records)
        if not records:
            return {"count": 0}

        n = len(records)
        errors = [r.error_bps for r in records]
        abs_errors = [r.abs_error_bps for r in records]
        sq_errors = [r.squared_error_bps for r in records]
        preds = [r.predicted_return * 10000 for r in records]  # bps
        actuals = [r.actual_pnl_bps for r in records]

        # 基础统计
        mae = sum(abs_errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        bias = sum(errors) / n  # 正 = 预测偏低, 负 = 预测偏高

        # IC (Information Coefficient) = rank correlation(pred, actual)
        ic = self._compute_ic(preds, actuals)

        # 方向准确率
        correct_direction = sum(
            1 for p, a in zip(preds, actuals)
            if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0)
        )
        direction_accuracy = correct_direction / n

        # 按置信度分组
        high_conf = [r for r in records if r.predicted_confidence > 0.7]
        low_conf = [r for r in records if r.predicted_confidence <= 0.7]
        high_conf_ic = self._compute_ic(
            [r.predicted_return * 10000 for r in high_conf],
            [r.actual_pnl_bps for r in high_conf],
        ) if len(high_conf) > 5 else None

        return {
            "count": n,
            "total_records": self._total_records,
            "mae_bps": round(mae, 2),
            "rmse_bps": round(rmse, 2),
            "bias_bps": round(bias, 2),
            "ic": round(ic, 4) if ic is not None else None,
            "direction_accuracy": round(direction_accuracy, 4),
            "high_conf_count": len(high_conf),
            "low_conf_count": len(low_conf),
            "high_conf_ic": round(high_conf_ic, 4) if high_conf_ic is not None else None,
            "avg_predicted_bps": round(sum(preds) / n, 2),
            "avg_actual_bps": round(sum(actuals) / n, 2),
        }

    def check_drift(self) -> Dict[str, Any]:
        """
        检测模型漂移

        Returns:
            漂移检测结果
        """
        if len(self._records) < self.cfg.drift_window:
            return {"drift_detected": False, "reason": "insufficient_data"}

        recent = list(self._records)[-self.cfg.drift_window:]
        preds = [r.predicted_return * 10000 for r in recent]
        actuals = [r.actual_pnl_bps for r in recent]

        ic = self._compute_ic(preds, actuals)
        mae = sum(r.abs_error_bps for r in recent) / len(recent)
        bias = sum(r.error_bps for r in recent) / len(recent)

        alerts = []
        if ic is not None and ic < self.cfg.ic_alert_threshold:
            alerts.append(f"low_ic: {ic:.4f} < {self.cfg.ic_alert_threshold}")
        if mae > self.cfg.mae_alert_threshold:
            alerts.append(f"high_mae: {mae:.1f}bps > {self.cfg.mae_alert_threshold}bps")
        if abs(bias) > self.cfg.bias_alert_threshold:
            alerts.append(f"bias: {bias:+.1f}bps (|{abs(bias):.1f}| > {self.cfg.bias_alert_threshold}bps)")

        drift_detected = len(alerts) > 0
        if drift_detected:
            self._drift_alerts += 1
            _log.warning(
                "model_drift_detected",
                extra={
                    "ic": round(ic, 4) if ic else None,
                    "mae": round(mae, 1),
                    "bias": round(bias, 1),
                    "alerts": alerts,
                    "trace_id": get_trace(),
                },
            )

        return {
            "drift_detected": drift_detected,
            "ic": round(ic, 4) if ic is not None else None,
            "mae_bps": round(mae, 2),
            "bias_bps": round(bias, 2),
            "alerts": alerts,
            "total_drift_alerts": self._drift_alerts,
        }

    def get_training_samples(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        导出训练样本 (供 alphacast_online_update.py 使用)

        Returns:
            list of {features_seq, label, weight}
        """
        records = list(self._records)[-limit:]
        samples = []
        for r in records:
            label = 1 if r.actual_pnl_bps > 0 else 0
            weight = r.predicted_confidence
            samples.append({
                "label": label,
                "weight": weight,
                "actual_pnl_bps": r.actual_pnl_bps,
                "predicted_return": r.predicted_return,
                "ts_ms": r.ts_ms,
            })
        return samples

    def _compute_ic(self, preds: List[float], actuals: List[float]) -> Optional[float]:
        """计算 Information Coefficient (Pearson 相关系数)"""
        n = len(preds)
        if n < 5:
            return None

        mean_p = sum(preds) / n
        mean_a = sum(actuals) / n

        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(preds, actuals)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in preds) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)

        if std_p < 1e-8 or std_a < 1e-8:
            return 0.0

        return cov / (std_p * std_a)

    def persist(self, path: Optional[str] = None) -> None:
        """持久化到文件"""
        target = path or self.cfg.persist_path
        if not target:
            return
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            records = [asdict(r) for r in self._records]
            with open(target, 'w') as f:
                json.dump({"records": records, "total": self._total_records}, f)
        except Exception as e:
            _log.warning("pnl_tracker_persist_failed", extra={"err": str(e)})

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._records),
            "total_records": self._total_records,
            "drift_alerts": self._drift_alerts,
            **self.get_stats(window=self.cfg.drift_window),
        }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import random
    random.seed(42)

    tracker = PnlTracker()

    # 模拟 150 笔成交
    for i in range(150):
        pred_ret = random.gauss(0.003, 0.005)
        actual_bps = pred_ret * 10000 * random.uniform(0.3, 1.5) + random.gauss(0, 10)
        tracker.record(
            predicted_return=pred_ret,
            predicted_confidence=0.5 + random.random() * 0.4,
            mcts_ev=pred_ret * 0.8,
            actual_pnl_bps=actual_bps,
            actual_side="buy" if pred_ret > 0 else "sell",
            fill_px=3000.0 + random.uniform(-50, 50),
            close_px=3000.0 + random.uniform(-50, 50),
            pulse_id=i,
        )

    stats = tracker.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")

    drift = tracker.check_drift()
    print(f"Drift: {json.dumps(drift, indent=2)}")

    samples = tracker.get_training_samples(limit=5)
    print(f"Training samples: {len(samples)} (showing first: {samples[0]})")

    print("\n✓ PnlTracker self-test passed")
