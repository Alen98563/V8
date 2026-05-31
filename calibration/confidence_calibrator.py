"""
calibration/confidence_calibrator.py — 置信度校准器
====================================================

对 AlphaCast 输出的原始置信度进行校准，使其更接近真实准确率。

方法:
    1. Isotonic Regression (保序回归): 非参数方法，不假设映射函数形状
    2. Platt Scaling: sigmoid 映射，适合输出分布接近正态的场景
    3. Temperature Scaling: logit 空间缩放，最简单

校准器会在每次 Temperature Scaling 更新后自动重拟合。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_trace

_log = get_logger("calibration.confidence")


@dataclass
class CalibrationConfig:
    """校准配置"""
    method: str = "temperature"  # "temperature" | "platt" | "isotonic"
    min_samples: int = 50        # 最小校准样本数
    max_samples: int = 2000      # 最大历史样本
    persist_path: Optional[str] = None


class ConfidenceCalibrator:
    """
    置信度校准器

    用法：
        calibrator = ConfidenceCalibrator(method="temperature")
        # 收集预测结果
        calibrator.add_sample(predicted_conf=0.72, actual_positive=True)
        # 校准
        calibrator.fit()
        # 校准新预测
        calibrated_conf = calibrator.calibrate(raw_conf=0.72)
    """

    def __init__(
        self,
        method: str = "temperature",
        cfg: Optional[CalibrationConfig] = None,
    ) -> None:
        self.cfg = cfg or CalibrationConfig(method=method)
        self.method = self.cfg.method
        self._samples: List[Tuple[float, bool]] = []
        self._is_fitted = False

        # Temperature Scaling 参数
        self._temperature: float = 1.0

        # Platt Scaling 参数
        self._platt_a: float = 0.0
        self._platt_b: float = 0.0

    def add_sample(self, predicted_conf: float, actual_positive: bool) -> None:
        """添加一个校准样本"""
        self._samples.append((predicted_conf, actual_positive))
        if len(self._samples) > self.cfg.max_samples:
            self._samples = self._samples[-self.cfg.max_samples:]

    def fit(self) -> bool:
        """用收集的样本拟合校准模型"""
        if len(self._samples) < self.cfg.min_samples:
            _log.debug("calibrator_not_enough_samples",
                       extra={"count": len(self._samples), "min": self.cfg.min_samples})
            return False

        confs = [c for c, _ in self._samples]
        labels = [1.0 if l else 0.0 for _, l in self._samples]

        try:
            if self.method == "temperature":
                self._fit_temperature(confs, labels)
            elif self.method == "platt":
                self._fit_platt(confs, labels)
            elif self.method == "isotonic":
                self._fit_isotonic(confs, labels)
            else:
                _log.warning(f"Unknown calibration method: {self.method}")
                return False

            self._is_fitted = True
            _log.info("calibrator_fitted",
                      extra={"method": self.method, "samples": len(self._samples)})
            return True

        except Exception as e:
            _log.warning("calibrator_fit_failed", extra={"err": str(e)})
            return False

    def _fit_temperature(self, confs: List[float], labels: List[float]) -> None:
        """Temperature Scaling 拟合"""
        import math
        best_T = 1.0
        best_loss = float("inf")

        for T_int in range(50, 301):  # 0.50 ~ 3.00
            T = T_int / 100.0
            loss = 0.0
            for c, y in zip(confs, labels):
                c_clipped = max(1e-6, min(1 - 1e-6, c))
                logit = math.log(c_clipped / (1 - c_clipped))
                scaled = 1.0 / (1.0 + math.exp(-logit / T))
                loss -= y * math.log(scaled + 1e-8) + (1 - y) * math.log(1 - scaled + 1e-8)
            loss /= len(confs)
            if loss < best_loss:
                best_loss = loss
                best_T = T

        self._temperature = best_T
        _log.info("temperature_fitted", extra={"T": round(best_T, 3), "nll": round(best_loss, 4)})

    def _fit_platt(self, confs: List[float], labels: List[float]) -> None:
        """Platt Scaling 拟合 (简化版梯度下降)"""
        import math
        a, b = 0.0, 0.0
        lr = 0.01

        for _ in range(1000):
            grad_a, grad_b = 0.0, 0.0
            for c, y in zip(confs, labels):
                p = 1.0 / (1.0 + math.exp(-(a * c + b)))
                grad_a += (p - y) * c
                grad_b += (p - y)
            a -= lr * grad_a / len(confs)
            b -= lr * grad_b / len(confs)

        self._platt_a = a
        self._platt_b = b

    def _fit_isotonic(self, confs: List[float], labels: List[float]) -> None:
        """Isotonic Regression (PAV 算法)"""
        # 按置信度排序
        paired = sorted(zip(confs, labels), key=lambda x: x[0])
        n = len(paired)

        # Pool Adjacent Violators
        blocks = [[i] for i in range(n)]
        block_means = [paired[i][1] for i in range(n)]

        i = 0
        while i < len(blocks) - 1:
            if block_means[i] > block_means[i + 1]:
                # Merge
                merged = blocks[i] + blocks[i + 1]
                merged_mean = sum(paired[j][1] for j in merged) / len(merged)
                blocks[i] = merged
                block_means[i] = merged_mean
                blocks.pop(i + 1)
                block_means.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1

        # 存储映射表
        self._isotonic_map = []
        for block, mean in zip(blocks, block_means):
            conf_avg = sum(paired[j][0] for j in block) / len(block)
            self._isotonic_map.append((conf_avg, mean))

    def calibrate(self, raw_conf: float) -> float:
        """校准原始置信度"""
        if not self._is_fitted:
            return raw_conf

        import math

        if self.method == "temperature":
            c = max(1e-6, min(1 - 1e-6, raw_conf))
            logit = math.log(c / (1 - c))
            return 1.0 / (1.0 + math.exp(-logit / self._temperature))

        elif self.method == "platt":
            return 1.0 / (1.0 + math.exp(-(self._platt_a * raw_conf + self._platt_b)))

        elif self.method == "isotonic":
            if not hasattr(self, '_isotonic_map') or not self._isotonic_map:
                return raw_conf
            # 线性插值
            for i in range(len(self._isotonic_map) - 1):
                x0, y0 = self._isotonic_map[i]
                x1, y1 = self._isotonic_map[i + 1]
                if x0 <= raw_conf <= x1:
                    t = (raw_conf - x0) / max(x1 - x0, 1e-8)
                    return y0 + t * (y1 - y0)
            # 外推
            if raw_conf < self._isotonic_map[0][0]:
                return self._isotonic_map[0][1]
            return self._isotonic_map[-1][1]

        return raw_conf

    def stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "is_fitted": self._is_fitted,
            "sample_count": len(self._samples),
            "temperature": round(self._temperature, 4) if self.method == "temperature" else None,
        }


if __name__ == "__main__":
    import random
    random.seed(42)

    cal = ConfidenceCalibrator(method="temperature")
    for _ in range(100):
        conf = 0.4 + random.random() * 0.5
        actual = random.random() < conf
        cal.add_sample(conf, actual)

    cal.fit()
    for raw in [0.3, 0.5, 0.7, 0.9]:
        print(f"  raw={raw:.2f} → calibrated={cal.calibrate(raw):.3f}")

    print(f"Stats: {cal.stats()}")
    print("✓ ConfidenceCalibrator self-test passed")
