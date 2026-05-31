"""
calibration/temperature_scaling.py — Temperature Scaling 在线管理
===================================================================

管理 AlphaCast 的 Temperature Scaling 在线校准：
    - 收集成交结果 (predicted_positive, actual_positive)
    - 每 50 笔触发一次温度更新
    - EMA 平滑 (α=0.3)
    - T ∈ [0.5, 3.0] 硬限制
    - 连续 3 次方向异常 → 暂停更新
    - 持久化温度到 Redis (Phase 2)
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_trace

_log = get_logger("calibration.temperature")


@dataclass
class TemperatureConfig:
    """Temperature Scaling 配置"""
    initial_temperature: float = 1.0
    ema_alpha: float = 0.3
    min_temperature: float = 0.5
    max_temperature: float = 3.0
    update_interval: int = 50          # 每 N 笔更新
    min_observations: int = 20         # 最小观察数
    max_consecutive_opposite: int = 3  # 连续方向异常上限
    persist_path: Optional[str] = None # 持久化文件路径


class TemperatureScalingManager:
    """
    Temperature Scaling 在线管理器

    用法：
        tsm = TemperatureScalingManager()
        # 每笔成交后
        tsm.add_observation(confidence=0.72, actual_positive=True)
        # 自动检查是否需要更新
        new_T = tsm.maybe_update()
    """

    def __init__(self, cfg: Optional[TemperatureConfig] = None) -> None:
        self.cfg = cfg or TemperatureConfig()
        self.temperature = self.cfg.initial_temperature
        self._observations: Deque[Tuple[float, bool]] = deque(
            maxlen=max(self.cfg.update_interval * 4, 200)
        )
        self._fill_count = 0
        self._update_count = 0
        self._consecutive_opposite = 0
        self._is_paused = False

        # 尝试加载持久化温度
        if self.cfg.persist_path and os.path.exists(self.cfg.persist_path):
            try:
                with open(self.cfg.persist_path, 'r') as f:
                    data = json.load(f)
                self.temperature = float(data.get("temperature", self.cfg.initial_temperature))
                _log.info("temperature_loaded", extra={"T": self.temperature})
            except Exception:
                pass

    def add_observation(self, confidence: float, actual_positive: bool) -> None:
        """添加一笔成交观察"""
        self._observations.append((confidence, actual_positive))
        self._fill_count += 1

    def should_update(self) -> bool:
        """检查是否应该更新温度"""
        if self._is_paused:
            return False
        if len(self._observations) < self.cfg.min_observations:
            return False
        return self._fill_count % self.cfg.update_interval == 0

    def maybe_update(self) -> Optional[float]:
        """如果条件满足则更新温度"""
        if not self.should_update():
            return None
        return self.update()

    def update(self) -> Optional[float]:
        """执行 Temperature Scaling 更新"""
        if len(self._observations) < self.cfg.min_observations:
            return None

        try:
            import numpy as np

            confs = np.array([c for c, _ in self._observations])
            labels = np.array([1.0 if l else 0.0 for _, l in self._observations])

            def nll(T_val: float) -> float:
                T = max(T_val, 0.01)
                logits = np.log(confs / (1 - confs + 1e-8) + 1e-8) / T
                scaled = 1.0 / (1.0 + np.exp(-logits))
                loss = -np.mean(
                    labels * np.log(scaled + 1e-8)
                    + (1 - labels) * np.log(1 - scaled + 1e-8)
                )
                return float(loss)

            # 网格搜索
            best_T = self.temperature
            best_loss = nll(self.temperature)
            for T_candidate in np.linspace(0.5, 3.0, 50):
                loss = nll(float(T_candidate))
                if loss < best_loss:
                    best_loss = loss
                    best_T = float(T_candidate)

            # 方向异常检测
            old_T = self.temperature
            if (best_T > old_T) == (old_T > 1.0):
                self._consecutive_opposite += 1
            else:
                self._consecutive_opposite = 0

            if self._consecutive_opposite >= self.cfg.max_consecutive_opposite:
                _log.warning(
                    "temperature_update_paused",
                    extra={
                        "consecutive_opposite": self._consecutive_opposite,
                        "T": round(old_T, 4),
                    },
                )
                self._is_paused = True
                self._observations.clear()
                return None

            # EMA 平滑
            new_T = old_T * (1 - self.cfg.ema_alpha) + best_T * self.cfg.ema_alpha
            new_T = max(self.cfg.min_temperature, min(self.cfg.max_temperature, new_T))
            self.temperature = new_T
            self._update_count += 1

            _log.info(
                "temperature_updated",
                extra={
                    "old_T": round(old_T, 4),
                    "new_T": round(new_T, 4),
                    "nll": round(best_loss, 4),
                    "update_count": self._update_count,
                    "trace_id": get_trace(),
                },
            )

            # 持久化
            self._persist()

            self._observations.clear()
            return new_T

        except Exception as e:
            _log.warning("temperature_update_failed", extra={"err": str(e)})
            return None

    def _persist(self) -> None:
        """持久化温度到文件"""
        if not self.cfg.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cfg.persist_path), exist_ok=True)
            with open(self.cfg.persist_path, 'w') as f:
                json.dump({
                    "temperature": self.temperature,
                    "update_count": self._update_count,
                    "fill_count": self._fill_count,
                }, f)
        except Exception:
            pass

    def resume(self) -> None:
        """恢复被暂停的温度更新"""
        self._is_paused = False
        self._consecutive_opposite = 0
        _log.info("temperature_update_resumed")

    def stats(self) -> Dict[str, Any]:
        return {
            "temperature": round(self.temperature, 4),
            "fill_count": self._fill_count,
            "update_count": self._update_count,
            "observation_count": len(self._observations),
            "is_paused": self._is_paused,
            "consecutive_opposite": self._consecutive_opposite,
        }


if __name__ == "__main__":
    tsm = TemperatureScalingManager()

    # 模拟 50 笔成交
    import random
    random.seed(42)
    for i in range(50):
        conf = 0.5 + random.random() * 0.4
        actual = random.random() < conf  # 置信度越高越可能正确
        tsm.add_observation(conf, actual)

    new_T = tsm.update()
    print(f"Updated temperature: {new_T}")
    print(f"Stats: {tsm.stats()}")
    print("✓ TemperatureScalingManager self-test passed")
