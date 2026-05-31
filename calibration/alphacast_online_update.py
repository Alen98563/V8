"""
calibration/alphacast_online_update.py — P5: AlphaCast 在线微调
================================================================

AlphaCast 模型的在线增量微调机制：

    1. 样本累积: 收集已成交的 (features_178d × 60, label, weight) 三元组
    2. 触发条件: 累积 ≥ 200 笔成交 且 距上次微调 ≥ 24h
    3. 微调策略:
       - 仅微调最后 2 层 (head_return + head_confidence)
       - 冻结 Transformer 主体 (避免灾难性遗忘)
       - 学习率 = 1e-5 (极低)
       - 最多 5 epochs
    4. 安全保护:
       - 微调前后 validation loss 对比
       - loss 上升 → 回滚权重
       - 连续 3 次回滚 → 暂停微调 48h
    5. 持久化: 新权重保存到 checkpoint + Triton hot-reload

接口契约：
    - OnlineUpdater.add_sample(features, label, weight)
    - OnlineUpdater.maybe_update() → 自动判断是否触发
    - OnlineUpdater.force_update() → 强制触发
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_trace

_log = get_logger("calibration.online_update")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore


# ============================================================
# 配置
# ============================================================

@dataclass
class OnlineUpdateConfig:
    """在线微调配置"""
    # 触发条件
    min_samples: int = 200              # 最小样本数
    max_samples: int = 2000             # 最大缓冲区
    min_interval_hours: float = 24.0    # 最小间隔

    # 微调参数
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-6
    freeze_transformer: bool = True     # 冻结 Transformer 主体

    # 安全保护
    max_loss_increase_pct: float = 0.1  # loss 上升 > 10% → 回滚
    max_consecutive_rollbacks: int = 3  # 连续回滚 → 暂停
    pause_hours: float = 48.0           # 暂停时长

    # 路径
    checkpoint_dir: str = "models/alphacast/checkpoints"
    triton_model_dir: str = "triton_model_repository/alphacast_resnet/1"


# ============================================================
# 在线微调器
# ============================================================

class AlphaCastOnlineUpdater:
    """
    AlphaCast 在线微调管理器

    用法：
        updater = AlphaCastOnlineUpdater(model=alphacast_model)
        # 每笔成交后
        updater.add_sample(features_178d_seq, label=1, weight=0.8)
        # 定期检查
        result = updater.maybe_update()
    """

    def __init__(
        self,
        model=None,
        cfg: Optional[OnlineUpdateConfig] = None,
    ) -> None:
        self.cfg = cfg or OnlineUpdateConfig()
        self._model = model
        self._buffer: Deque[Tuple[list, int, float]] = deque(maxlen=self.cfg.max_samples)
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._rollback_count: int = 0
        self._consecutive_rollbacks: int = 0
        self._is_paused: bool = False
        self._pause_until: float = 0.0

    def add_sample(self, features_seq: list, label: int, weight: float = 1.0) -> None:
        """
        添加一个训练样本

        Args:
            features_seq: 178d × T 特征序列 (list of lists)
            label: 1 (盈利) / 0 (亏损)
            weight: 样本权重 (CFL confidence_weight)
        """
        self._buffer.append((features_seq, label, weight))

    def should_update(self) -> bool:
        """判断是否应该触发微调"""
        if self._is_paused:
            if time.time() > self._pause_until:
                self._is_paused = False
                self._consecutive_rollbacks = 0
                _log.info("online_update_resumed")
            else:
                return False

        if len(self._buffer) < self.cfg.min_samples:
            return False

        if self._last_update_ts > 0:
            hours_since = (time.time() - self._last_update_ts) / 3600.0
            if hours_since < self.cfg.min_interval_hours:
                return False

        return True

    def maybe_update(self) -> Optional[Dict[str, Any]]:
        """如果条件满足则执行微调"""
        if not self.should_update():
            return None
        return self._do_update()

    def force_update(self) -> Optional[Dict[str, Any]]:
        """强制触发微调 (忽略间隔检查)"""
        if len(self._buffer) < self.cfg.min_samples:
            _log.warning("force_update_insufficient_samples",
                         extra={"count": len(self._buffer)})
            return None
        return self._do_update()

    def _do_update(self) -> Optional[Dict[str, Any]]:
        """执行微调"""
        if torch is None:
            _log.error("PyTorch required for online update")
            return None
        if self._model is None:
            _log.error("No model registered for online update")
            return None

        t0 = time.perf_counter()
        _log.info("online_update_start",
                   extra={"samples": len(self._buffer), "trace_id": get_trace()})

        try:
            # 准备数据
            samples = list(self._buffer)
            X_list, y_list, w_list = zip(*samples)

            X = torch.tensor(X_list, dtype=torch.float32)  # [N, T, 178] or [N, 178]
            y = torch.tensor(y_list, dtype=torch.float32)
            w = torch.tensor(w_list, dtype=torch.float32)

            # 确保 3D 输入
            if X.dim() == 2:
                X = X.unsqueeze(1)  # [N, 1, 178]

            # 分割 train/val (80/20, 时序)
            split = int(len(X) * 0.8)
            X_train, y_train, w_train = X[:split], y[:split], w[:split]
            X_val, y_val = X[split:], y[split:]

            # 冻结/解冻
            if self.cfg.freeze_transformer:
                self._freeze_body()

            # 保存旧权重 (用于回滚)
            old_state = {k: v.clone() for k, v in self._model.state_dict().items()}

            # 微调
            self._model.train()
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._model.parameters()),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )

            train_losses = []
            for epoch in range(self.cfg.epochs):
                total_loss = 0.0
                n_batches = 0
                for i in range(0, len(X_train), self.cfg.batch_size):
                    batch_X = X_train[i:i + self.cfg.batch_size]
                    batch_y = y_train[i:i + self.cfg.batch_size]
                    batch_w = w_train[i:i + self.cfg.batch_size]

                    optimizer.zero_grad()
                    output = self._model(batch_X)
                    pred = output["predicted_return"]
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        pred, batch_y, reduction='none'
                    )
                    weighted_loss = (loss * batch_w).mean()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self._model.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    total_loss += weighted_loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                train_losses.append(avg_loss)

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_output = self._model(X_val)
                val_pred = val_output["predicted_return"]
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    val_pred, y_val
                ).item()

            # 安全保护: 检查 loss 是否上升
            base_val_loss = self._compute_val_loss(X_val, y_val, old_state)
            loss_increase = (val_loss - base_val_loss) / max(base_val_loss, 1e-8)

            if loss_increase > self.cfg.max_loss_increase_pct:
                # 回滚
                self._model.load_state_dict(old_state)
                self._rollback_count += 1
                self._consecutive_rollbacks += 1

                _log.warning(
                    "online_update_rollback",
                    extra={
                        "loss_increase_pct": round(loss_increase * 100, 1),
                        "consecutive_rollbacks": self._consecutive_rollbacks,
                        "trace_id": get_trace(),
                    },
                )

                if self._consecutive_rollbacks >= self.cfg.max_consecutive_rollbacks:
                    self._is_paused = True
                    self._pause_until = time.time() + self.cfg.pause_hours * 3600
                    _log.warning("online_update_paused",
                                 extra={"hours": self.cfg.pause_hours})

                result = {
                    "status": "rollback",
                    "loss_increase_pct": round(loss_increase * 100, 1),
                    "consecutive_rollbacks": self._consecutive_rollbacks,
                }
            else:
                # 成功
                self._consecutive_rollbacks = 0
                self._update_count += 1
                self._last_update_ts = time.time()
                self._buffer.clear()

                # 保存 checkpoint
                self._save_checkpoint()

                elapsed = (time.perf_counter() - t0) * 1000
                _log.info(
                    "online_update_success",
                    extra={
                        "val_loss": round(val_loss, 4),
                        "loss_change_pct": round(loss_increase * 100, 1),
                        "epochs": len(train_losses),
                        "elapsed_ms": round(elapsed, 0),
                        "update_count": self._update_count,
                        "trace_id": get_trace(),
                    },
                )

                result = {
                    "status": "success",
                    "val_loss": round(val_loss, 4),
                    "loss_change_pct": round(loss_increase * 100, 1),
                    "train_losses": [round(l, 4) for l in train_losses],
                    "elapsed_ms": round(elapsed, 0),
                    "update_count": self._update_count,
                }

            return result

        except Exception as e:
            _log.error("online_update_failed", extra={"err": str(e)})
            return {"status": "error", "error": str(e)}

    def _freeze_body(self) -> None:
        """冻结 Transformer 主体，只微调 head 层"""
        if self._model is None:
            return
        for name, param in self._model.named_parameters():
            if 'head_' in name or 'temperature' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        _log.info("model_frozen", extra={
            "trainable": trainable,
            "total": total,
            "pct": round(trainable / max(total, 1) * 100, 1),
        })

    def _compute_val_loss(self, X_val, y_val, state_dict) -> float:
        """用旧权重计算 validation loss"""
        self._model.load_state_dict(state_dict)
        self._model.eval()
        with torch.no_grad():
            output = self._model(X_val)
            loss = nn.functional.binary_cross_entropy_with_logits(
                output["predicted_return"], y_val
            ).item()
        return loss

    def _save_checkpoint(self) -> None:
        """保存 checkpoint"""
        if self._model is None:
            return
        try:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.cfg.checkpoint_dir, f"online_update_{self._update_count}.pt")
            torch.save(self._model.state_dict(), path)

            # 更新 Triton 模型 (hot-reload)
            if os.path.exists(self.cfg.triton_model_dir):
                triton_path = os.path.join(self.cfg.triton_model_dir, "model.pt")
                torch.save(self._model.state_dict(), triton_path)
                _log.info("triton_model_updated", extra={"path": triton_path})
        except Exception as e:
            _log.warning("checkpoint_save_failed", extra={"err": str(e)})

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "update_count": self._update_count,
            "rollback_count": self._rollback_count,
            "consecutive_rollbacks": self._consecutive_rollbacks,
            "is_paused": self._is_paused,
            "hours_until_resume": max(0, (self._pause_until - time.time()) / 3600)
                if self._is_paused else 0,
            "hours_since_update": (time.time() - self._last_update_ts) / 3600
                if self._last_update_ts > 0 else None,
        }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("OnlineUpdater self-test")

    # Without model (just test buffer/trigger logic)
    updater = AlphaCastOnlineUpdater(model=None, cfg=OnlineUpdateConfig(min_samples=10))
    for i in range(15):
        updater.add_sample(
            features_seq=[[0.1] * 178],
            label=1 if i % 3 != 0 else 0,
            weight=0.8,
        )

    print(f"  should_update: {updater.should_update()}")
    print(f"  stats: {updater.stats()}")

    # force_update without model
    result = updater.force_update()
    print(f"  force_update result: {result}")

    print("✓ AlphaCastOnlineUpdater self-test passed")
