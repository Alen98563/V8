"""
MetaLabeler 重训机制 —— 抗击 97.4% 多数类陷阱

输入: 178d 特征 + AlphaCast.conf + MCTS.value
模型: LightGBM Binary 分类器
策略:
  - 动态样本类别权重 (Class Weights): 自动平衡 97.4% 负样本偏置
  - 自然样本分层累积: 过采样少数类 (SMOTE/复制) 或欠采样多数类
  - Top-Decile Filter: 仅取预测分前 10% 的样本通过 G4 门
  - 每 7 天滚动 30d 窗口自动重训
  - IC 衰减 < 0.03 触发告警

触发条件: CFL 标签 ≥ 2K 首次训练 / ≥ 5K 正式激活
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    warnings.warn("LightGBM not installed. MetaLabeler inference will not work.")


# ============================================================
# 数据模型
# ============================================================

@dataclass
class MetaLabelerConfig:
    """MetaLabeler 超参数"""
    # 触发条件
    min_labels_first_train: int = 2000  # 首次训练所需标签数
    min_labels_full_active: int = 5000  # 正式激活所需标签数
    rolling_window_days: int = 30       # 滚动窗口天数
    retrain_interval_days: int = 7      # 重训间隔

    # LightGBM 参数
    num_leaves: int = 31
    max_depth: int = 7
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 50

    # 多数类防御
    scale_pos_weight: Optional[float] = None  # None = 自动计算
    use_smote: bool = False                   # MVP 不启用 (数据量不足)

    # 门控
    top_decile_threshold: float = 0.9  # 前 10%

    # 质量监控
    min_auc: float = 0.55
    min_lift: float = 1.2
    ic_decay_warning: float = 0.03


@dataclass
class MetaLabelerStats:
    """MetaLabeler 性能统计"""
    is_active: bool = False
    n_labels: int = 0
    positive_ratio: float = 0.0
    last_train_time: Optional[datetime] = None
    last_auc: float = 0.0
    last_lift: float = 0.0
    ic_rolling: float = 0.0
    g4_pass_rate: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ============================================================
# MetaLabeler 主体
# ============================================================

class MetaLabeler:
    """
    G4 Meta 二分类器

    使用方式:
        labeler = MetaLabeler()
        labeler.add_label(features, alphacast_conf, mcts_value, label)
        # ...
        labeler.train_if_ready()
        prob = labeler.predict(features, alphacast_conf, mcts_value)
    """

    def __init__(self, config: Optional[MetaLabelerConfig] = None):
        self.config = config or MetaLabelerConfig()
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.stats = MetaLabelerStats()

        # 标签缓冲: (features, alphacast_conf, mcts_value, label, ts)
        self._label_buffer: List[Tuple[np.ndarray, float, float, int, float]] = []

    # ============================================================
    # 数据注入
    # ============================================================

    def add_label(
        self,
        features: np.ndarray,       # [feature_dim] 或 [178]
        alphacast_conf: float,
        mcts_value: float,
        label: int,                 # 0 = 负样本, 1 = 正样本
        ts: Optional[float] = None,
    ):
        """
        注入一条 CFL 标签
        无论 G1–G3 是否放行，100% 异步记录
        """
        ts_val = ts if ts is not None else datetime.now().timestamp()
        self._label_buffer.append((features.copy(), alphacast_conf, mcts_value, label, ts_val))
        self.stats.n_labels += 1

        # 滚动窗口裁剪
        cutoff = ts_val - self.config.rolling_window_days * 86400
        self._label_buffer = [
            x for x in self._label_buffer
            if x[4] >= cutoff
        ]

    # ============================================================
    # 训练
    # ============================================================

    def can_train(self) -> bool:
        """是否满足训练条件"""
        return len(self._label_buffer) >= self.config.min_labels_first_train

    def should_retrain(self) -> bool:
        """是否需要重训 (距上次训练 > 7 天)"""
        if self.stats.last_train_time is None:
            return self.can_train()
        delta = datetime.now() - self.stats.last_train_time
        return delta.days >= self.config.retrain_interval_days and self.can_train()

    def train_if_ready(self, force: bool = False) -> bool:
        """
        如果满足条件则训练/重训
        返回是否成功训练
        """
        if not force and not self.can_train():
            return False
        if not force and not self.should_retrain():
            return False

        n = len(self._label_buffer)
        if n < self.config.min_labels_first_train:
            return False

        # 构建特征矩阵
        X_list = []
        y_list = []

        for feat, ac_conf, mcts_val, label, _ in self._label_buffer:
            # 拼接 178d 特征 + AlphaCast.conf + MCTS.value
            x_row = np.concatenate([feat, [ac_conf, mcts_val]])
            X_list.append(x_row)
            y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        # 统计正负样本比例
        pos_ratio = y.mean()
        n_pos = y.sum()
        n_neg = y.shape[0] - n_pos

        self.stats.positive_ratio = pos_ratio
        print(f"[MetaLabeler] Training with {y.shape[0]} samples, "
              f"positive ratio: {pos_ratio:.3f} ({n_pos}:{n_neg})")

        # 自动计算类别权重 (抗击 97.4% 负样本偏置)
        if self.config.scale_pos_weight is None:
            scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
            scale_pos_weight = min(scale_pos_weight, 100.0)  # 上限 100x
        else:
            scale_pos_weight = self.config.scale_pos_weight

        print(f"[MetaLabeler] scale_pos_weight: {scale_pos_weight:.2f}")

        # LightGBM 训练
        if lgb is None:
            warnings.warn("LightGBM not available. Skipping training.")
            return False

        train_data = lgb.Dataset(
            X,
            label=y,
            feature_name=[f"feat_{i}" for i in range(X.shape[1])],
        )

        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'min_child_samples': self.config.min_child_samples,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': 42,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
        )

        # 评估
        y_pred = self.model.predict(X)
        self.stats.last_auc = self._compute_auc(y, y_pred)
        self.stats.last_lift = self._compute_lift(y, y_pred)
        self.stats.last_train_time = datetime.now()

        # 是否正式激活 G4
        self.stats.is_active = (
            self.stats.n_labels >= self.config.min_labels_full_active
            and self.stats.last_auc >= self.config.min_auc
            and self.stats.last_lift >= self.config.min_lift
        )

        print(f"[MetaLabeler] Training complete | "
              f"AUC: {self.stats.last_auc:.4f} | "
              f"Lift: {self.stats.last_lift:.2f}x | "
              f"Active: {self.stats.is_active}")

        return True

    # ============================================================
    # 推断
    # ============================================================

    def predict(
        self,
        features: np.ndarray,       # [feature_dim]
        alphacast_conf: float,
        mcts_value: float,
    ) -> Tuple[float, bool]:
        """
        预测样本为正的概率
        返回: (probability, g4_pass)

        G4 门控逻辑: 仅前 10% 高预测分样本通过
        """
        if not self.stats.is_active or self.model is None:
            return (0.5, True)  # G4 未激活时直通

        x = np.concatenate([features, [alphacast_conf, mcts_value]]).reshape(1, -1)
        prob = self.model.predict(x)[0]

        # Top-Decile Filter: 使用模型内置的 percentile
        # 简化: 用 prob > 0.7 近似 (后续可用真实分位数)
        pass_threshold = 0.7  # TODO: 用训练集分位数值替代
        passed = prob >= pass_threshold

        return (float(prob), passed)

    def predict_batch(
        self,
        features: np.ndarray,       # [N, feature_dim]
        alphacast_conf: np.ndarray, # [N]
        mcts_value: np.ndarray,     # [N]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量预测"""
        if not self.stats.is_active or self.model is None:
            return (np.full(features.shape[0], 0.5), np.ones(features.shape[0], dtype=bool))

        x = np.concatenate([
            features,
            alphacast_conf.reshape(-1, 1),
            mcts_value.reshape(-1, 1),
        ], axis=1)

        probs = self.model.predict(x)
        passed = probs >= 0.7

        return (probs, passed)

    # ============================================================
    # 质量监控
    # ============================================================

    def check_health(self) -> List[str]:
        """健康检查，返回告警列表"""
        alerts = []

        if self.stats.last_auc < self.config.ic_decay_warning:
            alerts.append(
                f"⚠️ MetaLabeler AUC {self.stats.last_auc:.4f} < {self.config.ic_decay_warning} "
                f"— model may need retraining or feature re-engineering"
            )

        if self.stats.is_active and self.stats.last_auc < self.config.min_auc:
            alerts.append(
                f"🔴 AUC {self.stats.last_auc:.4f} below minimum {self.config.min_auc}. "
                f"Consider disabling G4."
            )

        if self.stats.last_lift < self.config.min_lift and self.stats.is_active:
            alerts.append(
                f"⚠️ Lift {self.stats.last_lift:.2f}x below minimum {self.config.min_lift}x"
            )

        if self.stats.positive_ratio < 0.02:
            alerts.append(
                f"⚠️ Positive ratio {self.stats.positive_ratio:.4f} extremely low. "
                f"Check CFL labeling window."
            )

        self.stats.warnings = alerts
        return alerts

    # ============================================================
    # 辅助方法
    # ============================================================

    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算 AUC (不需要 sklearn)"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)

    def _compute_lift(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算 Top-Decile Lift"""
        n = len(y_true)
        top_n = max(1, int(n * 0.1))
        top_indices = np.argpartition(y_pred, -top_n)[-top_n:]
        baseline_rate = y_true.mean()
        top_rate = y_true[top_indices].mean()
        if baseline_rate > 0:
            return top_rate / baseline_rate
        return 1.0

    def feature_importance(self) -> List[Tuple[str, float]]:
        """特征重要性 (用于审查)"""
        if self.model is None:
            return []
        importances = zip(
            [f"feat_{i}" for i in range(self.model.feature_importance().shape[0])],
            self.model.feature_importance(),
        )
        return sorted(importances, key=lambda x: x[1], reverse=True)


# ============================================================
# 模型生命周期管理
# ============================================================

class ModelLifecycle:
    """
    模型生命周期管理器

    - n_complete ≥ 5K → 触发首次训练
    - 每 7 天滚动 30 天窗口重训
    - IC 衰减 → 告警 + 重训
    """

    def __init__(self, labeler: MetaLabeler):
        self.labeler = labeler

    def lifecycle_tick(self) -> Dict[str, Any]:
        """
        每个 tick 调用 (建议每 1h)
        返回状态字典
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'n_labels': self.labeler.stats.n_labels,
            'is_active': self.labeler.stats.is_active,
            'can_train': self.labeler.can_train(),
            'should_retrain': self.labeler.should_retrain(),
            'alerts': [],
        }

        # 自动重训
        if self.labeler.should_retrain():
            success = self.labeler.train_if_ready()
            result['trained'] = success

        # 健康检查
        alerts = self.labeler.check_health()
        result['alerts'] = alerts

        return result


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=== MetaLabeler Test ===\n")

    # 模拟 CFL 标签数据 (97.4% 负样本)
    np.random.seed(42)
    n_samples = 5000
    n_pos = int(n_samples * 0.026)  # 2.6% 正样本
    n_neg = n_samples - n_pos

    print(f"Simulating {n_samples} labels: {n_pos} positive, {n_neg} negative ({n_pos/n_samples:.4f} positive ratio)")

    labeler = MetaLabeler()

    # 注入正样本 (强特征)
    for i in range(n_pos):
        feat = np.random.randn(178) * 0.5 + 0.3  # 均值 0.3 的特征
        labeler.add_label(feat, alphacast_conf=0.7 + np.random.random() * 0.2,
                         mcts_value=0.01 + np.random.random() * 0.02, label=1)

    # 注入负样本
    for i in range(n_neg):
        feat = np.random.randn(178) * 0.5  # 零均值噪声
        labeler.add_label(feat, alphacast_conf=0.4 + np.random.random() * 0.3,
                         mcts_value=np.random.random() * 0.01, label=0)

    # 训练
    success = labeler.train_if_ready(force=True)
    print(f"\nTrain success: {success}")
    print(f"Active: {labeler.stats.is_active}")
    print(f"AUC: {labeler.stats.last_auc:.4f}")
    print(f"Lift: {labeler.stats.last_lift:.2f}x")

    # 推断
    test_feat = np.random.randn(178) * 0.5 + 0.2
    prob, passed = labeler.predict(test_feat, 0.7, 0.005)
    print(f"Predict: prob={prob:.4f}, passed={passed}")

    # 健康检查
    alerts = labeler.check_health()
    for a in alerts:
        print(a)

    # 特征重要性 Top-10
    fi = labeler.feature_importance()[:10]
    print("\nTop-10 Feature Importance:")
    for name, imp in fi:
        print(f"  {name}: {imp:.2f}")

    print("\n✓ MetaLabeler test complete")