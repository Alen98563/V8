"""
AlphaCast 时序预测 Transformer —— 6层8头 · 多任务输出

输入: 178d 融合特征 × 60步
输出:
  - predicted_return (ŷ): 预测收益
  - uncertainty (σ): 不确定性
  - confidence (conf): 置信度 [0, 1]
  - market_state: 市场状态标签

推断: 导出 TorchScript → Triton gRPC (localhost:8002, <10ms)

设计约束:
  - 多数类陷阱: CFL 97.4% 负样本偏置 → 在线 Temperature Scaling 校准
  - 置信度过滤: conf < 0.55 → 拒绝; σ 过大 → 降仓; 收益/风险比 < 1.0 → 放弃
  - 每笔成交后累积 50 笔触发 Temperature Scaling 系数更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ============================================================
# 位置编码
# ============================================================

class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 120, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================
# AlphaCast Transformer 主体
# ============================================================

class AlphaCastModel(nn.Module):
    """
    AlphaCast: 6层8头多任务 Transformer

    Input:  [B, T, D]  D=178, T=60
    Output: Dict[str, Tensor]
      - predicted_return: [B]  预测收益
      - uncertainty:     [B]  不确定性 σ (>0)
      - confidence:      [B]  置信度 [0, 1]
      - market_state:    [B]  状态 logits (4 类)
    """

    def __init__(
        self,
        input_dim: int = 178,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_market_states: int = 4,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # CLS Token (可学习)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 多任务头
        self.head_return = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.head_uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # 保证 σ > 0
        )

        self.head_confidence = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # [0, 1]
        )

        self.head_market_state = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_market_states),
        )

        # Temperature Scaling 参数 (在线校准用)
        self.temperature = nn.Parameter(torch.ones(1))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, D]  D=178, T=60
        returns: Dict with keys: predicted_return, uncertainty, confidence, market_state
        """
        B, T, D = x.shape

        # 输入投影
        x = self.input_proj(x)  # [B, T, d_model]

        # 拼接 CLS Token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]

        # 位置编码
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)  # [B, T+1, d_model]

        # 取 CLS 输出
        cls_out = x[:, 0, :]  # [B, d_model]

        # 多任务头
        pred_return = self.head_return(cls_out).squeeze(-1)      # [B]
        uncertainty = self.head_uncertainty(cls_out).squeeze(-1) # [B]
        confidence = self.head_confidence(cls_out).squeeze(-1)   # [B]
        market_state = self.head_market_state(cls_out)           # [B, 4]

        # Temperature Scaling 校准 (默认 T=1, 在线微调时更新)
        pred_return_calibrated = pred_return / self.temperature

        return {
            'predicted_return': pred_return_calibrated,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'market_state': market_state,
            'temperature': self.temperature.expand(B),
        }


# ============================================================
# 二次校准模块
# ============================================================

class AlphaCastRecalib:
    """
    AlphaCast 输出二次校准

    过滤规则:
    - confidence < 0.55 → 拒绝
    - uncertainty 过大 → 降仓 (position *= 0.5)
    - 收益/风险比 < 1.0 → 放弃
    - MCTS 最优路径后二次校准
    """

    MIN_CONFIDENCE = 0.55
    MAX_UNCERTAINTY = 0.05  # σ 上限
    MIN_RISK_REWARD = 1.0

    @classmethod
    def evaluate(
        cls,
        predicted_return: float,
        uncertainty: float,
        confidence: float,
        mcts_ev: float = 0.0,
    ) -> Dict[str, object]:
        """
        返回校准后的决策:
        {
            'action': 'pass' | 'reject' | 'reduce',
            'position_multiplier': float,
            'reason': str,
            'calibrated_return': float,
        }
        """
        # 1. 置信度过滤
        if confidence < cls.MIN_CONFIDENCE:
            return {
                'action': 'reject',
                'position_multiplier': 0.0,
                'reason': f'confidence {confidence:.3f} < {cls.MIN_CONFIDENCE}',
                'calibrated_return': 0.0,
            }

        # 2. 不确定性过滤
        position_mult = 1.0
        if uncertainty > cls.MAX_UNCERTAINTY:
            position_mult = 0.5
            # 不完全拒绝，但降仓

        # 3. 收益/风险比
        risk_reward = abs(predicted_return) / max(uncertainty, 1e-8)
        if risk_reward < cls.MIN_RISK_REWARD and mcts_ev <= 0:
            return {
                'action': 'reject',
                'position_multiplier': 0.0,
                'reason': f'risk_reward {risk_reward:.2f} < {cls.MIN_RISK_REWARD}',
                'calibrated_return': 0.0,
            }

        # 4. MCTS 修正 (如有)
        calibrated_return = predicted_return
        if mcts_ev != 0:
            # MCTS 期望收益作为二次校准信号
            calibrated_return = 0.6 * predicted_return + 0.4 * mcts_ev

        return {
            'action': 'reduce' if position_mult < 1.0 else 'pass',
            'position_multiplier': position_mult * confidence,
            'reason': 'ok' if position_mult >= 1.0 else f'high uncertainty σ={uncertainty:.4f}',
            'calibrated_return': calibrated_return,
        }


# ============================================================
# Temperature Scaling 在线校准
# ============================================================

class TemperatureScaling:
    """
    在线 Temperature Scaling 校准

    每 50 笔成交触发一次 L-BFGS 优化 NLL loss
    EMA 平滑 (α=0.3)
    T ∈ [0.5, 3.0] 硬限制
    """

    def __init__(self, initial_temp: float = 1.0, ema_alpha: float = 0.3):
        self.temperature = initial_temp
        self.ema_alpha = ema_alpha
        self.min_temp = 0.5
        self.max_temp = 3.0
        self._buffer = []  # 收集 (confidence, label) 对
        self._buffer_size = 50
        self._consecutive_opposite = 0

    def add_observation(self, confidence: float, actual_positive: bool):
        """添加一笔成交观察"""
        label = 1.0 if actual_positive else 0.0
        self._buffer.append((confidence, label))

    def should_update(self) -> bool:
        return len(self._buffer) >= self._buffer_size

    def update(self) -> Optional[float]:
        """执行 Temperature Scaling 更新，返回新 T 或 None"""
        if not self.should_update():
            return None

        import numpy as np
        confs = np.array([c for c, _ in self._buffer])
        labels = np.array([l for _, l in self._buffer])

        # NLL 优化
        from scipy.optimize import minimize

        def nll(T):
            T = T[0]
            if T <= 0:
                return 1e10
            scaled = 1.0 / (1.0 + np.exp(-np.log(conds / (1 - confs + 1e-8) + 1e-8) / T))
            nll_val = -np.mean(labels * np.log(scaled + 1e-8) + (1 - labels) * np.log(1 - scaled + 1e-8))
            return nll_val

        try:
            result = minimize(nll, [self.temperature], method='L-BFGS-B',
                            bounds=[(self.min_temp, self.max_temp)])
            new_T = result.x[0]
        except Exception:
            new_T = self.temperature

        # 方向检测
        if (new_T > self.temperature) == (self.temperature > 1.0):
            self._consecutive_opposite += 1
        else:
            self._consecutive_opposite = 0

        # 连续 3 次方向相反 → 暂停
        if self._consecutive_opposite >= 3:
            self._buffer.clear()
            return self.temperature

        # EMA 平滑
        old_T = self.temperature
        self.temperature = old_T * (1 - self.ema_alpha) + new_T * self.ema_alpha
        self.temperature = max(self.min_temp, min(self.max_temp, self.temperature))

        self._buffer.clear()
        return self.temperature


# ============================================================
# 模型导出
# ============================================================

def export_torchscript(
    model: AlphaCastModel,
    path: str = "models/alphacast/alphacast_model.pt",
):
    """导出为 TorchScript (供 Triton 推断)"""
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f"✓ AlphaCast TorchScript exported: {path}")


def export_onnx(
    model: AlphaCastModel,
    path: str = "models/alphacast/alphacast_model.onnx",
    seq_len: int = 60,
    input_dim: int = 178,
):
    """导出为 ONNX (备用)"""
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["features"],
        output_names=["predicted_return", "uncertainty", "confidence", "market_state"],
        dynamic_axes={
            "features": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"✓ AlphaCast ONNX exported: {path}")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    model = AlphaCastModel(input_dim=178, d_model=256, nhead=8, num_layers=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AlphaCastModel: {total_params:,} params")

    # 前向传播
    x = torch.randn(4, 60, 178)
    out = model(x)
    print(f"Input: {x.shape}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    assert out['predicted_return'].shape == (4,)
    assert out['uncertainty'].shape == (4,)
    assert out['confidence'].shape == (4,)
    assert (out['confidence'] >= 0).all() and (out['confidence'] <= 1).all()
    assert (out['uncertainty'] > 0).all(), "Uncertainty must be positive"

    # 校准测试
    result = AlphaCastRecalib.evaluate(0.01, 0.02, 0.7, mcts_ev=0.005)
    print(f"Recalib: {result}")

    result_reject = AlphaCastRecalib.evaluate(0.01, 0.02, 0.3)  # 低置信度
    print(f"Recalib (reject): {result_reject}")

    # 延迟测试
    import time
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 60, 178)
        for _ in range(10):
            model(x)
        start = time.perf_counter()
        for _ in range(1000):
            model(x)
        elapsed = (time.perf_counter() - start) / 1000 * 1000
        print(f"Latency: {elapsed:.2f} ms (target <10ms)")

    print("✓ All AlphaCast tests passed")
