"""
models/resnet/feature_fusion.py — P2: 注意力门控特征融合 (50d⊕128d → 100d)
===========================================================================

将 ResNet 的 128d 深度嵌入与 50d 经验特征通过注意力门控融合为 100d 向量。

数据流:
    FeatureEngine (50d 微观特征)  ──┐
                                    ├── AttentionGateFusion ──→ 100d ──→ AlphaCast
    ResNetEncoder (128d 嵌入)     ──┘

100d 融合向量是 features/feature_fusion.py 中 DIM_RESNET=100 的来源。

架构:
    1. 50d → Linear(64) → proj_empirical
    2. 128d → Linear(64) → proj_deep
    3. concat(64+64=128) → Attention(2) → softmax → weights
    4. weighted_sum → Linear(100) → LayerNorm → output

门控逻辑:
    - 高波动 regime → 更依赖经验特征 (OBI/OFI 反应快)
    - 低波动 regime → 更依赖深度嵌入 (捕捉长程模式)
    - 自动学习最优权重分配
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from common.logging_setup import get_logger

_log = get_logger("models.resnet.feature_fusion")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore


# ── 维度常量 ─────────────────────────────────────────────────

DIM_EMPIRICAL = 50    # FeatureEngine 50d 微观特征
DIM_DEEP = 128        # ResNetEncoder 128d 嵌入
DIM_OUTPUT = 100      # 融合输出维度 (DIM_RESNET in feature_fusion.py)
DIM_PROJ = 64         # 投影中间维度


# ============================================================
# 注意力门控融合层
# ============================================================

if nn is not None:

    class ResNetFeatureFusion(nn.Module):
        """
        50d⊕128d 注意力门控融合

        Input:
            empirical: [B, 50]  — FeatureEngine 经验特征
            deep_emb:  [B, 128] — ResNetEncoder 深度嵌入

        Output:
            fused: [B, 100] — 融合后的向量
        """

        def __init__(
            self,
            dim_empirical: int = DIM_EMPIRICAL,
            dim_deep: int = DIM_DEEP,
            dim_output: int = DIM_OUTPUT,
            dim_proj: int = DIM_PROJ,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.dim_empirical = dim_empirical
            self.dim_deep = dim_deep
            self.dim_output = dim_output

            # 投影层
            self.proj_empirical = nn.Sequential(
                nn.Linear(dim_empirical, dim_proj),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_deep = nn.Sequential(
                nn.Linear(dim_deep, dim_proj),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # 注意力打分
            self.attn_net = nn.Sequential(
                nn.Linear(dim_proj * 2, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1),
            )

            # 可学习源偏置
            self.source_bias = nn.Parameter(torch.zeros(2))

            # 输出投影
            self.output_proj = nn.Sequential(
                nn.Linear(dim_empirical + dim_deep, dim_output),
                nn.LayerNorm(dim_output),
                nn.Dropout(dropout),
            )

            # Regime 调节 (可选，通过波动率等外部信号调节注意力)
            self.regime_proj = nn.Linear(4, 2, bias=False)  # 4d regime → 2d bias

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(
            self,
            empirical: torch.Tensor,      # [B, 50]
            deep_emb: torch.Tensor,       # [B, 128]
            regime_signal: Optional[torch.Tensor] = None,  # [B, 4] 可选
        ) -> torch.Tensor:
            """
            Returns: [B, 100]
            """
            B = empirical.size(0)

            # 投影
            p_emp = self.proj_empirical(empirical)   # [B, 64]
            p_deep = self.proj_deep(deep_emb)        # [B, 64]

            # 注意力打分
            combined = torch.cat([p_emp, p_deep], dim=-1)  # [B, 128]
            attn_weights = self.attn_net(combined)          # [B, 2]

            # 源偏置
            attn_weights = attn_weights + self.source_bias.unsqueeze(0)

            # Regime 调节 (如果有)
            if regime_signal is not None:
                regime_bias = self.regime_proj(regime_signal)  # [B, 2]
                attn_weights = attn_weights + regime_bias

            # Softmax 重新归一化
            attn_weights = F.softmax(attn_weights, dim=-1)   # [B, 2]

            # 加权源
            w_emp = attn_weights[:, 0:1]   # [B, 1]
            w_deep = attn_weights[:, 1:2]  # [B, 1]

            # 缩放 (补偿 softmax 均值 0.5)
            emp_scaled = empirical * w_emp * 2.0   # [B, 50]
            deep_scaled = deep_emb * w_deep * 2.0   # [B, 128]

            # 拼接 + 投影
            fused_raw = torch.cat([emp_scaled, deep_scaled], dim=-1)  # [B, 178]
            output = self.output_proj(fused_raw)  # [B, 100]

            return output

        def get_attention_weights(
            self,
            empirical: torch.Tensor,
            deep_emb: torch.Tensor,
            regime_signal: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """获取注意力权重 (用于可视化/分析)"""
            p_emp = self.proj_empirical(empirical)
            p_deep = self.proj_deep(deep_emb)
            combined = torch.cat([p_emp, p_deep], dim=-1)
            attn = self.attn_net(combined) + self.source_bias.unsqueeze(0)
            if regime_signal is not None:
                attn = attn + self.regime_proj(regime_signal)
            return F.softmax(attn, dim=-1)


# ============================================================
# 无 PyTorch 的简单拼接融合 (fallback)
# ============================================================

class SimpleFeatureFusion:
    """
    无 PyTorch 依赖的简单融合 (PCA 降维或加权拼接)

    用于 PyTorch 不可用时的 fallback，或快速原型验证。
    """

    def __init__(self, output_dim: int = DIM_OUTPUT):
        self.output_dim = output_dim

    def fuse(
        self,
        empirical: list,    # 50d
        deep_emb: list,     # 128d
    ) -> list:
        """简单加权拼接 + 截断到 output_dim"""
        # 加权: 经验特征权重 0.4, 深度嵌入权重 0.6
        emp_w = [v * 0.4 for v in empirical[:DIM_EMPIRICAL]]
        deep_w = [v * 0.6 for v in deep_emb[:DIM_DEEP]]
        combined = emp_w + deep_w  # 178d

        # 截断或填充到 output_dim
        if len(combined) >= self.output_dim:
            return combined[:self.output_dim]
        return combined + [0.0] * (self.output_dim - len(combined))


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    # Test SimpleFeatureFusion
    simple = SimpleFeatureFusion()
    result = simple.fuse([0.1] * 50, [0.2] * 128)
    print(f"SimpleFusion: {len(result)}d (expected {DIM_OUTPUT})")
    assert len(result) == DIM_OUTPUT

    # Test ResNetFeatureFusion
    if torch is not None:
        fusion = ResNetFeatureFusion()
        total_params = sum(p.numel() for p in fusion.parameters())
        print(f"ResNetFeatureFusion: {total_params:,} params")

        emp = torch.randn(4, DIM_EMPIRICAL)
        deep = torch.randn(4, DIM_DEEP)
        out = fusion(emp, deep)
        print(f"Output: {out.shape} (expected [4, {DIM_OUTPUT}])")
        assert out.shape == (4, DIM_OUTPUT)

        # Test with regime signal
        regime = torch.tensor([[0.5, 0.02, 0.001, 0.0]] * 4)
        out_regime = fusion(emp, deep, regime)
        assert out_regime.shape == (4, DIM_OUTPUT)

        # Test attention weights
        weights = fusion.get_attention_weights(emp, deep)
        print(f"Attention weights: {weights[0].tolist()}")
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)

        # Latency test
        import time as _time
        fusion.eval()
        with torch.no_grad():
            emp = torch.randn(1, DIM_EMPIRICAL)
            deep = torch.randn(1, DIM_DEEP)
            for _ in range(10):
                fusion(emp, deep)
            start = _time.perf_counter()
            for _ in range(1000):
                fusion(emp, deep)
            elapsed = (_time.perf_counter() - start) / 1000 * 1000
            print(f"Latency: {elapsed:.2f} ms (target <2ms)")

        print("✓ ResNetFeatureFusion test passed")
    else:
        print("⚠ PyTorch not available, skipping neural fusion test")

    print("✓ feature_fusion self-test passed")
