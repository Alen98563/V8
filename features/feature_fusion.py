"""
features/feature_fusion.py — Task 3b: 注意力门控特征融合
=========================================================

将多源特征融合为 AlphaCast 的 178d 输入向量：

    50d 微观特征 (FeatureEngine)
  + 12d Alpha 信号 (OBI + OFI + FundingRate + 衍生)
  + 16d 门控元信息 (G1–G5 状态 + 时间特征)
  + 100d ResNet 编码 (market embedding, Phase 2)
  = 178d 融合特征

融合方式：注意力门控 (Attention Gate)
    - 每个特征源有可学习权重
    - 门控信号来自市场状态 (波动率/流动性 regime)
    - 输出 LayerNorm + Dropout

可独立运行，也可嵌入 AlphaCast 训练管线。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from common.logging_setup import get_logger

_log = get_logger("features.fusion")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore


# ============================================================
# 特征维度常量
# ============================================================

DIM_MICRO = 50       # FeatureEngine 50d 微观特征
DIM_ALPHA = 12       # Alpha 信号衍生特征
DIM_GATE = 16        # 门控元信息 + 时间特征
DIM_RESNET = 100     # ResNet 编码 (Phase 2)
DIM_TOTAL = DIM_MICRO + DIM_ALPHA + DIM_GATE + DIM_RESNET  # 178d


# ============================================================
# 注意力门控融合层
# ============================================================

if nn is not None:

    class AttentionGateFusion(nn.Module):
        """
        注意力门控融合

        对每个特征源计算 attention weight，加权求和后 LayerNorm。
        门控信号来自波动率 regime (高波 → 降低 Alpha 权重，提高门控权重)。
        """

        def __init__(
            self,
            dim_micro: int = DIM_MICRO,
            dim_alpha: int = DIM_ALPHA,
            dim_gate: int = DIM_GATE,
            dim_resnet: int = DIM_RESNET,
            output_dim: int = DIM_TOTAL,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.dim_micro = dim_micro
            self.dim_alpha = dim_alpha
            self.dim_gate = dim_gate
            self.dim_resnet = dim_resnet
            self.output_dim = output_dim

            # 各源投影到统一维度 (用于 attention 计算)
            proj_dim = 64
            self.proj_micro = nn.Linear(dim_micro, proj_dim)
            self.proj_alpha = nn.Linear(dim_alpha, proj_dim)
            self.proj_gate = nn.Linear(dim_gate, proj_dim)
            self.proj_resnet = nn.Linear(dim_resnet, proj_dim)

            # Attention 打分 (4 个源)
            self.attn_score = nn.Sequential(
                nn.Linear(proj_dim * 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 4),
                nn.Softmax(dim=-1),
            )

            # 源缩放 (attention weight × 投影)
            self.source_scale = nn.Parameter(torch.ones(4))

            # 最终拼接投影
            total_input_dim = dim_micro + dim_alpha + dim_gate + dim_resnet
            self.output_proj = nn.Sequential(
                nn.Linear(total_input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )

        def forward(
            self,
            micro: torch.Tensor,       # [B, dim_micro]
            alpha: torch.Tensor,       # [B, dim_alpha]
            gate_info: torch.Tensor,   # [B, dim_gate]
            resnet_emb: torch.Tensor,  # [B, dim_resnet]
        ) -> torch.Tensor:
            """
            输出: [B, output_dim]  (默认 178d)
            """
            B = micro.size(0)

            # 各源投影
            p_micro = self.proj_micro(micro)      # [B, 64]
            p_alpha = self.proj_alpha(alpha)      # [B, 64]
            p_gate = self.proj_gate(gate_info)    # [B, 64]
            p_resnet = self.proj_resnet(resnet_emb)  # [B, 64]

            # Attention 打分
            combined = torch.cat([p_micro, p_alpha, p_gate, p_resnet], dim=-1)  # [B, 256]
            attn_weights = self.attn_score(combined)  # [B, 4]

            # 缩放
            scaled_weights = attn_weights * self.source_scale.unsqueeze(0)  # [B, 4]
            # 重新归一化
            scaled_weights = F.softmax(scaled_weights, dim=-1)

            # 加权 (broadcast 到各源维度)
            w_micro = scaled_weights[:, 0:1]    # [B, 1]
            w_alpha = scaled_weights[:, 1:2]
            w_gate = scaled_weights[:, 2:3]
            w_resnet = scaled_weights[:, 3:4]

            # 源加权缩放 (保持原始维度)
            micro_w = micro * w_micro * 4.0     # ×4 补偿 softmax 均值 0.25
            alpha_w = alpha * w_alpha * 4.0
            gate_w = gate_info * w_gate * 4.0
            resnet_w = resnet_emb * w_resnet * 4.0

            # 拼接 + 投影
            fused = torch.cat([micro_w, alpha_w, gate_w, resnet_w], dim=-1)  # [B, total]
            output = self.output_proj(fused)  # [B, output_dim]

            return output


# ============================================================
# 特征组装器 (无 PyTorch 依赖)
# ============================================================

class FeatureAssembler:
    """
    从各模块收集原始特征，组装为 178d 向量

    用法：
        assembler = FeatureAssembler()
        features_178d = assembler.assemble(
            micro_50d=fe.get_features_50d(),
            obi_signal=obi_engine.on_snapshot(snap),
            funding_signal=funding_engine.get_last_signal(),
            gate_result=gating.evaluate(ctx),
            resnet_embedding=None,  # Phase 2
        )
    """

    def __init__(self) -> None:
        pass

    def _pad_or_truncate(self, vec: List[float], target_dim: int) -> List[float]:
        """将向量填充或截断到目标维度"""
        if len(vec) >= target_dim:
            return vec[:target_dim]
        return vec + [0.0] * (target_dim - len(vec))

    def assemble_alpha_features(
        self,
        obi_signal=None,
        funding_signal=None,
    ) -> List[float]:
        """从 Alpha 信号提取 12d 特征"""
        feats = []

        # OBI/OFI 相关 (6d)
        if obi_signal is not None:
            feats.extend([
                obi_signal.raw_signal,
                obi_signal.confidence,
                obi_signal.obi,
                obi_signal.ofi,
                abs(obi_signal.raw_signal),  # 信号强度
                obi_signal.raw_signal * obi_signal.confidence,  # 加权信号
            ])
        else:
            feats.extend([0.0] * 6)

        # 资金费率相关 (6d)
        if funding_signal is not None:
            feats.extend([
                funding_signal.raw_signal,
                funding_signal.confidence,
                funding_signal.funding_rate * 10000,  # 放大到 bps 级
                funding_signal.funding_rate_zscore,
                funding_signal.hours_to_settlement / 8.0,  # 归一化到 [0,1]
                funding_signal.raw_signal * funding_signal.confidence,
            ])
        else:
            feats.extend([0.0] * 6)

        return self._pad_or_truncate(feats, DIM_ALPHA)

    def assemble_gate_features(
        self,
        gate_result=None,
        ts_ms: int = 0,
    ) -> List[float]:
        """从门控结果 + 时间信息提取 16d 特征"""
        import datetime

        feats = []

        # 门控状态 (6d): G1–G5 pass/fail + 综合结果
        if gate_result is not None:
            gate_str = str(getattr(gate_result, 'gate', '-'))
            reason = str(getattr(gate_result, 'reason', '-'))
            passed = 1.0 if getattr(gate_result, 'passed', False) else 0.0
            feats.extend([
                passed,
                1.0 if 'G1' in gate_str else 0.0,
                1.0 if 'G2' in gate_str else 0.0,
                1.0 if 'G3' in gate_str else 0.0,
                1.0 if 'G4' in gate_str else 0.0,
                1.0 if 'G5' in gate_str else 0.0,
            ])
        else:
            feats.extend([0.0] * 6)

        # 时间特征 (10d)
        if ts_ms > 0:
            dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000.0)
            hour = dt.hour
            minute = dt.minute
            day_of_week = dt.weekday()
            # 周期编码
            feats.extend([
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * minute / 60),
                math.cos(2 * math.pi * minute / 60),
                math.sin(2 * math.pi * day_of_week / 7),
                math.cos(2 * math.pi * day_of_week / 7),
                1.0 if hour in [0, 8, 16] else 0.0,   # 结算小时
                1.0 if 0 <= hour < 8 else 0.0,         # 亚洲时段
                1.0 if 8 <= hour < 16 else 0.0,        # 欧洲时段
                1.0 if 16 <= hour < 24 else 0.0,       # 美洲时段
            ])
        else:
            feats.extend([0.0] * 10)

        return self._pad_or_truncate(feats, DIM_GATE)

    def assemble(
        self,
        micro_50d: bytes | List[float] = b"",
        obi_signal=None,
        funding_signal=None,
        gate_result=None,
        ts_ms: int = 0,
        resnet_embedding: Optional[List[float]] = None,
    ) -> List[float]:
        """
        组装完整 178d 特征向量

        Returns: 178d float list
        """
        import struct

        # 50d 微观特征
        if isinstance(micro_50d, bytes) and micro_50d:
            micro = list(struct.unpack(f"<{len(micro_50d)//4}f", micro_50d))
        elif isinstance(micro_50d, list):
            micro = micro_50d
        else:
            micro = []
        micro = self._pad_or_truncate(micro, DIM_MICRO)

        # 12d Alpha 特征
        alpha = self.assemble_alpha_features(obi_signal, funding_signal)

        # 16d 门控 + 时间特征
        gate = self.assemble_gate_features(gate_result, ts_ms)

        # 100d ResNet 编码 (Phase 2, 目前零填充)
        resnet = self._pad_or_truncate(resnet_embedding or [], DIM_RESNET)

        # 拼接
        fused = micro + alpha + gate + resnet
        assert len(fused) == DIM_TOTAL, f"Expected {DIM_TOTAL}d, got {len(fused)}d"

        return fused


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    # 测试 FeatureAssembler (无 PyTorch 依赖)
    assembler = FeatureAssembler()

    features = assembler.assemble(
        micro_50d=[0.1] * 50,
        ts_ms=1700000000000,
    )
    print(f"Assembled: {len(features)}d (expected {DIM_TOTAL})")
    assert len(features) == DIM_TOTAL

    # 测试 Alpha 子特征
    alpha = assembler.assemble_alpha_features()
    print(f"Alpha features: {len(alpha)}d (expected {DIM_ALPHA})")
    assert len(alpha) == DIM_ALPHA

    # 测试门控子特征
    gate = assembler.assemble_gate_features(ts_ms=1700000000000)
    print(f"Gate features: {len(gate)}d (expected {DIM_GATE})")
    assert len(gate) == DIM_GATE

    # 测试 AttentionGateFusion (如果有 PyTorch)
    if torch is not None:
        fusion = AttentionGateFusion()
        micro_t = torch.randn(2, DIM_MICRO)
        alpha_t = torch.randn(2, DIM_ALPHA)
        gate_t = torch.randn(2, DIM_GATE)
        resnet_t = torch.randn(2, DIM_RESNET)
        out = fusion(micro_t, alpha_t, gate_t, resnet_t)
        print(f"AttentionGateFusion output: {out.shape} (expected [2, {DIM_TOTAL}])")
        assert out.shape == (2, DIM_TOTAL)
        print("✓ AttentionGateFusion test passed")
    else:
        print("⚠ PyTorch not available, skipping AttentionGateFusion test")

    print("✓ FeatureFusion self-test passed")
