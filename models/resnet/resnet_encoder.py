"""
ResNet 1D 时间序列卷积残差网络 —— 6 层残差块 · 128d 嵌入 · 多尺度 Conv1D

特征流: 50d 经验特征 × 60步 → ResNet → 128d 深度嵌入
融合: 50d ⊕ 128d → 注意力门控 → 178d 融合向量 (feature_fusion.py)
推断: 导出 ONNX → Triton gRPC (localhost:8001, <5ms)

架构:
  Input: [B, 50, T]  (T=60, 5m@1s)
  → MultiScaleConv1D [30/60/120/240s] → Concat
  → ResBlock × 6 (Conv1D → BN → LeakyReLU → Conv1D → BN + Skip)
  → AdaptiveAvgPool1D → Linear → 128d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# 多尺度卷积前端
# ============================================================

class MultiScaleConv1D(nn.Module):
    """多尺度时序感受野: [30s, 60s, 120s, 240s] 并行卷积 + 拼接"""

    def __init__(self, in_channels: int = 50, out_channels: int = 32):
        super().__init__()
        self.conv_30 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_60 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_120 = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.conv_240 = nn.Conv1d(in_channels, out_channels, kernel_size=17, padding=8)

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]  C=50, T=60
        returns: [B, 128, T]
        """
        c30 = self.conv_30(x)
        c60 = self.conv_60(x)
        c120 = self.conv_120(x)
        c240 = self.conv_240(x)
        out = torch.cat([c30, c60, c120, c240], dim=1)  # [B, 128, T]
        return self.act(self.bn(out))


# ============================================================
# 残差块
# ============================================================

class ResBlock1D(nn.Module):
    """1D 残差块: Conv1D → BN → LeakyReLU → Conv1D → BN + Skip → LeakyReLU"""

    def __init__(self, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # skip connection
        return self.act(out)


# ============================================================
# ResNet 编码器主体
# ============================================================

class ResNetEncoder(nn.Module):
    """
    ResNet 1D 时序编码器

    Input:  [B, feature_dim, seq_len]  (50, 60)
    Output: [B, embedding_dim]         (128)
    """

    def __init__(
        self,
        feature_dim: int = 50,
        seq_len: int = 60,
        embedding_dim: int = 128,
        num_res_blocks: int = 6,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # 多尺度前端
        self.multi_scale = MultiScaleConv1D(feature_dim, hidden_channels // 4)

        # 6 层残差块
        self.res_blocks = nn.Sequential(*[
            ResBlock1D(hidden_channels) for _ in range(num_res_blocks)
        ])

        # 全局池化 + 映射
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, embedding_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, feature_dim, seq_len]
        returns: [B, embedding_dim]
        """
        # 多尺度前端
        x = self.multi_scale(x)       # [B, 128, T]
        # 残差块
        x = self.res_blocks(x)        # [B, 128, T]
        # 全局池化
        x = self.global_pool(x)       # [B, 128, 1]
        x = x.squeeze(-1)             # [B, 128]
        # 映射
        x = self.fc(x)                # [B, 128]
        return x

    def forward_with_skip(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回嵌入 + 最后一层残差块特征 (用于注意力融合)"""
        x = self.multi_scale(x)
        x = self.res_blocks(x)
        feat = self.global_pool(x).squeeze(-1)
        embed = self.fc(feat)
        return embed, feat


# ============================================================
# 注意力门控融合
# ============================================================

class FeatureFusion(nn.Module):
    """
    经验 Alpha 50d ⊕ ResNet 128d → 注意力门控 → 178d 融合向量

    gate = σ(W_g · [empirical; resnet] + b_g)
    fused = gate ⊙ [empirical; resnet]
    """

    def __init__(self, empirical_dim: int = 50, resnet_dim: int = 128, fused_dim: int = 178):
        super().__init__()
        self.empirical_dim = empirical_dim
        self.resnet_dim = resnet_dim
        self.fused_dim = fused_dim

        total_dim = empirical_dim + resnet_dim  # 178
        self.gate_fc = nn.Linear(total_dim, total_dim)
        self.gate_act = nn.Sigmoid()

    def forward(
        self, empirical: torch.Tensor, resnet_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        empirical: [B, 50]  经验特征
        resnet_embed: [B, 128]  ResNet 深度嵌入
        returns: (fused [B, 178], alpha [B, 178])  融合向量 + 注意力权重
        """
        concat = torch.cat([empirical, resnet_embed], dim=-1)  # [B, 178]
        alpha = self.gate_act(self.gate_fc(concat))             # [B, 178]
        fused = alpha * concat                                   # [B, 178]
        return fused, alpha


# ============================================================
# 模型导出工具
# ============================================================

def export_onnx(
    model: ResNetEncoder,
    path: str = "models/resnet/resnet_encoder.onnx",
    seq_len: int = 60,
    feature_dim: int = 50,
):
    """导出为 ONNX 格式 (供 Triton 推断服务)"""
    model.eval()
    dummy = torch.randn(1, feature_dim, seq_len)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={
            "features": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"✓ ResNet ONNX exported: {path}")


def export_torchscript(
    model: ResNetEncoder,
    path: str = "models/resnet/resnet_encoder.pt",
):
    """导出为 TorchScript 格式"""
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f"✓ ResNet TorchScript exported: {path}")


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    # 构建模型
    model = ResNetEncoder(feature_dim=50, seq_len=60, embedding_dim=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNetEncoder: {total_params:,} params")

    # 前向传播测试
    x = torch.randn(4, 50, 60)  # batch=4
    embed = model(x)
    print(f"Input: {x.shape} → Output: {embed.shape}")
    assert embed.shape == (4, 128), f"Expected (4, 128), got {embed.shape}"

    # 融合测试
    fusion = FeatureFusion(50, 128, 178)
    emp = torch.randn(4, 50)
    fused, alpha = fusion(emp, embed)
    print(f"Fusion: empirical{emp.shape} + resnet{embed.shape} → fused{fused.shape}")
    assert fused.shape == (4, 178)

    # 延迟测试
    import time
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 50, 60)
        # 预热
        for _ in range(10):
            model(x)
        # 测量
        start = time.perf_counter()
        for _ in range(1000):
            model(x)
        elapsed = (time.perf_counter() - start) / 1000 * 1000
        print(f"Latency: {elapsed:.2f} ms (target <5ms)")

    print("✓ All ResNet tests passed")
