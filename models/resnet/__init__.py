"""
Feature Fusion: 注意力门控融合

Empirical Alpha 50d ⊕ ResNet 128d → 178d 融合向量

gate = σ(W_g · [empirical; resnet] + b_g)
fused = gate ⊙ [empirical; resnet]

参照: models/resnet/resnet_encoder.py 的 FeatureFusion 类
可以直接从 resnet_encoder 导入使用，此文件为独立暴露接口

注意: 需要安装 torch (Phase 2 组件)
"""

try:
    from models.resnet.resnet_encoder import FeatureFusion, ResNetEncoder
    TORCH_AVAILABLE = True
except ImportError:
    # torch 未安装，提供占位符
    FeatureFusion = None
    ResNetEncoder = None
    TORCH_AVAILABLE = False

__all__ = ['FeatureFusion', 'ResNetEncoder', 'TORCH_AVAILABLE']