"""
AlphaCast 模块

包含:
- AlphaCastModel: Transformer 6L8H 多任务预测
- AlphaCastRecalib: 二次校准 (置信度过滤/仓位精炼)
- TemperatureScaling: 在线温度缩放校准

注意: 需要安装 torch (Phase 2 组件)
"""

try:
    from models.alphacast.alphacast_model import (
        AlphaCastModel,
        AlphaCastRecalib,
        TemperatureScaling,
        PositionalEncoding,
    )
    TORCH_AVAILABLE = True
except ImportError:
    # torch 未安装，提供占位符
    AlphaCastModel = None
    AlphaCastRecalib = None
    TemperatureScaling = None
    PositionalEncoding = None
    TORCH_AVAILABLE = False

__all__ = [
    "AlphaCastModel",
    "AlphaCastRecalib",
    "TemperatureScaling",
    "PositionalEncoding",
    "TORCH_AVAILABLE",
]