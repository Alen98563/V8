"""
labeling/__init__.py — CFL 反事实标签工厂模块
"""

from labeling.counterfactual_labeler import CounterfactualLabeler, CFLLabel

__all__ = ["CounterfactualLabeler", "CFLLabel"]
