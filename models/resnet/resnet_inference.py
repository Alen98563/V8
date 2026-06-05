"""
Replace ResNetInference with PyTorch-based fine-tuned model.
Matches architecture from local training (hidden=32, scales=(3,5,7), 3 resblocks).
Drop-in replacement for models/resnet/resnet_inference.py
"""

from __future__ import annotations
import logging, os
from typing import Optional, List
import numpy as np

_log = logging.getLogger("models.resnet.inference")

DIM_EMPIRICAL = 50
DIM_DEEP = 128
DIM_FUSED = 178
SEQ_LEN = 60


# ── ResNetEncoder (matches fine-tuned weights) ──
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding="same")
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding="same")
        self.b2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = F.relu(self.b1(self.c1(x)))
        r = self.b2(self.c2(r))
        return F.relu(r + self.skip(x))


class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, kernel, padding="same")
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, kernel, padding="same")
        self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        r = F.relu(self.b1(self.c1(x)))
        r = self.b2(self.c2(r))
        return F.relu(r + x)


class ResNetEncoder(nn.Module):
    """50×T → multi-scale Conv1d → 3 ResBlocks → 128d embedding"""
    def __init__(self, in_ch=50, hidden=32, out_dim=128, scales=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([ConvBlock(in_ch, hidden, ks) for ks in scales])
        self.fuse = nn.Conv1d(hidden * len(scales), hidden, 1)
        self.res_blocks = nn.Sequential(*[ResBlock1D(hidden) for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        cs = [conv(x) for conv in self.convs]
        xf = F.relu(self.fuse(torch.cat(cs, dim=1)))
        xf = self.res_blocks(xf)
        return self.fc(self.pool(xf).squeeze(-1))


class ResNetInference:
    """Fine-tuned ResNetEncoder PyTorch 推理器 (replaces ONNX)"""

    _instance: Optional["ResNetInference"] = None
    _lock = __import__("threading").Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._init()
                    cls._instance = obj
        return cls._instance

    def _init(self):
        self._model: Optional[ResNetEncoder] = None
        self._loaded = False
        self._load()

    def _load(self):
        try:
            self._model = ResNetEncoder()
            self._model.eval()

            # Try fine-tuned weights first, fall back to random init
            pt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", "cfl_classifier", "resnet_finetuned.pt",
            )
            if os.path.exists(pt_path):
                sd = torch.load(pt_path, map_location="cpu")
                self._model.load_state_dict(sd)
                self._loaded = True
                _log.info(f"ResNet loaded (fine-tuned): {pt_path}")
            else:
                _log.warning("No fine-tuned weights found; using random init")
                self._loaded = True
        except Exception as e:
            _log.error(f"ResNet load failed: {e}")
            self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def encode(self, features_50d_60t: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded or self._model is None:
            return None
        try:
            if features_50d_60t.shape == (50, 60):
                inp = torch.from_numpy(features_50d_60t).unsqueeze(0).float()
            else:
                inp = (
                    torch.from_numpy(np.asarray(features_50d_60t, dtype=np.float32))
                    .reshape(1, 50, 60)
                )
            with torch.no_grad():
                out = self._model(inp)
            return out.numpy().ravel().astype(np.float64)
        except Exception as e:
            _log.warning(f"ResNet infer error: {e}")
            return None


class FeatureFusion178:
    """50d empirical ⊕ 128d deep embedding → 178d fused vector"""

    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha

    def fuse(self, empirical: List[float], deep_emb: Optional[List[float]]) -> List[float]:
        emp_50 = list(empirical[:50])
        if len(emp_50) < 50:
            emp_50 += [0.0] * (50 - len(emp_50))
        if deep_emb is None:
            deep_128 = [0.0] * 128
        else:
            deep_128 = list(deep_emb[:128])
            if len(deep_128) < 128:
                deep_128 += [0.0] * (128 - len(deep_128))
        alpha = max(0.0, min(1.0, self.alpha))
        beta = 1.0 - alpha
        return [v * alpha for v in emp_50] + [v * beta for v in deep_128]
