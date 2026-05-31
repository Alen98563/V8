"""
models/resnet/serve_resnet.py — Task 5: ResNet Triton 部署 + 异步客户端
====================================================================

Exports DeepSeek's ResNetEncoder (50d×60 → 128d embedding) to ONNX, writes the
Triton config.pbtxt (dynamic batching + GPU instance group), and provides an
async client returning the 128d embedding with <5ms target latency.

CLI:
    python -m models.resnet.serve_resnet export   # build ONNX + config.pbtxt
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from common.logging_setup import get_logger
from models.triton_common import TritonAsyncClient, write_config_pbtxt

_log = get_logger("models.resnet.serve")

MODEL_NAME = "resnet_encoder"
REPO_DIR = os.path.join("infra", "triton_models", MODEL_NAME)
SEQ_LEN = 60
FEATURE_DIM = 50
EMBED_DIM = 128


def export(repo_dir: str = REPO_DIR) -> str:
    """Export ResNetEncoder to <repo>/1/model.onnx and write config.pbtxt."""
    import torch  # noqa: F401

    from models.resnet.resnet_encoder import ResNetEncoder, export_onnx

    version_dir = os.path.join(repo_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    model = ResNetEncoder(feature_dim=FEATURE_DIM, seq_len=SEQ_LEN, embedding_dim=EMBED_DIM)
    onnx_path = os.path.join(version_dir, "model.onnx")
    export_onnx(model, path=onnx_path, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM)

    write_config_pbtxt(
        model_dir=repo_dir,
        name=MODEL_NAME,
        platform="onnxruntime_onnx",
        inputs=[{"name": "features", "data_type": "TYPE_FP32", "dims": [FEATURE_DIM, SEQ_LEN]}],
        outputs=[{"name": "embedding", "data_type": "TYPE_FP32", "dims": [EMBED_DIM]}],
        max_batch_size=32,
        preferred_batch=(4, 8, 16),
        max_queue_delay_us=1500,
        gpu_instances=1,
    )
    _log.info("resnet_exported", extra={"onnx": onnx_path})
    return onnx_path


class ResNetClient:
    def __init__(self, url: str = "localhost:8001") -> None:
        self._c = TritonAsyncClient(url=url, model=MODEL_NAME)

    async def embed(self, features: Any) -> Any:
        """features: numpy [B, 50, 60] (or [50,60]); returns [B,128]."""
        import numpy as np

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        out = await self._c.infer({"features": arr}, outputs=["embedding"])
        return out["embedding"]

    async def close(self) -> None:
        await self._c.close()


def _main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export()
    else:
        # smoke: run a stub inference
        import numpy as np

        async def _smoke():
            c = ResNetClient()
            emb = await c.embed(np.zeros((50, 60), dtype=np.float32))
            print("embedding shape:", getattr(emb, "shape", None))
            await c.close()

        asyncio.run(_smoke())


if __name__ == "__main__":
    _main()
