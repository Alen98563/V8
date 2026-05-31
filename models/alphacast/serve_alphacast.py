"""
models/alphacast/serve_alphacast.py — Task 5: AlphaCast Triton 部署 + 客户端
=========================================================================

Exports DeepSeek's AlphaCast Transformer (178d×60 → multi-task) to TorchScript +
ONNX, writes config.pbtxt with dynamic batching, and provides an async client
returning {predicted_return, uncertainty, confidence, market_state} with <10ms
target latency. This client is what the MCTS rollout_fn calls.

CLI:
    python -m models.alphacast.serve_alphacast export
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from common.logging_setup import get_logger
from models.triton_common import TritonAsyncClient, write_config_pbtxt

_log = get_logger("models.alphacast.serve")

MODEL_NAME = "alphacast"
REPO_DIR = os.path.join("infra", "triton_models", MODEL_NAME)
SEQ_LEN = 60
INPUT_DIM = 178


def export(repo_dir: str = REPO_DIR) -> str:
    import torch  # noqa: F401

    from models.alphacast.alphacast_model import AlphaCastModel, export_onnx

    version_dir = os.path.join(repo_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    model = AlphaCastModel(input_dim=INPUT_DIM, d_model=256, nhead=8, num_layers=6)
    onnx_path = os.path.join(version_dir, "model.onnx")
    export_onnx(model, path=onnx_path, seq_len=SEQ_LEN, input_dim=INPUT_DIM)

    write_config_pbtxt(
        model_dir=repo_dir,
        name=MODEL_NAME,
        platform="onnxruntime_onnx",
        inputs=[{"name": "features", "data_type": "TYPE_FP32", "dims": [SEQ_LEN, INPUT_DIM]}],
        outputs=[
            {"name": "predicted_return", "data_type": "TYPE_FP32", "dims": [1]},
            {"name": "uncertainty", "data_type": "TYPE_FP32", "dims": [1]},
            {"name": "confidence", "data_type": "TYPE_FP32", "dims": [1]},
            {"name": "market_state", "data_type": "TYPE_FP32", "dims": [4]},
        ],
        max_batch_size=16,
        preferred_batch=(2, 4, 8),
        max_queue_delay_us=2000,
        gpu_instances=1,
    )
    _log.info("alphacast_exported", extra={"onnx": onnx_path})
    return onnx_path


class AlphaCastClient:
    """Async AlphaCast inference. ``rollout_bytes`` is the adapter the MCTS pool
    (Rust or fallback) calls per simulation: state JSON in, AlphaCastOutput JSON
    out.
    """

    OUTPUTS = ["predicted_return", "uncertainty", "confidence", "market_state"]

    def __init__(self, url: str = "localhost:8001", temperature: float = 1.0) -> None:
        self._c = TritonAsyncClient(url=url, model=MODEL_NAME)
        self.temperature = temperature

    async def predict(self, features: Any) -> dict:
        import numpy as np

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        out = await self._c.infer({"features": arr}, outputs=self.OUTPUTS)
        pr = float(np.asarray(out["predicted_return"]).ravel()[0])
        return {
            "predicted_return": pr / max(self.temperature, 1e-6),
            "uncertainty": float(np.asarray(out["uncertainty"]).ravel()[0]),
            "confidence": float(np.asarray(out["confidence"]).ravel()[0]),
            "market_state": np.asarray(out["market_state"]).ravel().tolist(),
            "temperature": self.temperature,
            "latency_ms": out.get("_latency_ms", 0.0),
        }

    def make_rollout_fn(self, feature_provider):
        """Build a sync rollout_fn(state_bytes)->bytes for MctsPool.run_sync.

        ``feature_provider(state: dict) -> numpy[60,178]`` supplies the model
        input for a given MCTS state. Runs the async predict on a private loop
        (MctsPool calls this from a worker thread / executor).
        """

        def _rollout(state_bytes: bytes) -> bytes:
            try:
                state = json.loads(state_bytes.decode("utf-8")) if state_bytes else {}
            except Exception:
                state = {}
            feats = feature_provider(state)
            res = asyncio.run(self.predict(feats))
            return json.dumps(res).encode("utf-8")

        return _rollout

    async def close(self) -> None:
        await self._c.close()


def _main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export()
    else:
        import numpy as np

        async def _smoke():
            c = AlphaCastClient()
            res = await c.predict(np.zeros((SEQ_LEN, INPUT_DIM), dtype=np.float32))
            print(json.dumps(res, indent=2))
            await c.close()

        asyncio.run(_smoke())


if __name__ == "__main__":
    _main()
