"""
models/triton_common.py — shared Triton/ONNX export + async client helpers
==========================================================================

Keeps Task 5 (serve_resnet.py / serve_alphacast.py) DRY: model→ONNX/TorchScript
export, config.pbtxt generation (dynamic_batching + GPU instance group), and an
async gRPC inference client targeting <10ms.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from common.logging_setup import get_logger

_log = get_logger("models.triton")


def write_config_pbtxt(
    model_dir: str,
    name: str,
    platform: str,
    inputs: Sequence[dict],
    outputs: Sequence[dict],
    max_batch_size: int = 32,
    preferred_batch: Sequence[int] = (4, 8, 16),
    max_queue_delay_us: int = 2000,
    gpu_instances: int = 1,
) -> str:
    """Emit a Triton config.pbtxt with dynamic batching + instance group."""
    def _io_block(io: dict) -> str:
        dims = ", ".join(str(d) for d in io["dims"])
        return (
            "  {\n"
            f"    name: \"{io['name']}\"\n"
            f"    data_type: {io['data_type']}\n"
            f"    dims: [ {dims} ]\n"
            "  }"
        )

    inputs_s = ",\n".join(_io_block(i) for i in inputs)
    outputs_s = ",\n".join(_io_block(o) for o in outputs)
    pref = ", ".join(str(b) for b in preferred_batch)
    body = f"""name: "{name}"
platform: "{platform}"
max_batch_size: {max_batch_size}
input [
{inputs_s}
]
output [
{outputs_s}
]
dynamic_batching {{
  preferred_batch_size: [ {pref} ]
  max_queue_delay_microseconds: {max_queue_delay_us}
}}
instance_group [
  {{
    count: {gpu_instances}
    kind: KIND_GPU
  }}
]
"""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "config.pbtxt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    _log.info("config_pbtxt_written", extra={"path": path, "model": name})
    return path


class TritonAsyncClient:
    """Thin async wrapper over tritonclient.grpc.aio with a graceful no-server
    path (returns a deterministic stub) so the orchestrator/dry-run works
    without a live Triton.
    """

    def __init__(self, url: str = "localhost:8001", model: str = "", version: str = "") -> None:
        self.url = url
        self.model = model
        self.version = version
        self._client = None

    async def _ensure(self) -> bool:
        if self._client is not None:
            return True
        try:
            import tritonclient.grpc.aio as grpcclient  # type: ignore

            self._client = grpcclient.InferenceServerClient(url=self.url)
            return True
        except Exception as exc:  # no tritonclient or no server
            _log.warning("triton_unavailable", extra={"err": str(exc), "url": self.url})
            return False

    async def infer(self, inputs: dict, outputs: Sequence[str]) -> dict:
        """inputs: {name: numpy array}. Returns {name: numpy array}."""
        import time as _t

        t0 = _t.perf_counter()
        ok = await self._ensure()
        if not ok or self._client is None:
            return self._stub(inputs, outputs, latency_ms=(_t.perf_counter() - t0) * 1000)
        try:
            import numpy as np
            import tritonclient.grpc.aio as grpcclient  # type: ignore

            infer_inputs = []
            for name, arr in inputs.items():
                arr = np.ascontiguousarray(arr.astype(np.float32))
                tin = grpcclient.InferInput(name, list(arr.shape), "FP32")
                tin.set_data_from_numpy(arr)
                infer_inputs.append(tin)
            infer_outputs = [grpcclient.InferRequestedOutput(o) for o in outputs]
            res = await self._client.infer(
                model_name=self.model,
                inputs=infer_inputs,
                outputs=infer_outputs,
                model_version=self.version or "",
            )
            out = {o: res.as_numpy(o) for o in outputs}
            out["_latency_ms"] = (_t.perf_counter() - t0) * 1000  # type: ignore
            return out
        except Exception as exc:
            _log.warning("triton_infer_failed", extra={"err": str(exc)})
            return self._stub(inputs, outputs, latency_ms=(_t.perf_counter() - t0) * 1000)

    def _stub(self, inputs: dict, outputs: Sequence[str], latency_ms: float) -> dict:
        import numpy as np

        # deterministic-ish stub derived from input mean
        seed = 0.0
        for arr in inputs.values():
            try:
                seed = float(np.asarray(arr).mean())
            except Exception:
                pass
        out: dict = {}
        for o in outputs:
            if "return" in o:
                out[o] = np.array([float(np.tanh(seed))], dtype=np.float32)
            elif "uncertain" in o:
                out[o] = np.array([0.02], dtype=np.float32)
            elif "confidence" in o:
                out[o] = np.array([0.6], dtype=np.float32)
            elif "state" in o:
                out[o] = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
            elif "embedding" in o:
                out[o] = np.zeros((1, 128), dtype=np.float32)
            else:
                out[o] = np.zeros((1,), dtype=np.float32)
        out["_latency_ms"] = latency_ms
        out["_stub"] = True
        return out

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass


__all__ = ["write_config_pbtxt", "TritonAsyncClient"]
