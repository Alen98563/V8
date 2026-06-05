"""
harness/pipeline_v2.py —Task 6b: 全链�?v2 (特征融合 + �?Alpha + 校准)
=========================================================================

�?pipeline_v1 (Harness trace/span 包装�? 基础上扩展：
    - �?Alpha 融合: OBI + FundingRate �?加权信号
    - 特征组装: 178d 融合特征 (FeatureAssembler)
    - AlphaCast 二次校准 (Recalibrator)
    - MCTS Worker 调度 (自动�?native / fallback)
    - 性能追踪: 各阶段延�?+ 全链路延�?
v1 是无副作用的 trace 包装器；v2 是有业务逻辑的完�?pipeline�?
接口契约�?    - PipelineV2.run_pulse(snap, features_bytes) �?PulseResult
    - �?orchestrator/main_loop.py �?Orchestrator._on_bar_close() 串联
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from common.logging_setup import get_logger, get_trace, new_trace_id, set_pulse, set_trace
from harness.pipeline_v1 import Harness

# Alpha 引擎
from alpha.crypto.obi_v2 import ObiV2Engine, AlphaSignal
from alpha.crypto.funding_rate_arb import FundingRateArbEngine, FundingSignal

# 特征
from features.feature_fusion import FeatureAssembler, DIM_TOTAL

# 校准
from models.alphacast.alphacast_recalib import AlphaCastRecalibrator, RecalibResult

# MCTS
from models.mcts.mcts_worker import MctsWorker

_log = get_logger("harness.pipeline_v2")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PulseResult:
    """单次 pulse 的完整结�?""
    pulse_id: int
    trace_id: str
    ts_ms: int

    # Alpha 信号
    obi_signal: Optional[AlphaSignal] = None
    funding_signal: Optional[FundingSignal] = None
    fused_signal: float = 0.0          # 融合后方向信�?    fused_confidence: float = 0.0      # 融合后置信度

    # 特征
    features_178d: Optional[list] = None

    # 校准
    recalib_result: Optional[RecalibResult] = None

    # MCTS
    mcts_action: str = "hold"
    mcts_ev: float = 0.0
    mcts_position: float = 0.0

    # 门控 (由外�?HardGating 填充)
    gate_passed: bool = False
    gate_reason: str = "-"

    # 执行 (由外�?OrderSender 填充)
    order_sent: bool = False
    fill_receipt: Optional[dict] = None

    # 性能
    latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        d = {
            "pulse_id": self.pulse_id,
            "trace_id": self.trace_id,
            "ts_ms": self.ts_ms,
            "fused_signal": self.fused_signal,
            "fused_confidence": self.fused_confidence,
            "mcts_action": self.mcts_action,
            "mcts_ev": self.mcts_ev,
            "mcts_position": self.mcts_position,
            "gate_passed": self.gate_passed,
            "gate_reason": self.gate_reason,
            "order_sent": self.order_sent,
            "latency_ms": self.latency_ms,
            "total_latency_ms": round(self.total_latency_ms, 3),
        }
        if self.recalib_result:
            d["recalib"] = self.recalib_result.as_dict()
        return d


# ============================================================
# Pipeline V2
# ============================================================

class PipelineV2:
    """
    全链�?Pipeline V2

    用法�?        pipe = PipelineV2()
        result = pipe.run_pulse(
            pulse_id=1,
            snap={"last_px": 3000.0, "bid1": 2999.5, "ask1": 3000.5, ...},
            features_bytes=fe.get_features_50d(),
        )
    """

    def __init__(
        self,
        inst_id: str = "BTC-USDT-SWAP",
        use_funding_arb: bool = True,
        use_recalib: bool = True,
        mcts_force_fallback: bool = False,
    ) -> None:
        self.inst_id = inst_id
        self.harness = Harness("pipeline_v2")

        # Alpha 引擎
        self.obi = ObiV2Engine(inst_id)
        self.funding = FundingRateArbEngine(inst_id) if use_funding_arb else None
        self._use_funding = use_funding_arb

        # 特征组装
        self.assembler = FeatureAssembler()

        # 校准
        self.recalib = AlphaCastRecalibrator() if use_recalib else None
        self._use_recalib = use_recalib

        # MCTS
        self.mcts_worker = MctsWorker(force_fallback=mcts_force_fallback)

        # 信号融合权重
        self._obi_weight = 0.6
        self._funding_weight = 0.4

    def run_pulse(
        self,
        pulse_id: int,
        snap: dict,
        features_bytes: bytes,
        gate_result: Any = None,
        funding_rate: Optional[float] = None,
    ) -> PulseResult:
        """
        执行一次完�?pulse

        流程:
        1. Alpha 信号计算 (OBI + FundingRate)
        2. 信号融合
        3. 特征组装 (178d)
        4. AlphaCast 校准 (如果启用)
        5. MCTS 规划
        """
        t0 = time.perf_counter()
        trace_id = self.harness.begin_pulse(pulse_id)
        ts_ms = int(snap.get("ts_ms", time.time() * 1000))
        latencies = {}

        result = PulseResult(
            pulse_id=pulse_id,
            trace_id=trace_id,
            ts_ms=ts_ms,
        )

        # 1. Alpha 信号
        t1 = time.perf_counter()
        obi_sig = self.harness.stage("alpha_obi", self.obi.on_snapshot, snap)
        latencies["alpha_obi"] = round((time.perf_counter() - t1) * 1000, 3)
        result.obi_signal = obi_sig

        funding_sig = None
        if self._use_funding and self.funding is not None:
            t1 = time.perf_counter()
            rate = funding_rate or 0.0001  # 默认费率
            funding_sig = self.funding.on_funding_rate(rate, ts_ms)
            latencies["alpha_funding"] = round((time.perf_counter() - t1) * 1000, 3)
            result.funding_signal = funding_sig

        # 2. 信号融合
        t1 = time.perf_counter()
        fused_signal = self._obi_weight * obi_sig.raw_signal
        fused_conf = obi_sig.confidence
        if funding_sig is not None:
            fused_signal += self._funding_weight * funding_sig.raw_signal
            fused_conf = 0.5 * obi_sig.confidence + 0.5 * funding_sig.confidence
        result.fused_signal = round(fused_signal, 6)
        result.fused_confidence = round(fused_conf, 4)
        latencies["fusion"] = round((time.perf_counter() - t1) * 1000, 3)

        # 3. 特征组装
        t1 = time.perf_counter()
        features_178d = self.assembler.assemble(
            micro_50d=features_bytes,
            obi_signal=obi_sig,
            funding_signal=funding_sig,
            gate_result=gate_result,
            ts_ms=ts_ms,
        )
        result.features_178d = features_178d
        latencies["assemble"] = round((time.perf_counter() - t1) * 1000, 3)

        # 4. AlphaCast 校准
        if self._use_recalib and self.recalib is not None:
            t1 = time.perf_counter()
            # 使用融合信号作为 predicted_return 的代�?            # (正式版本中这里会调用 Triton AlphaCast 推理)
            recalib_result = self.recalib.evaluate(
                predicted_return=fused_signal * 0.01,  # 信号 �?收益估计
                uncertainty=max(0.01, 1.0 - fused_conf),
                confidence=fused_conf,
            )
            result.recalib_result = recalib_result
            latencies["recalib"] = round((time.perf_counter() - t1) * 1000, 3)

            # 校准拒绝 �?直接 hold
            if recalib_result.action == "reject":
                result.mcts_action = "hold"
                result.total_latency_ms = (time.perf_counter() - t0) * 1000
                result.latency_ms = latencies
                _log.info("pulse_rejected_by_recalib",
                          extra={"reason": recalib_result.reason, "trace_id": trace_id})
                return result

        # 5. MCTS 规划
        t1 = time.perf_counter()
        import struct

        def rollout_fn(fb: bytes) -> bytes:
            return json.dumps({
                "predicted_return": fused_signal * 0.01,
                "confidence": fused_conf,
                "uncertainty": max(0.01, 1.0 - fused_conf),
                "market_state": [fused_signal, 0.02, fused_signal * 0.5, 0.0],
            }).encode("utf-8")

        mcts_bytes = self.mcts_worker.run(features_bytes, rollout_fn)
        mcts_result = json.loads(mcts_bytes.decode("utf-8"))
        latencies["mcts"] = round((time.perf_counter() - t1) * 1000, 3)

        result.mcts_action = mcts_result.get("best_action", "hold")
        result.mcts_ev = mcts_result.get("expected_value", 0.0)
        result.mcts_position = mcts_result.get("best_position", 0.0)

        # 校准降仓
        if result.recalib_result and result.recalib_result.action == "reduce":
            result.mcts_position *= result.recalib_result.position_multiplier

        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        result.latency_ms = latencies

        _log.info(
            "pulse_v2_done",
            extra={
                "pulse_id": pulse_id,
                "action": result.mcts_action,
                "ev": round(result.mcts_ev, 6),
                "pos": round(result.mcts_position, 4),
                "total_ms": round(result.total_latency_ms, 1),
                "trace_id": trace_id,
            },
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mcts": self.mcts_worker.get_stats(),
            "recalib": self.recalib.stats() if self.recalib else {},
        }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import struct

    pipe = PipelineV2(mcts_force_fallback=True)

    # 模拟 3 �?pulse
    for i in range(1, 4):
        snap = {
            "ts_ms": int(time.time() * 1000),
            "last_px": 3000.0 + i * 2,
            "bid1": 2999.5 + i * 2,
            "ask1": 3000.5 + i * 2,
            "bid1_sz": 15.0,
            "ask1_sz": 12.0,
            "spread": 1.0,
        }
        features = struct.pack(f"<{50}f", *([0.05 * i] * 50))

        result = pipe.run_pulse(
            pulse_id=i,
            snap=snap,
            features_bytes=features,
            funding_rate=0.0003,
        )

        print(f"Pulse {i}: action={result.mcts_action}, ev={result.mcts_ev:.6f}, "
              f"pos={result.mcts_position:.4f}, latency={result.total_latency_ms:.1f}ms")
        print(f"  Signal: fused={result.fused_signal:+.4f}, conf={result.fused_confidence:.3f}")
        if result.recalib_result:
            print(f"  Recalib: {result.recalib_result.action} ({result.recalib_result.reason})")

    print(f"\nStats: {json.dumps(pipe.get_stats(), indent=2, default=str)}")
    print("�?PipelineV2 self-test passed")
