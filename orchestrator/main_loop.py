"""
orchestrator/main_loop.py — Task 7: 中央编排器主循环 (V8 Orchestrator)
======================================================================

The single entry point that wires every V8 stage into one pulse-driven loop:

    OkxWsClient (WS ingest)
        → FeatureEngine.on_tick (微结构 → 50d 特征)
        → ShmReader.push        (zero-copy SHM ring)
        → ObiV2Engine.on_snapshot   (经验 alpha: OBI/OFI)
        ── every bar close (5m) ──
        → FeatureEngine.on_5m_close → get_features_50d
        → MctsPool.run_sync(rollout = AlphaCast)   (规划 + 期望值)
        → HardGating.evaluate (G1–G5 fail-closed)
        → OrderSender.place    (signed OKX request / dry-run sim fill)
        → PnlAggregator.on_fill (实时对账 + 50笔在线温度标定)
        → Alerter (异常/风控告警)

Every stage runs inside the Harness span so a single ``trace_id`` + ``pulse_id``
is greppable across the whole pipeline. The loop is dry-run safe end-to-end:
with the fallback engine + dry_run=True it drives synthetic snapshots, simulates
fills, and closes the P&L books without ever touching the network.

CLI:
    python -m orchestrator.main_loop                 # run with config/v8.yaml
    python -m orchestrator.main_loop --pulses 5      # bounded run (CI / smoke)
    python -m orchestrator.main_loop --tick-hz 50    # ingest cadence override
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from common.config import V8Config, load_config
from common.engine import (
    FeatureEngine,
    MctsPool,
    OkxChannel,
    OkxWsClient,
    OrderFSM,
    backend,
)
from common.logging_setup import get_logger, get_trace, new_trace_id
from data.shm_bridge import ShmReader
from alpha.crypto.obi_v2 import ObiV2Engine
from gating.hard_gating import GateContext, HardGating
from harness.pipeline_v1 import Harness
from execution.channels.order_sender import OrderSender
from execution.settlement.pnl_aggregator import Fill, PnlAggregator
from orchestrator.alerting import Alerter

_log = get_logger("orchestrator.main_loop")


# ---------------------------------------------------------------------------
# Live, observable state — fed to the dashboard / status endpoint.
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorState:
    backend: str = backend()
    running: bool = False
    pulses: int = 0
    ticks: int = 0
    last_px: float = 0.0
    last_signal: float = 0.0
    last_confidence: float = 0.0
    last_action: str = "hold"
    last_gate: str = "-"
    last_gate_reason: str = "-"
    orders_sent: int = 0
    fills: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position: float = 0.0
    sharpe: float = 0.0
    started_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Orchestrator:
    """Pulse-driven central coordinator. One instance owns the whole pipeline."""

    def __init__(self, cfg: Optional[V8Config] = None) -> None:
        self.cfg = cfg or load_config()
        self.inst_id = self.cfg.inst_id
        self.bar_seconds = self.cfg.bar_seconds

        # --- engine + IO seams (native .pyd or pure-Python fallback) ---------
        self.ws = OkxWsClient(self.inst_id)
        self.fe = FeatureEngine(self.inst_id)
        self.shm = ShmReader(self.inst_id)

        # --- alpha + risk ----------------------------------------------------
        self.alpha = ObiV2Engine(self.inst_id)
        self.gating = HardGating(self.cfg.gating)

        # --- planning (MCTS) + model rollout ---------------------------------
        self.mcts = MctsPool(workers=8, timeout_ms=100)
        self._alpha_client = None  # lazy AlphaCast Triton client (Phase 2)

        # --- execution + settlement ------------------------------------------
        self.channel = OkxChannel(
            self.cfg.okx.api_key,
            self.cfg.okx.secret_key,
            self.cfg.okx.passphrase,
            is_demo=self.cfg.okx.is_demo,
        )
        self.dry_run = self.cfg.dry_run or not self.cfg.okx.configured
        self.sender = OrderSender(self.channel, dry_run=self.dry_run)
        self.pnl = PnlAggregator(on_recalibrate=self._on_recalibrate)

        # --- observability ---------------------------------------------------
        self.harness = Harness("v8_main")
        self.alerter = Alerter()
        self.state = OrchestratorState()

        self._stop = asyncio.Event()

    # =======================================================================
    # Rollout adapter for the MCTS planner.
    # =======================================================================
    def _rollout_fn(self, features_50d: bytes):
        """Build a sync rollout_fn(state_bytes)->bytes for MctsPool.run_sync.

        On native + Triton we hand MCTS the AlphaCast client. Until that path is
        wired (Phase 2), use a deterministic momentum/vol heuristic over the live
        50d feature vector so the planner produces real, non-degenerate EV.
        """

        import struct

        def _rollout(_state_bytes: bytes) -> bytes:
            feats = list(struct.unpack(f"<{len(features_50d)//4}f", features_50d)) if features_50d else []
            momentum = feats[5] if len(feats) > 5 else 0.0     # mean return proxy
            vol = feats[6] if len(feats) > 6 else 0.0           # realized vol proxy
            last_ret = feats[7] if len(feats) > 7 else 0.0
            # predicted_return: momentum nudged by last tick, damped by vol
            denom = 1.0 + 50.0 * abs(vol)
            predicted = (0.7 * momentum + 0.3 * last_ret) / denom
            confidence = max(0.0, min(1.0, 1.0 - 20.0 * vol))
            uncertainty = min(0.2, vol)
            return json.dumps(
                {
                    "predicted_return": predicted,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "market_state": [momentum, vol, last_ret, 0.0],
                }
            ).encode("utf-8")

        return _rollout

    def _on_recalibrate(self, fill_count: int) -> None:
        """Every 50 fills: hook for online AlphaCast temperature scaling."""
        _log.info("temperature_recalibrate", extra={"fill_count": fill_count, "trace_id": get_trace()})

    # =======================================================================
    # Per-tick ingest: WS snapshot → features → SHM → alpha.
    # =======================================================================
    def _ingest_tick(self) -> Optional[dict]:
        raw = self.ws.latest_snapshot()
        if not raw:
            return None
        snap = json.loads(raw) if isinstance(raw, str) else raw

        # feature engine consumes the raw microstructure tick
        self.fe.on_tick(json.dumps(snap).encode("utf-8"))

        # push the current 50d vector into the SHM ring (zero-copy on native)
        import struct

        feats_bytes = self.fe.get_features_50d()
        feats = list(struct.unpack(f"<{len(feats_bytes)//4}f", feats_bytes))
        self.shm.push(snap.get("ts_ms", int(time.time() * 1000)), feats)

        # empirical alpha signal (OBI/OFI)
        sig = self.alpha.on_snapshot(snap)

        self.state.ticks += 1
        self.state.last_px = float(snap.get("last_px", self.state.last_px))
        self.state.last_signal = sig.raw_signal
        self.state.last_confidence = sig.confidence
        return {"snap": snap, "signal": sig, "features": feats_bytes}

    # =======================================================================
    # Per-pulse (bar close): plan → gate → execute → settle.
    # =======================================================================
    async def _on_bar_close(self, last_snap: dict, features_bytes: bytes) -> None:
        pulse_id = self.state.pulses + 1
        self.harness.begin_pulse(pulse_id)
        self.fe.on_5m_close()

        # ---- 1. MCTS planning (rollout = AlphaCast / heuristic) -------------
        plan_bytes = self.harness.stage(
            "mcts",
            self.mcts.run_sync,
            features_bytes,
            self._rollout_fn(features_bytes),
        )
        plan = json.loads(plan_bytes.decode("utf-8"))
        action = plan.get("best_action", "hold")
        position = float(plan.get("best_position", 0.0))
        ev = float(plan.get("expected_value", 0.0))
        self.state.last_action = action

        if action == "hold" or position <= 0:
            _log.info("pulse_hold", extra={"pulse_id": pulse_id, "ev": ev, "trace_id": get_trace()})
            self.state.pulses = pulse_id
            self.pnl.mark(self.state.last_px)
            return

        # ---- 2. HardGating G1–G5 (fail-closed) ------------------------------
        spread = float(last_snap.get("spread", 0.0))
        mid = float(last_snap.get("last_px", 0.0)) or 1.0
        ctx = GateContext(
            spread_bps=(spread / mid * 1e4) if mid else 0.0,
            bid_depth_10=float(last_snap.get("bid1_sz", 0.0)),
            ask_depth_10=float(last_snap.get("ask1_sz", 0.0)),
            realized_vol=abs(self.state.last_signal) * 0.01,
            confidence=self.state.last_confidence,
            uncertainty=max(0.0, 1.0 - self.state.last_confidence) * 0.05,
            is_open_intent=True,
        )
        gate = self.harness.stage("gating", self.gating.evaluate, ctx)
        self.state.last_gate = gate.gate
        self.state.last_gate_reason = gate.reason
        if not gate.passed:
            await self.alerter.info(f"pulse {pulse_id} blocked at {gate.gate}: {gate.reason}")
            self.state.pulses = pulse_id
            self.pnl.mark(self.state.last_px)
            return

        # ---- 3. Execution: build UnifiedOrder, sign + send ------------------
        side = "buy" if action == "buy" else "sell"
        px = self.state.last_px
        sz = round(position, 4)
        cl_ord_id = new_trace_id("o")[:32]
        order = {
            "inst_id": self.inst_id,
            "td_mode": "cross",
            "side": side,
            "order_type": "limit",
            "px": str(px),
            "sz": str(sz),
            "cl_ord_id": cl_ord_id,
            "tag": "v8mcts",
        }
        fsm = OrderFSM(cl_ord_id, self.inst_id, get_trace())
        receipt = await self.harness.astage("execution", self.sender.place, order, fsm)
        self.state.orders_sent += 1

        # ---- 4. Settlement: book the fill -----------------------------------
        if str(receipt.get("code")) == "0" and receipt.get("state") == "FILLED":
            fill = Fill(
                trace_id=get_trace(),
                cl_ord_id=cl_ord_id,
                side=side,
                fill_px=float(receipt.get("fill_px", px)),
                fill_sz=float(receipt.get("fill_sz", sz)),
                fee=float(receipt.get("fee", 0.0)),
                intended_px=px,
            )
            snap = self.harness.stage("settlement", self.pnl.on_fill, fill)
            self.state.fills = snap.fill_count
            self.state.realized_pnl = snap.realized_pnl
            self.state.position = snap.position
            self.state.sharpe = snap.sharpe
        else:
            await self.alerter.warn(
                f"order not filled: code={receipt.get('code')} msg={receipt.get('msg', '')}"
            )

        self.pnl.mark(self.state.last_px)
        self.state.unrealized_pnl = self.pnl.unrealized()
        self.state.pulses = pulse_id
        _log.info(
            "pulse_done",
            extra={
                "pulse_id": pulse_id,
                "action": action,
                "realized": self.state.realized_pnl,
                "position": self.state.position,
                "trace_id": get_trace(),
            },
        )

    # =======================================================================
    # Main run loop.
    # =======================================================================
    async def run(self, max_pulses: int = 0, tick_hz: float = 20.0) -> OrchestratorState:
        """Drive the pipeline. ``max_pulses=0`` runs until stopped (SIGINT)."""
        self.ws.start()
        self.state.running = True
        tick_dt = 1.0 / max(tick_hz, 1.0)
        ticks_per_bar = max(1, int(self.bar_seconds * tick_hz))
        # In dry-run/CI we compress the 5m bar to ~`tick_hz` ticks so pulses fire
        # quickly; live native runs use the real wall-clock bar boundary.
        if self.dry_run:
            ticks_per_bar = max(20, int(tick_hz))

        await self.alerter.info(
            f"V8 orchestrator start: backend={backend()} dry_run={self.dry_run} "
            f"inst={self.inst_id} pulses={max_pulses or '∞'}"
        )
        _log.info(
            "orchestrator_start",
            extra={"backend": backend(), "dry_run": self.dry_run, "inst": self.inst_id},
        )

        last_ingest: Optional[dict] = None
        tick_in_bar = 0
        try:
            while not self._stop.is_set():
                ing = self._ingest_tick()
                if ing is not None:
                    last_ingest = ing
                    tick_in_bar += 1

                if tick_in_bar >= ticks_per_bar and last_ingest is not None:
                    tick_in_bar = 0
                    await self._on_bar_close(last_ingest["snap"], last_ingest["features"])
                    if max_pulses and self.state.pulses >= max_pulses:
                        break

                await asyncio.sleep(tick_dt)
        except asyncio.CancelledError:  # pragma: no cover
            pass
        finally:
            await self.shutdown()

        return self.state

    async def shutdown(self) -> None:
        if not self.state.running:
            return
        self.ws.stop()
        self.state.running = False
        snap = self.pnl.snapshot()
        _log.info(
            "orchestrator_stop",
            extra={
                "pulses": self.state.pulses,
                "ticks": self.state.ticks,
                "fills": snap.fill_count,
                "realized": snap.realized_pnl,
                "sharpe": snap.sharpe,
                "trace_id": get_trace(),
            },
        )
        await self.alerter.info(
            f"V8 orchestrator stop: pulses={self.state.pulses} fills={snap.fill_count} "
            f"realized={snap.realized_pnl} sharpe={snap.sharpe}"
        )

    def request_stop(self) -> None:
        self._stop.set()

    # =======================================================================
    # Synchronous run path — bypasses asyncio entirely.
    # Needed on Windows machines where asyncio's socketpair() triggers a hard
    # crash (security software DLL injection on loopback). In dry-run mode
    # there is no real concurrency: WS gives synthetic snapshots, OrderSender
    # simulates fills instantly, and Alerter is fire-and-forget.
    # =======================================================================
    def _alerter_fire(self, coro) -> None:
        """Sync wrapper for alerter. In sync mode, just log and skip.

        asyncio.new_event_loop() also calls socketpair() internally on Windows,
        which crashes on this machine. So in sync mode we skip alerter entirely
        — it's non-critical (just notifications).
        """
        # In sync mode we cannot create any event loop. Just log the intent.
        try:
            # Extract the message from the coroutine for logging
            if hasattr(coro, 'cr_frame') and coro.cr_frame:
                _log.debug("alerter_skipped_sync", extra={"note": "alerter disabled in sync mode"})
            coro.close()  # properly close the coroutine to avoid RuntimeWarning
        except Exception:
            pass

    def _on_bar_close_sync(self, last_snap: dict, features_bytes: bytes) -> None:
        """Synchronous equivalent of _on_bar_close. No await anywhere."""
        pulse_id = self.state.pulses + 1
        self.harness.begin_pulse(pulse_id)
        self.fe.on_5m_close()

        # ---- 1. MCTS planning ------------------------------------------------
        plan_bytes = self.harness.stage(
            "mcts",
            self.mcts.run_sync,
            features_bytes,
            self._rollout_fn(features_bytes),
        )
        plan = json.loads(plan_bytes.decode("utf-8"))
        action = plan.get("best_action", "hold")
        position = float(plan.get("best_position", 0.0))
        ev = float(plan.get("expected_value", 0.0))
        self.state.last_action = action

        if action == "hold" or position <= 0:
            _log.info("pulse_hold", extra={"pulse_id": pulse_id, "ev": ev, "trace_id": get_trace()})
            self.state.pulses = pulse_id
            self.pnl.mark(self.state.last_px)
            return

        # ---- 2. HardGating G1–G5 --------------------------------------------
        spread = float(last_snap.get("spread", 0.0))
        mid = float(last_snap.get("last_px", 0.0)) or 1.0
        ctx = GateContext(
            spread_bps=(spread / mid * 1e4) if mid else 0.0,
            bid_depth_10=float(last_snap.get("bid1_sz", 0.0)),
            ask_depth_10=float(last_snap.get("ask1_sz", 0.0)),
            realized_vol=abs(self.state.last_signal) * 0.01,
            confidence=self.state.last_confidence,
            uncertainty=max(0.0, 1.0 - self.state.last_confidence) * 0.05,
            is_open_intent=True,
        )
        gate = self.harness.stage("gating", self.gating.evaluate, ctx)
        self.state.last_gate = gate.gate
        self.state.last_gate_reason = gate.reason
        if not gate.passed:
            self._alerter_fire(self.alerter.info(f"pulse {pulse_id} blocked at {gate.gate}: {gate.reason}"))
            self.state.pulses = pulse_id
            self.pnl.mark(self.state.last_px)
            return

        # ---- 3. Execution: build order, simulate fill (sync) -----------------
        side = "buy" if action == "buy" else "sell"
        px = self.state.last_px
        sz = round(position, 4)
        cl_ord_id = new_trace_id("o")[:32]
        order = {
            "inst_id": self.inst_id,
            "td_mode": "cross",
            "side": side,
            "order_type": "limit",
            "px": str(px),
            "sz": str(sz),
            "cl_ord_id": cl_ord_id,
            "tag": "v8mcts",
        }
        fsm = OrderFSM(cl_ord_id, self.inst_id, get_trace())

        # In sync mode we inline the dry-run simulation from OrderSender.place
        # to avoid creating any asyncio event loop (which crashes on this machine).
        if self.dry_run:
            receipt = {
                "code": "0",
                "cl_ord_id": cl_ord_id,
                "ord_id": f"sim-{int(time.time()*1000)}",
                "state": "FILLED",
                "fill_px": px,
                "fill_sz": sz,
                "fee": round(px * sz * 0.0005, 6),
                "trace_id": get_trace(),
                "simulated": True,
            }
            fsm.set_ord_id(receipt["ord_id"])
            fsm.transition(2, "posted")
            fsm.update_fill(px, sz, receipt["fee"])
            fsm.transition(4, "filled")
            _log.info("order_simulated_sync", extra={"trace_id": get_trace(), "cl_ord_id": cl_ord_id, "fill_px": px, "fill_sz": sz})
        else:
            # Live mode requires async — cannot run in sync path
            _log.error("sync_mode_live_unsupported", extra={"trace_id": get_trace()})
            self.state.pulses = pulse_id
            return
        self.state.orders_sent += 1

        # ---- 4. Settlement ---------------------------------------------------
        if str(receipt.get("code")) == "0" and receipt.get("state") == "FILLED":
            fill = Fill(
                trace_id=get_trace(),
                cl_ord_id=cl_ord_id,
                side=side,
                fill_px=float(receipt.get("fill_px", px)),
                fill_sz=float(receipt.get("fill_sz", sz)),
                fee=float(receipt.get("fee", 0.0)),
                intended_px=px,
            )
            snap = self.harness.stage("settlement", self.pnl.on_fill, fill)
            self.state.fills = snap.fill_count
            self.state.realized_pnl = snap.realized_pnl
            self.state.position = snap.position
            self.state.sharpe = snap.sharpe
        else:
            self._alerter_fire(
                self.alerter.warn(f"order not filled: code={receipt.get('code')} msg={receipt.get('msg', '')}")
            )

        self.pnl.mark(self.state.last_px)
        self.state.unrealized_pnl = self.pnl.unrealized()
        self.state.pulses = pulse_id
        _log.info(
            "pulse_done",
            extra={
                "pulse_id": pulse_id,
                "action": action,
                "realized": self.state.realized_pnl,
                "position": self.state.position,
                "trace_id": get_trace(),
            },
        )

    def run_sync(self, max_pulses: int = 0, tick_hz: float = 20.0) -> OrchestratorState:
        """Synchronous main loop. No asyncio event loop, no socketpair."""
        self.ws.start()
        self.state.running = True
        tick_dt = 1.0 / max(tick_hz, 1.0)
        ticks_per_bar = max(1, int(self.bar_seconds * tick_hz))
        if self.dry_run:
            ticks_per_bar = max(20, int(tick_hz))

        self._alerter_fire(
            self.alerter.info(
                f"V8 orchestrator start (SYNC): backend={backend()} dry_run={self.dry_run} "
                f"inst={self.inst_id} pulses={max_pulses or '∞'}"
            )
        )
        _log.info(
            "orchestrator_start_sync",
            extra={"backend": backend(), "dry_run": self.dry_run, "inst": self.inst_id},
        )

        self._stop_sync = False
        last_ingest: Optional[dict] = None
        tick_in_bar = 0
        try:
            while not self._stop_sync:
                ing = self._ingest_tick()
                if ing is not None:
                    last_ingest = ing
                    tick_in_bar += 1

                if tick_in_bar >= ticks_per_bar and last_ingest is not None:
                    tick_in_bar = 0
                    self._on_bar_close_sync(last_ingest["snap"], last_ingest["features"])
                    if max_pulses and self.state.pulses >= max_pulses:
                        break

                time.sleep(tick_dt)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown_sync()

        return self.state

    def _shutdown_sync(self) -> None:
        if not self.state.running:
            return
        self.ws.stop()
        self.state.running = False
        snap = self.pnl.snapshot()
        _log.info(
            "orchestrator_stop_sync",
            extra={
                "pulses": self.state.pulses,
                "ticks": self.state.ticks,
                "fills": snap.fill_count,
                "realized": snap.realized_pnl,
                "sharpe": snap.sharpe,
                "trace_id": get_trace(),
            },
        )
        self._alerter_fire(
            self.alerter.info(
                f"V8 orchestrator stop (SYNC): pulses={self.state.pulses} fills={snap.fill_count} "
                f"realized={snap.realized_pnl} sharpe={snap.sharpe}"
            )
        )


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V8 central orchestrator main loop")
    p.add_argument("--pulses", type=int, default=0, help="bounded run; 0 = run forever")
    p.add_argument("--tick-hz", type=float, default=20.0, help="ingest cadence (ticks/sec)")
    p.add_argument("--config", type=str, default=None, help="path to v8.yaml")
    p.add_argument("--sync", action="store_true", default=False, help="force synchronous mode (no asyncio)")
    return p.parse_args(argv)


async def _amain(argv: Optional[list] = None) -> OrchestratorState:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    orch = Orchestrator(cfg)

    # graceful SIGINT/SIGTERM
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        s = getattr(signal, sig_name, None)
        if s is not None:
            try:
                loop.add_signal_handler(s, orch.request_stop)
            except (NotImplementedError, RuntimeError):  # Windows / no loop signal support
                pass

    state = await orch.run(max_pulses=args.pulses, tick_hz=args.tick_hz)
    print(json.dumps(state.as_dict(), indent=2, ensure_ascii=False))
    return state


def _smain(argv: Optional[list] = None) -> OrchestratorState:
    """Synchronous entry point — no asyncio.run(), no socketpair."""
    args = _parse_args(argv)
    cfg = load_config(args.config)
    orch = Orchestrator(cfg)
    state = orch.run_sync(max_pulses=args.pulses, tick_hz=args.tick_hz)
    print(json.dumps(state.as_dict(), indent=2, ensure_ascii=False))
    return state


def main(argv: Optional[list] = None) -> None:
    """Auto-detect: try async first, fall back to sync on crash."""
    args = _parse_args(argv)
    # If --sync flag or if we know asyncio is broken, go sync directly
    if getattr(args, 'sync', False):
        _smain(argv)
        return
    try:
        asyncio.run(_amain(argv))
    except Exception:
        _log.warning("asyncio failed, falling back to sync mode")
        _smain(argv)


if __name__ == "__main__":
    main()
