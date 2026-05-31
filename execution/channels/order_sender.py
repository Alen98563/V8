"""
execution/channels/order_sender.py â€?async sender for OkxChannel signed requests
=================================================================================

DeepSeek's ``OkxChannel`` (Rust) only *signs* and returns request JSON bytes â€?it never touches the network (to avoid nesting an async runtime inside Python's
loop). This module takes that signed request and actually sends it via httpx,
keeping the FSM in step.

Dry-run safe: when ``dry_run=True`` or no creds, it simulates an immediate fill
so the pipeline closes end-to-end without hitting OKX.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from common.engine import OkxChannel, OrderFSM
from common.logging_setup import get_logger, get_trace

_log = get_logger("execution.order_sender")


class OrderSender:
    def __init__(self, channel: OkxChannel, dry_run: bool = True) -> None:
        self._ch = channel
        self.dry_run = dry_run

    async def _send_signed(self, signed_bytes: bytes) -> dict:
        req = json.loads(signed_bytes.decode("utf-8"))
        try:
            import httpx  # type: ignore

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.request(
                    req["method"], req["url"], headers=req["headers"], content=req["body"]
                )
                return r.json()
        except Exception as exc:
            _log.warning("okx_send_failed", extra={"err": str(exc)})
            return {"code": "-1", "msg": str(exc), "data": []}

    async def place(self, order: dict, fsm: Optional[OrderFSM] = None) -> dict:
        """order: UnifiedOrder-shaped dict. Returns a receipt dict."""
        signed = self._ch.place_order(json.dumps(order).encode("utf-8"))

        if self.dry_run:
            # simulate an immediate full fill at intended px
            px = float(order.get("px") or 0.0)
            sz = float(order.get("sz") or 0.0)
            receipt = {
                "code": "0",
                "cl_ord_id": order.get("cl_ord_id", ""),
                "ord_id": f"sim-{int(time.time()*1000)}",
                "state": "FILLED",
                "fill_px": px,
                "fill_sz": sz,
                "fee": round(px * sz * 0.0005, 6),
                "trace_id": get_trace(),
                "simulated": True,
            }
            if fsm is not None:
                fsm.set_ord_id(receipt["ord_id"])
                fsm.transition(2, "posted")          # POSTED
                fsm.update_fill(px, sz, receipt["fee"])
                fsm.transition(4, "filled")          # FILLED
            _log.info("order_simulated", extra={"trace_id": get_trace(), **{k: receipt[k] for k in ("cl_ord_id", "fill_px", "fill_sz")}})
            return receipt

        resp = await self._send_signed(signed)
        ok = str(resp.get("code")) == "0"
        if fsm is not None:
            if ok:
                fsm.transition(2, "posted")
            else:
                fsm.transition(7, resp.get("msg", "rejected"))
        _log.info("order_sent", extra={"ok": ok, "trace_id": get_trace(), "resp_code": resp.get("code")})
        return resp

    async def cancel(self, ord_id: str, inst_id: str = "BTC-USDT-SWAP") -> dict:
        signed = self._ch.cancel_order(ord_id, inst_id)
        if self.dry_run:
            return {"code": "0", "ord_id": ord_id, "state": "CANCELED", "simulated": True}
        return await self._send_signed(signed)


__all__ = ["OrderSender"]
