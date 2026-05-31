"""
orchestrator/alerting.py — async Telegram / 钉钉 告警组件
======================================================

Fire-and-forget async alerts on httpx. No-network safe: if httpx is missing or
no webhook is configured, it logs the alert instead of raising. Every alert
carries the active trace_id.
"""

from __future__ import annotations

import os
from typing import Optional

from common.logging_setup import get_logger, get_trace

_log = get_logger("orchestrator.alerting")


class Alerter:
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        dingtalk_webhook: Optional[str] = None,
    ) -> None:
        self.tg_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.tg_chat = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.dingtalk = dingtalk_webhook or os.getenv("DINGTALK_WEBHOOK")

    async def _post(self, url: str, payload: dict) -> bool:
        try:
            import httpx  # type: ignore

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(url, json=payload)
                return r.status_code < 400
        except Exception as exc:
            _log.warning("alert_post_failed", extra={"err": str(exc)})
            return False

    async def send(self, level: str, text: str) -> None:
        msg = f"[V8/{level}] {text} (trace={get_trace()})"
        sent = False
        if self.tg_token and self.tg_chat:
            sent |= await self._post(
                f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                {"chat_id": self.tg_chat, "text": msg},
            )
        if self.dingtalk:
            sent |= await self._post(
                self.dingtalk, {"msgtype": "text", "text": {"content": msg}}
            )
        if not sent:
            _log.warning("alert_unrouted", extra={"level": level, "text": text})

    async def info(self, text: str) -> None:
        await self.send("INFO", text)

    async def warn(self, text: str) -> None:
        await self.send("WARN", text)

    async def critical(self, text: str) -> None:
        await self.send("CRIT", text)


__all__ = ["Alerter"]
