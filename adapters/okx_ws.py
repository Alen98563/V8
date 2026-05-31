"""
V8 量化交易系统 - OKX WebSocket 适配器

职责：
- 封装 OKX WebSocket 连接管理
- 处理订阅/取消订阅
- 自动重连和心跳维护
- 消息分发到回调函数

注意：
- 公共频道无需认证（行情、深度、K 线）
- 私有频道需要登录认证（订单、持仓、账户）
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

import websockets
from websockets.client import WebSocketClientProtocol

from common.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class WsSubscription:
    """WebSocket 订阅配置"""
    channel: str           # 频道名称（tickers/books5/candle5m 等）
    inst_id: str           # 交易对 ID
    callback: Callable     # 消息回调函数
    is_private: bool = False  # 是否私有频道（需要认证）


class OkxWsAdapter:
    """
    OKX WebSocket 适配器

    功能：
    - 多频道订阅管理
    - 自动心跳（30s ping）
    - 断线自动重连
    - 消息路由分发
    """

    def __init__(
        self,
        ws_url: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        reconnect_interval: int = 5,
        heartbeat_interval: int = 25,
    ):
        """
        初始化 WebSocket 适配器

        Args:
            ws_url: WebSocket 地址
            api_key: API 密钥（私有频道需要）
            secret_key: 密钥（私有频道需要）
            passphrase: 口令（私有频道需要）
            reconnect_interval: 重连间隔（秒）
            heartbeat_interval: 心跳间隔（秒，默认 25s，OKX 要求 30s 内）
        """
        self.ws_url = ws_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        self.ws: Optional[WebSocketClientProtocol] = None
        self.subscriptions: Dict[str, WsSubscription] = {}  # key: "channel:inst_id"
        self.is_running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    def _make_subscription_key(self, channel: str, inst_id: str) -> str:
        """生成订阅唯一键"""
        return f"{channel}:{inst_id}"

    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            logger.info(f"正在连接 OKX WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(self.ws_url)
            self.is_running = True
            logger.info("OKX WebSocket 连接成功")

            # 启动心跳和接收任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            raise

    async def disconnect(self):
        """关闭 WebSocket 连接"""
        self.is_running = False

        # 取消后台任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()

        # 关闭连接
        if self.ws:
            await self.ws.close()
            logger.info("OKX WebSocket 已断开")

    async def subscribe(self, channel: str, inst_id: str, callback: Callable, is_private: bool = False):
        """
        订阅频道

        Args:
            channel: 频道名称
            inst_id: 交易对 ID
            callback: 消息回调函数
            is_private: 是否私有频道
        """
        sub_key = self._make_subscription_key(channel, inst_id)

        # 记录订阅
        self.subscriptions[sub_key] = WsSubscription(
            channel=channel,
            inst_id=inst_id,
            callback=callback,
            is_private=is_private,
        )

        # 发送订阅请求
        sub_msg = {
            "op": "subscribe",
            "args": [{"channel": channel, "instId": inst_id}]
        }
        await self.ws.send(json.dumps(sub_msg))
        logger.info(f"已订阅频道: {channel} {inst_id}")

    async def unsubscribe(self, channel: str, inst_id: str):
        """取消订阅"""
        sub_key = self._make_subscription_key(channel, inst_id)

        if sub_key in self.subscriptions:
            # 发送取消订阅请求
            unsub_msg = {
                "op": "unsubscribe",
                "args": [{"channel": channel, "instId": inst_id}]
            }
            await self.ws.send(json.dumps(unsub_msg))
            del self.subscriptions[sub_key]
            logger.info(f"已取消订阅: {channel} {inst_id}")

    async def _heartbeat_loop(self):
        """心跳循环 - 定期发送 ping 保持连接"""
        try:
            while self.is_running:
                await asyncio.sleep(self.heartbeat_interval)
                if self.ws:
                    await self.ws.send("ping")
                    logger.debug("发送心跳 ping")
        except asyncio.CancelledError:
            logger.debug("心跳任务已取消")
        except Exception as e:
            logger.error(f"心跳循环异常: {e}")

    async def _receive_loop(self):
        """接收循环 - 处理 WebSocket 消息"""
        try:
            while self.is_running:
                try:
                    message = await self.ws.recv()

                    # 处理 pong 响应
                    if message == "pong":
                        logger.debug("收到心跳 pong")
                        continue

                    # 解析 JSON 消息
                    data = json.loads(message)

                    # 处理订阅确认
                    if "event" in data:
                        event = data["event"]
                        if event == "subscribe":
                            logger.info(f"订阅确认: {data.get('arg', {})}")
                        elif event == "unsubscribe":
                            logger.info(f"取消订阅确认: {data.get('arg', {})}")
                        elif event == "error":
                            logger.error(f"WebSocket 错误: {data.get('msg')}")
                        continue

                    # 处理数据推送
                    if "arg" in data and "data" in data:
                        arg = data["arg"]
                        channel = arg.get("channel")
                        inst_id = arg.get("instId")
                        sub_key = self._make_subscription_key(channel, inst_id)

                        # 路由到对应回调
                        if sub_key in self.subscriptions:
                            callback = self.subscriptions[sub_key].callback
                            await callback(data["data"])
                        else:
                            logger.warning(f"收到未订阅的消息: {sub_key}")

                except websockets.ConnectionClosed:
                    logger.warning("WebSocket 连接断开，准备重连")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"消息解析失败: {e}, 原始消息: {message}")
                except Exception as e:
                    logger.error(f"消息处理异常: {e}")

        except asyncio.CancelledError:
            logger.debug("接收任务已取消")

    async def auto_reconnect(self):
        """自动重连循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.reconnect_interval)

                # 检查连接状态
                if not self.ws or self.ws.closed:
                    logger.warning("检测到连接断开，尝试重连...")
                    await self.connect()

                    # 重新订阅所有频道
                    for sub_key, sub in self.subscriptions.items():
                        await self.subscribe(sub.channel, sub.inst_id, sub.callback, sub.is_private)
                    logger.info(f"重连成功，已恢复 {len(self.subscriptions)} 个订阅")

            except Exception as e:
                logger.error(f"重连失败: {e}，{self.reconnect_interval}s 后重试")


# ========== 便捷工厂函数 ==========

def create_public_ws_adapter(demo: bool = False) -> OkxWsAdapter:
    """
    创建公共频道 WebSocket 适配器（无需认证）

    Args:
        demo: 是否使用模拟盘

    Returns:
        OkxWsAdapter 实例
    """
    url = "wss://wspap.okx.com:8443/ws/v5/public" if demo else "wss://ws.okx.com:8443/ws/v5/public"
    return OkxWsAdapter(ws_url=url)


def create_private_ws_adapter(
    api_key: str,
    secret_key: str,
    passphrase: str,
    demo: bool = False,
) -> OkxWsAdapter:
    """
    创建私有频道 WebSocket 适配器（需要认证）

    Args:
        api_key: API 密钥
        secret_key: 密钥
        passphrase: 口令
        demo: 是否使用模拟盘

    Returns:
        OkxWsAdapter 实例
    """
    url = "wss://wspap.okx.com:8443/ws/v5/private" if demo else "wss://ws.okx.com:8443/ws/v5/private"
    return OkxWsAdapter(
        ws_url=url,
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
    )
