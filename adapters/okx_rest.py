"""
V8 量化交易系统 - OKX REST API 适配器

职责：
- 封装 OKX REST API v5 调用
- 处理签名、请求、响应解析
- 支持模拟盘和实盘切换
- 错误处理和重试逻辑

注意：
- API 密钥从环境变量读取，不硬编码
- 所有请求带 trace_id 用于链路追踪
"""

import hashlib
import hmac
import base64
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import aiohttp
from common.logging_setup import get_logger, get_trace

logger = get_logger(__name__)


class OkxSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OkxOrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"  # 只做 Maker


class OkxPositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class OkxOrder:
    """OKX 统一订单结构"""
    inst_id: str                    # 交易对 ID（如 BTC-USDT-SWAP）
    side: OkxSide                   # 买卖方向
    position_side: OkxPositionSide  # 持仓方向
    order_type: OkxOrderType        # 订单类型
    size: str                       # 委托数量（张数）
    price: Optional[str] = None     # 委托价格（限价单必填）
    client_order_id: Optional[str] = None  # 客户端自定义订单 ID
    reduce_only: bool = False       # 是否只减仓

    def to_api_dict(self) -> Dict[str, Any]:
        """转换为 OKX API 请求体"""
        d = {
            "instId": self.inst_id,
            "tdMode": "cross",  # 全仓模式
            "side": self.side.value,
            "posSide": self.position_side.value,
            "ordType": self.order_type.value,
            "sz": self.size,
        }
        if self.price:
            d["px"] = self.price
        if self.client_order_id:
            d["clOrdId"] = self.client_order_id
        if self.reduce_only:
            d["reduceOnly"] = True
        return d


class OkxRestAdapter:
    """
    OKX REST API 适配器

    功能：
    - 市场数据查询（行情、K 线、深度）
    - 账户信息（余额、持仓）
    - 订单管理（下单、撤单、查询）
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = "https://www.okx.com",
        demo: bool = False,
        timeout: int = 10,
    ):
        """
        初始化 OKX REST 适配器

        Args:
            api_key: API 密钥
            secret_key: 密钥
            passphrase: 口令
            base_url: REST API 基础 URL
            demo: 是否使用模拟盘
            timeout: 请求超时（秒）
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")
        self.demo = demo
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

        if demo:
            logger.info("OKX REST 适配器初始化 [模拟盘模式]")
        else:
            logger.info("OKX REST 适配器初始化 [实盘模式]")

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self.session

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        生成 OKX API 签名

        签名算法：HMAC SHA256 + Base64

        Args:
            timestamp: ISO 格式时间戳
            method: HTTP 方法（GET/POST）
            path: 请求路径（如 /api/v5/trade/order）
            body: 请求体（POST 时为 JSON 字符串）

        Returns:
            Base64 编码的签名字符串
        """
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        生成 OKX API 请求头

        包含：
        - OK-ACCESS-KEY: API 密钥
        - OK-ACCESS-SIGN: 签名
        - OK-ACCESS-TIMESTAMP: 时间戳
        - OK-ACCESS-PASSPHRASE: 口令
        - x-simulated-trading: 模拟盘标志（如果启用）
        """
        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        sign = self._sign(timestamp, method, path, body)

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        # 模拟盘需要额外 header
        if self.demo:
            headers["x-simulated-trading"] = "1"

        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        发送 HTTP 请求

        Args:
            method: HTTP 方法
            path: API 路径
            params: GET 查询参数
            body: POST 请求体

        Returns:
            API 响应 JSON

        Raises:
            Exception: API 错误或网络错误
        """
        session = await self._get_session()
        url = self.base_url + path

        body_str = json.dumps(body) if body else ""
        headers = self._headers(method, path, body_str)

        # 添加 trace_id 到日志上下文
        trace_id = get_trace()
        logger.debug(f"OKX 请求: {method} {path}", extra={"trace_id": trace_id})

        try:
            async with session.request(method, url, headers=headers, params=params, data=body_str) as resp:
                data = await resp.json()

                if data.get("code") != "0":
                    logger.error(f"OKX API 错误: {data.get('msg', 'Unknown error')}", extra={"code": data.get("code")})
                    raise Exception(f"OKX API Error: {data.get('msg')} (code: {data.get('code')})")

                logger.debug(f"OKX 响应成功: {path}", extra={"data_len": len(data.get("data", []))})
                return data

        except aiohttp.ClientError as e:
            logger.error(f"OKX 网络错误: {e}")
            raise

    # ========== 市场数据接口 ==========

    async def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        """
        获取最新行情

        Args:
            inst_id: 交易对 ID

        Returns:
            行情数据（包含最新价、买一卖一、成交量等）
        """
        path = "/api/v5/market/ticker"
        resp = await self._request("GET", path, params={"instId": inst_id})
        return resp["data"][0] if resp.get("data") else {}

    async def get_orderbook(self, inst_id: str, depth: int = 20) -> Dict[str, Any]:
        """
        获取盘口深度

        Args:
            inst_id: 交易对 ID
            depth: 深度档数（5/10/20/40）

        Returns:
            深度数据（bids/asks 数组）
        """
        path = "/api/v5/market/books"
        resp = await self._request("GET", path, params={"instId": inst_id, "sz": str(depth)})
        return resp["data"][0] if resp.get("data") else {}

    async def get_candles(self, inst_id: str, bar: str = "5m", limit: int = 100) -> List[Dict]:
        """
        获取 K 线数据

        Args:
            inst_id: 交易对 ID
            bar: K 线周期（1m/5m/15m/1H/4H/1D 等）
            limit: 返回数量（最大 300）

        Returns:
            K 线数组（每条包含 ts/o/h/l/c/vol 等字段）
        """
        path = "/api/v5/market/candles"
        resp = await self._request("GET", path, params={"instId": inst_id, "bar": bar, "limit": str(limit)})
        return resp.get("data", [])

    # ========== 账户接口 ==========

    async def get_balance(self, ccy: Optional[str] = None) -> Dict[str, Any]:
        """
        获取账户余额

        Args:
            ccy: 币种（可选，不传返回所有币种）

        Returns:
            余额数据
        """
        path = "/api/v5/account/balance"
        params = {"ccy": ccy} if ccy else None
        resp = await self._request("GET", path, params=params)
        return resp["data"][0] if resp.get("data") else {}

    async def get_positions(self, inst_id: Optional[str] = None) -> List[Dict]:
        """
        获取持仓信息

        Args:
            inst_id: 交易对 ID（可选）

        Returns:
            持仓数组
        """
        path = "/api/v5/account/positions"
        params = {"instId": inst_id} if inst_id else None
        resp = await self._request("GET", path, params=params)
        return resp.get("data", [])

    # ========== 交易接口 ==========

    async def place_order(self, order: OkxOrder) -> Dict[str, Any]:
        """
        下单

        Args:
            order: 订单对象

        Returns:
            下单结果（包含 ordId 等）
        """
        path = "/api/v5/trade/order"
        body = order.to_api_dict()
        resp = await self._request("POST", path, body=body)
        return resp["data"][0] if resp.get("data") else {}

    async def cancel_order(self, inst_id: str, ord_id: str) -> Dict[str, Any]:
        """
        撤单

        Args:
            inst_id: 交易对 ID
            ord_id: 订单 ID

        Returns:
            撤单结果
        """
        path = "/api/v5/trade/cancel-order"
        body = {"instId": inst_id, "ordId": ord_id}
        resp = await self._request("POST", path, body=body)
        return resp["data"][0] if resp.get("data") else {}

    async def get_order(self, inst_id: str, ord_id: str) -> Dict[str, Any]:
        """
        查询订单详情

        Args:
            inst_id: 交易对 ID
            ord_id: 订单 ID

        Returns:
            订单详情
        """
        path = "/api/v5/trade/order"
        resp = await self._request("GET", path, params={"instId": inst_id, "ordId": ord_id})
        return resp["data"][0] if resp.get("data") else {}

    async def close(self):
        """关闭 HTTP 会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("OKX REST 会话已关闭")
