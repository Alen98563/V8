"""
V8 量化交易系统 - 市场数据模块

职责：
- 从 OKX REST API 获取历史行情数据
- K 线数据加载和缓存
- 资金费率查询
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from common.logging_setup import get_logger
from adapters.okx_rest import OkxRestAdapter

logger = get_logger(__name__)


class MarketDataLoader:
    """
    市场数据加载器

    功能：
    - 历史 K 线数据批量拉取
    - 数据缓存（避免重复请求）
    - 资金费率查询
    """

    def __init__(self, rest_adapter: OkxRestAdapter, cache_ttl: int = 300):
        """
        初始化市场数据加载器

        Args:
            rest_adapter: OKX REST 适配器实例
            cache_ttl: 缓存有效期（秒，默认 5 分钟）
        """
        self.rest = rest_adapter
        self.cache_ttl = cache_ttl
        self._candles_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}

    async def get_candles(
        self,
        inst_id: str,
        bar: str = "5m",
        limit: int = 100,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取 K 线数据

        Args:
            inst_id: 交易对 ID
            bar: K 线周期（1m/5m/15m/1H/4H/1D）
            limit: 返回数量
            use_cache: 是否使用缓存

        Returns:
            DataFrame，列：ts, open, high, low, close, vol, volCcy
        """
        cache_key = f"{inst_id}_{bar}_{limit}"

        # 检查缓存
        if use_cache and cache_key in self._candles_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if (datetime.now().timestamp() - cache_time) < self.cache_ttl:
                logger.debug(f"使用缓存 K 线: {cache_key}")
                return self._candles_cache[cache_key]

        # 从 API 拉取
        logger.info(f"拉取 K 线数据: {inst_id} {bar} limit={limit}")
        raw_data = await self.rest.get_candles(inst_id, bar, limit)

        if not raw_data:
            logger.warning(f"未获取到 K 线数据: {inst_id}")
            return pd.DataFrame()

        # 解析为 DataFrame
        df = pd.DataFrame(raw_data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "vol": float,
            "volCcy": float,
        })
        df = df.sort_values("ts").reset_index(drop=True)

        # 更新缓存
        self._candles_cache[cache_key] = df
        self._cache_timestamps[cache_key] = datetime.now().timestamp()

        return df

    async def get_funding_rate(self, inst_id: str) -> Optional[float]:
        """
        获取当前资金费率

        Args:
            inst_id: 交易对 ID

        Returns:
            资金费率（小数形式，如 0.0001 表示 0.01%）
        """
        try:
            path = "/api/v5/public/funding-rate"
            resp = await self.rest._request("GET", path, params={"instId": inst_id})
            data = resp.get("data", [])
            if data:
                funding_rate = float(data[0].get("fundingRate", 0))
                logger.debug(f"{inst_id} 资金费率: {funding_rate:.6f}")
                return funding_rate
        except Exception as e:
            logger.error(f"获取资金费率失败: {e}")
        return None

    async def get_next_funding_time(self, inst_id: str) -> Optional[datetime]:
        """
        获取下次资金费率结算时间

        Args:
            inst_id: 交易对 ID

        Returns:
            下次结算时间
        """
        try:
            path = "/api/v5/public/funding-rate"
            resp = await self.rest._request("GET", path, params={"instId": inst_id})
            data = resp.get("data", [])
            if data:
                next_ts = int(data[0].get("nextFundingTime", 0))
                if next_ts > 0:
                    return datetime.fromtimestamp(next_ts / 1000)
        except Exception as e:
            logger.error(f"获取下次资金费率时间失败: {e}")
        return None

    def clear_cache(self):
        """清空缓存"""
        self._candles_cache.clear()
        self._cache_timestamps.clear()
        logger.debug("市场数据缓存已清空")
