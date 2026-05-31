"""
V8 量化交易系统 - 盘口深度管理

职责：
- 维护本地盘口快照
- 计算盘口特征（深度、价差、不平衡度）
- 提供给 Alpha 信号计算使用
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from common.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookSnapshot:
    """盘口快照数据"""
    inst_id: str
    ts: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...] 买盘从大到小
    asks: List[Tuple[float, float]]  # [(price, size), ...] 卖盘从小到大

    @property
    def best_bid(self) -> Optional[float]:
        """买一价"""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """卖一价"""
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """中间价"""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """买卖价差（绝对值）"""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """买卖价差（基点，1 bp = 0.01%）"""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None


class OrderBookManager:
    """
    盘口深度管理器

    功能：
    - 接收并存储盘口快照
    - 计算盘口特征指标
    - 提供历史盘口数据查询
    """

    def __init__(self, max_snapshots: int = 100):
        """
        初始化盘口管理器

        Args:
            max_snapshots: 每个交易对保留的最大快照数量
        """
        self.max_snapshots = max_snapshots
        self._snapshots: Dict[str, List[OrderBookSnapshot]] = {}

    def update_snapshot(self, snapshot: OrderBookSnapshot):
        """
        更新盘口快照

        Args:
            snapshot: 盘口快照数据
        """
        inst_id = snapshot.inst_id

        if inst_id not in self._snapshots:
            self._snapshots[inst_id] = []

        # 添加新快照
        self._snapshots[inst_id].append(snapshot)

        # 限制历史记录数量
        if len(self._snapshots[inst_id]) > self.max_snapshots:
            self._snapshots[inst_id] = self._snapshots[inst_id][-self.max_snapshots:]

    def get_latest(self, inst_id: str) -> Optional[OrderBookSnapshot]:
        """获取最新盘口快照"""
        snapshots = self._snapshots.get(inst_id, [])
        return snapshots[-1] if snapshots else None

    def calc_depth(self, inst_id: str, levels: int = 10) -> Tuple[float, float]:
        """
        计算盘口深度

        Args:
            inst_id: 交易对 ID
            levels: 计算前 N 档

        Returns:
            (bid_depth, ask_depth) 买卖盘深度（基础货币单位）
        """
        snap = self.get_latest(inst_id)
        if not snap:
            return (0.0, 0.0)

        bid_depth = sum(size for _, size in snap.bids[:levels])
        ask_depth = sum(size for _, size in snap.asks[:levels])

        return (bid_depth, ask_depth)

    def calc_order_book_imbalance(self, inst_id: str, levels: int = 10) -> float:
        """
        计算盘口不平衡度（OBI）

        公式：OBI = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        范围：[-1, 1]
        - OBI > 0: 买盘强于卖盘（看涨信号）
        - OBI < 0: 卖盘强于买盘（看跌信号）
        - OBI ≈ 0: 买卖均衡

        Args:
            inst_id: 交易对 ID
            levels: 计算前 N 档

        Returns:
            OBI 值
        """
        bid_depth, ask_depth = self.calc_depth(inst_id, levels)

        total = bid_depth + ask_depth
        if total == 0:
            return 0.0

        obi = (bid_depth - ask_depth) / total
        return obi

    def calc_weighted_price(self, inst_id: str, side: str, levels: int = 5) -> float:
        """
        计算加权价格（VWAP）

        Args:
            inst_id: 交易对 ID
            side: "bid" 或 "ask"
            levels: 计算前 N 档

        Returns:
            加权价格
        """
        snap = self.get_latest(inst_id)
        if not snap:
            return 0.0

        data = snap.bids if side == "bid" else snap.asks
        data = data[:levels]

        if not data:
            return 0.0

        total_size = sum(size for _, size in data)
        if total_size == 0:
            return 0.0

        weighted_price = sum(price * size for price, size in data) / total_size
        return weighted_price

    def get_history(self, inst_id: str, count: int = 10) -> List[OrderBookSnapshot]:
        """
        获取历史盘口快照

        Args:
            inst_id: 交易对 ID
            count: 返回数量

        Returns:
            盘口快照列表（时间从早到晚）
        """
        snapshots = self._snapshots.get(inst_id, [])
        return snapshots[-count:] if len(snapshots) > count else snapshots

    def clear(self, inst_id: Optional[str] = None):
        """
        清空盘口数据

        Args:
            inst_id: 交易对 ID（None 表示清空所有）
        """
        if inst_id:
            self._snapshots.pop(inst_id, None)
            logger.debug(f"已清空 {inst_id} 盘口数据")
        else:
            self._snapshots.clear()
            logger.debug("已清空所有盘口数据")


def parse_okx_orderbook(raw_data: Dict, inst_id: str) -> OrderBookSnapshot:
    """
    解析 OKX WebSocket 盘口数据

    Args:
        raw_data: OKX WebSocket 推送的原始数据
        inst_id: 交易对 ID

    Returns:
        OrderBookSnapshot 实例
    """
    ts = datetime.fromtimestamp(int(raw_data.get("ts", 0)) / 1000)

    bids_raw = raw_data.get("bids", [])
    asks_raw = raw_data.get("asks", [])

    # OKX 格式：[[price, size, 0, ordersCount], ...]
    bids = [(float(p), float(s)) for p, s, *_ in bids_raw]
    asks = [(float(p), float(s)) for p, s, *_ in asks_raw]

    return OrderBookSnapshot(
        inst_id=inst_id,
        ts=ts,
        bids=bids,
        asks=asks,
    )
