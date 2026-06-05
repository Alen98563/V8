"""Data models for crypto market TimescaleDB tables.

Each record maps 1:1 to a column set in the corresponding crypto.* table.
Column tuples are used by writer.py for executemany() parameter binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


@dataclass(slots=True)
class TickRecord:
    """crypto.tick — 逐笔成交"""

    ts: datetime
    ts_ns: int
    ticker: str
    exchange: str
    trade_id: str
    price: float
    size: float
    side: str
    trade_mode: str = ""
    source: str = "ws"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ts_ns", "ticker", "exchange", "trade_id", "price", "size", "side", "trade_mode", "source")


@dataclass(slots=True)
class OrderbookRecord:
    """crypto.orderbook — 盘口快照"""

    ts: datetime
    ts_ns: int
    ticker: str
    side: str
    level: int
    price: float
    size: float
    count: Optional[int] = None
    seq_id: Optional[int] = None
    action: str = "snapshot"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ts_ns", "ticker", "side", "level", "price", "size", "count", "seq_id", "action")


@dataclass(slots=True)
class OhlcvRecord:
    """crypto.ohlcv — K线"""

    ts: datetime
    ticker: str
    bar: str
    open: float
    high: float
    low: float
    close: float
    vol: float
    vol_ccy: float = 0.0
    vol_ccy_quote: Optional[float] = None
    confirm: bool = True
    source: str = "ws"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "bar", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm", "source")


@dataclass(slots=True)
class FundingRateRecord:
    """crypto.funding_rate — 资金费率"""

    ts: datetime
    ticker: str
    funding_rate: float
    next_funding_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None
    method: str = "next_period_min"
    realized_rate: Optional[float] = None
    source: str = "rest"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "funding_rate", "next_funding_rate", "next_funding_time", "method", "realized_rate", "source")


@dataclass(slots=True)
class MarkPriceRecord:
    """crypto.mark_price — 标记价格"""

    ts: datetime
    ticker: str
    mark_px: float
    index_px: Optional[float] = None
    source: str = "ws"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "mark_px", "index_px", "source")


@dataclass(slots=True)
class OpenInterestRecord:
    """crypto.open_interest — 持仓量"""

    ts: datetime
    ticker: str
    oi: float
    oi_ccy: Optional[float] = None
    oi_usd: Optional[float] = None
    source: str = "rest"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "oi", "oi_ccy", "oi_usd", "source")


@dataclass(slots=True)
class LiquidationRecord:
    """crypto.liquidations — 强平数据"""

    ts: datetime
    ticker: str
    side: str
    bk_px: float
    sz: float
    bk_loss: Optional[float] = None
    source: str = "ws"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "side", "bk_px", "sz", "bk_loss", "source")


@dataclass(slots=True)
class LsRatioRecord:
    """crypto.ls_ratio — 多空比"""

    ts: datetime
    ticker: str
    ratio_type: str
    long_ratio: float
    short_ratio: float
    source: str = "rest"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "ratio_type", "long_ratio", "short_ratio", "source")


@dataclass(slots=True)
class RegimeRecord:
    """crypto.regime — 市场状态"""

    ts: datetime
    ticker: str
    regime: str
    regime_score: Optional[float] = None
    hurst: Optional[float] = None
    vol_regime: Optional[str] = None
    vol_percentile: Optional[float] = None
    detection_model: str = "hmm"

    @classmethod
    def columns(cls) -> Tuple[str, ...]:
        return ("ts", "ticker", "regime", "regime_score", "hurst", "vol_regime", "vol_percentile", "detection_model")
