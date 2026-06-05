"""
execution/exchange/exchange_config.py — OKX 交易所配置

合约规格、手续费率、Tick Size 等交易规则参数。
在 _on_bar_close_sync 中供 exit_manager 和 settlement 使用。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ExchangeConfig:
    """交易所规则配置（从 YAML exchange: 段读取）"""

    # ── 手续费 ──
    fee_taker: float = 0.0005     # Taker 费率 (5bps VIP0)
    fee_maker: float = 0.0002     # Maker 费率 (2bps VIP0)

    # ── 合约规格 ──
    tick_size: float = 0.1        # 最小价格变动 (BTC=0.1, ETH=0.01)
    lot_size: float = 1.0         # 合约乘数
    min_size: float = 0.01        # 最小下单量 (BTC=0.01, ETH=0.1)

    # ── 资金费率 ──
    funding_interval_h: float = 8.0   # 资金费率结算间隔 (小时)
    funding_rate_cache: float = 0.0   # 运行时从 Redis 读取

    def round_px(self, px: float) -> float:
        """将价格四舍五入到 tick_size"""
        return round(px / self.tick_size) * self.tick_size

    def round_sz(self, sz: float) -> float:
        """将数量截断到 min_size 的整数倍"""
        return max(round(sz / self.min_size) * self.min_size, 0.0)

    def exit_fee(self, position: float, px: float) -> float:
        """预估平仓手续费 = |position| * px * fee_taker"""
        return abs(position) * px * self.fee_taker

    def entry_fee(self, sz: float, px: float) -> float:
        """开仓手续费 = sz * px * fee_taker"""
        return abs(sz) * px * self.fee_taker

    def funding_payment(self, position: float, px: float,
                        funding_rate: float) -> float:
        """单次资金费率支付 (负 = 多头付, 正 = 空头付)
           payment = position * px * funding_rate
           funding_rate > 0: 多头付空头 → 多头 payment 为负
        """
        return -position * px * funding_rate

    def round_trip_fee(self, position: float, entry_px: float,
                       exit_px: float) -> float:
        """往返手续费 = 开仓费 + 平仓费"""
        return self.entry_fee(position, entry_px) + self.exit_fee(position, exit_px)