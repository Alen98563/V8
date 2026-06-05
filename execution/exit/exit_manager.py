"""
execution/exit/exit_manager.py — V8 退出逻辑

止盈、止损、持仓时间限制、信号反转退出。
在 _on_bar_close_sync 中 MCTS 规划之后、仓位计算之前注入。

v2 (fee-aware): 退出阈值扣除预估平仓手续费，基于净未实现盈亏判断。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    from execution.exchange.exchange_config import ExchangeConfig
except ModuleNotFoundError:
    from exchange_config import ExchangeConfig  # local dev


@dataclass
class ExitConfig:
    """退出条件配置（从 YAML exit: 段读取）"""

    take_profit_pct: float = 0.02       # 止盈：净未实现盈利/仓位名义价值 ≥ 2%
    stop_loss_pct: float = 0.01         # 止损：净未实现亏损/仓位名义价值 ≥ 1%
    max_hold_bars: int = 0              # 最大持仓 bar 数（0=不限）
    signal_reversal_exit: bool = True   # MCTS 信号反转时是否退出
    trailing_stop_pct: float = 0.0      # 回撤止损：从 MFE 高点回撤 ≥ X%（0=关闭）


class ExitManager:
    """退出条件管理器：在每次 bar close 时检查退出条件。

    v2 改进：扣除预估平仓 taker 手续费后再判断 net 止盈 / net 止损。
    """

    def __init__(self, config: ExitConfig, xchg: ExchangeConfig) -> None:
        self.config = config
        self.xchg = xchg
        self._bars_in_position: int = 0
        self._entry_ms: int = 0
        self._mfe_px: float = 0.0       # MFE high water mark (in price terms)
        self._total_funding: float = 0.0 # 累计资金费率支付

    def check(self, position: float, avg_entry: float, unrealized: float,
              last_px: float, equity: float, signal_action: str,
              funding_rate: float = 0.0) -> tuple[str | None, float, float]:
        """检查退出条件。

        Args:
            position: 当前仓位 (+多/-空)
            avg_entry: 平均开仓价
            unrealized: 未实现盈亏 (gross, before fees)
            last_px: 最新成交价
            equity: 账户权益
            signal_action: MCTS 信号方向 ("buy"/"sell")
            funding_rate: 当前资金费率 (from Redis)

        Returns:
            (exit_side, exit_sz, exit_reason_code)
            exit_side: "sell" 平多, "buy" 平空, None 不退出
            exit_reason_code: 0=不退出, 1=止盈, 2=止损, 3=最大持仓, 4=信号反转, 5=回撤止损
        """
        cfg = self.config

        # ── 无仓位 → 重置计数器 ──
        if position == 0:
            self._reset()
            return None, 0.0, 0

        is_long = position > 0
        exit_side = "sell" if is_long else "buy"

        # ── 更新持仓 bar 计数 ──
        self._bars_in_position += 1

        # ── 净未实现盈亏 (扣除预估平仓 taker 费) ──
        exit_fee = self.xchg.exit_fee(position, last_px)
        net_unrealized = unrealized - exit_fee
        pos_notional = max(abs(position) * avg_entry, 1.0)
        net_upnl_pct = net_unrealized / pos_notional
        gross_upnl_pct = unrealized / pos_notional

        # ── 更新 MFE 高点（用于回撤止损） ──
        if cfg.trailing_stop_pct > 0:
            if is_long and last_px > self._mfe_px:
                self._mfe_px = last_px
            elif not is_long and last_px < self._mfe_px:
                self._mfe_px = last_px

            if self._mfe_px > 0:
                drawdown = abs(last_px - self._mfe_px) / self._mfe_px
                if drawdown >= cfg.trailing_stop_pct:
                    return exit_side, abs(position), 5  # trailing stop

        # ── 1. 止盈 (net) ──
        if cfg.take_profit_pct > 0 and net_upnl_pct >= cfg.take_profit_pct:
            return exit_side, abs(position), 1

        # ── 2. 止损 (net) ──
        if cfg.stop_loss_pct > 0 and net_upnl_pct <= -cfg.stop_loss_pct:
            return exit_side, abs(position), 2

        # ── 3. 最大持仓 bar 数 ──
        if cfg.max_hold_bars > 0 and self._bars_in_position >= cfg.max_hold_bars:
            return exit_side, abs(position), 3

        # ── 4. 信号反转退出 ──
        if cfg.signal_reversal_exit:
            if (is_long and signal_action == "sell") or (not is_long and signal_action == "buy"):
                return exit_side, abs(position), 4

        return None, 0.0, 0

    def on_position_closed(self) -> None:
        """持仓完全平仓后调用，重置状态。"""
        self._reset()

    def _reset(self) -> None:
        self._bars_in_position = 0
        self._entry_ms = 0
        self._mfe_px = 0.0
        self._total_funding = 0.0