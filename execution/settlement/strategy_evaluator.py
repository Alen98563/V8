"""
execution/settlement/strategy_evaluator.py — 策略效能分析模型
============================================================
增强 PnlAggregator，在每笔成交结算后实时输出多维财务指标。
兼容现有 Fill/PnlSnapshot 接口，不破坏已有数据流。

评估维度：
    V1 胜率与盈亏    — win_rate, profit_factor, expectancy
    V2 风险收益      — max_drawdown, calmar, sortino
    V3 资金成本      — funding_cost, opportunity_cost, margin_cost
    V4 时间效率      — bars_traded_pct, turns_per_day, active_time
    V5 综合评分      — 0-100 加权综合分
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque

from common.logging_setup import get_logger

_log = get_logger("settlement.evaluator")


@dataclass
class TradeRecord:
    """单笔完整往返交易（开仓→平仓）。"""
    side: str                 # "long" | "short"
    entry_px: float
    exit_px: float
    sz: float                 # 成交数量
    realized_pnl: float
    fee_entry: float
    fee_exit: float
    entry_ts_ms: int
    exit_ts_ms: int
    bars_held: int = 0
    mae: float = 0.0          # 最大浮亏 (adverse excursion)
    mfe: float = 0.0          # 最大浮盈 (favorable excursion)

    @property
    def total_fee(self) -> float:
        return self.fee_entry + self.fee_exit


@dataclass
class StrategyReport:
    """完整策略评估报告。"""
    # ── V0 基础 ──
    inst_id: str = ""
    total_realized: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    fill_count: int = 0
    closed_trades: int = 0

    # ── V1 胜率与盈亏 ──
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0          # 胜率 = wins / closed_trades
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0    # 盈亏比 = gross_profit / |gross_loss|
    expectancy: float = 0.0        # 期望值 = win_rate * avg_win - (1-win_rate) * |avg_loss|
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # ── V2 风险收益 ──
    max_drawdown: float = 0.0     # 最大回撤 (绝对值)
    max_drawdown_pct: float = 0.0 # 最大回撤百分比
    max_dd_bars: int = 0          # 最大回撤持续 bar 数
    sharpe: float = 0.0           # 年化 Sharpe
    sortino: float = 0.0          # 年化 Sortino (downside-only)
    calmar: float = 0.0           # Calmar = ann_return / max_drawdown_pct
    var_95: float = 0.0           # 95% VaR (per trade)

    # ── V3 资金成本 ──
    max_position: float = 0.0     # 历史最大持仓
    avg_position: float = 0.0     # 平均持仓
    funding_paid: float = 0.0     # 已付资金费率
    funding_est: float = 0.0      # 预估总资金成本
    margin_cost: float = 0.0      # 保证金机会成本 (按无风险利率)
    cost_adjusted_pnl: float = 0.0 # 成本调整后净利

    # ── V4 时间效率 ──
    total_bars: int = 0
    active_bars: int = 0          # 有持仓的 bar 数
    bars_traded_pct: float = 0.0  # 持仓时间占比
    turns_per_day: float = 0.0    # 日均周转次数
    avg_bars_held: float = 0.0    # 平均持仓 bar 数

    # ── V5 综合评分 ──
    composite_score: float = 0.0  # 0-100

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class StrategyEvaluator:
    """
    策略效能评估器。

    用法:
        evaluator = StrategyEvaluator(inst_id="BTC-USDT-SWAP", funding_rate_annual=0.10)
        ... on each fill: evaluator.on_fill(fill, position, avg_entry, realized_since_last)
        report = evaluator.report()
    """

    def __init__(
        self,
        inst_id: str = "",
        funding_rate_annual: float = 0.10,   # OKX 平均年化费率 ~10%
        risk_free_rate: float = 0.03,         # 无风险利率 3%
        margin_ratio: float = 0.05,           # 保证金率 5%
    ) -> None:
        self.inst_id = inst_id
        self.funding_rate_8h = funding_rate_annual / (365 * 3)  # 每 8 小时
        self.risk_free_daily = risk_free_rate / 365
        self.margin_ratio = margin_ratio

        # Trade tracking
        self._trades: List[TradeRecord] = []

        # Real-time tracking (for open trade MAE/MFE)
        self._open_entry: Optional[float] = None
        self._open_side: str = ""
        self._open_sz: float = 0.0
        self._open_fee: float = 0.0
        self._open_ts: int = 0
        self._open_bar_count: int = 0
        self._open_max_px: float = 0.0  # for MAE/MFE
        self._open_min_px: float = 0.0

        # Cumulative
        self.total_realized: float = 0.0
        self.total_fees: float = 0.0
        self.fill_count: int = 0
        self.bar_count: int = 0
        self.positions_in_bar: int = 0

        # Position tracking for drawdown
        self._peak_equity: float = 0.0
        self._max_dd: float = 0.0
        self._max_dd_pct: float = 0.0
        self._dd_start_bar: int = 0
        self._max_dd_bars: int = 0
        self._current_dd_bars: int = 0

        # Position stats
        self._position_history: Deque[float] = deque(maxlen=10000)
        self.max_position: float = 0.0
        self.avg_position: float = 0.0

    # ===================================================================
    # Public API
    # ===================================================================
    def on_fill(self, fill, position_before: float, position_after: float,
                avg_entry: float, last_px: float) -> Optional[TradeRecord]:
        """
        每笔成交调用。

        Returns TradeRecord if a round-trip closed, else None.
        """
        self.fill_count += 1
        self.total_fees += fill.fee

        signed = fill.fill_sz if fill.side == "buy" else -fill.fill_sz
        pos_before = position_before
        pos_after = position_after

        # Position tracking
        self._position_history.append(abs(pos_after))
        self.max_position = max(self.max_position, abs(pos_after))

        closed_trade = None

        if pos_before == 0 and pos_after != 0:
            # ── Opening ──
            self._open_entry = fill.fill_px
            self._open_side = "long" if pos_after > 0 else "short"
            self._open_sz = abs(signed)
            self._open_fee = fill.fee
            self._open_ts = fill.ts_ms
            self._open_bar_count = 0
            self._open_max_px = fill.fill_px
            self._open_min_px = fill.fill_px
            self.positions_in_bar += 1

        elif pos_before != 0 and pos_after != 0 and (pos_before > 0) == (pos_after > 0):
            # ── Adding to position — update MAE/MFE ──
            self._open_max_px = max(self._open_max_px, fill.fill_px)
            self._open_min_px = min(self._open_min_px, fill.fill_px)

        elif pos_before != 0 and pos_after == 0:
            # ── Closing ──
            mae = (self._open_entry - self._open_min_px) if self._open_side == "long" else (self._open_max_px - self._open_entry)
            mfe = (self._open_max_px - self._open_entry) if self._open_side == "long" else (self._open_entry - self._open_min_px)

            closed_trade = TradeRecord(
                side=self._open_side,
                entry_px=self._open_entry or fill.fill_px,
                exit_px=fill.fill_px,
                sz=self._open_sz,
                realized_pnl=self.total_realized - getattr(self, '_last_realized', 0),
                fee_entry=self._open_fee,
                fee_exit=fill.fee,
                entry_ts_ms=self._open_ts,
                exit_ts_ms=fill.ts_ms,
                bars_held=self._open_bar_count,
                mae=mae,
                mfe=mfe,
            )
            self._trades.append(closed_trade)
            self._open_entry = None
            self._open_side = ""
            self._open_sz = 0.0

        # Update equity tracking
        self._update_equity(avg_entry, last_px, pos_after)

        return closed_trade

    def on_bar_close(self, last_px: float, position: float) -> None:
        """每个 bar 结束时调用，用于时间统计和 MAE/MFE 追踪。"""
        self.bar_count += 1
        self._open_bar_count += 1

        if self._open_entry is not None and position != 0:
            self._open_max_px = max(self._open_max_px, last_px)
            self._open_min_px = min(self._open_min_px, last_px)

        if abs(position) > 0.001:
            self.positions_in_bar += 1

    def report(self) -> StrategyReport:
        """生成完整评估报告。"""
        trades = self._trades
        r = StrategyReport()
        r.inst_id = self.inst_id
        r.total_realized = self.total_realized
        r.total_fees = self.total_fees
        r.net_pnl = self.total_realized - self.total_fees
        r.fill_count = self.fill_count
        r.closed_trades = len(trades)

        if not trades:
            return r

        # ── V1 胜率与盈亏 ──
        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl <= 0]
        r.win_count = len(wins)
        r.loss_count = len(losses)
        r.win_rate = len(wins) / len(trades) if trades else 0.0
        r.avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0.0
        r.avg_loss = sum(t.realized_pnl for t in losses) / len(losses) if losses else 0.0
        r.largest_win = max((t.realized_pnl for t in trades), default=0.0)
        r.largest_loss = min((t.realized_pnl for t in trades), default=0.0)

        gross_profit = sum(t.realized_pnl for t in wins)
        gross_loss = abs(sum(t.realized_pnl for t in losses))
        r.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        r.expectancy = r.win_rate * r.avg_win + (1 - r.win_rate) * r.avg_loss

        # ── V2 风险指标 ──
        r.max_drawdown = self._max_dd
        r.max_drawdown_pct = self._max_dd_pct
        r.max_dd_bars = self._max_dd_bars

        # Sortino (downside deviation only)
        rets = [t.realized_pnl for t in trades]
        r.sortino = self._compute_sortino(rets)
        r.sharpe = self._compute_sharpe(rets)

        # Calmar
        if r.max_drawdown_pct > 0:
            ann_return = (r.net_pnl / self.bar_count) * 288 * 365 if self.bar_count > 0 else 0
            r.calmar = ann_return / r.max_drawdown_pct
        else:
            r.calmar = float('inf') if r.net_pnl > 0 else 0.0

        # VaR 95%
        sorted_rets = sorted(rets)
        idx = int(len(rets) * 0.05)
        r.var_95 = sorted_rets[idx] if idx < len(rets) and sorted_rets else 0.0

        # ── V3 资金成本 ──
        r.max_position = self.max_position
        r.avg_position = sum(self._position_history) / len(self._position_history) if self._position_history else 0.0

        # 预估资金费率 (每 8 小时收取，持仓期间)
        bars_with_pos = self.positions_in_bar
        funding_events = bars_with_pos / 288 * 3  # 288 bars/day, 3 times/day
        r.funding_est = self.max_position * self.funding_rate_8h * funding_events

        # 保证金机会成本
        margin_used = self.max_position * self.margin_ratio * r.avg_win if r.avg_win > 0 else self.max_position * self.margin_ratio * 10000
        bars_held_total = sum(t.bars_held for t in trades)
        days_held = bars_held_total / 288 if bars_held_total > 0 else 0
        r.margin_cost = margin_used * self.risk_free_daily * days_held

        r.cost_adjusted_pnl = r.net_pnl - r.funding_est - r.margin_cost

        # ── V4 时间效率 ──
        r.total_bars = self.bar_count
        r.active_bars = self.positions_in_bar
        r.bars_traded_pct = self.positions_in_bar / self.bar_count * 100 if self.bar_count else 0.0
        r.turns_per_day = len(trades) / (self.bar_count / 288) if self.bar_count > 0 else 0.0
        r.avg_bars_held = sum(t.bars_held for t in trades) / len(trades) if trades else 0.0

        # ── V5 综合评分 (0-100) ──
        r.composite_score = self._compute_composite(r)

        return r

    # ===================================================================
    # Internal
    # ===================================================================
    def _update_equity(self, avg_entry: float, last_px: float, position: float) -> None:
        """Update peak equity and drawdown."""
        equity = self.total_realized - self.total_fees
        if position != 0:
            equity += (last_px - avg_entry) * position

        if equity > self._peak_equity:
            self._peak_equity = equity
            self._current_dd_bars = 0
        else:
            dd = self._peak_equity - equity
            dd_pct = dd / self._peak_equity * 100 if self._peak_equity > 0 else 0
            self._current_dd_bars += 1
            if dd > self._max_dd:
                self._max_dd = dd
                self._max_dd_pct = dd_pct
                self._max_dd_bars = self._current_dd_bars

    @staticmethod
    def _compute_sharpe(rets: list) -> float:
        if len(rets) < 3:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        sd = math.sqrt(var)
        return (mean / sd) * math.sqrt(288 * 365) if sd > 0 else 0.0

    @staticmethod
    def _compute_sortino(rets: list) -> float:
        """Sortino: only penalises downside volatility."""
        if len(rets) < 3:
            return 0.0
        mean = sum(rets) / len(rets)
        downside = [r - mean for r in rets if r < mean]
        if not downside:
            return float('inf') if mean > 0 else 0.0
        var_down = sum(d ** 2 for d in downside) / len(downside)
        sd_down = math.sqrt(var_down)
        return (mean / sd_down) * math.sqrt(288 * 365) if sd_down > 0 else 0.0

    def _compute_composite(self, r: StrategyReport) -> float:
        """
        综合评分 (0-100)，权重：
            V1 胜率与盈亏  — 35%
            V2 风险控制    — 35%
            V3 资金效率    — 15%
            V4 时间效率    — 15%
        """
        score = 0.0

        # V1: Win rate + Profit factor (35)
        wr = r.win_rate
        wr_score = min(1.0, (wr - 0.35) / 0.3) if wr > 0.35 else 0.0  # 65% win = 1.0
        pf = r.profit_factor
        pf_score = min(1.0, (pf - 1.0) / 1.5) if pf > 1.0 else 0.0    # pf=2.5 → 1.0
        score += (wr_score * 0.175 + pf_score * 0.175)

        # V2: Sharpe/Sortino + Drawdown control (35)
        sharpe_ratio = min(1.0, max(0.0, r.sharpe / 2.0))  # sharpe=2.0 → 1.0
        sortino_ratio = min(1.0, max(0.0, r.sortino / 3.0)) if r.sortino < float('inf') else 1.0
        dd_ratio = max(0.0, 1.0 - r.max_drawdown_pct / 20.0)  # dd < 20% → ok
        score += (sharpe_ratio * 0.12 + sortino_ratio * 0.12 + dd_ratio * 0.11)

        # V3: Cost efficiency (15)
        cost_ratio = max(0.0, 1.0 - (r.funding_est + r.margin_cost) / max(abs(r.total_realized), 1.0))
        score += cost_ratio * 0.15

        # V4: Time efficiency (15)
        time_ratio = min(1.0, r.bars_traded_pct / 50.0)  # 50% active → 1.0
        turn_ratio = min(1.0, r.turns_per_day / 20.0)     # 20 turns/day → 1.0
        score += (time_ratio * 0.075 + turn_ratio * 0.075)

        return round(score * 100, 1)


__all__ = ["StrategyEvaluator", "StrategyReport", "TradeRecord"]
