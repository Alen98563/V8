#!/usr/bin/env python3
"""
tools/eval_replay.py v2 — 从 fill_settled 日志生成策略评估报告
"""

import json, sys, os, math
from dataclasses import dataclass

@dataclass
class Trade:
    side: str
    entry_px: float
    exit_px: float
    sz: float
    realized: float
    entry_ts: float
    exit_ts: float
    bars: int = 0

@dataclass
class Report:
    inst_id: str = ""
    fills: int = 0; closed_trades: int = 0
    total_realized: float = 0.0; total_fees: float = 0.0; net_pnl: float = 0.0
    win_count: int = 0; loss_count: int = 0
    win_rate: float = 0.0; profit_factor: float = 0.0; expectancy: float = 0.0
    avg_win: float = 0.0; avg_loss: float = 0.0
    max_drawdown: float = 0.0; max_dd_pct: float = 0.0
    sharpe: float = 0.0; sortino: float = 0.0; calmar: float = 0.0; var_95: float = 0.0
    max_position: float = 0.0; avg_position: float = 0.0
    funding_est: float = 0.0; margin_cost: float = 0.0; cost_adj_pnl: float = 0.0
    total_bars: int = 0; active_bars: int = 0; bar_active_pct: float = 0.0
    turns_per_day: float = 0.0; avg_bars_held: float = 0.0
    score: float = 0.0


def _close_segment(fills, start, end, trades):
    if start < 0 or end < start:
        return
    first = fills[start]
    last = fills[end]
    side = "long" if first["position"] > 0 else "short"
    entry_px = first["px"]
    exit_px = last["px"]
    sz = abs(first["position"])  # starting size of this segment
    realized = fills[end]["realized"] - fills[max(0, start-1)]["realized"]
    bars = max(1, int((last["ts"] - first["ts"]) / 300))
    trades.append(Trade(side=side, entry_px=entry_px, exit_px=exit_px,
                        sz=sz, realized=realized,
                        entry_ts=first["ts"], exit_ts=last["ts"], bars=bars))


def parse_log(log_path, inst_id="BTC-USDT-SWAP"):
    fills = []
    with open(log_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("msg") != "fill_settled":
                continue
            fills.append(ev)

    if not fills:
        return [], Report(inst_id=inst_id)

    trades = []
    pos_vals = []
    realized_history = []
    equity_curve = []
    peak_equity = -float('inf')

    # Find position PROGRESSION changes
    prev_pos = 0.0
    for i, f in enumerate(fills):
        pos = f["position"]
        pos_vals.append(abs(pos))
        realized_history.append(f["realized"])
        eq = f["realized"]
        equity_curve.append(eq)
        if eq > peak_equity:
            peak_equity = eq
        prev_pos = pos

    # Segment detection: direction changes
    seg_start = -1
    prev_dir = 0  # -1 short, 0 flat, 1 long
    for i, f in enumerate(fills):
        pos = f["position"]
        cur_dir = 0 if abs(pos) < 0.0001 else (1 if pos > 0 else -1)

        if cur_dir != 0 and prev_dir == 0:
            # Opened new position
            seg_start = i
        elif cur_dir == 0 and prev_dir != 0 and seg_start >= 0:
            # Closed position
            _close_segment(fills, seg_start, i, trades)
            seg_start = -1
        prev_dir = cur_dir

    # Handle still-open
    if seg_start >= 0:
        _close_segment(fills, seg_start, len(fills)-1, trades)

    # ── Build Report ──
    r = Report(inst_id=inst_id)
    r.fills = len(fills)
    r.closed_trades = len(trades)
    r.total_realized = fills[-1]["realized"] if fills else 0.0
    r.net_pnl = r.total_realized - r.total_fees
    r.max_position = max(pos_vals) if pos_vals else 0
    r.avg_position = sum(pos_vals)/len(pos_vals) if pos_vals else 0

    # V1
    if trades:
        wins = [t for t in trades if t.realized > 0]
        losses = [t for t in trades if t.realized <= 0]
        r.win_count = len(wins)
        r.loss_count = len(losses)
        r.win_rate = len(wins) / len(trades)
        r.avg_win = sum(t.realized for t in wins)/len(wins) if wins else 0
        r.avg_loss = sum(t.realized for t in losses)/len(losses) if losses else 0
        gp = sum(t.realized for t in wins)
        gl = abs(sum(t.realized for t in losses))
        r.profit_factor = gp/gl if gl > 0 else (float('inf') if gp > 0 else 0)
        r.expectancy = r.win_rate * r.avg_win + (1-r.win_rate) * r.avg_loss

    # V2
    if equity_curve and peak_equity > 0:
        r.max_drawdown = max((peak_equity-eq for eq in equity_curve), default=0.0)
        r.max_dd_pct = r.max_drawdown / peak_equity * 100

        rets = [equity_curve[i]-equity_curve[i-1] for i in range(1, len(equity_curve))]
        if rets and len(rets) >= 3:
            mean = sum(rets)/len(rets)
            sd = math.sqrt(sum((x-mean)**2 for x in rets)/len(rets))
            sc = math.sqrt(288*365)
            r.sharpe = (mean/sd)*sc if sd > 0 else 0.0

            # Sortino per-trade
            if trades:
                tr = [t.realized for t in trades]
                tm = sum(tr)/len(tr)
                dwn = [min(0, x-tm) for x in tr]
                vd = sum(d**2 for d in dwn)/len(dwn)
                sdd = math.sqrt(vd)
                r.sortino = (tm/sdd)*math.sqrt(288*365) if sdd > 0 else 99.0

            # Calmar
            if r.max_dd_pct > 0:
                daily_ret = r.net_pnl / (len(rets) / 288) if len(rets) >= 288 else r.net_pnl
                ann_ret = daily_ret * 365
                r.calmar = ann_ret / r.max_dd_pct

        if trades:
            r.var_95 = sorted(t.realized for t in trades)[max(0, int(len(trades)*0.05))]

    # V3 & V4: Time-based from timestamp range
    if fills:
        ts0 = fills[0]["ts"]
        ts1 = fills[-1]["ts"]
        r.total_bars = max(1, int((ts1-ts0)/300))
        # Active bars: count unique 5-min windows with position > 0
        active_windows = set()
        for f in fills:
            if abs(f["position"]) > 0.001:
                bar_idx = int((f["ts"] - ts0) / 300)
                active_windows.add(bar_idx)
        r.active_bars = len(active_windows)
        r.bar_active_pct = r.active_bars/r.total_bars*100 if r.total_bars else 0

        funding_8h = 0.10/(365*3)
        avg_px = sum(f["px"] for f in fills)/len(fills)
        funding_events = r.total_bars/288*3
        r.funding_est = r.avg_position * avg_px * funding_8h * funding_events
        r.margin_cost = r.max_position * avg_px * 0.05 * 0.03/365 * (r.total_bars/288)
        r.cost_adj_pnl = r.net_pnl - r.funding_est - r.margin_cost

    if trades:
        r.turns_per_day = len(trades)/(r.total_bars/288) if r.total_bars else 0
        r.avg_bars_held = sum(t.bars for t in trades)/len(trades)

    # V5
    wr_s = min(1.0, max(0, (r.win_rate-0.35)/0.3))
    pf_s = min(1.0, max(0, (r.profit_factor-1.0)/1.5))
    sharpe_s = min(1.0, max(0, r.sharpe/2.0))
    dd_s = max(0, 1.0 - r.max_dd_pct/20.0)
    cost_s = max(0, 1.0 - (r.funding_est+r.margin_cost)/max(abs(r.total_realized), 1))
    time_s = min(1.0, r.bar_active_pct/50.0)
    r.score = round((wr_s*17.5 + pf_s*17.5 + sharpe_s*24 + dd_s*11 + cost_s*15 + time_s*15), 1)

    return trades, r


def print_report(r: Report):
    print("=" * 60)
    print(f"  策略评估报告 — {r.inst_id}")
    print("=" * 60)
    print()
    print("── 基础指标 ──")
    print(f"  成交笔数:       {r.fills:>8}")
    print(f"  闭环交易数:     {r.closed_trades:>8}")
    print(f"  已实现盈亏:     {r.total_realized:>+12.2f} USDT")
    print(f"  总手续费:       {r.total_fees:>12.2f} USDT")
    print(f"  净利:           {r.net_pnl:>+12.2f} USDT")
    print()
    print("── V1 胜率与盈亏 ──")
    print(f"  胜率:           {r.win_rate*100:>8.1f}% ({r.win_count}W/{r.closed_trades}T)")
    pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 999 else "∞"
    print(f"  盈亏比:         {pf_str:>8}")
    print(f"  平均盈利:       {r.avg_win:>+12.2f} USDT")
    print(f"  平均亏损:       {r.avg_loss:>+12.2f} USDT")
    print(f"  每笔期望:       {r.expectancy:>+12.2f} USDT")
    print()
    print("── V2 风险收益 ──")
    print(f"  最大回撤:       {r.max_drawdown:>12.2f} USDT ({r.max_dd_pct:.1f}%)")
    print(f"  Sharpe:         {r.sharpe:>8.2f}")
    so_str = f"{r.sortino:.2f}" if r.sortino < 999 else "∞"
    print(f"   Sortino:        {so_str:>8}")
    print(f"  Calmar:         {r.calmar:>8.2f}")
    print(f"  VaR 95%:        {r.var_95:>+12.2f} USDT")
    print()
    print("── V3 资金成本 ──")
    print(f"  最大持仓:       {r.max_position:>12.4f} BTC")
    print(f"  平均持仓:       {r.avg_position:>12.4f} BTC")
    print(f"  预估资金费率:   {r.funding_est:>12.2f} USDT")
    print(f"  保证金机会成本: {r.margin_cost:>12.2f} USDT")
    print(f"  调整后净利:     {r.cost_adj_pnl:>+12.2f} USDT")
    print()
    print("── V4 时间效率 ──")
    print(f"  总 Bars:        {r.total_bars:>8}")
    print(f"  活跃 Bars:      {r.active_bars:>8} ({r.bar_active_pct:.1f}%)")
    print(f"  日均周转:       {r.turns_per_day:>8.1f} 次")
    print(f"  平均持仓:       {r.avg_bars_held:>8.1f} bars")
    print()
    print("── V5 综合评分 ──")
    bar = "█" * int(r.score/5) + "░" * (20 - int(r.score/5))
    print(f"  {r.score:>5.1f}/100  [{bar}]")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/eval_replay.py <log_file> [inst_id]")
        sys.exit(1)
    trades, r = parse_log(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "BTC-USDT-SWAP")
    print_report(r)