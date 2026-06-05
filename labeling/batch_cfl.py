#!/usr/bin/env python3
"""
batch_cfl.py — Batch CFL labeling for multi-instrument feature parquets
Usage: python3 batch_cfl.py --input features_btc.parquet --output cfl_btc.parquet
"""

import argparse, sys
import numpy as np
import polars as pl
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFLConfig:
    horizon_bars: int = 12
    taker_fee_bps: float = 5.0
    profit_threshold_bps: float = 3.0
    loss_threshold_bps: float = 1.0

def compute_cfl(entry_px, future_prices, config):
    if not future_prices or entry_px <= 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    round_trip = config.taker_fee_bps * 2
    returns_bps = [(px / entry_px - 1.0) * 10000.0 for px in future_prices]
    mfe, mae = max(returns_bps), min(returns_bps)
    pnl_net = mfe - round_trip
    
    if pnl_net > config.profit_threshold_bps:
        label = 1
    elif pnl_net < -config.loss_threshold_bps:
        label = -1
    else:
        label = 0
    
    raw = abs(pnl_net) / (abs(pnl_net) + round_trip) if abs(pnl_net) > 0 else 0.0
    conf = float(np.clip(raw, 0.0, 1.0))
    return (label, pnl_net, mfe, round_trip, mae, mfe, conf)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, default=12)
    args = p.parse_args()

    df = pl.read_parquet(args.input)
    n = df.height
    horizon = args.horizon
    config = CFLConfig(horizon_bars=horizon)
    
    prices = df['close'].to_numpy()
    nrows = n - horizon
    
    labels, pnl_nets, pnl_gross, costs, maes, mfes, confs = [], [], [], [], [], [], []
    
    for i in range(nrows):
        entry = prices[i]
        future = prices[i+1:i+1+horizon].tolist()
        label, pnl_net, pnl_g, cost, mae, mfe, conf = compute_cfl(entry, future, config)
        labels.append(label)
        pnl_nets.append(pnl_net)
        pnl_gross.append(pnl_g)
        costs.append(cost)
        maes.append(mae)
        mfes.append(mfe)
        confs.append(conf)
    
    # Build labels DF
    label_df = pl.DataFrame({
        'ts_ms': df['ts_ms'][:nrows],
        'inst_id': df['inst_id'][:nrows],
        'label': labels,
        'pnl_net': pnl_nets,
        'pnl_gross': pnl_gross,
        'cost_bps': costs,
        'max_adverse_excursion': maes,
        'max_favorable_excursion': mfes,
        'confidence_weight': confs,
        'horizon_bars': [horizon] * nrows,
    })
    
    # Merge
    merged = df[:nrows].join(label_df, on=['ts_ms', 'inst_id'], how='left')
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(args.output)
    
    vc = label_df['label'].value_counts().sort('label')
    total = label_df.height
    pos = label_df.filter(pl.col('label') == 1).height
    neg = label_df.filter(pl.col('label') == -1).height
    zero = label_df.filter(pl.col('label') == 0).height
    
    print(f"{args.input} -> {args.output}")
    print(f"  Bars: {nrows} | +1={pos}({100*pos/total:.1f}%) 0={zero}({100*zero/total:.1f}%) -1={neg}({100*neg/total:.1f}%)")
    print(f"  Mean PnL_net: {np.mean(pnl_nets):.2f} bps | Mean confidence: {np.mean(confs):.3f}")

if __name__ == '__main__':
    main()
