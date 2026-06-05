#!/usr/bin/env python3
"""
tick_flow_calc.py — 从 bid_sz/ask_sz 变化推导买卖量 + CVD

V8 tick DB 无 per-tick volume → 用挂单量变化推断:
  - ask_sz 减少 → 市价买入（吃掉 ask）
  - bid_sz 减少 → 市价卖出（吃掉 bid）

输出（per-bar 聚合 flow 特征, 注入到 OF 特征矩阵）
"""

import polars as pl
import numpy as np
from pathlib import Path


def compute_tick_flow_volumes(df: pl.DataFrame) -> pl.DataFrame:
    """
    从 bid_sz/ask_sz 变化推算每 tick 的买卖量。
    """
    df = df.sort('ts')
    bid_sz = df['bid_sz'].to_numpy()
    ask_sz = df['ask_sz'].to_numpy()
    last_px = df['last'].to_numpy()
    
    buy_vol = np.zeros(len(df), dtype=np.float64)
    sell_vol = np.zeros(len(df), dtype=np.float64)
    
    for i in range(1, len(df)):
        d_bid = bid_sz[i] - bid_sz[i-1]
        d_ask = ask_sz[i] - ask_sz[i-1]
        px = last_px[i]
        
        # ask 被吃掉 → 市价买入 (数量×last价格)
        if d_ask < -1e-10:
            buy_vol[i] = abs(d_ask) * px
        
        # bid 被吃掉 → 市价卖出
        if d_bid < -1e-10:
            sell_vol[i] = abs(d_bid) * px
    
    df = df.with_columns([
        pl.Series('buy_vol', buy_vol),
        pl.Series('sell_vol', sell_vol),
    ])
    df = df.with_columns([
        (pl.col('buy_vol') - pl.col('sell_vol')).alias('net_flow'),
        (pl.col('buy_vol') - pl.col('sell_vol')).cum_sum().alias('CVD'),
    ])
    return df


def aggregate_flow_to_bars(df: pl.DataFrame, bar_ms: int = 300000) -> pl.DataFrame:
    """将 tick 级 flow 聚合为 bar 级特征。"""
    df = df.with_columns([(pl.col('ts') // bar_ms).alias('bar_id')])
    
    bars = df.group_by('bar_id').agg([
        pl.col('buy_vol').sum().alias('bar_buy_vol'),
        pl.col('sell_vol').sum().alias('bar_sell_vol'),
        pl.col('net_flow').sum().alias('bar_net_flow'),
        pl.col('CVD').last().alias('bar_CVD'),
        pl.col('ts').first().alias('ts_ms'),
        pl.col('ts').count().alias('tick_count'),
    ]).sort('ts_ms')
    
    bars = bars.with_columns([
        (pl.col('bar_buy_vol') / (pl.col('bar_buy_vol') + pl.col('bar_sell_vol') + 1e-12)).alias('buy_ratio'),
        (pl.col('bar_net_flow') / (pl.col('bar_buy_vol') + pl.col('bar_sell_vol') + 1e-12)).alias('net_flow_ratio'),
        ((pl.col('bar_buy_vol') + pl.col('bar_sell_vol')) / (pl.col('tick_count') + 1e-12)).alias('avg_trade_size'),
        pl.col('bar_CVD').diff().alias('CVD_delta'),
        (pl.col('bar_buy_vol') + pl.col('bar_sell_vol')).alias('total_vol'),
    ])
    return bars


def inject_flow_features(feat_df: pl.DataFrame, bar_flow: pl.DataFrame):
    """
    将 flow 特征注入 OF 特征矩阵。
    Returns: (merged_df, [(feat_name, desc), ...], total_feat_count)
    """
    flow_cols = ['bar_buy_vol', 'bar_sell_vol', 'bar_net_flow', 'bar_CVD',
                 'CVD_delta', 'buy_ratio', 'net_flow_ratio', 'avg_trade_size', 'total_vol']
    bar_subset = bar_flow.select(['ts_ms'] + flow_cols)

    merged = feat_df.join(bar_subset, on='ts_ms', how='left')
    for c in flow_cols:
        merged = merged.with_columns([pl.col(c).fill_null(0).alias(c)])

    n_existing = sum(1 for c in feat_df.columns if c.startswith('feat_'))
    rename = {}
    pairs = []
    for i, col in enumerate(flow_cols):
        new = f'feat_{n_existing + i}'
        rename[col] = new
        pairs.append((new, col))

    merged = merged.rename(rename)
    return merged, pairs, n_existing + len(flow_cols)
