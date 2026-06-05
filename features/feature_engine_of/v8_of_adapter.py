#!/usr/bin/env python3
"""
v8_of_adapter.py — V8 tick DB → OF FeatureEngine v3.1 适配器 + Flow特征注入
──────────────────────────────────────────────────────────
端到端流程:
  1. find_db_files(inst_id, days) → 匹配 SQLite 文件列表
  2. load_to_buffer(db_paths, inst_id) → MarketStateBuffer (含全量 tick)
  3. FeatureEngine.compute(buffer) → 77维特征 dict
  4. compute_flow_features(db_paths, bar_ms) → 从bid_sz/ask_sz推导买卖量+CVD
  5. inject_flow_into_features(features, flow) → 合并注入
  6. run_on_ticks(db_paths, inst_id) → DataFrame (ts_ms, inst_id, f0-f76+flow)

用法:
  python3 -m features.feature_engine_of.v8_of_adapter \\
    --instrument BTC-USDT-SWAP \\
    --days 2 \\
    --data-dir /home/jerry/V8/data/ticks_db \\
    --output /tmp/of_features_btc.parquet
"""

from __future__ import annotations

import argparse
import sqlite3 as _sql
import sys
import time as _time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Dual-module support: works as package (-m) or standalone script
if __package__:
    from .tick_snapshot import MarketStateBuffer, BufferRegistry, MarketSnapshot
    from ._feature_engine import FeatureEngine, all_feature_names
else:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from features.feature_engine_of.tick_snapshot import MarketStateBuffer, BufferRegistry, MarketSnapshot
    from features.feature_engine_of._feature_engine import FeatureEngine, all_feature_names

FEATURE_NAMES = all_feature_names()
TOTAL_OF_FEATURES = len(FEATURE_NAMES)  # 77


def find_db_files(
    inst_id: str,
    days: int,
    data_dir: str = "/home/jerry/V8/data/ticks_db",
) -> List[Path]:
    """Find SQLite tick DB files for the given instrument and date range."""
    db_root = Path(data_dir) / inst_id
    if not db_root.exists():
        return []

    today = datetime.now(timezone.utc).date()
    files = []
    for d in range(days):
        dt = today - timedelta(days=d)
        # Try both year formats (data is 2026, code used 2025 in prev runs)
        for yr_fmt in [dt.strftime('%Y'), dt.strftime('%y')]:
            pattern = f"{inst_id}_{yr_fmt}{dt.strftime('%m%d')}.db"
            fpath = db_root / pattern
            if fpath.exists() and fpath.stat().st_size > 0:
                files.append(fpath)
                break

    files.sort()
    return files


def load_to_buffer(
    db_paths: List[Path],
    inst_id: str,
    max_snapshots: Optional[int] = None,
) -> MarketStateBuffer:
    """Load ticks from SQLite DBs into MarketStateBuffer."""
    if max_snapshots is None:
        total = 0
        for db_path in db_paths:
            conn = _sql.connect(str(db_path))
            total += conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            conn.close()
        max_snapshots = max(total + 1000, 7200)
        print(f"  Auto-sized buffer: {max_snapshots} snapshots ({total} ticks in DB)")

    buffer = MarketStateBuffer(inst_id, max_snapshots)
    for db_path in db_paths:
        n = buffer.load_from_db(str(db_path))
        if n > 0:
            print(f"  Loaded {n} ticks from {db_path.name}")
    return buffer


def timeline_to_features(
    buffer: MarketStateBuffer,
    bar_duration_ms: int = 300_000,
    universe: Optional[BufferRegistry] = None,
) -> List[dict]:
    """Convert tick timeline → bar-level feature vectors."""
    engine = FeatureEngine()
    all_ticks = list(buffer._buf)

    if len(all_ticks) < 2:
        return []

    features_list = []
    bar_start_ts = all_ticks[0].timestamp * 1000

    in_bar = []
    for snap in all_ticks:
        ts_ms = snap.timestamp * 1000
        if ts_ms >= bar_start_ts + bar_duration_ms:
            feat = _snap_features(engine, in_bar, buffer.market_id, universe)
            feat["ts_ms"] = int(bar_start_ts + bar_duration_ms)
            feat["inst_id"] = buffer.market_id
            features_list.append(feat)
            bar_start_ts += bar_duration_ms
            cutoff = (bar_start_ts - bar_duration_ms) / 1000.0
            in_bar = [s for s in in_bar if s.timestamp >= cutoff]
        in_bar.append(snap)

    if len(in_bar) >= 2:
        feat = _snap_features(engine, in_bar, buffer.market_id, universe)
        feat["ts_ms"] = int(all_ticks[-1].timestamp * 1000)
        feat["inst_id"] = buffer.market_id
        features_list.append(feat)

    return features_list


def _snap_features(
    engine: FeatureEngine,
    ticks: List[MarketSnapshot],
    inst_id: str,
    universe: Optional[BufferRegistry],
) -> dict:
    """Compute 77-dim features for current tick window."""
    buf = MarketStateBuffer(inst_id, max_snapshots=len(ticks) + 10)
    for s in ticks:
        buf.push(s)

    feats = engine.compute(buf, universe)
    result = {}
    for i, name in enumerate(FEATURE_NAMES):
        result[f"feat_{i}"] = feats.get(name, 0.0)
    return result


# ─────────────────────────────────────────────
# Flow Feature Injection (from tick_flow_calc logic)
# ─────────────────────────────────────────────

FLOW_FEATURE_NAMES = [
    'bar_buy_vol', 'bar_sell_vol', 'bar_net_flow', 'bar_CVD',
    'CVD_delta', 'buy_ratio', 'net_flow_ratio', 'avg_trade_size', 'total_vol'
]


def compute_flow_features(
    db_paths: List[Path],
    bar_ms: int = 300_000,
) -> Optional[pl.DataFrame]:
    """
    Load ticks from SQLite DBs, compute inferred flow volumes from bid_sz/ask_sz delta,
    then aggregate to bar-level flow features.

    Returns polars DataFrame with columns: ts_ms, + 9 flow features
    """
    # Load all ticks into polars
    frames = []
    for db_path in db_paths:
        conn = _sql.connect(str(db_path))
        df = pl.read_database("SELECT ts, bid_sz, ask_sz, last FROM ticks ORDER BY ts", conn)
        conn.close()
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    df = pl.concat(frames).sort('ts').unique(subset=['ts'], keep='last')

    # Compute tick-level inferred volumes
    bid_sz = df['bid_sz'].to_numpy()
    ask_sz = df['ask_sz'].to_numpy()
    last_px = df['last'].to_numpy()
    n = len(df)

    buy_vol = np.zeros(n, dtype=np.float64)
    sell_vol = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        d_ask = ask_sz[i] - ask_sz[i - 1]
        d_bid = bid_sz[i] - bid_sz[i - 1]
        px = last_px[i]

        # ask reduced → market buy
        if d_ask < -1e-10:
            buy_vol[i] = abs(d_ask) * px
        # bid reduced → market sell
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

    # Aggregate to bars
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


def inject_flow_into_features(
    features: List[dict],
    bar_flow: pl.DataFrame,
) -> Tuple[List[dict], int]:
    """
    Merge flow features into existing feature dicts by matching ts_ms.
    Adds feat_77 through feat_85 to each dict.
    Returns (updated_features, new_total_count).
    """
    if bar_flow is None or len(bar_flow) == 0:
        return features, TOTAL_OF_FEATURES

    # Build lookup: ts_ms → {flow values}
    flow_cols_raw = ['bar_buy_vol', 'bar_sell_vol', 'bar_net_flow', 'bar_CVD',
                     'CVD_delta', 'buy_ratio', 'net_flow_ratio', 'avg_trade_size', 'total_vol']
    flow_dict = {}
    for row in bar_flow.iter_rows(named=True):
        ts = row['ts_ms']
        flow_dict[ts] = {c: row.get(c, 0.0) for c in flow_cols_raw}

    # Merge: for each bar dict, add flow features with next indices
    for feat in features:
        ts = feat.get('ts_ms', 0)
        flow_vals = flow_dict.get(ts, {})
        # Find nearest match within ±bar window
        if not flow_vals:
            best_ts = None
            best_dist = float('inf')
            for fts in flow_dict:
                d = abs(fts - ts)
                if d < best_dist and d < 150_000:  # within 2.5 min
                    best_dist = d
                    best_ts = fts
            if best_ts:
                flow_vals = flow_dict[best_ts]
            else:
                flow_vals = {c: 0.0 for c in flow_cols_raw}

        for i, col_name in enumerate(flow_cols_raw):
            feat[f'feat_{TOTAL_OF_FEATURES + i}'] = flow_vals.get(col_name, 0.0)

    total = TOTAL_OF_FEATURES + len(flow_cols_raw)
    return features, total


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_on_ticks(
    db_paths: List[Path],
    inst_id: str,
    bar_duration_ms: int = 300_000,
) -> Optional[dict]:
    """
    Full pipeline: load ticks → compute OF features → inject flow features.
    """
    # Step 1: OF features
    buffer = load_to_buffer(db_paths, inst_id)
    if len(buffer) < 2:
        print(f"  ⚠ Not enough ticks ({len(buffer)}), need >= 2")
        return None

    features = timeline_to_features(buffer, bar_duration_ms)

    # Cross-section
    db_root = db_paths[0].parent if db_paths else None
    universe = None
    if db_root and db_root.exists():
        universe = BufferRegistry()
        for inst_dir in db_root.parent.iterdir():
            if inst_dir.is_dir():
                other_id = inst_dir.name
                today_db = inst_dir / f"{other_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.db"
                # Try year 2026 format too
                if not today_db.exists():
                    today_db = inst_dir / f"{other_id}_2026{datetime.now(timezone.utc).strftime('%m%d')}.db"
                if today_db.exists() and today_db.stat().st_size > 0:
                    other_buf = MarketStateBuffer(other_id, max_snapshots=300)
                    other_buf.load_from_db(str(today_db), end_s=_time.time())
                    if len(other_buf) > 0:
                        universe.add(other_buf)
        if len(universe.active_markets()) > 1:
            print(f"  Cross-section: {len(universe.active_markets())} instruments")
            features = timeline_to_features(buffer, bar_duration_ms, universe)

    bar_count_initial = len(features)

    # Step 2: Flow features
    print(f"  Computing flow features from bid_sz/ask_sz delta...")
    t_flow = _time.time()
    bar_flow = compute_flow_features(db_paths, bar_ms=bar_duration_ms)
    flow_elapsed = _time.time() - t_flow

    if bar_flow is not None and len(bar_flow) > 0:
        flow_bars = len(bar_flow)
        flow_nonzero = sum(
            1 for c in ['bar_buy_vol', 'bar_sell_vol']
            if abs(bar_flow[c].sum()) > 1e-10
        )
        print(f"  Flow: {flow_bars} bars, {flow_nonzero}/2 vol cols non-zero ({flow_elapsed:.2f}s)")
        features, total_feats = inject_flow_into_features(features, bar_flow)
    else:
        total_feats = TOTAL_OF_FEATURES
        print(f"  ⚠ Flow computation produced no bars ({flow_elapsed:.2f}s)")

    return {
        "features": features,
        "bar_count": len(features),
        "tick_count": len(buffer),
        "feature_names": FEATURE_NAMES + FLOW_FEATURE_NAMES,
        "total_features": total_feats,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V8 OF FeatureEngine adapter + Flow injection")
    parser.add_argument("--instrument", default="BTC-USDT-SWAP")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--bar-ms", type=int, default=300_000)
    parser.add_argument("--data-dir", default="/home/jerry/V8/data/ticks_db")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"[OF Adapter] Instrument: {args.instrument}, Days: {args.days}, Bar: {args.bar_ms}ms")

    db_paths = find_db_files(args.instrument, args.days, args.data_dir)
    print(f"[OF Adapter] Found {len(db_paths)} DB files: {[p.name for p in db_paths]}")

    if not db_paths:
        print("[OF Adapter] ERROR: No DB files found")
        sys.exit(1)

    t0 = _time.time()
    result = run_on_ticks(db_paths, args.instrument, args.bar_ms)
    elapsed = _time.time() - t0

    if result is None:
        print("[OF Adapter] ERROR: Failed to compute features")
        sys.exit(1)

    print(f"[OF Adapter] Done: {result['bar_count']} bars, {result['tick_count']} ticks, "
          f"{result['total_features']} features ({TOTAL_OF_FEATURES} OF + 9 flow) in {elapsed:.2f}s")

    # Sample diagnostics
    if result["features"]:
        sample = result["features"][0]
        total_feat_cols = [k for k in sample if k.startswith("feat_")]
        non_zero = sum(1 for k in total_feat_cols if abs(sample[k]) > 1e-10)
        print(f"[OF Adapter] Sample bar: {non_zero}/{len(total_feat_cols)} features non-zero")

        # Show top OF features
        of_cols = [k for k in total_feat_cols if int(k.split("_")[1]) < TOTAL_OF_FEATURES]
        top5_of = sorted(
            [(k, sample[k]) for k in of_cols if abs(sample[k]) > 0.01],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        for k, v in top5_of:
            idx = int(k.split("_")[1])
            name = result['feature_names'][idx]
            print(f"  {k} ({name}): {v:.6f}")

        # Show flow features
        flow_cols = [k for k in total_feat_cols if int(k.split("_")[1]) >= TOTAL_OF_FEATURES]
        print(f"[OF Adapter] Flow features ({len(flow_cols)}):")
        for k in flow_cols:
            idx = int(k.split("_")[1])
            name = result['feature_names'][idx] if idx < len(result['feature_names']) else "?"
            print(f"  {k} ({name}): {sample[k]:.6f}")

    # Save
    if args.output:
        import pandas as pd
        df = pd.DataFrame(result["features"])
        df.to_parquet(args.output)
        print(f"[OF Adapter] Saved to {args.output}")

    return result


if __name__ == "__main__":
    main()
