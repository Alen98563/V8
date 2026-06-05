#!/usr/bin/env python3
"""
V8 遗传编程进化管道 — 端到端自动策略发现

流程:
1. 从 SQLite tick 数据库加载指定品种的数据
2. 特征引擎: tick → 5min bars → 基础特征
3. FeatureGP: 搜索高 IC 特征
4. 构建完整特征矩阵 (基础 + GP 发现)
5. CFL 标签: 生成 forward_return 标签
6. StrategyGP: 进化 entry/exit 规则树
7. 输出策略基因组 → genome_registry JSON

用法:
  python3 -m evolution.pipeline --instrument BTC-USDT-SWAP --days 3

Author: Hermes
Date: 2026-06-04
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 将 V8 根加入路径
V8_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V8_ROOT))

from features.feature_gp import FeatureGpEngine
from labeling.counterfactual_labeler import CFLConfig, CounterfactualLabeler
from models.strategy_gp import StrategyGpEngine
from features.feature_engine_of.v8_of_adapter import run_on_ticks, find_db_files as find_of_db_files

# ============================================================
# 路径
# ============================================================
TICKS_DB = V8_ROOT / "data" / "ticks_db"
EVOLUTION_DIR = V8_ROOT / "evolution"
GENOME_DIR = EVOLUTION_DIR / "genomes"
FEATURE_POOL_PATH = V8_ROOT / "features" / "feature_gp" / "gp_feature_pool.json"

EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
GENOME_DIR.mkdir(parents=True, exist_ok=True)


def find_db_paths(instrument: str, days: int) -> List[str]:
    """找到指定品种近 N 天的 SQLite DB 文件路径"""
    tick_dir = TICKS_DB / instrument
    if not tick_dir.exists():
        print(f"[Pipeline] 无数据: {tick_dir}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    db_paths = []

    for db_file in sorted(tick_dir.glob(f"{instrument}_*.db")):
        try:
            date_str = db_file.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if file_date < cutoff.replace(hour=0, minute=0, second=0, microsecond=0):
            continue
        db_paths.append(str(db_file))

    if not db_paths:
        print(f"[Pipeline] {instrument} 近 {days} 天无 DB 文件")
    return db_paths


def load_ohlcv_from_ticks(instrument: str, db_paths: List[str], bar_ms: int = 300_000) -> pd.DataFrame:
    """从 tick DB 聚合 OHLCV bars (不使用特征)"""
    all_rows = []
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT ts, inst_id, last, bid, ask, bid_sz, ask_sz, vol_24h FROM ticks ORDER BY ts"):
            all_rows.append(dict(row))
        conn.close()
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    bar_rule = f"{bar_ms}ms"
    bars = df["last"].resample(bar_rule).ohlc()
    bars.columns = ["open", "high", "low", "close"]
    bars["vol"] = df["bid_sz"].resample(bar_rule).sum().fillna(0) + df["ask_sz"].resample(bar_rule).sum().fillna(0)
    bars["ts_ms"] = bars.index.astype("int64") // 1_000_000
    bars["inst_id"] = instrument
    bars = bars.reset_index(drop=True)
    bars = bars.dropna(subset=["close"])
    return bars


def run_feature_engine(instrument: str, db_paths: List[str], bar_ms: int = 300_000) -> pd.DataFrame:
    """运行 OF FeatureEngine v3.1: tick DB → 77-dim 微观特征 + OHLCV"""
    t0 = time.time()

    # 1. OF 77-dim features
    result = run_on_ticks(
        [Path(p) for p in db_paths],
        instrument,
        bar_duration_ms=bar_ms,
    )
    if result is None or not result["features"]:
        print("[Pipeline] OF 特征引擎失败, fallback 到空特征")
        # Fallback: just load OHLCV
        bars = load_ohlcv_from_ticks(instrument, db_paths, bar_ms)
        bars["ts"] = bars["ts_ms"]
        bars["instrument"] = instrument
        print(f"[Pipeline] 仅 OHLCV: {len(bars)} bars")
        return bars

    feat_df = pd.DataFrame(result["features"])
    print(f"[Pipeline] OF特征引擎: {len(feat_df)} bars, {result['bar_count']} bars,  features ({result['tick_count']} ticks) in {time.time()-t0:.1f}s")

    # 2. OHLCV bars (for CFL labels + forward_return)
    ohlcv = load_ohlcv_from_ticks(instrument, db_paths, bar_ms)

    # 3. Merge on ts_ms (round to bar boundary)
    if len(ohlcv) > 0 and len(feat_df) > 0:
        feat_df["bar_ts"] = (feat_df["ts_ms"] // bar_ms) * bar_ms
        ohlcv["bar_ts"] = (ohlcv["ts_ms"] // bar_ms) * bar_ms
        bars = pd.merge(feat_df, ohlcv, on="bar_ts", how="inner", suffixes=("", "_ohlcv"))
        bars["ts_ms"] = bars["ts_ms_ohlcv"]
        bars["inst_id"] = bars["inst_id_ohlcv"]
        drop_cols = [c for c in bars.columns if c.endswith("_ohlcv") and c != "ts_ms"]
        bars = bars.drop(columns=[c for c in drop_cols + ["bar_ts"] if c in bars.columns])
    else:
        bars = feat_df
        bars["open"] = bars["high"] = bars["low"] = bars["close"] = 0.0
        bars["vol"] = 0.0
        bars["tick_count"] = 0
        bars["bid1"] = bars["ask1"] = bars["bid1_sz"] = bars["ask1_sz"] = 0.0

    bars["ts"] = bars["ts_ms"]
    bars["instrument"] = instrument
    print(f"[Pipeline] 合并后: {len(bars)} bars (每 {bar_ms/1000}s)")
    return bars


def run_feature_gp(bars: pd.DataFrame) -> bool:
    """运行 FeatureGP 搜索新特征"""
    if "forward_return" not in bars.columns:
        print("[Pipeline] 跳过 FeatureGP: 缺少 forward_return 列")
        return False

    n_bars = len(bars)
    if n_bars < 150:  # TEMP: lowered for testing
        print(f"[Pipeline] 跳过 FeatureGP: {n_bars} bars < 200")
        return False

    engine = FeatureGpEngine(bars, label_col="forward_return")
    features = engine.run(max_gens=30, pop_size=200, verbose=True)

    if features:
        FeatureGpEngine.save_pool(features, FEATURE_POOL_PATH)
        print(f"[Pipeline] FeatureGP: 发现 {len(features)} 个特征 → {FEATURE_POOL_PATH}")
        return True
    return False


def run_strategy_gp(bars: pd.DataFrame) -> List[Dict]:
    """运行 StrategyGP 搜索策略"""
    if len(bars) < 150:
        print(f"[Pipeline] 跳过 StrategyGP: {len(bars)} bars < 200")
        return []

    # 排除非特征列
    exclude = {"ts", "ts_ms", "instrument", "inst_id", "close_time", "forward_return", "cf_label",
               "open", "high", "low", "close", "vol", "tick_count",
               "bid1", "ask1", "bid1_sz", "ask1_sz"}
    feature_cols = [c for c in bars.columns
                    if bars[c].dtype in ("float64", "float32", "int64") and c not in exclude]

    engine = StrategyGpEngine(bars, feature_cols)
    strategies = engine.run(max_gens=30, pop_size=300, verbose=True)

    result = []
    if strategies:
        output_dir = GENOME_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        engine.save_genomes(strategies, output_dir)

        for bp in strategies:
            result.append({
                "gene_id": bp.gene_id,
                "strategy_name": bp.strategy_name,
                "entry_tree": bp.entry_tree,
                "exit_tree": bp.exit_tree,
                "stop_loss": bp.stop_loss,
                "take_profit": bp.take_profit,
                "position_size": bp.position_size,
                "timeout_minutes": bp.timeout_minutes,
                "fitness": bp.fitness,
                "sharpe": bp.sharpe,
                "win_rate": bp.win_rate,
                "max_drawdown": bp.max_drawdown,
                "total_trades": bp.total_trades,
                "total_pnl": bp.total_pnl,
            })

        print(f"[Pipeline] StrategyGP: 发现 {len(strategies)} 个策略 → {output_dir}")

    return result


def run_pipeline(instrument: str, days: int = 3, bar_ms: int = 300_000,
                 skip_feature_gp: bool = False, skip_strategy_gp: bool = False):
    """完整管道"""

    print(f"\n{'='*60}")
    print(f"V8 进化管道: {instrument} ({days}天数据)")
    print(f"{'='*60}")

    # Step 1: 找 DB 文件
    db_paths = find_db_paths(instrument, days)
    if not db_paths:
        return
    print(f"[Pipeline] 找到 {len(db_paths)} 个 DB 文件")

    # Step 2: 特征引擎 (内部加载 tick)
    bars = run_feature_engine(instrument, db_paths, bar_ms)

    # Step 3: CFL 标签
    try:
        cfl_config = CFLConfig(horizon_bars=12, taker_fee_bps=5.0, profit_threshold_bps=3.0)
        labeler = CounterfactualLabeler(cfl_config)
        # CFL label_dataframe 需要 polars, 列: ts_ms, inst_id, last_px
        import polars as pl
        import labeling.counterfactual_labeler as cfl_mod
        print(f"[Pipeline] CFL module: {cfl_mod.__file__} (lines: {len(open(cfl_mod.__file__).readlines())})")
        df_pl = pl.from_pandas(bars[['ts_ms','inst_id','close']].rename(columns={'close':'last_px'}))
        labels = labeler.label_dataframe(df_pl)
        bars['cf_label'] = labels['cfl_label'].to_list()
        bars['forward_return'] = (bars['close'].shift(-cfl_config.horizon_bars) - bars['close']) / bars['close']
        print(f"[Pipeline] CFL标签: {labels['cfl_label'].drop_nulls().len()} 个")
    except Exception as e:
        print(f"[Pipeline] CFL标签失败: {e}, 用简化标签")
        import traceback; traceback.print_exc()
        bars["forward_return"] = (bars["close"].shift(-12) / bars["close"] - 1)

    # Step 4: FeatureGP
    if not skip_feature_gp:
        run_feature_gp(bars)

    # Step 5: StrategyGP
    if not skip_strategy_gp:
        strategies = run_strategy_gp(bars)

        # 保存报告
        report = {
            "instrument": instrument,
            "bar_count": len(bars),
            "bar_ms": bar_ms,
            "days": days,
            "strategies": strategies,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        report_path = GENOME_DIR / f"report_{instrument}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] 报告 → {report_path}")

    print(f"\n{'='*60}")
    print(f"管道完成")
    print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V8 Evolution Pipeline")
    ap.add_argument("--instrument", default="BTC-USDT-SWAP", help="品种")
    ap.add_argument("--days", type=int, default=3, help="使用近N天数据")
    ap.add_argument("--bar-ms", type=int, default=300_000, help="bar 周期 (ms)")
    ap.add_argument("--skip-feature-gp", action="store_true")
    ap.add_argument("--skip-strategy-gp", action="store_true")
    args = ap.parse_args()

    run_pipeline(
        instrument=args.instrument,
        days=args.days,
        bar_ms=args.bar_ms,
        skip_feature_gp=args.skip_feature_gp,
        skip_strategy_gp=args.skip_strategy_gp,
    )