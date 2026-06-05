# -*- coding: utf-8 -*-
"""
offline_feature_engine.py — Python batch port of Rust FeatureEngine
Recalculates 50d features from SQLite tick DBs.
Outputs parquet with bars + features, ready for CFL labeling.

Supports: --bar-ms to switch bar duration (300000=5m, 900000=15m, 3600000=1h)
"""
import sqlite3, argparse, sys, time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Deque
import numpy as np
import pandas as pd
from scipy import stats

# —— Constants (matching feature_engine.rs) ——————
FEATURE_DIM = 50
# Tick-level window constants (in seconds of tick data)
WINDOW_1M  = 60    # ~1 min of 1s ticks
WINDOW_5M  = 300   # ~5 min
WINDOW_15M = 900   # ~15 min
WINDOW_1H  = 3600  # ~1 hour

BAR_LABEL = {300_000: '5m', 900_000: '15m', 1_800_000: '30m',
             3_600_000: '1h', 14_400_000: '4h'}

# —— Bar dataclass ——————————————

@dataclass
class Bar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    vol: float = 0.0
    tick_count: int = 0
    bid1: float = 0.0
    bid1_sz: float = 0.0
    ask1: float = 0.0
    ask1_sz: float = 0.0

    def update(self, px, sz, bid1, ask1, bid1_sz, ask1_sz):
        self.high = max(self.high, px)
        self.low = min(self.low, px)
        self.close = px
        self.vol += sz
        self.tick_count += 1
        self.bid1 = bid1
        self.bid1_sz = bid1_sz
        self.ask1 = ask1
        self.ask1_sz = ask1_sz

# —— Feature computations (pure functions, matching Rust) ——

def momentum(prices: np.ndarray, w: int) -> float:
    if len(prices) <= w or w == 0: return 0.0
    curr, past = prices[-1], prices[-1 - w]
    return float((curr - past) / past) if past > 0 else 0.0

def historical_vol(prices: np.ndarray, w: int) -> float:
    if len(prices) < w + 2: return 0.0
    rets = np.diff(prices[-w:]) / prices[-w:-1]
    if len(rets) < 2: return 0.0
    annual_factor = np.sqrt(252 * 288)
    return float(np.std(rets) * annual_factor)

def compute_hurst(prices: np.ndarray, w: int = WINDOW_5M) -> float:
    n = min(len(prices), w)
    if n < 60: return 0.5
    rets = np.diff(prices[-n:])
    mean = rets.mean()
    cum_dev = np.cumsum(rets - mean)
    R = cum_dev.max() - cum_dev.min()
    S = rets.std()
    if S > 0:
        return max(0.0, min(1.0, float(np.log(R / S) / np.log(n))))
    return 0.5

def vwap_deviation(prices: np.ndarray, volumes: np.ndarray, w: int) -> float:
    n = min(len(prices), w, len(volumes))
    if n < 2: return 0.0
    ps = prices[-n:]; vs = volumes[-n:]
    den = vs.sum()
    if den > 0:
        vwap_val = np.average(ps, weights=vs)
        return float((ps[-1] - vwap_val) / vwap_val) if vwap_val > 0 else 0.0
    return 0.0

def realized_skewness(prices: np.ndarray, w: int) -> float:
    rets = get_returns(prices, w)
    if len(rets) < 3: return 0.0
    return float(stats.skew(rets))

def realized_kurtosis(prices: np.ndarray, w: int) -> float:
    rets = get_returns(prices, w)
    if len(rets) < 4: return 0.0
    return float(stats.kurtosis(rets))

def vol_profile(volumes: np.ndarray, w: int) -> float:
    n = min(len(volumes), w)
    if n < 1: return 0.0
    recent = volumes[-n:].sum()
    total = volumes.sum()
    return float(recent / total) if total > 0 else 0.0

def vol_price_corr(prices: np.ndarray, volumes: np.ndarray, w: int) -> float:
    n = min(len(prices), w, len(volumes))
    if n < 10: return 0.0
    ps = prices[-n:]; vs = volumes[-n:]
    corr = np.corrcoef(ps, vs)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def price_acceleration(prices: np.ndarray) -> float:
    if len(prices) < 60: return 0.0
    now = momentum(prices, 30)
    prev = momentum(prices[:-30], 30)
    return float(now - prev)

def log_return(prices: np.ndarray, w: int) -> float:
    if len(prices) <= w: return 0.0
    curr, past = prices[-1], prices[-1 - w]
    return float(np.log(curr / past)) if past > 0 else 0.0

def amihud(prices: np.ndarray, volumes: np.ndarray, w: int) -> float:
    n = min(len(prices), w, len(volumes))
    if n < 10: return 0.0
    ps = prices[-n:]; vs = volumes[-n:]
    rets = np.abs(np.diff(ps) / ps[:-1])
    dv = vs[1:] * ps[1:]
    dv[dv == 0] = 1.0
    return float((rets / dv).mean())

def vol_mean_reversion(prices: np.ndarray) -> float:
    short = historical_vol(prices, WINDOW_5M)
    long_vol = historical_vol(prices, WINDOW_1H)
    return float((short - long_vol) / long_vol) if long_vol > 0 else 0.0

def bollinger_position(prices: np.ndarray, w: int) -> float:
    n = min(len(prices), w)
    if n < 5: return 0.0
    ps = prices[-n:]
    mean = ps.mean()
    std = ps.std()
    curr = ps[-1]
    return float((curr - mean) / (2 * std)) if std > 0 else 0.0

def atr(prices: np.ndarray, w: int) -> float:
    if len(prices) < w + 2: return 0.0
    n = min(len(prices), w)
    ps = prices[-n:]
    tr = np.abs(np.diff(ps))
    return float(tr.mean())

def get_returns(prices: np.ndarray, w: int) -> np.ndarray:
    n = min(len(prices), w)
    return np.diff(prices[-n:]) / prices[-n:-1]

# —— Stateful feature helpers (require orderbook context) ——

def compute_obi(prices: np.ndarray, volumes: np.ndarray) -> float:
    if len(prices) < 2: return 0.0
    buy_vol = sell_vol = 0.0
    for i in range(1, len(prices)):
        v = volumes[i] if i < len(volumes) else 0.0
        if prices[i] > prices[i-1]:
            buy_vol += v
        elif prices[i] < prices[i-1]:
            sell_vol += v
    tot = buy_vol + sell_vol
    return float((buy_vol - sell_vol) / tot) if tot > 0 else 0.0

def compute_ofi(prices: np.ndarray, volumes: np.ndarray) -> float:
    n = min(len(prices), WINDOW_5M)
    if n < 2: return 0.0
    ps = prices[-n:]
    vs = volumes[-(n):] if len(volumes) >= n else np.zeros(n)
    dp = np.diff(ps)
    ofi = (np.sign(dp) * vs[1:]).sum()
    return float(ofi / n)

def compute_spread_z(prices: np.ndarray) -> float:
    n = min(len(prices), WINDOW_5M)
    if n < 2: return 0.0
    ps = prices[-n:]
    hi, lo = ps.max(), ps.min()
    mid = (hi + lo) / 2
    return float((hi - lo) / mid) if mid > 0 else 0.0

def compute_trade_intensity(ts_arr: np.ndarray) -> float:
    if len(ts_arr) < 2: return 0.0
    dur = ts_arr[-1] - ts_arr[0]
    return float(len(ts_arr) / dur * 1000) if dur > 0 else 0.0

# ============================================================
# Main engine
# ============================================================

class OfflineFeatureEngine:
    def __init__(self, inst_id: str, db_paths: List[str], bar_duration_ms: int = 300_000):
        self.inst_id = inst_id
        self.bar_duration_ms = bar_duration_ms
        self.bar_label = BAR_LABEL.get(bar_duration_ms, f'{bar_duration_ms}ms')
        self.ticks = self._load_ticks(db_paths)
        print(f"Loaded {len(self.ticks)} ticks for {inst_id} (bar={self.bar_label})")

    def _load_ticks(self, db_paths):
        frames = []
        for db_path in sorted(db_paths):
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("""
                SELECT ts, last as px, last_sz as sz,
                       bid, ask, bid_sz, ask_sz,
                       inst_id
                FROM ticks ORDER BY ts
            """, conn)
            conn.close()
            if len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=['ts','px','sz','bid','ask','bid_sz','ask_sz'])
        return pd.concat(frames, ignore_index=True)

    def run(self) -> pd.DataFrame:
        price_buf: Deque[float] = deque(maxlen=WINDOW_1H)
        vol_buf: Deque[float] = deque(maxlen=WINDOW_1H)
        ts_buf: Deque[int] = deque(maxlen=WINDOW_1H)
        bars: Deque[Bar] = deque(maxlen=288)
        current_bar: Optional[Bar] = None
        results: List[dict] = []

        t0 = time.time()
        for idx, t in self.ticks.iterrows():
            ts = int(t['ts'])
            px = float(t['px']) if pd.notna(t['px']) else 0.0
            sz = float(t['sz']) if pd.notna(t['sz']) else 0.0
            bid1 = float(t['bid']) if pd.notna(t['bid']) else px - 0.5
            ask1 = float(t['ask']) if pd.notna(t['ask']) else px + 0.5
            bid1_sz = float(t['bid_sz']) if pd.notna(t['bid_sz']) else 0.0
            ask1_sz = float(t['ask_sz']) if pd.notna(t['ask_sz']) else 0.0

            price_buf.append(px)
            vol_buf.append(sz)
            ts_buf.append(ts)

            bar_ts = (ts // self.bar_duration_ms) * self.bar_duration_ms

            if current_bar and current_bar.ts_ms != bar_ts:
                bars.append(current_bar)
                features = self._compute_all(
                    np.array(price_buf, dtype=np.float64),
                    np.array(vol_buf, dtype=np.float64),
                    np.array(ts_buf, dtype=np.int64),
                    bars,
                    current_bar,
                )
                results.append({
                    'ts_ms': current_bar.ts_ms,
                    'inst_id': self.inst_id,
                    'open': current_bar.open,
                    'high': current_bar.high,
                    'low': current_bar.low,
                    'close': current_bar.close,
                    'vol': current_bar.vol,
                    'tick_count': current_bar.tick_count,
                    'bid1': current_bar.bid1,
                    'ask1': current_bar.ask1,
                    'bid1_sz': current_bar.bid1_sz,
                    'ask1_sz': current_bar.ask1_sz,
                })
                for k, v in enumerate(features):
                    results[-1][f'f{k}'] = v
                current_bar = None

            if current_bar is None:
                current_bar = Bar(
                    ts_ms=bar_ts, open=px, high=px, low=px, close=px,
                    vol=sz, tick_count=1,
                    bid1=bid1, ask1=ask1, bid1_sz=bid1_sz, ask1_sz=ask1_sz
                )
            else:
                current_bar.update(px, sz, bid1, ask1, bid1_sz, ask1_sz)

        elapsed = time.time() - t0
        bar_label = self.bar_label
        print(f"Processed {len(self.ticks)} ticks -> {len(results)} {bar_label} bars in {elapsed:.2f}s")
        return pd.DataFrame(results)

    def _compute_all(self, prices, volumes, timestamps, bars, current_bar):
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        n = len(prices)
        if n < 10: return f

        f[0:5] = [current_bar.open, current_bar.high, current_bar.low,
                   current_bar.close, current_bar.vol]
        f[5] = compute_obi(prices, volumes)
        f[6] = compute_ofi(prices, volumes)
        f[7] = compute_spread_z(prices)
        bt = current_bar.bid1 * current_bar.bid1_sz
        at = current_bar.ask1 * current_bar.ask1_sz
        tot = bt + at
        f[8] = float((bt - at) / tot) if tot > 0 else 0.0
        f[9] = momentum(prices, WINDOW_1M)
        f[10] = momentum(prices, WINDOW_5M)
        f[11] = momentum(prices, WINDOW_15M)
        f[12] = momentum(prices, WINDOW_1H)
        f[13] = historical_vol(prices, WINDOW_5M)
        f[14] = historical_vol(prices, WINDOW_15M)
        f[15] = historical_vol(prices, WINDOW_1H)
        f[16] = compute_hurst(prices)
        f[17] = vwap_deviation(prices, volumes, WINDOW_5M)
        f[18] = realized_skewness(prices, WINDOW_5M)
        f[19] = realized_kurtosis(prices, WINDOW_5M)
        f[20] = (current_bar.bid1 / current_bar.ask1) if current_bar.ask1 > 0 else 0.5
        f[21] = vol_profile(volumes, WINDOW_5M)
        for i, w in enumerate([10, 30, 60, 120, 300, 900]):
            f[22 + i] = momentum(prices, w)
        f[28] = compute_trade_intensity(timestamps)
        f[29] = vol_price_corr(prices, volumes, WINDOW_5M)
        f[30] = price_acceleration(prices)
        f[31] = log_return(prices, WINDOW_1M)
        f[32] = log_return(prices, WINDOW_5M)
        f[33] = log_return(prices, WINDOW_15M)
        f[34] = log_return(prices, WINDOW_1H)
        f[35] = 0.0
        f[36] = amihud(prices, volumes, WINDOW_5M)
        f[37] = vol_mean_reversion(prices)
        f[38] = bollinger_position(prices, WINDOW_5M)
        f[39] = 0.0
        f[40] = atr(prices, WINDOW_5M)
        return f

# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Offline FeatureEngine')
    parser.add_argument('--inst', required=True, help='e.g. BTC-USDT-SWAP')
    parser.add_argument('--db-dir', required=True, help='Dir with instrument tick DBs')
    parser.add_argument('--output', required=True, help='Output parquet path')
    parser.add_argument('--bar-ms', type=int, default=300_000,
                        help='Bar duration in ms: 300000=5m(default), 900000=15m, 3600000=1h')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.bar_ms <= 0 or args.bar_ms > 86_400_000:
        print(f"ERROR: invalid bar-ms={args.bar_ms} (must be 1ms..86400000ms)")
        sys.exit(1)

    db_dir = Path(args.db_dir) / args.inst
    if not db_dir.exists():
        print(f"ERROR: db dir not found: {db_dir}")
        sys.exit(1)

    db_paths = sorted(db_dir.glob('*.db'))
    if not db_paths:
        print(f"ERROR: no .db files in {db_dir}")
        sys.exit(1)

    bar_label = BAR_LABEL.get(args.bar_ms, f'{args.bar_ms}ms')
    print(f"Found {len(db_paths)} DB files for {args.inst} (bar={bar_label})")
    engine = OfflineFeatureEngine(args.inst, [str(p) for p in db_paths], bar_duration_ms=args.bar_ms)
    df = engine.run()

    if len(df) == 0:
        print("ERROR: 0 bars produced")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)

    feature_cols = [c for c in df.columns if c.startswith('f')]
    print(f"Output: {out_path} ({len(df)} bars, {len(feature_cols)} features)")
    print(f"Time range: {df['ts_ms'].min()} -> {df['ts_ms'].max()}")
    if args.verbose:
        print("\nFeature stats:")
        for col in feature_cols:
            vals = df[col]
            nonzero = (vals != 0).sum()
            print(f"  {col}: {nonzero}/{len(vals)} nonzero, mean={vals.mean():.6f}")

if __name__ == '__main__':
    main()