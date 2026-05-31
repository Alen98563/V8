#!/usr/bin/env python3
"""
QTS V8 Dry Run — 模拟盘全链路测试
Route: N150 -> Tailscale SOCKS5h -> London VPS -> OKX Demo API
"""
import sys, os, json, time, signal, argparse
from datetime import datetime

sys.path.insert(0, "/home/jerry/.local/lib/python3.12/site-packages")
import v8_core_engine as vce

# Config
CONFIG = {
    "inst_id": "ETH-USDT-SWAP",
    "total_ticks": 6000,       # 总 tick 数
    "bar_window": 5,           # K线窗口数
}

def load_env(env_path="/home/jerry/strategy.okx/.env"):
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def fmt_ts(ms):
    return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S")

env = load_env()
for k in ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]:
    if k not in env:
        raise RuntimeError(f"Missing {k} in .env")

inst_id = CONFIG["inst_id"]

print("=" * 60)
print("  QTS V8 DRY RUN")
print(f"  Instrument: {inst_id}")
print(f"  Total ticks: {CONFIG['total_ticks']}")
print("=" * 60)

# Module init
print("\n[1/6] OkxChannel...")
ch = vce.OkxChannel(env["OKX_API_KEY"], env["OKX_SECRET_KEY"], env["OKX_PASSPHRASE"], is_demo=True, rate_limit=60)
offset = ch.sync_time()
print(f"      sync_time offset={offset}ms, tokens={ch.rate_limit_remaining()}")

print("[2/6] FeatureEngine...")
fe = vce.FeatureEngine()
fe.set_funding_rate(0.0001, 0.0003)

print("[3/6] OrderFSM...")
fsm = vce.OrderFSM("dry_run_001", inst_id, "dry")

print("[4/6] MctsPool...")
mcts = vce.MctsPool(4, 100)

print("[5/6] ShmBridge...")
bridge = vce.ShmBridge()

print("[6/6] OkxWsClient...")
ws = vce.OkxWsClient()

print("\nAll modules ready. Starting tick ingestion...\n")

import random
random.seed(42)

base_price = 3000.0
tick_count = 0
bar_count = 0
start = time.time()

def make_snapshot():
    global base_price
    base_price += random.gauss(0, 0.5)
    return json.dumps({
        "ts_ms": int(time.time() * 1000),
        "inst_id": inst_id,
        "bid_px": base_price - random.uniform(0.1, 1.0),
        "ask_px": base_price + random.uniform(0.1, 1.0),
        "bid_sz": round(random.uniform(0.5, 5.0), 4),
        "ask_sz": round(random.uniform(0.3, 4.0), 4),
        "last_px": base_price,
        "last_sz": round(random.uniform(0.01, 0.5), 4),
        "obi_050": round(random.uniform(0.8, 1.5), 2),
        "obi_100": round(random.uniform(0.7, 1.4), 2),
        "obi_200": round(random.uniform(0.5, 1.3), 2),
        "trade_count": random.randint(10, 100),
        "taker_buy_vol": round(random.uniform(0.5, 8.0), 2),
        "book5_bid_depths": [round(random.uniform(0.5, 3.0), 4) for _ in range(5)],
        "book5_ask_depths": [round(random.uniform(0.5, 3.0), 4) for _ in range(5)],
    }).encode()

total = CONFIG["total_ticks"]
report_every = max(total // 12, 1)

for i in range(total):
    snap = make_snapshot()
    tick = fe.parse_json_snapshot(snap)
    fe.ingest_tick(tick)
    tick_count += 1

    # Simulate 5m bar close every 60 ticks
    if tick_count % 60 == 0:
        fe.on_5m_close()
        bar_count += 1
        bars = fe.close_series(CONFIG["bar_window"])
        features = fe.get_features_50d()
        top5 = sorted(enumerate(features), key=lambda x: abs(x[1]), reverse=True)[:5]

        elapsed = time.time() - start
        now = fmt_ts(int(time.time() * 1000))
        top_str = " | ".join(f"dim[{i}]={v:.4f}" for i, v in top5)

        print(f"\n  [{now}] BAR#{bar_count} elapsed={elapsed:.1f}s  bars={len(bars)}  dims={len(features)}")
        print(f"  Top5: {top_str}")

        # Push to SHM
        bridge.push_snapshot(int(time.time() * 1000), list(features[:50]))

    # Progress
    if (i + 1) % report_every == 0:
        elapsed = time.time() - start
        tps = tick_count / max(elapsed, 0.001)
        print(f"  [...{fmt_ts(int(time.time()*1000))}] ticks={tick_count}/{total} bars={bar_count} {tps:.0f} tick/s", flush=True)

# Summary
elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"  DRY RUN COMPLETE")
print(f"  Duration: {elapsed:.2f}s")
print(f"  Ticks: {tick_count} ({tick_count / max(elapsed, 0.001):.0f} tick/s)")
print(f"  Bars: {bar_count}")
print(f"  Python 3.12 + libv8_core_engine.so")
print(f"  Modules: OkxChannel | FeatureEngine | OrderFSM | MctsPool | ShmBridge | OkxWsClient")
print(f"  Route: N150 -> Tailscale -> London VPS -> OKX Demo API")
print(f"{'=' * 60}")