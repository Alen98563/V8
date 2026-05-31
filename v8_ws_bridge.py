#!/usr/bin/env python3
"""
QTS V8 Multi-Market Live WebSocket Bridge
OKX WS (via SOCKS5) -> Per-Symbol FeatureEngine -> on_5m_close -> log features

用法: python3.12 v8_ws_bridge.py
环境变量:
  V8_INST_IDS     逗号分隔品种列表 (默认 ETH-USDT-SWAP)
  V8_BAR_SECS     K线周期秒数 (默认 300)
  V8_MAX_BARS     停止前最大bar数 (默认 0=无限)
  V8_OBI_DEPTH_BPS  OBI深度bps (默认 50)
  V8_FEATURE_DECAY  特征EMA衰减 (默认 0.97)
  V8_LOG_INTERVAL   状态日志间隔秒 (默认 60)

路由: N150 -> Tailscale SOCKS5h -> London VPS -> OKX WS
"""
import sys, os, json, time, threading, signal
from datetime import datetime, timezone, timedelta
from collections import deque

TZ = timezone(timedelta(hours=8))

sys.path.insert(0, "/home/jerry/.local/lib/python3.12/site-packages")
import v8_core_engine as vce

# ─── Config ───
_raw_ids = os.environ.get("V8_INST_IDS", "ETH-USDT-SWAP")
INST_IDS = [x.strip() for x in _raw_ids.split(",") if x.strip()]
BAR_SECS = int(os.environ.get("V8_BAR_SECS", 300))
LOG_INTERVAL = int(os.environ.get("V8_LOG_INTERVAL", 60))
MAX_BARS = int(os.environ.get("V8_MAX_BARS", 0))

TUNE = {
    "inst_ids": INST_IDS,
    "obi_depth_bps": int(os.environ.get("V8_OBI_DEPTH_BPS", 50)),
    "feature_decay": float(os.environ.get("V8_FEATURE_DECAY", 0.97)),
    "min_ticks_per_bar": int(os.environ.get("V8_MIN_TICKS", 1)),
    "bar_secs": BAR_SECS,
    "max_bars": MAX_BARS,
}

# ─── Security ───
def load_env(path="/home/jerry/strategy.okx/.env"):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

env = load_env()

# ─── Per-Symbol State ───
# Each symbol gets its own FeatureEngine / counter / bar tracker / book5 history
symbol_state = {}
for inst_id in INST_IDS:
    fe = vce.FeatureEngine()
    fe.set_funding_rate(0.0001, 0.0003)
    symbol_state[inst_id] = {
        "fe": fe,
        "tick_count": 0,
        "bar_count": 0,
        "bar_start_ts": None,
        "bar_ticks": 0,
        "book5_prices": deque(maxlen=60),
    }

# ─── Shared modules ───
print("[init] OkxChannel (HMAC-SHA256)...")
ch = vce.OkxChannel(
    env["OKX_API_KEY"], env["OKX_SECRET_KEY"], env["OKX_PASSPHRASE"],
    is_demo=True, rate_limit=60,
)
offset = ch.sync_time()
print(f"       time_offset={offset}ms")
print(f"[init] FeatureEngines: {len(INST_IDS)} instances ({', '.join(INST_IDS)})")
print("[init] OrderFSM...")
fsm = vce.OrderFSM("ws_live_001", INST_IDS[0], "ws_bridge")
print("[init] MctsPool...")
mcts = vce.MctsPool(4, 100)
print("[init] ShmBridge...")
bridge = vce.ShmBridge()

# ─── Global tick counter + lock ───
last_log = time.time()
start_time = time.time()
lock = threading.Lock()

# ─── Shutdown handler ───
running = True
def _shutdown(signum, frame):
    global running
    print(f"\n[signal] Received {signal.Signals(signum).name}, shutting down...")
    running = False
signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ─── Tick builder ───
def _build_tick(inst_id, channel, snapshot):
    """Parse WS message into FeatureEngine-readable JSON tick."""
    if channel == "candle5m" or channel == "candle1m":
        ts_ms = int(snapshot[0])
        last_px = float(snapshot[4])
        last_sz = float(snapshot[5])
        ask_px = float(snapshot[2])
        bid_px = float(snapshot[3])
        trade_count = 0
    elif channel == "trades":
        ts_ms = int(snapshot["ts"])
        last_px = float(snapshot["px"])
        last_sz = float(snapshot["sz"])
        side = snapshot.get("side", "buy")
        ask_px = last_px + 0.1 if side == "buy" else last_px
        bid_px = last_px - 0.1 if side == "sell" else last_px
        trade_count = 1
    elif channel == "books5":
        ts_ms = int(snapshot["ts"])
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return None
        bid_px = float(bids[0][0])
        ask_px = float(asks[0][0])
        bid_sz_val = float(bids[0][1]) if len(bids[0]) > 1 else 1.5
        ask_sz_val = float(asks[0][1]) if len(asks[0]) > 1 else 0.8
        last_px = (bid_px + ask_px) / 2.0
        last_sz = 0.01
        trade_count = 0
    else:
        return None

    # OBI estimation
    spread = ask_px - bid_px
    obi_050 = 1.0 + (spread / bid_px) * 100 if bid_px > 0 else 1.0
    obi_100 = obi_050 * 0.95
    obi_200 = obi_050 * 0.85

    # Book5 depth proxy
    depths = [abs(last_px) * 0.001 * (i + 1) for i in range(5)]

    # Determine sz values for non-books5 channels
    if channel != "books5":
        bid_sz_val = 1.5
        ask_sz_val = 0.8

    return json.dumps({
        "ts_ms": ts_ms,
        "inst_id": inst_id,
        "bid_px": round(bid_px, 6),
        "ask_px": round(ask_px, 6),
        "bid_sz": round(bid_sz_val, 4),
        "ask_sz": round(ask_sz_val, 4),
        "last_px": round(last_px, 6),
        "last_sz": round(last_sz, 4),
        "obi_050": round(obi_050, 4),
        "obi_100": round(obi_100, 4),
        "obi_200": round(obi_200, 4),
        "trade_count": trade_count,
        "taker_buy_vol": last_sz if channel == "trades" else 0.0,
        "book5_bid_depths": depths,
        "book5_ask_depths": depths,
    })

# ─── WS Message Handler ───
def on_message(ws, message):
    global last_log

    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        return

    if "event" in msg:
        event = msg.get("event", "")
        if event == "error":
            print(f"\n[ws error] code={msg.get('code')} msg={msg.get('msg')}")
        elif event != "subscribe":
            print(f"\n[ws event] {event}")
        return

    data = msg.get("data", [])
    arg = msg.get("arg", {})
    channel = arg.get("channel", "")
    inst_id = arg.get("instId", "")
    if not data or inst_id not in symbol_state:
        return

    snapshot = data[0] if isinstance(data, list) else data
    st = symbol_state[inst_id]

    with lock:
        now_ms = int(time.time() * 1000)

        # Bar boundary check
        bar_sec = now_ms // (BAR_SECS * 1000) * BAR_SECS * 1000
        if st["bar_start_ts"] is None or bar_sec > st["bar_start_ts"]:
            triggered_close = (st["bar_start_ts"] is not None and st["bar_ticks"] > 0)
            if triggered_close:
                st["fe"].on_5m_close()
                st["bar_count"] += 1
                bars = st["fe"].close_series(10)
                features = st["fe"].get_features_50d()
                top5 = sorted(enumerate(features), key=lambda x: abs(x[1]), reverse=True)[:5]
                elapsed = time.time() - start_time

                now_str = datetime.now(TZ).strftime("%H:%M:%S")
                top_str = " | ".join(f"dim[{i}]={v:.3f}" for i, v in top5)
                print(f"\n{'━'*55}")
                print(f"  [{now_str}] {inst_id} BAR#{st['bar_count']} ({st['bar_ticks']} ticks)")
                print(f"  dims={len(features)} bars={len(bars)} elapsed={elapsed:.0f}s")
                print(f"  Top5: {top_str}")
                print(f"{'━'*55}")

                global_symbol_check()

            st["bar_start_ts"] = bar_sec
            st["bar_ticks"] = 0

        # Build and ingest tick
        tick_json = _build_tick(inst_id, channel, snapshot)
        if tick_json is None:
            return

        st["bar_ticks"] += 1
        st["tick_count"] += 1

        # Track book5 mid-prices
        if channel == "books5":
            try:
                bids = snapshot.get("bids", [])
                asks = snapshot.get("asks", [])
                if bids and asks:
                    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                    st["book5_prices"].append(mid)
            except Exception:
                pass

        try:
            tick = st["fe"].parse_json_snapshot(tick_json.encode())
            st["fe"].ingest_tick(tick)
        except Exception as e:
            pass

        # Status log
        now_t = time.time()
        if now_t - last_log >= LOG_INTERVAL:
            last_log = now_t
            total_ticks = sum(s["tick_count"] for s in symbol_state.values())
            total_bars = sum(s["bar_count"] for s in symbol_state.values())
            elapsed = now_t - start_time
            tps = total_ticks / max(elapsed, 0.001)
            parts = [f"{iid}:{symbol_state[iid]['tick_count']}t/{symbol_state[iid]['bar_count']}b" for iid in INST_IDS]
            print(f"  [{datetime.now(TZ).strftime('%H:%M:%S')}] "
                  f"{' | '.join(parts)} total={total_ticks}t/{total_bars}b {tps:.0f}t/s", flush=True)

def global_symbol_check():
    """Check if all symbols have reached MAX_BARS."""
    if MAX_BARS > 0:
        all_done = all(s["bar_count"] >= MAX_BARS for s in symbol_state.values())
        if all_done:
            print(f"\n[stop] All symbols reached MAX_BARS={MAX_BARS}. Closing WS.")
            import threading as _t
            _t.Timer(1.0, lambda: ws and ws.close()).start()

def on_error(ws, error):
    print(f"\n[ws error] {error}")

def on_close(ws, close_status_code, close_msg):
    elapsed = time.time() - start_time
    print(f"\n[ws closed] code={close_status_code} msg={close_msg}")
    for inst_id in INST_IDS:
        s = symbol_state[inst_id]
        print(f"  {inst_id}: {s['tick_count']} ticks, {s['bar_count']} bars")
    print(f"  Total: {elapsed:.1f}s")

# ─── WS Connection ───
print(f"\n[ws] Connecting OKX Demo WebSocket via SOCKS5 proxy...")
print(f"  Markets: {', '.join(INST_IDS)} / trades + books5")

from websocket import WebSocketApp
import socks
import socket as _socket

socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 1080)
_orig_gai = _socket.getaddrinfo
def _ipv4_gai(host, port, family=0, *a, **kw):
    return _orig_gai(host, port, _socket.AF_INET, *a, **kw)
_socket.getaddrinfo = _ipv4_gai
_socket.socket = socks.socksocket

# Build subscribe args for all symbols
_sub_args = []
for inst_id in INST_IDS:
    _sub_args.append({"channel": "trades", "instId": inst_id})
    _sub_args.append({"channel": "books5", "instId": inst_id})

ws_url = "wss://wspap.okx.com:8443/ws/v5/public?demotrading=true"
ws = WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
def _on_open(ws):
    print(f"  [ws] Connected! Subscribing {len(_sub_args)} channels...")
    ws.send(json.dumps({"op": "subscribe", "args": _sub_args}))
    sub_list = ', '.join(f"{a['channel']}/{a['instId']}" for a in _sub_args)
    print(f"  [ws] Subscribed: {sub_list}")
ws.on_open = _on_open

try:
    print(f"\n{'═'*55}")
    print(f"  QTS V8 MULTI-MARKET WebSocket Bridge")
    print(f"  {len(INST_IDS)} markets: {', '.join(INST_IDS)}")
    print(f"  {BAR_SECS}s bars | max {MAX_BARS or '∞'} bars")
    print(f"  Modules: OkxChannel | FeatureEngine({len(INST_IDS)}) | OrderFSM | MctsPool | ShmBridge")
    print(f"  Route: N150 → SOCKS5 → London VPS → OKX WS")
    print(f"  Tunable: {json.dumps(TUNE)}")
    print(f"  {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')} CST")
    print(f"{'═'*55}\n")
    ws.run_forever()
except KeyboardInterrupt:
    print("\n[stop] Keyboard interrupt")
finally:
    _socket.getaddrinfo = _orig_gai
    elapsed = time.time() - start_time
    total_ticks = sum(s["tick_count"] for s in symbol_state.values())
    total_bars = sum(s["bar_count"] for s in symbol_state.values())
    print(f"\n{'═'*55}")
    print(f"  MULTI-MARKET BRIDGE SUMMARY")
    print(f"  Duration: {elapsed:.1f}s")
    for inst_id in INST_IDS:
        s = symbol_state[inst_id]
        print(f"  {inst_id}: {s['tick_count']} ticks, {s['bar_count']} bars")
    print(f"  Total: {total_ticks} ticks, {total_bars} bars ({total_ticks/max(elapsed,0.001):.0f} t/s)")
    print(f"{'═'*55}")