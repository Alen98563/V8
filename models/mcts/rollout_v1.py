"""
rollout_v1.1 — Phase 1 (normalized returns + dynamic σ + light mean-reversion)
Adapted: 2026-06-02 14:15 CST
"""

import json
import struct
import numpy as np

FEATURE_DIM = 50
THRESH = 0.0003   # ~3 bps threshold


def rollout_v1(state_bytes: bytes) -> bytes:
    features = struct.unpack(f"<{FEATURE_DIM}f", state_bytes)
    px = np.array(features, dtype=np.float64)
    current_px = px[-1]

    # Guard: feature buffer not yet filled (current_px == 0 or NaN)
    if current_px == 0.0 or np.isnan(current_px) or np.isinf(current_px):
        return json.dumps({
            "predicted_return": 0.0,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "market_state": "uninitialized",
        }).encode()

    # Guard: Python FeatureEngine stand-in pads feats[10..48] with zeros.
    # After px/current_px normalization, zeros create 100% momentum artifact.
    # Replace zero slots with current_px -> normalized value = 1.0 (neutral).
    zero_mask = (px == 0.0)
    if zero_mask.any():
        safe = px.copy()
        safe[zero_mask] = current_px
        px = safe

    half_idx = FEATURE_DIM // 2

    # normalise to fractional returns
    r = px / current_px
    half = r[half_idx:]

    # momentum: fractional return over half-window
    momentum = r[-1] - r[half_idx]

    # trend: per-tick slope
    t = np.arange(len(half), dtype=np.float64)
    trend = np.polyfit(t, half, 1)[0]

    # volatility & z-score
    vol = float(np.std(r) + 1e-8)
    z_score = (r[-1] - float(np.mean(r))) / vol

    # light mean-reversion: ~15% pull toward mean
    mr = -0.15 * z_score * vol

    # composite signal
    signal = 0.6 * momentum + 0.2 * trend + mr

    # dynamic σ — proportional to |signal|, floor at 2bps
    sigma = max(0.0002, 0.5 * abs(signal) + 0.0002)
    noise = np.random.normal(0.0, sigma)
    predicted_return = signal + noise

    # Guard: NaN/Inf from numerical instability
    if np.isnan(predicted_return) or np.isinf(predicted_return):
        predicted_return = 0.0

    # market state
    if z_score > 2.0:
        market_state = "overbought"
    elif z_score < -2.0:
        market_state = "oversold"
    elif predicted_return > THRESH:
        market_state = "trending_up"
    elif predicted_return < -THRESH:
        market_state = "trending_down"
    else:
        market_state = "ranging"

    # Convert fractional return → price-level delta for MCTS reward compat
    predicted_return_px = predicted_return * current_px

    confidence = 1.0 / (1.0 + abs(predicted_return) * 300.0)
    uncertainty = float(np.std(r[-5:]) + 1e-8)

    result = json.dumps(
        {
            "predicted_return": float(predicted_return_px),  # price units for MCTS
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "market_state": str(market_state),
        }
    )
    return result.encode()


# ── self-test ──
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    def gen(drift_tick, noise_tick):
        steps = drift_tick + noise_tick * rng.randn(FEATURE_DIM)
        return 70000.0 + np.cumsum(steps)

    def stats(state, label):
        state_bytes = struct.pack(f"<{FEATURE_DIM}f", *state.astype(np.float32))
        values = []
        actions = {"buy": 0, "sell": 0, "hold": 0}
        ms = []
        for _ in range(2000):
            out = json.loads(rollout_v1(state_bytes))
            v = out["predicted_return"]
            values.append(v)
            ms.append(out["market_state"])
            if v > THRESH:
                actions["buy"] += 1
            elif v < -THRESH:
                actions["sell"] += 1
            else:
                actions["hold"] += 1

        avg, std = np.mean(values), np.std(values)
        print(f"  [{label}] buy={actions['buy']/20:.0f}%  sell={actions['sell']/20:.0f}%  hold={actions['hold']/20:.0f}%")
        print(f"         mean_return={avg:.6f}  σ={std:.6f}")
        sc = dict()
        for m in ms:
            sc[m] = sc.get(m, 0) + 1
        print(f"         states={sc}")
        return actions

    print("=== self-test: 2000 rollouts each ===\n")

    # strong uptrend (SNR ~10)
    a1 = stats(gen(6.0, 6.0), "strong uptrend")
    # strong downtrend
    a2 = stats(gen(-6.0, 7.0), "strong downtrend")
    # ranging
    a3 = stats(gen(0.0, 6.0), "ranging")
    # mild uptrend (SNR ~1.5)
    a4 = stats(gen(1.5, 8.0), "mild uptrend")

    def check(cond, msg):
        if not cond:
            print(f"  FAIL: {msg}")
            return False
        return True

    # hold threshold uses fractional THRESH; output is price units → hold=0 expected
    passes = all([
        check(a1["buy"] > a1["sell"] + 200, "strong uptrend: buy >> sell"),
        check(a2["sell"] > a2["buy"] + 200, "strong downtrend: sell >> buy"),
        check(a1["sell"] > 10, "uptrend: some sells (exploration)"),
        check(a2["buy"] > 10, "downtrend: some buys (exploration)"),
        check(a3["buy"] > 10 and a3["sell"] > 10, "ranging: explore both sides"),
    ])

    if passes:
        print("\nPASS")
    else:
        print("\nFAIL")
        exit(1)
