from __future__ import annotations
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from features.feature_engine_of.tick_snapshot import MarketStateBuffer

if TYPE_CHECKING:
    from features.feature_engine_of.tick_snapshot import BufferRegistry

VELOCITY_WINDOWS_S = [5, 15, 30, 60, 120, 300]
ROLLING_SPREAD_WINDOW_S = 300
REGIME_WINDOW_S = 1_800
MIN_SAMPLES = 1

class FeatureEngine:
    def compute(self, buffer, universe=None):
        if len(buffer) < MIN_SAMPLES: return {}
        n = len(buffer)
        features = {}
        if n >= 2: features.update(self._velocity_features(buffer))
        elif universe is not None: features.update(self._cs_velocity(buffer, universe))
        if n >= 10: features.update(self._regime_features(buffer))
        elif universe is not None: features.update(self._cs_regime(buffer, universe))
        else:
            features["realized_vol_5m"] = buffer.realized_vol(300)
            features["realized_vol_30m_direct"] = buffer.realized_vol(REGIME_WINDOW_S)
        features.update(self._microstructure_features(buffer))
        if n >= 2: features.update(self._flow_features(buffer))
        elif universe is not None: features.update(self._cs_flow(buffer, universe))
        else: features.update(self._cvd_features(buffer))
        for _name in all_feature_names(): features.setdefault(_name, 0.0)
        return features

    def _cs_velocity(self, buffer, universe):
        f = {}; latest = buffer.get_latest()
        if not latest: return f
        others_mid, others_obi, others_depth, others_spread = [], [], [], []
        for buf in universe.all_buffers().values():
            if buf.market_id == buffer.market_id: continue
            snap = buf.get_latest()
            if snap and snap.mid_price > 0:
                others_mid.append(snap.mid_price); others_obi.append(snap.obi)
                others_depth.append(snap.bid_depth + snap.ask_depth); others_spread.append(snap.spread)
        if not others_mid: return f
        arr_mid = np.array(others_mid); arr_obi = np.array(others_obi)
        mean_mid = float(arr_mid.mean()); std_mid = float(arr_mid.std()) + 1e-10
        mean_obi = float(arr_obi.mean()); std_obi = float(arr_obi.std()) + 1e-10
        cs_price_z = (latest.mid_price - mean_mid) / std_mid
        cs_obi_z = (latest.obi - mean_obi) / std_obi
        for w in VELOCITY_WINDOWS_S:
            f[f"price_vel_{w}s"] = float(cs_price_z * 0.01)
            f[f"price_acc_{w}s"] = 0.0
            f[f"price_mom_{w}s"] = float(np.sign(cs_price_z))
            f[f"obi_vel_{w}s"] = float(cs_obi_z * 0.01)
            f[f"obi_mean_{w}s"] = mean_obi; f[f"obi_std_{w}s"] = std_obi
            f[f"obi_latest_{w}s"] = float(latest.obi)
        return f

    def _cs_regime(self, buffer, universe):
        f = {}; latest = buffer.get_latest()
        if not latest: return f
        others = [snap.mid_price for buf in universe.all_buffers().values()
                   if buf.market_id != buffer.market_id and (snap := buf.get_latest()) and snap.mid_price > 0]
        if not others: return f
        arr = np.array(others)
        cs_vol = float(arr.std()); cs_pct = float((arr < latest.mid_price).mean())
        f.update({"realized_vol_30m": cs_vol, "hurst_rs": 0.5, "autocorr_lag1": 0.0})
        f.update({"price_pct_in_range_30m": cs_pct, "vol_ratio_5m_30m": 1.0})
        f.update({"realized_vol_5m": cs_vol, "realized_vol_30m_direct": cs_vol})
        return f

    def _cs_flow(self, buffer, universe):
        f = self._cvd_features(buffer); latest = buffer.get_latest()
        if not latest: return f
        others_depth = [snap.bid_depth + snap.ask_depth for buf in universe.all_buffers().values()
                         if buf.market_id != buffer.market_id and (snap := buf.get_latest())]
        if not others_depth: return f
        arr = np.array(others_depth)
        depth_z = (float(latest.bid_depth + latest.ask_depth) - float(arr.mean())) / (float(arr.std()) + 1e-10)
        for w in [30, 120, 300]:
            f[f"buy_ratio_{w}s"] = 0.5
            f[f"net_flow_{w}s"] = float(np.clip(depth_z * 0.02, -1, 1))
            f[f"log_volume_{w}s"] = float(np.log1p(latest.bid_depth + latest.ask_depth))
            f[f"trade_rate_{w}s"] = 0.0
        return f

    def _cvd_features(self, buffer):
        return {f"cvd_{w}s": buffer.cvd(w) for w in [30, 120, 300]}

    def _velocity_features(self, buffer):
        f = {}
        for w in VELOCITY_WINDOWS_S:
            prices = buffer.price_series(w); obis = buffer.obi_series(w)
            if len(prices) >= 2 and prices[0] > 0:
                f[f"price_vel_{w}s"] = (prices[-1] - prices[0]) / prices[0]
                f[f"price_acc_{w}s"] = float(np.diff(prices).mean())
                diffs = np.diff(prices); f[f"price_mom_{w}s"] = float(np.sign(diffs).mean())
            if len(obis) >= 2:
                f[f"obi_vel_{w}s"] = float(obis[-1] - obis[0])
                f[f"obi_mean_{w}s"] = float(obis.mean()); f[f"obi_std_{w}s"] = float(obis.std())
                f[f"obi_latest_{w}s"] = float(obis[-1])
        return f

    def _regime_features(self, buffer):
        f = {}
        prices_30m = buffer.price_series(REGIME_WINDOW_S)
        if len(prices_30m) >= 10:
            log_rets = np.diff(np.log(np.maximum(prices_30m, 1e-10)))
            f["realized_vol_30m"] = float(log_rets.std())
            f["hurst_rs"] = self._hurst_rs(prices_30m)
            if len(log_rets) >= 4:
                f["autocorr_lag1"] = float(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1])
            p_min, p_max = prices_30m.min(), prices_30m.max(); rng = p_max - p_min
            if rng > 0: f["price_pct_in_range_30m"] = float((prices_30m[-1] - p_min) / rng)
        vol_5m = buffer.realized_vol(300); vol_30m = buffer.realized_vol(1800)
        if vol_30m > 0: f["vol_ratio_5m_30m"] = vol_5m / vol_30m
        f["realized_vol_5m"] = vol_5m; f["realized_vol_30m_direct"] = buffer.realized_vol(REGIME_WINDOW_S)
        return f

    def _microstructure_features(self, buffer):
        f = {}; latest = buffer.get_latest()
        if not latest: return f
        total_depth = latest.bid_depth + latest.ask_depth
        f["depth_imbalance"] = ((latest.bid_depth - latest.ask_depth) / total_depth) if total_depth > 0 else 0.0
        f["total_depth"] = float(total_depth); f["log_depth"] = float(np.log1p(total_depth))
        spreads = buffer.spread_series(ROLLING_SPREAD_WINDOW_S)
        if len(spreads) >= 2:
            mu, sigma = spreads.mean(), spreads.std()
            f["spread_current"] = float(latest.spread)
            f["spread_z_5m"] = float((latest.spread - mu) / (sigma + 1e-10))
            f["spread_pct_rank_5m"] = float((latest.spread > spreads).mean())
            f["spread_ma_5m"] = float(mu)
        f["obi_current"] = float(latest.obi)
        f["funding_rate"] = float(latest.funding_rate)
        if hasattr(buffer, 'oi_series'):
            oi_vals = buffer.oi_series(300)
            if len(oi_vals) >= 2 and oi_vals[0] > 0:
                f["oi_delta_5m"] = float((oi_vals[-1] - oi_vals[0]) / oi_vals[0])
        return f

    def _flow_features(self, buffer):
        f = {}
        for w in [30, 120, 300]:
            window = buffer.get_window(w)
            if len(window) < 2: continue
            buy_vol = sum(s.buy_volume for s in window)
            sell_vol = sum(s.sell_volume for s in window)
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                f[f"buy_ratio_{w}s"] = buy_vol / total_vol
                f[f"net_flow_{w}s"] = (buy_vol - sell_vol) / total_vol
                f[f"log_volume_{w}s"] = float(np.log1p(total_vol))
            trade_counts = [s.trade_count for s in window]
            f[f"trade_rate_{w}s"] = float(np.mean(trade_counts))
        f.update(self._cvd_features(buffer))
        return f

    @staticmethod
    def _hurst_rs(prices):
        if len(prices) < 20: return 0.5
        log_prices = np.log(np.maximum(prices, 1e-10))
        deviations = log_prices - log_prices.mean()
        cum_dev = np.cumsum(deviations)
        R = cum_dev.max() - cum_dev.min(); S = log_prices.std()
        return float(np.clip(np.log(R / S) / np.log(len(prices)), 0.0, 1.0)) if S >= 1e-10 else 0.5

def all_feature_names():
    names = []
    for w in VELOCITY_WINDOWS_S:
        for suffix in ["price_vel","price_acc","price_mom","obi_vel","obi_mean","obi_std","obi_latest"]:
            names.append(f"{suffix}_{w}s")
    names += ["realized_vol_30m","hurst_rs","autocorr_lag1","price_pct_in_range_30m","vol_ratio_5m_30m",
              "realized_vol_5m","realized_vol_30m_direct","depth_imbalance","total_depth","log_depth",
              "spread_current","spread_z_5m","spread_pct_rank_5m","spread_ma_5m","obi_current",
              "funding_rate","parity_deviation","time_to_resolution","log_ttr","oi_delta_5m",
              "buy_ratio_30s","net_flow_30s","log_volume_30s","trade_rate_30s","cvd_30s",
              "buy_ratio_120s","net_flow_120s","log_volume_120s","trade_rate_120s","cvd_120s",
              "buy_ratio_300s","net_flow_300s","log_volume_300s","trade_rate_300s","cvd_300s"]
    return names
