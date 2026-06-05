# ═══════════════════════════════════════════════════════════════════════════════
# gating/cross_section.py — G3 横截面时序百分位门
# ═══════════════════════════════════════════════════════════════════════════════
#
# Composite percentile gate that rejects signals when cross-market conditions
# are unfavourable relative to their own trailing history.
#
# Three sub-metrics fused into cs_composite ∈ [-1, +1]:
#   z_delta_obi  — z-score of delta_obi vs trailing 256-tick window
#   z_lead_lag   — z-score of lead_lag  vs trailing window
#   regime_score — spread + vol regime alignment between the two markets
#
# Gate passes when  cs_composite >= cs_composite_floor  (default 0.15).
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger

_log = get_logger("gating.cross_section")


@dataclass
class CrossSectionGateConfig:
    """Tuning knobs exposed via v8.yaml ``gating.cross_section``."""

    enabled: bool = True
    window: int = 256                # rolling z-score window
    cs_composite_floor: float = 0.15  # [-1..+1] → 0=P50, <0=below-median
    z_delta_weight: float = 0.4
    z_lag_weight: float = 0.4
    regime_weight: float = 0.2
    spread_ratio_max: float = 3.0    # BTC spread / ETH spread above this → regime penalty
    vol_ratio_max: float = 3.0


@dataclass
class RegimeScore:
    spread_ratio: float
    vol_ratio: float
    score: float                     # 1.0 = aligned, 0.0 = divergent


class CrossSectionGate:
    """Maintains rolling z-score windows for delta_obi and lead_lag,
    returns a composite gate decision on every bar close.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg_dict = config or {}
        self.cfg = CrossSectionGateConfig(
            enabled=cfg_dict.get("enabled", True),
            window=int(cfg_dict.get("window", 256)),
            cs_composite_floor=float(cfg_dict.get("cs_composite_floor", 0.15)),
            z_delta_weight=float(cfg_dict.get("z_delta_weight", 0.4)),
            z_lag_weight=float(cfg_dict.get("z_lag_weight", 0.4)),
            regime_weight=float(cfg_dict.get("regime_weight", 0.2)),
        )

        self._delta_hist: Deque[float] = deque(maxlen=self.cfg.window)
        self._lag_hist: Deque[float] = deque(maxlen=self.cfg.window)
        self._total_ticks = 0
        self._last_composite = 1.0    # optimistic default before history builds

    # ── z-score ────────────────────────────────────────────────────────────────
    @staticmethod
    def _z_score(x: float, hist: Deque[float]) -> float:
        n = len(hist)
        if n < 16:
            return 0.0          # not enough history → neutral
        mu = sum(hist) / n
        var = sum((v - mu) ** 2 for v in hist) / n
        sigma = math.sqrt(var + 1e-10)
        return (x - mu) / sigma

    @staticmethod
    def _regime_score(
        spread_a: float, spread_b: float,
        vol_a: float, vol_b: float,
        spread_ratio_max: float, vol_ratio_max: float,
    ) -> RegimeScore:
        """1.0 = spreads/vols within 1:2 ratio; decays toward 0 as ratios diverge."""
        s_ratio = (spread_a + 1e-6) / (spread_b + 1e-6)
        s_score = max(0.0, 1.0 - abs(math.log2(max(s_ratio, 1.0 / s_ratio))) / math.log2(spread_ratio_max))

        v_ratio = (vol_a + 1e-6) / (vol_b + 1e-6)
        v_score = max(0.0, 1.0 - abs(math.log2(max(v_ratio, 1.0 / v_ratio))) / math.log2(vol_ratio_max))

        return RegimeScore(
            spread_ratio=s_ratio,
            vol_ratio=v_ratio,
            score=0.5 * s_score + 0.5 * v_score,
        )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    def evaluate(
        self,
        delta_obi: float,
        lead_lag: float,
        spread_self: float = 0.0,
        spread_other: float = 0.0,
        vol_self: float = 0.0,
        vol_other: float = 0.0,
    ) -> Tuple[bool, str, float]:
        """Returns (passed, reason, cs_composite).

        ``cs_composite`` ∈ [-1, +1] fed into GateContext.cs_composite.
        """
        if not self.cfg.enabled:
            return True, "G3_disabled", 1.0

        self._total_ticks += 1

        # Rolling windows
        self._delta_hist.append(delta_obi)
        self._lag_hist.append(lead_lag)

        # z-scores (clamped to ±4)
        z_d = self._z_score(delta_obi, self._delta_hist)
        z_l = self._z_score(lead_lag, self._lag_hist)
        z_d = max(-4.0, min(4.0, z_d))
        z_l = max(-4.0, min(4.0, z_l))

        # z → [0, 1] via erf approximation:  0.5 + 0.5*tanh(z)
        # +z means cross-market condition is BETTER than its own history
        p_d = 0.5 + 0.5 * math.tanh(z_d)
        p_l = 0.5 + 0.5 * math.tanh(z_l)

        # regime alignment
        regime = self._regime_score(
            spread_self, spread_other, vol_self, vol_other,
            self.cfg.spread_ratio_max, self.cfg.vol_ratio_max,
        )

        # composite: weighted average mapped to [-1, 1] from [0, 1]
        raw = (
            self.cfg.z_delta_weight * p_d +
            self.cfg.z_lag_weight * p_l +
            self.cfg.regime_weight * regime.score
        )
        cs_composite = 2.0 * raw - 1.0     # [0, 1] → [-1, +1]
        self._last_composite = cs_composite

        passed = cs_composite >= self.cfg.cs_composite_floor

        reason = (
            "G3_pass"
            if passed
            else (
                f"G3_composite={cs_composite:.3f}<{self.cfg.cs_composite_floor:.3f}"
                f" z_d={z_d:.2f} z_l={z_l:.2f} regime={regime.score:.2f}"
            )
        )

        return passed, reason, cs_composite

    # ── Stats ──────────────────────────────────────────────────────────────────
    @property
    def last_composite(self) -> float:
        return self._last_composite

    @property
    def total_ticks(self) -> int:
        return self._total_ticks


__all__ = ["CrossSectionGate", "CrossSectionGateConfig", "RegimeScore"]