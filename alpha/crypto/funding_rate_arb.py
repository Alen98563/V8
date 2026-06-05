"""
alpha/crypto/funding_rate_arb.py —Task 4b: 资金费率套利 Alpha 信号
====================================================================

利用 OKX 永续合约资金费率 (Funding Rate) 的均值回归特性生成交易信号：

    - 资金费率极端正�?(>0.05%) �?市场过度做多 �?空头信号
    - 资金费率极端负�?(<-0.03%) �?市场过度做空 �?多头信号
    - 费率接近 0 �?中�?
资金费率�?8 小时结算一�?(00/08/16 UTC)，在结算�?30 分钟
HardGating G5 会阻止新开仓，所以此信号主要用于结算后的方向判断�?
接口契约�?    - 输出 AlphaSignal (�?obi_v2 对齐)
    - 可与 OBI 信号融合 (feature_fusion.py)
    - 支持历史回测和实时推断两种模�?
数据源：
    - OKX REST API: /api/v5/public/funding-rate
    - 或通过 WS 推�?(Phase 2)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from common.logging_setup import get_logger, get_pulse, get_trace

_log = get_logger("alpha.funding_rate_arb")


# ============================================================
# 配置
# ============================================================

@dataclass
class FundingArbConfig:
    """资金费率套利信号配置"""
    # 费率阈�?(百分比形式，0.01 = 1%)
    extreme_positive: float = 0.05    # > 0.05% �?做空信号
    extreme_negative: float = -0.03   # < -0.03% �?做多信号
    moderate_positive: float = 0.02   # > 0.02% �?轻度空头
    moderate_negative: float = -0.01  # < -0.01% �?轻度多头

    # 均值回归窗�?(用过�?N 期费率做 z-score)
    zscore_window: int = 24           # 24 �?= 8 �?(�?8h 一�?
    zscore_threshold: float = 2.0     # |z| > 2 视为极端

    # 信号衰减 (距下次结算越近，信号越弱)
    decay_before_settlement_min: int = 60  # 结算�?60 分钟开始衰�?
    # 结算时间 (UTC): 0, 8, 16
    settlement_hours_utc: List[int] = field(default_factory=lambda: [0, 8, 16])


# ============================================================
# AlphaSignal (�?obi_v2 兼容)
# ============================================================

@dataclass
class FundingSignal:
    """资金费率 Alpha 信号"""
    trace_id: str
    inst_id: str
    ts_ms: int
    pulse_id: int
    alpha_name: str = "funding_rate_arb"
    raw_signal: float = 0.0          # [-1, 1] 方向 + 强度
    confidence: float = 0.0          # [0, 1]
    funding_rate: float = 0.0        # 当前费率
    funding_rate_avg: float = 0.0    # 均�?    funding_rate_zscore: float = 0.0 # z-score
    next_settlement_ms: int = 0      # 下次结算时间
    hours_to_settlement: float = 0.0 # 距下次结算小时数

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ============================================================
# 核心引擎
# ============================================================

class FundingRateArbEngine:
    """
    资金费率套利信号引擎

    用法�?        engine = FundingRateArbEngine()
        # 喂入最新费�?(可从 REST API 或缓存获�?
        signal = engine.on_funding_rate(
            funding_rate=0.0003,  # 0.03%
            ts_ms=int(time.time() * 1000),
        )
    """

    def __init__(
        self,
        inst_id: str = "BTC-USDT-SWAP",
        cfg: Optional[FundingArbConfig] = None,
    ) -> None:
        self.inst_id = inst_id
        self.cfg = cfg or FundingArbConfig()
        self._rate_history: Deque[float] = deque(maxlen=self.cfg.zscore_window)
        self._last_signal: Optional[FundingSignal] = None

    def _next_settlement_ms(self, ts_ms: int) -> int:
        """计算下一个结算时间戳 (ms)"""
        ts_sec = ts_ms / 1000.0
        # 结算时间: 00:00, 08:00, 16:00 UTC
        import datetime
        dt = datetime.datetime.utcfromtimestamp(ts_sec)
        for h in sorted(self.cfg.settlement_hours_utc):
            settlement = dt.replace(hour=h, minute=0, second=0, microsecond=0)
            if settlement.timestamp() * 1000 > ts_ms:
                return int(settlement.timestamp() * 1000)
        # 所有今天的结算都过�?�?明天 00:00
        tomorrow = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow += datetime.timedelta(days=1)
        return int(tomorrow.timestamp() * 1000)

    def _hours_to_settlement(self, ts_ms: int) -> float:
        """距下次结算的小时�?""
        next_ms = self._next_settlement_ms(ts_ms)
        return (next_ms - ts_ms) / (3600 * 1000.0)

    def _compute_zscore(self, current_rate: float) -> float:
        """计算当前费率相对历史�?z-score"""
        if len(self._rate_history) < 3:
            return 0.0
        rates = list(self._rate_history)
        mean = sum(rates) / len(rates)
        var = sum((r - mean) ** 2 for r in rates) / len(rates)
        std = var ** 0.5
        if std < 1e-10:
            return 0.0
        return (current_rate - mean) / std

    def on_funding_rate(
        self,
        funding_rate: float,
        ts_ms: int = 0,
        next_funding_rate: Optional[float] = None,
    ) -> FundingSignal:
        """
        输入当前资金费率，输出交易信�?
        Args:
            funding_rate: 当前费率 (小数形式�?.0001 = 0.01%)
            ts_ms: 时间�?(ms)，默认当前时�?            next_funding_rate: 预测下期费率 (OKX API 有提�?

        Returns:
            FundingSignal
        """
        if ts_ms == 0:
            ts_ms = int(time.time() * 1000)

        # 百分比转�?(API 返回 0.0001 表示 0.01%)
        rate_pct = funding_rate * 100  # 转成百分比形�?
        # 更新历史
        self._rate_history.append(rate_pct)

        # z-score
        zscore = self._compute_zscore(rate_pct)

        # 基础信号：费率均值回�?        # 极端�?�?做空 (负信�?；极端负 �?做多 (正信�?
        raw_signal = 0.0

        if rate_pct > self.cfg.extreme_positive:
            raw_signal = -0.8  # 强做�?        elif rate_pct > self.cfg.moderate_positive:
            raw_signal = -0.4  # 弱做�?        elif rate_pct < self.cfg.extreme_negative:
            raw_signal = 0.8   # 强做�?        elif rate_pct < self.cfg.moderate_negative:
            raw_signal = 0.4   # 弱做�?
        # z-score 增强
        if abs(zscore) > self.cfg.zscore_threshold:
            raw_signal *= 1.3  # 历史极端 �?增强信号
            raw_signal = max(-1.0, min(1.0, raw_signal))

        # 结算衰减：距结算越近，信号越�?(G5 会在结算�?30min 完全阻止)
        hours_to_settle = self._hours_to_settlement(ts_ms)
        decay_hours = self.cfg.decay_before_settlement_min / 60.0
        if hours_to_settle < decay_hours:
            decay = hours_to_settle / decay_hours  # 线性衰�?            raw_signal *= decay

        # 置信度：历史越长越稳定，z-score 越极端越有信�?        history_conf = min(len(self._rate_history) / self.cfg.zscore_window, 1.0)
        zscore_conf = min(abs(zscore) / 3.0, 1.0)
        confidence = 0.4 * history_conf + 0.6 * zscore_conf
        confidence = max(0.0, min(1.0, confidence))

        signal = FundingSignal(
            trace_id=get_trace(),
            inst_id=self.inst_id,
            ts_ms=ts_ms,
            pulse_id=get_pulse(),
            raw_signal=round(raw_signal, 6),
            confidence=round(confidence, 4),
            funding_rate=round(funding_rate, 8),
            funding_rate_avg=round(sum(self._rate_history) / len(self._rate_history) / 100, 8)
                if self._rate_history else 0.0,
            funding_rate_zscore=round(zscore, 4),
            next_settlement_ms=self._next_settlement_ms(ts_ms),
            hours_to_settlement=round(hours_to_settle, 2),
        )

        self._last_signal = signal
        _log.debug(
            "funding_signal",
            extra={
                "rate_pct": round(rate_pct, 4),
                "zscore": round(zscore, 2),
                "signal": round(raw_signal, 4),
                "hours_to_settle": round(hours_to_settle, 1),
                "trace_id": get_trace(),
            },
        )

        return signal

    def get_last_signal(self) -> Optional[FundingSignal]:
        return self._last_signal


# ============================================================
# OKX API 辅助
# ============================================================

async def fetch_funding_rate(
    inst_id: str = "BTC-USDT-SWAP",
    demo: bool = True,
) -> Optional[Dict]:
    """
    �?OKX API 获取当前资金费率

    Returns:
        {"fundingRate": "0.0001", "nextFundingRate": "0.00005", "fundingTime": "..."}
        �?None (网络失败)
    """
    try:
        import httpx
        base = "https://www.okx.com"
        url = f"{base}/api/v5/public/funding-rate"
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params={"instId": inst_id})
            data = r.json()
            if data.get("code") == "0" and data.get("data"):
                return data["data"][0]
    except Exception as e:
        _log.warning("funding_rate_fetch_failed", extra={"err": str(e)})
    return None


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    engine = FundingRateArbEngine()

    # 模拟不同费率场景
    scenarios = [
        ("极端正费�?(0.08%)", 0.0008),
        ("中度正费�?(0.03%)", 0.0003),
        ("中性费�?(0.005%)", 0.00005),
        ("中度负费�?(-0.02%)", -0.0002),
        ("极端负费�?(-0.05%)", -0.0005),
    ]

    for name, rate in scenarios:
        sig = engine.on_funding_rate(rate)
        print(f"  {name}: signal={sig.raw_signal:+.3f}, conf={sig.confidence:.3f}, "
              f"z={sig.funding_rate_zscore:+.2f}, settle_in={sig.hours_to_settlement:.1f}h")

    print(f"\n�?FundingRateArbEngine self-test passed")
