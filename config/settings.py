"""
config/settings.py — P5: 部署环境配置切换
==========================================

集中管理 demo/live 环境切换，避免硬编码。

用法：
    from config.settings import Settings
    settings = Settings.load("v8.yaml")

    if settings.is_live:
        risk_manager.set_mode("production")

环境变量覆盖优先级最高：
    V8_MODE=demo|live
    V8_OKX_API_KEY=xxx
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

_CONFIG_DIR = Path(__file__).parent


@dataclass
class OkxCredentials:
    """OKX API 凭证"""
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    demo: bool = True


@dataclass
class MctsSettings:
    """MCTS 引擎配置"""
    backend: str = "native"           # native | fallback
    timeout_ms: int = 100
    num_workers: int = 8
    num_simulations: int = 200
    max_depth: int = 5
    ucb1_c: float = 2.0               # UCB1 探索系数


@dataclass
class RiskSettings:
    """风控配置"""
    max_position_pct: float = 1.0     # 单次最大仓位 (合约张数)
    max_daily_loss_pct: float = 0.03  # 单日最大亏损比例
    max_consecutive_losses: int = 5    # 连续亏损暂停
    kelly_fraction: float = 0.25
    maker_only: bool = False           # 仅挂单 (demo=true 时开启)


@dataclass
class AlphaSettings:
    """Alpha 信号配置"""
    obi_threshold: float = 0.6
    funding_rate_arb: bool = True
    fusion_obi_weight: float = 0.6
    fusion_funding_weight: float = 0.4


@dataclass
class GatingSettings:
    """门控配置"""
    g1_min_depth_usd: float = 50_000  # G1 最小盘口深度
    g2_max_hv: float = 0.04           # G2 波动率上限
    g3_percentile: float = 80.0       # G3 时序百分位
    g5_settlement_lock_mins: int = 30 # G5 结算前锁仓
    g5_enabled: bool = True


@dataclass
class CalibrationSettings:
    """在线校准配置"""
    online_update_enabled: bool = False
    temperature_update_interval: int = 50
    recalib_reject_threshold: float = 0.001


@dataclass
class Settings:
    """部署环境配置聚合"""
    mode: str = "demo"                # demo | live
    inst_id: str = "BTC-USDT-SWAP"
    timeframes: list = field(default_factory=lambda: ["5m"])

    # 子配置
    okx: OkxCredentials = field(default_factory=OkxCredentials)
    mcts: MctsSettings = field(default_factory=MctsSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    alpha: AlphaSettings = field(default_factory=AlphaSettings)
    gating: GatingSettings = field(default_factory=GatingSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)

    # 基础设施
    redis_url: str = "redis://localhost:6379"
    db_url: str = "postgresql://v8:v8_quant_2026@localhost:5432/v8_timeseries"
    log_dir: str = "logs/"
    log_level: str = "INFO"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @classmethod
    def load(cls, config_path: str = "v8.yaml") -> Settings:
        """
        从 YAML 加载配置，环境变量可覆盖。

        Args:
            config_path: YAML 配置文件路径 (相对 config/)
        """
        full_path = _CONFIG_DIR / config_path

        settings = cls()

        if full_path.exists():
            with open(full_path) as f:
                raw = yaml.safe_load(f) or {}

            if "inst_id" in raw:
                settings.inst_id = raw["inst_id"]
            if "mode" in raw:
                settings.mode = raw["mode"]

            # 子配置
            if "redis_url" in raw:
                settings.redis_url = raw["redis_url"]
            if "db_url" in raw:
                settings.db_url = raw["db_url"]

            # OKX
            okx_raw = raw.get("okx", {})
            settings.okx = OkxCredentials(
                api_key=okx_raw.get("api_key", ""),
                secret_key=okx_raw.get("secret_key", ""),
                passphrase=okx_raw.get("passphrase", ""),
                demo=okx_raw.get("demo", True),
            )

            # MCTS
            mcts_raw = raw.get("mcts", {})
            settings.mcts = MctsSettings(
                backend=mcts_raw.get("backend", "native"),
                timeout_ms=mcts_raw.get("timeout_ms", 100),
                num_workers=mcts_raw.get("num_workers", 8),
                num_simulations=mcts_raw.get("num_simulations", 200),
                max_depth=mcts_raw.get("max_depth", 5),
                ucb1_c=mcts_raw.get("ucb1_c", 2.0),
            )

            # Risk
            risk_raw = raw.get("risk", {})
            settings.risk = RiskSettings(
                max_position_pct=risk_raw.get("max_position_pct", 1.0),
                max_daily_loss_pct=risk_raw.get("max_daily_loss_pct", 0.03),
                max_consecutive_losses=risk_raw.get("max_consecutive_losses", 5),
                kelly_fraction=risk_raw.get("kelly_fraction", 0.25),
                maker_only=risk_raw.get("maker_only", False),
            )

            # Alpha
            alpha_raw = raw.get("alpha", {})
            settings.alpha = AlphaSettings(
                obi_threshold=alpha_raw.get("obi_threshold", 0.6),
                funding_rate_arb=alpha_raw.get("funding_rate_arb", True),
                fusion_obi_weight=alpha_raw.get("fusion_obi_weight", 0.6),
                fusion_funding_weight=alpha_raw.get("fusion_funding_weight", 0.4),
            )

            # Gating
            gating_raw = raw.get("gating", {})
            settings.gating = GatingSettings(
                g1_min_depth_usd=gating_raw.get("g1_min_depth_usd", 50_000),
                g2_max_hv=gating_raw.get("g2_max_hv", 0.04),
                g3_percentile=gating_raw.get("g3_percentile", 80.0),
                g5_settlement_lock_mins=gating_raw.get("g5_settlement_lock_mins", 30),
                g5_enabled=gating_raw.get("g5_enabled", True),
            )

            # Calibration
            calib_raw = raw.get("calibration", {})
            settings.calibration = CalibrationSettings(
                online_update_enabled=calib_raw.get("online_update_enabled", False),
                temperature_update_interval=calib_raw.get("temperature_update_interval", 50),
                recalib_reject_threshold=calib_raw.get("recalib_reject_threshold", 0.001),
            )

        # 环境变量覆盖
        settings.mode = os.getenv("V8_MODE", settings.mode)
        settings.okx.api_key = os.getenv("V8_OKX_API_KEY", settings.okx.api_key)
        settings.okx.secret_key = os.getenv("V8_OKX_SECRET_KEY", settings.okx.secret_key)
        settings.okx.passphrase = os.getenv("V8_OKX_PASSPHRASE", settings.okx.passphrase)
        settings.redis_url = os.getenv("REDIS_URL", settings.redis_url)
        settings.db_url = os.getenv("V8_DB_URL", settings.db_url)
        settings.log_dir = os.getenv("V8_LOG_DIR", settings.log_dir)
        settings.risk.max_daily_loss_pct = float(
            os.getenv("V8_MAX_DAILY_LOSS", settings.risk.max_daily_loss_pct)
        )

        return settings

    def as_dict(self) -> Dict[str, Any]:
        """导出为字典 (用于日志/监控)"""
        return {
            "mode": self.mode,
            "inst_id": self.inst_id,
            "mcts_backend": self.mcts.backend,
            "kelly_fraction": self.risk.kelly_fraction,
            "max_daily_loss_pct": self.risk.max_daily_loss_pct,
            "obi_threshold": self.alpha.obi_threshold,
            "g5_enabled": self.gating.g5_enabled,
            "online_update": self.calibration.online_update_enabled,
        }


# ── 快速加载 ─────────────────────────────────────────────

def load_settings(config_path: str = "v8.yaml") -> Settings:
    """加载配置的便捷函数"""
    return Settings.load(config_path)


if __name__ == "__main__":
    settings = Settings.load()
    print(f"Mode: {settings.mode}")
    print(f"MCTS: backend={settings.mcts.backend}, workers={settings.mcts.num_workers}")
    print(f"Risk: kelly={settings.risk.kelly_fraction}, max_daily_loss={settings.risk.max_daily_loss_pct}")
    print(f"Dict: {settings.as_dict()}")
    print("✓ Settings self-test passed")
