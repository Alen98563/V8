"""
V8 量化交易系统 - 配置加载器
=============================

功能：
- 加载 config/v8.yaml 运行时配置
- 加载 config/api_keys.yaml API 密钥环境变量名映射
- 从系统环境变量读取真实密钥（绝不在配置文件中硬编码）
- 支持 .env 文件（通过 python-dotenv）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

# 配置文件路径（相对于项目根目录）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
_DEF_PATH = os.path.join(_CONFIG_DIR, "v8.yaml")
_API_KEYS_PATH = os.path.join(_CONFIG_DIR, "api_keys.yaml")


@dataclass
class OkxCreds:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    is_demo: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.secret_key and self.passphrase)


@dataclass
class V8Config:
    """V8 系统配置"""
    inst_id: str = "BTC-USDT-SWAP"
    bar_seconds: int = 300
    redis_url: str = "redis://127.0.0.1:6379/0"
    triton_url: str = "localhost:8001"
    log_level: str = "INFO"
    dry_run: bool = True
    okx: OkxCreds = field(default_factory=OkxCreds)
    okx_demo: bool = True  # 是否使用模拟盘
    gating: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_config(path: Optional[str] = None) -> V8Config:
    """加载 V8 主配置文件"""
    # best-effort .env
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    data = _load_yaml(path or _DEF_PATH)

    okx_demo = _as_bool(os.getenv("OKX_DEMO", data.get("okx_demo", True)))

    cfg = V8Config(
        inst_id=str(data.get("inst_id", "BTC-USDT-SWAP")),
        bar_seconds=int(data.get("bar_seconds", 300)),
        redis_url=os.getenv("V8_REDIS_URL", data.get("redis_url", "redis://127.0.0.1:6379/0")),
        triton_url=os.getenv("V8_TRITON_URL", data.get("triton_url", "localhost:8001")),
        log_level=os.getenv("V8_LOG_LEVEL", data.get("log_level", "INFO")),
        dry_run=_as_bool(os.getenv("V8_DRY_RUN", data.get("dry_run", True))),
        okx_demo=okx_demo,
        gating=data.get("gating", {}) or {},
        raw=data,
    )

    # 根据 demo 模式选择对应的环境变量
    if okx_demo:
        cfg.okx = OkxCreds(
            api_key=os.getenv("OKX_DEMO_API_KEY", os.getenv("OKX_API_KEY", "")),
            secret_key=os.getenv("OKX_DEMO_SECRET_KEY", os.getenv("OKX_SECRET_KEY", "")),
            passphrase=os.getenv("OKX_DEMO_PASSPHRASE", os.getenv("OKX_PASSPHRASE", "")),
            is_demo=True,
        )
    else:
        cfg.okx = OkxCreds(
            api_key=os.getenv("OKX_API_KEY", ""),
            secret_key=os.getenv("OKX_SECRET_KEY", ""),
            passphrase=os.getenv("OKX_PASSPHRASE", ""),
            is_demo=False,
        )

    return cfg


def load_api_keys(path: Optional[str] = None) -> dict[str, Any]:
    """
    加载 API 密钥配置文件（config/api_keys.yaml）

    返回环境变量名映射，实际密钥需从 os.getenv() 读取
    """
    return _load_yaml(path or _API_KEYS_PATH)


def get_okx_endpoints(demo: bool = True) -> dict[str, str]:
    """
    获取 OKX API 端点地址

    Args:
        demo: 是否使用模拟盘

    Returns:
        端点字典（rest_base, ws_public, ws_private）
    """
    api_config = load_api_keys()
    endpoints = api_config.get("okx", {}).get("endpoints", {})

    if demo:
        return {
            "rest_base": endpoints.get("demo_rest_base", "https://www.okx.com"),
            "ws_public": endpoints.get("demo_ws_public", "wss://wspap.okx.com:8443/ws/v5/public"),
            "ws_private": endpoints.get("demo_ws_private", "wss://wspap.okx.com:8443/ws/v5/private"),
        }
    else:
        return {
            "rest_base": endpoints.get("rest_base", "https://www.okx.com"),
            "ws_public": endpoints.get("ws_public", "wss://ws.okx.com:8443/ws/v5/public"),
            "ws_private": endpoints.get("ws_private", "wss://ws.okx.com:8443/ws/v5/private"),
        }


def _as_bool(val, default: bool = False) -> bool:
    """Parse string/bool/int to bool. Case-insensitive.

    Accepts: true/false/1/0/yes/no/on/off (any case).
    Falls back to default for unrecognized values.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return default


__all__ = ["V8Config", "OkxCreds", "load_config", "load_api_keys", "get_okx_endpoints"]
