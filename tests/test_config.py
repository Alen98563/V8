"""
测试：配置加�?
验证 config/v8.yaml �?config/api_keys.yaml 的正确加�?"""

import pytest
import os
from common.config import load_config, V8Config


def test_load_v8_config():
    """测试加载 v8.yaml"""
    config = load_config()
    
    # 验证必要字段
    assert config.inst_id == "BTC-USDT-SWAP"
    assert config.bar_seconds == 300
    assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    # 验证门控配置
    assert config.gating.min_depth_10 > 0
    assert config.gating.max_spread_bps > 0
    assert config.gating.max_realized_vol > config.gating.min_realized_vol


def test_config_dry_run_default():
    """测试 dry_run 默认�?True（安全起见）"""
    config = load_config()
    assert config.dry_run is True, "默认应该启用 dry_run 模式"


def test_config_gating_blackout():
    """测试资金费率黑名单窗口配�?""
    config = load_config()
    assert config.gating.funding_blackout_min >= 30, "黑名单窗口应该至�?30 分钟"


@pytest.mark.skipif(
    not os.path.exists("config/api_keys.yaml"),
    reason="api_keys.yaml 不存�?
)
def test_api_keys_config_exists():
    """测试 api_keys.yaml 文件存在且可�?""
    from common.config import load_api_keys
    api_config = load_api_keys()
    
    assert "okx" in api_config
    assert "alerts" in api_config
    assert "triton" in api_config
