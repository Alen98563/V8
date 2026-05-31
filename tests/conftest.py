"""
V8 量化交易系统 - 单元测试

pytest 配置文件，提供共享 fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from adapters.okx_rest import OkxRestAdapter
from adapters.okx_ws import OkxWsAdapter
from data.orderbook import OrderBookManager
from execution.risk.risk_manager import RiskManager, RiskConfig


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_rest_adapter() -> OkxRestAdapter:
    """创建模拟 REST 适配器"""
    adapter = OkxRestAdapter(
        api_key="test_key",
        secret_key="test_secret",
        passphrase="test_pass",
        demo=True,
    )
    # Mock HTTP 请求
    adapter._request = AsyncMock()
    return adapter


@pytest.fixture
def mock_ws_adapter() -> OkxWsAdapter:
    """创建模拟 WebSocket 适配器"""
    adapter = OkxWsAdapter(ws_url="wss://test.example.com")
    adapter.connect = AsyncMock()
    adapter.disconnect = AsyncMock()
    adapter.subscribe = AsyncMock()
    return adapter


@pytest.fixture
def orderbook_manager() -> OrderBookManager:
    """创建盘口管理器"""
    return OrderBookManager(max_snapshots=10)


@pytest.fixture
def risk_manager() -> RiskManager:
    """创建风险管理器"""
    config = RiskConfig(
        max_risk_per_trade=0.01,
        max_daily_loss=0.05,
        max_open_positions=3,
    )
    return RiskManager(config)


@pytest.fixture
def sample_orderbook_data():
    """示例盘口数据"""
    return {
        "bids": [
            [2000.0, 10.0, 0, 5],
            [1999.5, 15.0, 0, 8],
            [1999.0, 20.0, 0, 12],
        ],
        "asks": [
            [2001.0, 12.0, 0, 6],
            [2001.5, 18.0, 0, 9],
            [2002.0, 25.0, 0, 15],
        ],
        "ts": "1717027200000",  # 2024-05-30 12:00:00 UTC
    }
