"""
测试：市场数据模�?
验证 K 线数据加载、缓存、资金费率查�?"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock
from data.market_data import MarketDataLoader


@pytest.mark.asyncio
async def test_get_candles_basic(mock_rest_adapter):
    """测试基本 K 线数据获�?""
    # Mock 响应
    mock_rest_adapter.get_candles = AsyncMock(return_value=[
        ["1717027200000", "2000", "2010", "1995", "2005", "100", "200000", "200000", "1"],
        ["1717027500000", "2005", "2015", "2000", "2010", "120", "240000", "240000", "1"],
    ])
    
    loader = MarketDataLoader(mock_rest_adapter)
    df = await loader.get_candles("BTC-USDT-SWAP", bar="5m", limit=10)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "open" in df.columns
    assert "close" in df.columns
    assert "ts" in df.columns


@pytest.mark.asyncio
async def test_candles_caching(mock_rest_adapter):
    """测试 K 线数据缓�?""
    mock_rest_adapter.get_candles = AsyncMock(return_value=[
        ["1717027200000", "2000", "2010", "1995", "2005", "100", "200000", "200000", "1"],
    ])
    
    loader = MarketDataLoader(mock_rest_adapter, cache_ttl=300)
    
    # 第一次请�?    df1 = await loader.get_candles("BTC-USDT-SWAP", use_cache=True)
    call_count_1 = mock_rest_adapter.get_candles.call_count
    
    # 第二次请求（应该使用缓存�?    df2 = await loader.get_candles("BTC-USDT-SWAP", use_cache=True)
    call_count_2 = mock_rest_adapter.get_candles.call_count
    
    assert call_count_1 == call_count_2, "第二次请求应该使用缓�?
    assert df1.equals(df2)


@pytest.mark.asyncio
async def test_candles_cache_bypass(mock_rest_adapter):
    """测试禁用缓存"""
    mock_rest_adapter.get_candles = AsyncMock(return_value=[
        ["1717027200000", "2000", "2010", "1995", "2005", "100", "200000", "200000", "1"],
    ])
    
    loader = MarketDataLoader(mock_rest_adapter)
    
    # 禁用缓存
    await loader.get_candles("BTC-USDT-SWAP", use_cache=False)
    await loader.get_candles("BTC-USDT-SWAP", use_cache=False)
    
    assert mock_rest_adapter.get_candles.call_count == 2


def test_clear_cache(mock_rest_adapter):
    """测试清空缓存"""
    loader = MarketDataLoader(mock_rest_adapter)
    loader._candles_cache["test"] = pd.DataFrame()
    loader._cache_timestamps["test"] = datetime.now().timestamp()
    
    loader.clear_cache()
    
    assert len(loader._candles_cache) == 0
    assert len(loader._cache_timestamps) == 0


@pytest.mark.asyncio
async def test_get_funding_rate(mock_rest_adapter):
    """测试资金费率查询"""
    mock_rest_adapter._request = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "fundingRate": "0.0001",
            "nextFundingTime": "1717056000000",
        }]
    })
    
    loader = MarketDataLoader(mock_rest_adapter)
    funding_rate = await loader.get_funding_rate("BTC-USDT-SWAP")
    
    assert funding_rate == pytest.approx(0.0001, rel=1e-6)


@pytest.mark.asyncio
async def test_get_next_funding_time(mock_rest_adapter):
    """测试下次资金费率时间查询"""
    mock_rest_adapter._request = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "fundingRate": "0.0001",
            "nextFundingTime": "1717056000000",  # 2024-05-30 20:00:00 UTC
        }]
    })
    
    loader = MarketDataLoader(mock_rest_adapter)
    next_time = await loader.get_next_funding_time("BTC-USDT-SWAP")
    
    assert next_time is not None
    assert isinstance(next_time, datetime)


@pytest.mark.asyncio
async def test_empty_candles_response(mock_rest_adapter):
    """测试�?K 线响应处�?""
    mock_rest_adapter.get_candles = AsyncMock(return_value=[])
    
    loader = MarketDataLoader(mock_rest_adapter)
    df = await loader.get_candles("BTC-USDT-SWAP")
    
    assert isinstance(df, pd.DataFrame)
    assert df.empty
