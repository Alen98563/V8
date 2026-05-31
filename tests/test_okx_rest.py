"""
测试：OKX REST 适配�?
验证签名、请求构造、响应解�?"""

import pytest
from adapters.okx_rest import (
    OkxRestAdapter,
    OkxOrder,
    OkxSide,
    OkxOrderType,
    OkxPositionSide,
)


def test_signature_generation(mock_rest_adapter):
    """测试 API 签名生成"""
    timestamp = "2024-05-30T12:00:00.000Z"
    method = "GET"
    path = "/api/v5/market/ticker"
    
    sign = mock_rest_adapter._sign(timestamp, method, path, "")
    
    # 签名应该是非空字符串
    assert sign
    assert isinstance(sign, str)
    # Base64 编码�?SHA256 签名长度固定
    assert len(sign) > 20


def test_headers_include_required_fields(mock_rest_adapter):
    """测试请求头包含必要字�?""
    headers = mock_rest_adapter._headers("GET", "/api/v5/test")
    
    assert "OK-ACCESS-KEY" in headers
    assert "OK-ACCESS-SIGN" in headers
    assert "OK-ACCESS-TIMESTAMP" in headers
    assert "OK-ACCESS-PASSPHRASE" in headers
    assert "Content-Type" in headers


def test_demo_mode_adds_simulated_header(mock_rest_adapter):
    """测试模拟盘模式添加额�?header"""
    assert mock_rest_adapter.demo is True
    
    headers = mock_rest_adapter._headers("GET", "/api/v5/test")
    assert "x-simulated-trading" in headers
    assert headers["x-simulated-trading"] == "1"


def test_order_to_api_dict():
    """测试订单转换�?API 请求�?""
    order = OkxOrder(
        inst_id="BTC-USDT-SWAP",
        side=OkxSide.BUY,
        position_side=OkxPositionSide.LONG,
        order_type=OkxOrderType.LIMIT,
        size="1",
        price="2000.0",
        client_order_id="test_order_001",
    )
    
    api_dict = order.to_api_dict()
    
    assert api_dict["instId"] == "BTC-USDT-SWAP"
    assert api_dict["side"] == "buy"
    assert api_dict["posSide"] == "long"
    assert api_dict["ordType"] == "limit"
    assert api_dict["sz"] == "1"
    assert api_dict["px"] == "2000.0"
    assert api_dict["clOrdId"] == "test_order_001"


def test_market_order_no_price():
    """测试市价单不需要价�?""
    order = OkxOrder(
        inst_id="BTC-USDT-SWAP",
        side=OkxSide.SELL,
        position_side=OkxPositionSide.SHORT,
        order_type=OkxOrderType.MARKET,
        size="2",
        price=None,
    )
    
    api_dict = order.to_api_dict()
    
    assert "px" not in api_dict, "市价单不应包含价格字�?


@pytest.mark.asyncio
async def test_get_ticker(mock_rest_adapter):
    """测试获取行情数据"""
    # Mock 响应
    mock_rest_adapter._request.return_value = {
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "last": "2000.5",
            "bidPx": "2000.0",
            "askPx": "2001.0",
        }]
    }
    
    ticker = await mock_rest_adapter.get_ticker("BTC-USDT-SWAP")
    
    assert ticker["instId"] == "BTC-USDT-SWAP"
    assert ticker["last"] == "2000.5"
    mock_rest_adapter._request.assert_called_once()
