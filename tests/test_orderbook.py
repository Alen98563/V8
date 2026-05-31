"""
测试：盘口深度管�?
验证盘口快照、特征计算、OBI 指标
"""

import pytest
from datetime import datetime
from data.orderbook import OrderBookManager, OrderBookSnapshot, parse_okx_orderbook


def test_orderbook_snapshot_properties():
    """测试盘口快照属性计�?""
    snapshot = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[(2000.0, 10.0), (1999.5, 15.0)],
        asks=[(2001.0, 12.0), (2001.5, 18.0)],
    )
    
    assert snapshot.best_bid == 2000.0
    assert snapshot.best_ask == 2001.0
    assert snapshot.mid_price == 2000.5
    assert snapshot.spread == 1.0
    assert snapshot.spread_bps == pytest.approx(4.998, rel=1e-3)


def test_orderbook_manager_update_and_get():
    """测试盘口管理器更新和获取"""
    manager = OrderBookManager(max_snapshots=5)
    
    snap1 = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[(2000.0, 10.0)],
        asks=[(2001.0, 12.0)],
    )
    
    manager.update_snapshot(snap1)
    latest = manager.get_latest("BTC-USDT-SWAP")
    
    assert latest is not None
    assert latest.best_bid == 2000.0


def test_orderbook_manager_max_snapshots():
    """测试盘口管理器限制快照数�?""
    manager = OrderBookManager(max_snapshots=3)
    
    for i in range(5):
        snap = OrderBookSnapshot(
            inst_id="BTC-USDT-SWAP",
            ts=datetime.now(),
            bids=[(2000.0 + i, 10.0)],
            asks=[(2001.0 + i, 12.0)],
        )
        manager.update_snapshot(snap)
    
    # 应该只保留最�?3 �?    history = manager.get_history("BTC-USDT-SWAP", count=10)
    assert len(history) == 3


def test_calc_depth(orderbook_manager):
    """测试盘口深度计算"""
    snap = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[(2000.0, 10.0), (1999.5, 15.0), (1999.0, 20.0)],
        asks=[(2001.0, 12.0), (2001.5, 18.0), (2002.0, 25.0)],
    )
    
    orderbook_manager.update_snapshot(snap)
    
    bid_depth, ask_depth = orderbook_manager.calc_depth("BTC-USDT-SWAP", levels=2)
    
    assert bid_depth == 25.0  # 10 + 15
    assert ask_depth == 30.0  # 12 + 18


def test_calc_obi(orderbook_manager):
    """测试盘口不平衡度计算"""
    # 买盘强于卖盘
    snap = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[(2000.0, 30.0)],
        asks=[(2001.0, 10.0)],
    )
    
    orderbook_manager.update_snapshot(snap)
    
    obi = orderbook_manager.calc_order_book_imbalance("BTC-USDT-SWAP", levels=1)
    
    # OBI = (30 - 10) / (30 + 10) = 0.5
    assert obi == pytest.approx(0.5, rel=1e-3)
    assert obi > 0  # 买盘强，看涨信号


def test_calc_weighted_price(orderbook_manager):
    """测试加权价格计算"""
    snap = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[(2000.0, 10.0), (1999.0, 20.0)],
        asks=[(2001.0, 15.0), (2002.0, 25.0)],
    )
    
    orderbook_manager.update_snapshot(snap)
    
    bid_vwap = orderbook_manager.calc_weighted_price("BTC-USDT-SWAP", side="bid", levels=2)
    ask_vwap = orderbook_manager.calc_weighted_price("BTC-USDT-SWAP", side="ask", levels=2)
    
    # 买盘 VWAP = (2000*10 + 1999*20) / 30 = 1999.33
    assert bid_vwap == pytest.approx(1999.33, rel=1e-2)
    # 卖盘 VWAP = (2001*15 + 2002*25) / 40 = 2001.625
    assert ask_vwap == pytest.approx(2001.625, rel=1e-2)


def test_parse_okx_orderbook(sample_orderbook_data):
    """测试解析 OKX WebSocket 盘口数据"""
    snapshot = parse_okx_orderbook(sample_orderbook_data, "BTC-USDT-SWAP")
    
    assert snapshot.inst_id == "BTC-USDT-SWAP"
    assert len(snapshot.bids) == 3
    assert len(snapshot.asks) == 3
    assert snapshot.best_bid == 2000.0
    assert snapshot.best_ask == 2001.0


def test_empty_orderbook():
    """测试空盘口处�?""
    snapshot = OrderBookSnapshot(
        inst_id="BTC-USDT-SWAP",
        ts=datetime.now(),
        bids=[],
        asks=[],
    )
    
    assert snapshot.best_bid is None
    assert snapshot.best_ask is None
    assert snapshot.mid_price is None
    assert snapshot.spread is None
