"""
测试：风险管理模块

验证仓位计算、风控规则、Kelly 公式
"""

import pytest
from execution.risk.risk_manager import RiskManager, RiskConfig, kelly_fraction


def test_position_size_calculation(risk_manager):
    """测试仓位大小计算"""
    position_size = risk_manager.calculate_position_size(
        account_balance=10000.0,  # 10k USDT
        entry_price=2000.0,
        stop_loss_price=1900.0,   # 止损 100 USDT
        confidence=1.0,
    )
    
    # 风险金额 = 10000 * 0.01 = 100 USDT
    # 单位风险 = 2000 - 1900 = 100 USDT
    # 仓位 = 100 / 100 = 1.0 ETH
    assert position_size == pytest.approx(1.0, rel=1e-2)


def test_position_size_with_confidence(risk_manager):
    """测试置信度影响仓位大小"""
    size_full = risk_manager.calculate_position_size(
        account_balance=10000.0,
        entry_price=2000.0,
        stop_loss_price=1900.0,
        confidence=1.0,
    )
    
    size_half = risk_manager.calculate_position_size(
        account_balance=10000.0,
        entry_price=2000.0,
        stop_loss_price=1900.0,
        confidence=0.5,
    )
    
    # 置信度 0.5 应该得到一半仓位
    assert size_half == pytest.approx(size_full / 2, rel=1e-2)


def test_position_size_max_limit(risk_manager):
    """测试最大仓位限制"""
    position_size = risk_manager.calculate_position_size(
        account_balance=10000.0,
        entry_price=2000.0,
        stop_loss_price=1000.0,  # 很大的止损距离
        confidence=1.0,
    )
    
    # 最大仓位 = 10000 * 0.10 / 2000 = 0.5 ETH
    max_size = (10000.0 * 0.10) / 2000.0
    assert position_size <= max_size


def test_risk_limits_pass(risk_manager):
    """测试风控规则通过"""
    allowed, reason = risk_manager.check_risk_limits()
    
    assert allowed is True
    assert reason == "OK"


def test_daily_loss_limit(risk_manager):
    """测试日亏损限制触发"""
    # 模拟亏损超过限制
    risk_manager.daily_pnl = -0.06  # -6%，超过 -5% 限制
    
    allowed, reason = risk_manager.check_risk_limits()
    
    assert allowed is False
    assert "日亏损限制" in reason


def test_daily_trades_limit(risk_manager):
    """测试日交易次数限制"""
    risk_manager.daily_trades = 50  # 达到限制
    
    allowed, reason = risk_manager.check_risk_limits()
    
    assert allowed is False
    assert "日交易次数限制" in reason


def test_open_positions_limit(risk_manager):
    """测试持仓数量限制"""
    risk_manager.open_positions = 3  # 达到限制
    
    allowed, reason = risk_manager.check_risk_limits()
    
    assert allowed is False
    assert "持仓数量限制" in reason


def test_trade_tracking(risk_manager):
    """测试交易记录跟踪"""
    # 开仓
    risk_manager.on_trade_open()
    assert risk_manager.daily_trades == 1
    assert risk_manager.open_positions == 1
    
    # 平仓
    risk_manager.on_trade_close(pnl=100.0)
    assert risk_manager.open_positions == 0
    assert risk_manager.daily_pnl == 100.0


def test_daily_reset(risk_manager):
    """测试日终重置"""
    risk_manager.daily_pnl = 500.0
    risk_manager.daily_trades = 10
    
    risk_manager.reset_daily()
    
    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.daily_trades == 0


def test_kelly_formula_positive_edge():
    """测试 Kelly 公式（正期望）"""
    # 胜率 60%，盈亏比 2:1
    kelly = kelly_fraction(win_rate=0.6, win_loss_ratio=2.0)
    
    # f = (0.6 * 2 - 0.4) / 2 = 0.4
    # 半 Kelly = 0.2
    assert kelly == pytest.approx(0.2, rel=1e-2)


def test_kelly_formula_negative_edge():
    """测试 Kelly 公式（负期望）"""
    # 胜率 30%，盈亏比 1:1
    kelly = kelly_fraction(win_rate=0.3, win_loss_ratio=1.0)
    
    # 负期望，应该返回 0
    assert kelly == 0.0


def test_kelly_formula_zero_inputs():
    """测试 Kelly 公式（无效输入）"""
    assert kelly_fraction(0.0, 2.0) == 0.0
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.0, 0.0) == 0.0


def test_zero_stop_loss_distance(risk_manager):
    """测试止损距离为零的情况"""
    position_size = risk_manager.calculate_position_size(
        account_balance=10000.0,
        entry_price=2000.0,
        stop_loss_price=2000.0,  # 止损距离为 0
        confidence=1.0,
    )
    
    assert position_size == 0.0
