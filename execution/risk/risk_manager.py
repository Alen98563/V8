"""
V8 量化交易系统 - 风险管理模块

职责：
- 仓位大小计算（Kelly 公式 / 固定比例 / ATR 动态调整）
- 单笔最大亏损限制
- 日累计亏损限制
- 持仓集中度控制
"""

from typing import Optional
from dataclasses import dataclass

from common.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RiskConfig:
    """风控参数配置"""
    # 单笔风险
    max_risk_per_trade: float = 0.01      # 单笔最大亏损占总资金比例（1%）
    max_position_size: float = 0.10       # 单笔最大仓位占总资金比例（10%）

    # 日度风险
    max_daily_loss: float = 0.05          # 日最大亏损占总资金比例（5%）
    max_daily_trades: int = 50            # 日最大交易次数

    # 持仓限制
    max_open_positions: int = 3           # 同时最大持仓数量
    max_correlation: float = 0.7          # 持仓间最大相关性（Phase 2）

    # 杠杆限制
    max_leverage: int = 5                 # 最大杠杆倍数
    default_leverage: int = 2             # 默认杠杆倍数


class RiskManager:
    """
    风险管理器

    功能：
    - 计算建议仓位大小
    - 检查风控规则
    - 跟踪日度损益
    """

    def __init__(self, config: RiskConfig):
        """
        初始化风险管理器

        Args:
            config: 风控参数配置
        """
        self.config = config
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.open_positions: int = 0

        logger.info(f"风险管理器初始化: 单笔风险={config.max_risk_per_trade:.2%}, 日风险={config.max_daily_loss:.2%}")

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 1.0,
    ) -> float:
        """
        计算建议仓位大小

        基于风险百分比法：
        仓位 = (账户余额 × 风险比例) / (入场价 - 止损价)

        Args:
            account_balance: 账户总余额（USDT）
            entry_price: 入场价格
            stop_loss_price: 止损价格
            confidence: 信号置信度（0-1，用于动态调整风险比例）

        Returns:
            建议仓位大小（基础货币单位）
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("价格无效，返回 0 仓位")
            return 0.0

        # 计算单位风险金额
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.warning("止损距离为 0，返回 0 仓位")
            return 0.0

        # 根据置信度动态调整风险比例
        adjusted_risk = self.config.max_risk_per_trade * min(confidence, 1.0)
        risk_amount = account_balance * adjusted_risk

        # 计算仓位大小
        position_size = risk_amount / risk_per_unit

        # 应用最大仓位限制
        max_size = (account_balance * self.config.max_position_size) / entry_price
        position_size = min(position_size, max_size)

        logger.debug(
            f"仓位计算: 余额={account_balance:.2f}, 风险={adjusted_risk:.2%}, "
            f"止损距离={risk_per_unit:.2f}, 仓位={position_size:.4f}"
        )

        return position_size

    def check_risk_limits(self) -> tuple[bool, str]:
        """
        检查是否触发风控限制

        Returns:
            (是否允许交易, 拒绝原因)
        """
        # 检查日亏损限制
        if self.daily_pnl < -self.config.max_daily_loss:
            reason = f"日亏损限制触发: 当前={self.daily_pnl:.2%}, 限制={self.config.max_daily_loss:.2%}"
            logger.warning(reason)
            return False, reason

        # 检查日交易次数
        if self.daily_trades >= self.config.max_daily_trades:
            reason = f"日交易次数限制: 当前={self.daily_trades}, 限制={self.config.max_daily_trades}"
            logger.warning(reason)
            return False, reason

        # 检查持仓数量
        if self.open_positions >= self.config.max_open_positions:
            reason = f"持仓数量限制: 当前={self.open_positions}, 限制={self.config.max_open_positions}"
            logger.warning(reason)
            return False, reason

        return True, "OK"

    def on_trade_open(self):
        """记录新开仓"""
        self.daily_trades += 1
        self.open_positions += 1
        logger.debug(f"开仓记录: 日交易={self.daily_trades}, 持仓={self.open_positions}")

    def on_trade_close(self, pnl: float):
        """记录平仓和损益"""
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += pnl
        logger.debug(f"平仓记录: 本次PnL={pnl:.2f}, 日PnL={self.daily_pnl:.2f}, 持仓={self.open_positions}")

    def reset_daily(self):
        """日终重置计数器"""
        logger.info(f"日终重置: 日PnL={self.daily_pnl:.2f}, 日交易={self.daily_trades}")
        self.daily_pnl = 0.0
        self.daily_trades = 0


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """
    Kelly 公式计算最优仓位比例

    公式：f = (p × b - q) / b
    - p: 胜率
    - q: 败率 (1 - p)
    - b: 盈亏比（平均盈利 / 平均亏损）

    Args:
        win_rate: 历史胜率（0-1）
        win_loss_ratio: 盈亏比

    Returns:
        建议仓位比例（0-1）
    """
    if win_loss_ratio <= 0 or win_rate <= 0:
        return 0.0

    p = win_rate
    q = 1 - p
    b = win_loss_ratio

    kelly = (p * b - q) / b

    # 实践中通常用半 Kelly 降低波动
    half_kelly = max(0, kelly / 2)

    logger.debug(f"Kelly 计算: 胜率={p:.2%}, 盈亏比={b:.2f}, Kelly={kelly:.2%}, 半Kelly={half_kelly:.2%}")

    return half_kelly
