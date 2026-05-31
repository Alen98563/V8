"""
MCTS 蒙特卡洛树搜索

配置参数:
- max_depth: N=5–20 步
- num_simulations: 200–1000 次
- discount_factor: γ=0.95
- risk_penalty: λ=0.3
- exploration_constant: c=√2≈1.414

正式 MCTS 在 Rust 侧由 v8_core_engine.MctsPlanner 导出
"""

# ============================================================
# MCTS 配置
# ============================================================

from dataclasses import dataclass


@dataclass
class MctsConfig:
    max_depth: int = 10
    num_simulations: int = 1000
    discount_factor: float = 0.95
    risk_penalty: float = 0.3
    exploration_constant: float = 1.414
    timeout_ms: int = 50
    num_workers: int = 8
    cache_ttl_seconds: int = 30


# 默认配置
DEFAULT_MCTS_CONFIG = MctsConfig()

# 降级配置 (快速但低精度)
DEGRADED_MCTS_CONFIG = MctsConfig(
    max_depth=5,
    num_simulations=100,
    timeout_ms=10,
    num_workers=2,
)