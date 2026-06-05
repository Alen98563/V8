"""
V8 StrategyGP — 遗传编程自动发现交易策略

输入: 特征矩阵 (baseline + GP-discovered)
输出: entry/exit 策略规则树, 导出为 StrategyGene JSON

使用 deap 的遗传编程进化完整的策略规则树。
每24小时运行一轮, 自动注册到 genome_registry。

Author: Hermes
Date: 2026-06-04
"""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

# ============================================================
# 配置
# ============================================================

STRATEGY_GP_DIR = Path(__file__).parent
GENOME_DIR = STRATEGY_GP_DIR / "genomes"

POP_SIZE = 400
MAX_GENERATIONS = 40
TOURNAMENT_SIZE = 3
ELITE_SIZE = 8
CROSSOVER_PROB = 0.65
MUTATION_PROB = 0.35
MAX_TREE_DEPTH = 8
MIN_TRADES = 10
EARLY_STOP_GENS = 8
N_ISLANDS = 4
MIGRATION_INTERVAL = 8

# 交易成本
TAKER_FEE = 0.0005        # 0.05%
SLIPPAGE = 0.00025        # 0.025%
MAKER_REBATE = 0.0002     # 0.02% (限价单)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class StrategyBlueprint:
    """遗传编程发现的策略模板"""
    gene_id: str
    strategy_name: str
    entry_tree: str           # 表达式树文本
    exit_tree: str
    stop_loss: float
    take_profit: float
    position_size: float
    timeout_minutes: int
    fitness: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    total_pnl: float
    in_sample: bool
    generation: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================
# 策略模拟器
# ============================================================

class TradeSimulator:
    """轻量级逐 bar 交易模拟器"""

    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 10_000,
                 max_position_frac: float = 0.10):
        self.data = data
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.max_pos_size = initial_capital * max_position_frac
        self.commission_rate = TAKER_FEE + SLIPPAGE

    def run(self,
            entry_signal: np.ndarray,
            exit_signal: np.ndarray,
            stop_loss: float,
            take_profit: float,
            timeout_minutes: int,
            direction: int = 0) -> Dict:
        """
        运行回测。

        direction: 1=LONG only, -1=SHORT only, 0=BOTH (signal sign matters)
        entry_signal[i]: >0 go long, <0 go short, 0 no signal
        exit_signal[i]: 非零则平仓
        """
        n = len(entry_signal)
        trades = []

        position = 0.0      # 持仓方向 * 数量
        entry_price = 0.0
        entry_bar = 0
        peak_capital = self.capital
        max_dd = 0.0

        for i in range(n):
            mid_price = self.data["mid_price"].iloc[i]

            if position == 0:
                sig = entry_signal[i]
                if sig == 0:
                    continue
                if direction == 1 and sig < 0:
                    continue
                if direction == -1 and sig > 0:
                    continue

                # 开仓
                position = self.max_pos_size / mid_price * np.sign(sig)
                entry_price = mid_price
                entry_bar = i
            else:
                # 检查 exit 条件
                pnl_pct = (mid_price - entry_price) / entry_price * np.sign(position)
                exit_reason = ""

                should_exit = False

                # 止损
                if pnl_pct <= -stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                # 止盈
                elif pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                # 信号反转
                elif exit_signal[i] != 0:
                    should_exit = True
                    exit_reason = "signal_exit"
                # 超时
                elif timeout_minutes > 0 and (i - entry_bar) >= timeout_minutes:
                    should_exit = True
                    exit_reason = "timeout"
                # 收盘前平仓 (最后 5 根 bar)
                elif i >= n - 5:
                    should_exit = True
                    exit_reason = "eod"

                if should_exit:
                    gross_pnl = position * (mid_price - entry_price)
                    commission = abs(position) * entry_price * self.commission_rate * 2
                    net_pnl = gross_pnl - commission

                    self.capital += net_pnl
                    peak_capital = max(peak_capital, self.capital)
                    dd = (peak_capital - self.capital) / peak_capital
                    max_dd = max(max_dd, dd)

                    trades.append({
                        "pnl": net_pnl,
                        "pnl_pct": pnl_pct,
                        "won": net_pnl > 0,
                        "exit_reason": exit_reason,
                        "bars_held": i - entry_bar,
                    })

                    position = 0.0

        # 强制平仓
        if position != 0:
            last_price = self.data["mid_price"].iloc[-1]
            gross_pnl = position * (last_price - entry_price)
            commission = abs(position) * entry_price * self.commission_rate * 2
            net_pnl = gross_pnl - commission
            self.capital += net_pnl

            trades.append({
                "pnl": net_pnl,
                "pnl_pct": (last_price - entry_price) / entry_price * np.sign(position),
                "won": net_pnl > 0,
                "exit_reason": "forced_close",
                "bars_held": n - entry_bar,
            })

        return {
            "trades": trades,
            "total_trades": len(trades),
            "total_pnl": self.capital - self.initial_capital,
            "sharpe": self._calc_sharpe(trades),
            "win_rate": sum(1 for t in trades if t["won"]) / max(len(trades), 1),
            "max_drawdown": max_dd,
            "profit_factor": self._calc_profit_factor(trades),
        }

    def _calc_sharpe(self, trades: List[Dict]) -> float:
        if len(trades) < 2:
            return 0.0
        returns = [t["pnl"] / self.initial_capital for t in trades]
        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return (mean / std) * math.sqrt(252 * 24 * 12)  # 年化 (5-min bars)

    def _calc_profit_factor(self, trades: List[Dict]) -> float:
        wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        return wins / losses if losses > 0 else float("inf")


# ============================================================
# GP 引擎
# ============================================================

class StrategyGpEngine:
    """遗传编程策略发现引擎"""

    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Args:
            data: bar DataFrame, 含特征列 + mid_price
            feature_cols: 可用特征列名
        """
        self.data = data
        self.feature_cols = feature_cols
        self.feature_vectors: Dict[str, np.ndarray] = {}
        self._build_feature_vectors()

        # 数据划分: 训练 50% / 验证 30% / 测试 20%
        n = len(data)
        self.train_end = int(n * 0.5)
        self.val_end = int(n * 0.8)
        self.train_data = data.iloc[:self.train_end]
        self.val_data = data.iloc[self.train_end:self.val_end]
        self.test_data = data.iloc[self.val_end:]

        # deap 设置: 两个 PSET — entry 和 exit
        self.entry_pset = self._build_pset("entry")
        self.exit_pset = self._build_pset("exit")

        creator.create("StratFitnessMax", base.Fitness, weights=(1.0,))
        creator.create("StrategyIndividual", list, fitness=creator.StratFitnessMax)

        self._setup_toolbox()

    def _build_feature_vectors(self):
        """构建特征向量"""
        for col in self.feature_cols:
            if col in self.data.columns:
                vals = self.data[col].values.astype(float)
                self.feature_vectors[col] = np.nan_to_num(vals, nan=0.0)

    def _build_pset(self, name: str) -> gp.PrimitiveSet:
        """构建 PrimitiveSet"""
        pset = gp.PrimitiveSet(f"MAIN_{name}", 0)

        # 比较操作符
        def _lt(a, b): return float(a < b)
        def _gt(a, b): return float(a > b)
        def _lte(a, b): return float(a <= b)
        def _gte(a, b): return float(a >= b)
        def _cross_above(a, b): return float(a[-1] > b[-1] and a[-2] <= b[-2]) if hasattr(a, '__len__') and hasattr(b, '__len__') and len(a) > 1 and len(b) > 1 else 0.0
        def _cross_below(a, b): return float(a[-1] < b[-1] and a[-2] >= b[-2]) if hasattr(a, '__len__') and hasattr(b, '__len__') and len(a) > 1 and len(b) > 1 else 0.0

        # 逻辑操作符
        def _and(a, b): return float(a > 0.5 and b > 0.5)
        def _or(a, b): return float(a > 0.5 or b > 0.5)
        def _not(a): return float(a < 0.5)

        pset.addPrimitive(_gt, 2, name="gt")
        pset.addPrimitive(_lt, 2, name="lt")
        pset.addPrimitive(_lte, 2, name="lte")
        pset.addPrimitive(_gte, 2, name="gte")
        pset.addPrimitive(_cross_above, 2, name="cross_above")
        pset.addPrimitive(_cross_below, 2, name="cross_below")
        pset.addPrimitive(_and, 2, name="AND")
        pset.addPrimitive(_or, 2, name="OR")
        pset.addPrimitive(_not, 1, name="NOT")

        # 数学操作符 (用于 entry/exit 表达式)
        pset.addPrimitive(np.add, 2, name="add")
        pset.addPrimitive(np.subtract, 2, name="sub")
        pset.addPrimitive(np.multiply, 2, name="mul")
        pset.addPrimitive(lambda a, b: a / b if abs(b) > 1e-10 else 0.0, 2, name="div")
        pset.addPrimitive(abs, 1, name="abs")

        # 特征终端
        for feat_name, feat_vec in self.feature_vectors.items():
            # 使用闭包捕获当前值
            pset.addEphemeralConstant(f"F_{feat_name[:12]}", lambda v=feat_vec: v)

        # 阈值常量
        for label, val in [("T0", 0.0), ("T05", 0.5), ("T1", 1.0), ("T2", 2.0),
                            ("T01", 0.01), ("T02", 0.02), ("T005", 0.005)]:
            pset.addTerminal(val, label)

        # 随机常数
        pset.addEphemeralConstant("RND", lambda: random.uniform(0.001, 3.0))

        return pset

    def _setup_toolbox(self):
        """设置 toolbox — 个体包含 (entry_tree, exit_tree, stop_loss, tp, pos_sz, timeout)"""
        self.toolbox = base.Toolbox()

        # 随机生成浮点参数
        def _gen_sl(): return random.uniform(0.005, 0.15)
        def _gen_tp(): return random.uniform(0.01, 0.25)
        def _gen_psz(): return random.uniform(0.005, 0.10)
        def _gen_timeout(): return random.randint(5, 480)

        # 个体: [entry_tree, exit_tree, stop_loss, take_profit, pos_size, timeout_min]
        def create_individual():
            entry_tree = gp.PrimitiveTree(gp.genHalfAndHalf(self.entry_pset, min_=1, max_=MAX_TREE_DEPTH // 2))
            exit_tree = gp.PrimitiveTree(gp.genHalfAndHalf(self.exit_pset, min_=1, max_=MAX_TREE_DEPTH // 2))
            return creator.StrategyIndividual([
                entry_tree, exit_tree,
                _gen_sl(), _gen_tp(), _gen_psz(), _gen_timeout()
            ])

        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile_entry", gp.compile, pset=self.entry_pset)
        self.toolbox.register("compile_exit", gp.compile, pset=self.exit_pset)

    def _simulate_on_data(self, individual, data: pd.DataFrame) -> Dict:
        """在数据子集上模拟交易"""
        entry_tree, exit_tree, sl, tp, psz, timeout = individual

        try:
            entry_func = self.toolbox.compile_entry(expr=entry_tree)
            exit_func = self.toolbox.compile_exit(expr=exit_tree)
        except Exception:
            return {"total_trades": 0, "sharpe": -10.0, "win_rate": 0.0, "max_drawdown": 1.0}

        n = len(data)
        entry_signal = np.zeros(n)
        exit_signal = np.zeros(n)

        for i in range(n):
            try:
                entry_signal[i] = entry_func()
                exit_signal[i] = exit_func()
            except Exception:
                pass

        sim = TradeSimulator(data)
        result = sim.run(entry_signal, exit_signal, sl, tp, int(timeout), direction=1)
        return result

    def evaluate(self, individual) -> Tuple[float]:
        """适应度: Sharpe主导 + 胜率 + 回撤惩罚"""
        result = self._simulate_on_data(individual, self.train_data)

        trades = result["total_trades"]
        if trades < MIN_TRADES:
            return (-10.0 + trades * 0.01,)  # 严重惩罚不活跃策略

        sharpe = result["sharpe"]
        win_rate = result["win_rate"]
        mdd = result["max_drawdown"]

        # 复杂度惩罚 (树深度)
        depth1 = individual[0].height
        depth2 = individual[1].height
        complexity_penalty = 0.02 * max(0, depth1 + depth2 - 10)

        # 综合适应度
        fitness = (
            max(0, sharpe) / 3.0 * 0.45 +
            win_rate * 0.25 +
            (1 - min(mdd, 0.25) / 0.25) * 0.15 +
            min(trades, 50) / 50 * 0.15 -
            complexity_penalty
        )

        return (fitness,)

    def _validate(self, individual) -> Tuple[float, Dict]:
        """验证集评估"""
        result = self._simulate_on_data(individual, self.val_data)
        if result["total_trades"] < MIN_TRADES:
            return -1.0, result
        return result["sharpe"], result

    def _test(self, individual) -> Dict:
        """测试集评估"""
        return self._simulate_on_data(individual, self.test_data)

    def run(self,
            max_gens: int = MAX_GENERATIONS,
            pop_size: int = POP_SIZE,
            verbose: bool = True) -> List[StrategyBlueprint]:
        """运行策略进化"""

        if len(self.data) < 200:
            if verbose:
                print(f"[StrategyGP] 数据不足: {len(self.data)} bars < 200")
            return []

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

        # 双树交叉/变异
        def mate_two_trees(ind1, ind2):
            for i in [0, 1]:  # entry + exit 树
                if random.random() < CROSSOVER_PROB:
                    gp.cxOnePoint(ind1[i], ind2[i])
            # 浮点参数: 算术交叉
            if random.random() < CROSSOVER_PROB:
                for j in [2, 3, 4, 5]:
                    if random.random() < 0.5:
                        a, b = ind1[j], ind2[j]
                        alpha = random.random()
                        ind1[j], ind2[j] = alpha*a + (1-alpha)*b, alpha*b + (1-alpha)*a
            return ind1, ind2

        def mutate_two_trees(ind):
            for i in [0, 1]:
                if random.random() < MUTATION_PROB:
                    ind[i], = gp.mutUniform(ind[i], expr=lambda pset, type_: gp.genFull(pset, min_=0, max_=2), pset=self.entry_pset if i == 0 else self.exit_pset)
            # 浮点参数: 高斯变异
            for j, (lo, hi) in [(2, (0.005, 0.15)), (3, (0.01, 0.25)), (4, (0.005, 0.10))]:
                if random.random() < MUTATION_PROB:
                    ind[j] += np.random.normal(0, (hi - lo) * 0.1)
                    ind[j] = max(lo, min(hi, ind[j]))
            if random.random() < MUTATION_PROB:  # timeout
                ind[5] = max(5, min(480, ind[5] + random.randint(-30, 30)))
            return ind,

        # 树大小限制
        def limit_tree_size(ind):
            for i in [0, 1]:
                if len(ind[i]) > MAX_TREE_DEPTH * 3:
                    ind[i] = gp.PrimitiveTree(gp.genHalfAndHalf(self.entry_pset if i == 0 else self.exit_pset, min_=1, max_=3))
            return ind

        self.toolbox.register("mate", mate_two_trees)
        self.toolbox.register("mutate", mutate_two_trees)

        pop = self.toolbox.population(n=pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        hof = tools.HallOfFame(ELITE_SIZE)
        best_fitness = -10.0
        no_improve = 0
        discovered = []

        if verbose:
            print(f"[StrategyGP] 开始进化: pop={pop_size}, gens={max_gens}, bars={len(self.data)}")

        for gen in range(max_gens):
            # 评估
            invalid = [ind for ind in pop if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # 限制树大小
            pop = [limit_tree_size(ind) for ind in pop]

            hof.update(pop)
            record = stats.compile(pop)

            if verbose and gen % 5 == 0:
                print(f"  gen {gen:3d}: best={record['max']:.4f}, avg={record['avg']:.4f}, min={record['min']:.4f}")

            # 验证最佳个体
            if record["max"] > best_fitness + 0.001:
                best_fitness = record["max"]
                no_improve = 0

                # 验证
                best_ind = hof[0]
                val_sharpe, val_result = self._validate(best_ind)
                if val_sharpe > 0.3:  # 验证集 Sharpe > 0.3 才保留
                    test_result = self._test(best_ind)
                    bp = StrategyBlueprint(
                        gene_id=str(uuid.uuid4())[:8],
                        strategy_name=f"gp_strat_{len(discovered):03d}",
                        entry_tree=str(best_ind[0]),
                        exit_tree=str(best_ind[1]),
                        stop_loss=round(best_ind[2], 4),
                        take_profit=round(best_ind[3], 4),
                        position_size=round(best_ind[4], 4),
                        timeout_minutes=int(best_ind[5]),
                        fitness=round(best_fitness, 4),
                        sharpe=round(test_result["sharpe"], 3),
                        win_rate=round(test_result["win_rate"], 3),
                        max_drawdown=round(test_result["max_drawdown"], 4),
                        total_trades=test_result["total_trades"],
                        total_pnl=round(test_result["total_pnl"], 2),
                        in_sample=False,
                        generation=gen,
                    )
                    discovered.append(bp)
                    if verbose:
                        print(f"    ✓ 新策略: Sharoe(IS)={record['max']:.3f}, Sharpe(OOS)={test_result['sharpe']:.3f}, "
                              f"trades={test_result['total_trades']}, WR={test_result['win_rate']:.1%}")
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_GENS:
                if verbose:
                    print(f"[StrategyGP] 早停: {no_improve} 代无改进")
                break

            # 下一代
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

            pop[:] = offspring

        # 排序
        discovered.sort(key=lambda s: s.sharpe, reverse=True)

        if verbose:
            print(f"[StrategyGP] 完成: 发现 {len(discovered)} 个策略")
            for bp in discovered[:5]:
                print(f"  {bp.strategy_name}: Sharpe={bp.sharpe:.3f}, "
                      f"WR={bp.win_rate:.1%}, Trades={bp.total_trades}, PnL={bp.total_pnl:.2f}")

        return discovered

    def save_genomes(self, strategies: List[StrategyBlueprint], output_dir: Path = GENOME_DIR):
        """保存策略基因组"""
        output_dir.mkdir(parents=True, exist_ok=True)
        for bp in strategies:
            path = output_dir / f"{bp.gene_id}.json"
            with open(path, "w") as f:
                json.dump({
                    "gene_id": bp.gene_id,
                    "strategy_name": bp.strategy_name,
                    "entry_tree": bp.entry_tree,
                    "exit_tree": bp.exit_tree,
                    "stop_loss": bp.stop_loss,
                    "take_profit": bp.take_profit,
                    "position_size": bp.position_size,
                    "timeout_minutes": bp.timeout_minutes,
                    "fitness": bp.fitness,
                    "sharpe": bp.sharpe,
                    "win_rate": bp.win_rate,
                    "max_drawdown": bp.max_drawdown,
                    "total_trades": bp.total_trades,
                    "total_pnl": bp.total_pnl,
                    "generation": bp.generation,
                    "created_at": bp.created_at,
                }, f, indent=2, ensure_ascii=False)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="V8 StrategyGP")
    ap.add_argument("--parquet", required=True, help="特征 parquet 文件路径")
    ap.add_argument("--output", default=str(GENOME_DIR), help="输出基因组目录")
    ap.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    ap.add_argument("--population", type=int, default=POP_SIZE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    # 过滤 NaN 列
    numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]
    numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > len(df) * 0.5]
    feature_cols = [c for c in numeric_cols if c not in ("ts", "instrument", "close_time", "forward_return", "Open", "High", "Low", "Close")]

    engine = StrategyGpEngine(df, feature_cols)
    strategies = engine.run(max_gens=args.generations, pop_size=args.population, verbose=not args.quiet)

    if strategies:
        engine.save_genomes(strategies, Path(args.output))
        print(f"\nSaved {len(strategies)} strategy genomes to {args.output}")
    else:
        print("No viable strategies discovered.")
