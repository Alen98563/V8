"""
V8 FeatureGP — 遗传编程自动发现高IC特征

输入: tick数据库 → bar聚合 → 基础原子特征
输出: 发现的特征池 JSON (特征表达式 + IC + IR)

使用 deap 的遗传编程引擎进化特征表达式树。
每12小时运行一轮，追加到特征池。

Author: Hermes
Date: 2026-06-04
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from scipy.stats import spearmanr

# ============================================================
# 配置
# ============================================================

FEATURE_GP_DIR = Path(__file__).parent
POOL_PATH = FEATURE_GP_DIR / "gp_feature_pool.json"

# 原子特征名 (与 FeatureEngine 对齐)
ATOMIC_FEATURES = [
    "mid_price", "bid_sz", "ask_sz", "spread", "obi", "net_flow",
    "price_vel", "realized_vol", "trade_rate", "depth_total",
    "bid_depth", "ask_depth", "cvd", "spread_z"
]

# GP 参数
POP_SIZE = 300
MAX_GENERATIONS = 50
TOURNAMENT_SIZE = 3
ELITE_SIZE = 5
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.3
MAX_TREE_DEPTH = 6
IC_TARGET = 0.06           # |IC| 达标即停止
EARLY_STOP_GENS = 10       # 连续无改进则停止
MIN_BARS_REQUIRED = 150  # TEMP    # 最少需要多少 bar

# 窗口大小池
WINDOW_SIZES = [5, 10, 20, 30, 60, 120, 300]


# ============================================================
# 数据结构
# ============================================================

@dataclass
class GpFeature:
    """遗传编程发现的特征"""
    id: str
    expression: str              # 人类可读表达式
    tree_str: str                # deap 树序列化
    ic: float                    # Spearman IC
    ic_std: float                # IC 标准差 (rolling)
    ir: float                    # Information Ratio
    generation: int              # 发现的代数
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================
# deap 基础设施
# ============================================================

def _safe_div(a, b):
    """安全除法, 避免 /0"""
    return a / b if abs(b) > 1e-10 else 0.0

def _safe_log(x):
    """安全对数"""
    return math.log(abs(x) + 1e-10)

def _safe_sqrt(x):
    """安全平方根"""
    return math.sqrt(abs(x))

def _rank(x):
    """按值排序 → rank/len"""
    if not hasattr(x, '__len__') or len(x) < 2:
        return 0.5
    s = np.argsort(np.argsort(x)).astype(float)
    return s[-1] / (len(x) - 1) if len(x) > 1 else 0.5

def _diff_n(x, n=1):
    """n 步差值"""
    if not hasattr(x, '__len__') or len(x) <= n:
        return 0.0
    return x[-1] - x[-(n+1)]

def _rolling_mean(x, n=20):
    if not hasattr(x, '__len__') or len(x) < n:
        return float(np.mean(x)) if hasattr(x, '__len__') else float(x)
    return float(np.mean(x[-n:]))

def _rolling_std(x, n=20):
    if not hasattr(x, '__len__') or len(x) < n:
        return float(np.std(x)) if hasattr(x, '__len__') else 0.0
    return float(np.std(x[-n:]))

def _rolling_max(x, n=20):
    if not hasattr(x, '__len__') or len(x) < n:
        return float(np.max(x)) if hasattr(x, '__len__') else float(x)
    return float(np.max(x[-n:]))

def _rolling_min(x, n=20):
    if not hasattr(x, '__len__') or len(x) < n:
        return float(np.min(x)) if hasattr(x, '__len__') else float(x)
    return float(np.min(x[-n:]))

def _zscore(x, n=60):
    """滚动 z-score"""
    if not hasattr(x, '__len__') or len(x) < max(n, 2):
        return 0.0
    mean = np.mean(x[-n:])
    std = np.std(x[-n:])
    return (x[-1] - mean) / std if std > 1e-10 else 0.0

def _cs_zscore(x):
    """截面 z-score"""
    if not hasattr(x, '__len__') or len(x) < 2:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    return (x[-1] - mean) / std if std > 1e-10 else 0.0


# ============================================================
# GP 引擎
# ============================================================

class FeatureGpEngine:
    """遗传编程特征发现引擎"""

    def __init__(self, data: pd.DataFrame, label_col: str = "forward_return"):
        """
        Args:
            data: bar DataFrame, columns = 原子特征 + label_col
            label_col: 标签列名
        """
        self.data = data
        self.label_col = label_col
        self.labels = data[label_col].values.astype(float)
        self.n_bars = len(data)

        # 为每个原子特征创建 "当前值向量" (每个 bar 的值)
        # 对于滚动窗口特征, 我们需要在每个 bar 上重建 window
        self.feature_vectors: Dict[str, np.ndarray] = {}
        self._build_feature_vectors()

        # deap 设置
        self.pset = gp.PrimitiveSet("MAIN", 0)
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(_safe_div, 2, name="div")
        self.pset.addPrimitive(np.maximum, 2, name="max")
        self.pset.addPrimitive(np.minimum, 2, name="min")
        self.pset.addPrimitive(_safe_log, 1, name="log")
        self.pset.addPrimitive(_safe_sqrt, 1, name="sqrt")
        self.pset.addPrimitive(np.abs, 1, name="abs")
        self.pset.addPrimitive(np.negative, 1, name="neg")

        # 原子特征终端
        for feat in ATOMIC_FEATURES:
            if feat in self.feature_vectors:
                self.pset.addTerminal(self.feature_vectors[feat], feat)

        # 随机常数
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=MAX_TREE_DEPTH)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def _build_feature_vectors(self):
        """为所有可用特征构建值向量 — 优先 OF 计算特征(feat_*), fallback 原子特征"""
        # 1. Auto-detect all feat_* columns (OF computed features)
        feat_cols = [c for c in self.data.columns if c.startswith('feat_') and self.data[c].dtype in ('float64', 'float32', 'int64')]
        
        if feat_cols:
            # Use OF features — these are our best bet
            for col in feat_cols:
                vals = np.nan_to_num(self.data[col].values.astype(float), nan=0.0)
                self.feature_vectors[col] = vals
        else:
            # Fallback: use ATOMIC_FEATURES
            for feat in ATOMIC_FEATURES:
                if feat in self.data.columns:
                    vals = self.data[feat].values.astype(float)
                    vals = np.nan_to_num(vals, nan=0.0)
                    self.feature_vectors[feat] = vals

        # 2. Also include any numeric non-feat columns for broader coverage
        skip = {'forward_return', 'cf_label', 'ts', 'ts_ms', 'instrument', 'inst_id', 
                'close_time', 'bar_ts', 'open_ts', 'close_ts'}
        for col in self.data.columns:
            if col in self.feature_vectors or col in skip:
                continue
            if self.data[col].dtype in ('float64', 'float32', 'int64'):
                vals = np.nan_to_num(self.data[col].values.astype(float), nan=0.0)
                if np.std(vals) > 1e-10:  # skip constant columns
                    self.feature_vectors[col] = vals

        # 3. For important raw features, precompute rolling windows
        window_feats = [c for c in self.data.columns if c in ATOMIC_FEATURES or c.startswith('feat_')]
        for feat in window_feats[:20]:  # limit to 20 to avoid explosion
            if feat in self.data.columns:
                for w in [10, 30, 60]:  # shorter window list
                    name = f"rm_{feat}_{w}"
                    series = self.data[feat].rolling(w, min_periods=1).mean()
                    vals = np.nan_to_num(series.values, nan=0.0)
                    if np.std(vals) > 1e-10:
                        self.feature_vectors[name] = vals

    def evaluate(self, individual) -> Tuple[float]:
        """向量化评估: 对整个时间序列计算特征值, 然后算 |IC|"""
        try:
            # Step 1: Get tree structure — replace feature names with numpy arrays
            # Walk the tree and substitute terminals with their vectors
            import copy
            tree = copy.deepcopy(individual)
            
            # Substitute feature terminals with actual vectors
            for i, node in enumerate(tree):
                if isinstance(node, gp.Terminal):
                    # Check if this terminal name maps to a feature vector
                    name = node.name if hasattr(node, 'name') else str(node.value)
                    # Try both the terminal value and name
                    vec = None
                    if hasattr(node, 'name'):
                        vec = self.feature_vectors.get(node.name)
                    if vec is None:
                        vec = self.feature_vectors.get(str(node.value))
                    
                    if vec is not None:
                        tree[i] = gp.Terminal(vec, False, object)
            
            # Step 2: Compile and evaluate
            func = self.toolbox.compile(expr=tree)
            result = func()
            
            # If result is a scalar (no feat terminals matched), try array-like
            if np.isscalar(result) or not hasattr(result, '__len__'):
                values = np.full(self.n_bars, float(result))
            else:
                values = np.array(result, dtype=float).flatten()
                if len(values) != self.n_bars:
                    values = values[:self.n_bars] if len(values) > self.n_bars else np.pad(values, (0, self.n_bars - len(values)))
            
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            # Remove constant features
            if np.std(values) < 1e-10:
                return (0.0,)

            # Spearman IC
            ic, _ = spearmanr(values, self.labels)
            if np.isnan(ic):
                return (0.0,)

            abs_ic = abs(ic)
            depth = individual.height
            depth_penalty = 0.02 * max(0, depth - 4)

            fitness = abs_ic - depth_penalty
            return (max(0.0, fitness),)

        except Exception as e:
            return (0.0,)

    def run(self,
            max_gens: int = MAX_GENERATIONS,
            pop_size: int = POP_SIZE,
            verbose: bool = True) -> List[GpFeature]:
        """运行进化并返回发现的特征"""

        if self.n_bars < MIN_BARS_REQUIRED:
            if verbose:
                print(f"[FeatureGP] 数据不足: {self.n_bars} bars < {MIN_BARS_REQUIRED}")
            return []

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # 限制树的高度
        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_TREE_DEPTH * 2))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_TREE_DEPTH * 2))

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(ELITE_SIZE)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        best_ever = 0.0
        best_individual = None
        generations_no_improve = 0
        discovered: Dict[str, GpFeature] = {}  # 去重

        if verbose:
            print(f"[FeatureGP] 开始进化: pop={pop_size}, gens={max_gens}, bars={self.n_bars}")

        for gen in range(max_gens):
            # 评估
            invalid = [ind for ind in pop if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Hall of Fame
            hof.update(pop)

            # 统计
            record = stats.compile(pop)
            current_best = record["max"]

            if verbose and gen % 5 == 0:
                print(f"  gen {gen:3d}: best_ic={current_best:.4f}, avg={record['avg']:.4f}")

            # 收集发现的特征
            for ind in hof:
                ic_val = ind.fitness.values[0]
                if ic_val > 0.01:
                    expr = str(ind)
                    if expr not in discovered and ic_val > discovered.get(expr, GpFeature(id="", expression="", tree_str="", ic=0, ic_std=0, ir=0, generation=0)).ic:
                        discovered[expr] = GpFeature(
                            id=f"gp_{len(discovered):04d}",
                            expression=expr,
                            tree_str=str(ind),
                            ic=round(ic_val, 4),
                            ic_std=0.0,  # 单次评估, 无滚动 std
                            ir=0.0,
                            generation=gen,
                        )

            # 早停
            if current_best > best_ever + 0.001:
                best_ever = current_best
                best_individual = str(hof[0])
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if current_best >= IC_TARGET:
                if verbose:
                    print(f"[FeatureGP] IC目标达成: {current_best:.4f} >= {IC_TARGET}")
                break

            if generations_no_improve >= EARLY_STOP_GENS:
                if verbose:
                    print(f"[FeatureGP] 早停: {generations_no_improve} 代无改进")
                break

            # 下一代
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PROB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTATION_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = offspring

        # 排序输出
        result = sorted(discovered.values(), key=lambda f: f.ic, reverse=True)
        if verbose:
            print(f"[FeatureGP] 完成: 发现 {len(result)} 个特征, 最佳 IC={result[0].ic if result else 0:.4f}")

        return result

    @staticmethod
    def load_pool(path: Path = POOL_PATH) -> Dict[str, GpFeature]:
        """加载已有特征池"""
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        return {
            item["expression"]: GpFeature(**item)
            for item in data.get("features", [])
        }

    @staticmethod
    def save_pool(features: List[GpFeature], path: Path = POOL_PATH):
        """保存特征池"""
        # 合并已有特征
        existing = FeatureGpEngine.load_pool(path)
        for f in features:
            if f.expression not in existing or f.ic > existing[f.expression].ic:
                existing[f.expression] = f

        result = sorted(existing.values(), key=lambda f: f.ic, reverse=True)
        # 只保留 top 50
        result = result[:50]

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_features": len(result),
            "features": [
                {
                    "id": f.id,
                    "expression": f.expression,
                    "tree_str": f.tree_str,
                    "ic": f.ic,
                    "ic_std": f.ic_std,
                    "ir": f.ir,
                    "generation": f.generation,
                    "created_at": f.created_at,
                }
                for f in result
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_feature_matrix(data: pd.DataFrame, features: List[GpFeature]) -> pd.DataFrame:
        """用发现的 GP 特征计算特征矩阵"""
        # 简化实现: 对于复杂表达式, 用 eval + context
        # 生产环境可能用 deap 的 compile
        return data  # placeholder


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="V8 FeatureGP")
    ap.add_argument("--parquet", required=True, help="特征 parquet 文件路径")
    ap.add_argument("--label-col", default="forward_return", help="标签列名")
    ap.add_argument("--output", default=str(POOL_PATH), help="输出特征池路径")
    ap.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    ap.add_argument("--population", type=int, default=POP_SIZE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    engine = FeatureGpEngine(df, label_col=args.label_col)
    features = engine.run(max_gens=args.generations, pop_size=args.population, verbose=not args.quiet)

    if features:
        FeatureGpEngine.save_pool(features, Path(args.output))
        print(f"\nTop 5 discovered features:")
        for f in features[:5]:
            print(f"  [{f.id}] IC={f.ic:.4f}  expr={f.expression[:80]}...")
    else:
        print("No features discovered.")
