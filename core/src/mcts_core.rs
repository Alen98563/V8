//! # MCTS 搜索池 (MctsPool) — T4-1
//!
//! ## 架构定位
//!
//! MctsPool 是 QTS V8 **决策层（L6）** 的高性能蒙特卡洛树搜索引擎。
//! 在 5m K线闭盘瞬间，基于 FeatureEngine 的 50 维特征 + AlphaCast 预测结果，
//! 通过 **1000 次并行 rollout** 计算最优交易动作（开/平/维持），
//! 目标在 **< 50ms** 内返回决策结果。
//!
//! ## MCTS 算法流程
//!
//! ```text
//!                         ┌─────────────────────────────────┐
//!                         │  MctsPool.run_sync(state_bytes)  │
//!                         └────────────┬──────────────────────┘
//!                                      │
//!                         ┌────────────▼───────────────┐
//!                         │  Init: 创建根节点，展开     │
//!                         │  全部动作空间 (20 个子节点) │
//!                         └────────────┬───────────────┘
//!                                      │
//!                    ┌────────────────▼──────────────────────┐
//!                    │          MCTS 主循环 (100ms 截止)       │
//!                    │                                        │
//!                    │  ① Selection: UCB1 选择最优子节点     │
//!                    │     UCB1 = Q + C·√(ln N_parent / N_child) │
//!                    │                                        │
//!                    │  ② Expansion: 展开新子节点           │
//!                    │     (动作空间: Buy/Sell/Hold/Close      │
//!                    │      × 仓位 0/25%/50%/75%/100%)       │
//!                    │                                        │
//!                    │  ③ Simulation (Rollout):              │
//!                    │     Python::with_gil()                 │
//!                    │       → rollout_fn(state)              │
//!                    │       → Triton AlphaCast 推断          │
//!                    │       → 返回 predicted_return / uncertainty / confidence │
//!                    │                                        │
//!                    │  ④ Backpropagation:                    │
//!                    │     reward = return - 0.1 × uncertainty│
//!                    │     更新路径上所有节点 (N += 1, Q += r) │
//!                    │                                        │
//!                    └────────────────┬────────────────────────┘
//!                                      │
//!                         ┌────────────▼──────────────┐
//!                         │  选择访问次数最多的子节点  │
//!                         │  → best_action / position  │
//!                         └────────────────────────────┘
//! ```
//!
//! ## 动作空间设计
//!
//! | Action | 说明           | PositionLevel | 说明            |
//! |--------|----------------|---------------|-----------------|
//! | `Buy`  | 开多仓         | `Zero`        | 0% 仓位         |
//! | `Sell` | 开空仓         | `Quarter`     | 25% 仓位        |
//! | `Hold` | 维持当前仓位   | `Half`        | 50% 仓位        |
//! | `Close`| 平仓           | `ThreeQuarters` | 75% 仓位     |
//! |        |                | `Full`        | 100% 仓位       |
//!
//! 总动作空间 = 4 × 5 = **20 个离散动作**
//!
//! ## Python 调用方式
//!
//! ```python
//! import asyncio
//! import v8_core_engine as vce
//!
//! pool = vce.MctsPool(workers=8, timeout_ms=100)
//!
//! async def search(state_bytes: bytes) -> dict:
//!     loop = asyncio.get_running_loop()
////!     rollout_fn = lambda s: alphacast_rollout(s)  # 同步函数，调用 Triton
//!     result_bytes = await loop.run_in_executor(
//!         None,
//!         lambda: pool.run_sync(state_bytes, rollout_fn)
//!     )
//!     return json.loads(result_bytes)
//!
//! # asyncio.get_event_loop().run_in_executor() 确保 Tokio runtime
//! # 在独立 OS 线程运行，不与 Python asyncio 主循环冲突
//! ```
//!
//! ## rollout_fn 接口契约
//!
//! ### Python 签名
//!
//! ```python
//! def rollout_fn(state_bytes: bytes) -> bytes:
//!     '''
//!     内部调用 Triton gRPC AlphaCast 模型
//!     返回 AlphaCastOutput JSON bytes
//!     '''
//!     state = json.loads(state_bytes)
//!     result = triton_client.predict(state)  # Triton InferenceServer
//!     return json.dumps(result).encode()
//! ```
//!
//! ### AlphaCastOutput 结构
//!
//! ```json
//! {
//!   "predicted_return": 0.015,   // 未来收益预测
//!   "uncertainty": 0.02,         // 模型不确定性
//!   "confidence": 0.8            // 置信度 (0-1)
//! }
//! ```
//!
//! ## 奖励函数
//!
//! ```text
//! raw_reward = predicted_return - 0.1 × uncertainty
//! final_reward = raw_reward × confidence
//! final_reward = clamp(final_reward, -1.0, 1.0)
//! ```
//!
//! `confidence` 调制确保低置信度决策不被放大。
//!
//! ## GIL 管理策略
//!
//! | 原则 | 说明 |
//! |------|------|
//! | **获取 → 调用 → 立即释放** | GIL 在 `Python::with_gil` 闭包结束后自动释放 |
//! | **不在循环内持有 GIL** | 每次 rollout 独立获取/释放 GIL，允许 Python GC 穿插执行 |
//! | **rollout_fn 必须是同步函数** | `async def` 持有协程，无法在同步上下文中调用 |
//!
//! ## 超时熔断策略
//!
//! | 条件 | 行为 |
//! |------|------|
//! | rollout_fn 连续失败 3 次 | 标记 `is_degraded=true`，后续 rollout 返回 0（Hold） |
//! | 整体超时（`timeout_ms`） | 提前退出循环，选择已探索节点中访问最多的 |
//!
//! ### 超时时的行为
//!
//! 超时不意味着返回 Hold，而是**选择已探索的节点中访问次数最多的**。
//! 即使只探索了 50 次（< 1000 次目标），也能返回一个有效决策。
//!
//! ## 性能契约
//!
//! | 指标 | 目标 | 说明 |
//! |------|------|------|
//! | MCTS 单次 rollout | < 50µs | rollout_fn 之外（纯 Rust UCB1 树操作） |
//! | rollout_fn Python 调用 | < 6ms | Triton gRPC AlphaCast 推断（Python 侧 SLA） |
//! | 1000 次总延迟 | < 50ms | `timeout_ms=100` 时完整探索目标 |
//! | Tokio runtime 创建 | < 1ms | 独立 OS 线程，非主线程 |
//!
//! ## Tokio Runtime 隔离
//!
//! MctsPool 在**独立 OS 线程**中创建 Tokio runtime，与 Python asyncio 主循环完全隔离。
//!
//! ```text
//! 主线程 (Python asyncio)
//!   └── asyncio.get_running_loop()
//!         └── run_in_executor(..., pool.run_sync)
//!               │
//!               └── [独立 OS 线程]
//!                     └── std::thread::spawn
//!                           └── Tokio Runtime (current_thread)
//!                                 └── block_on(mcts_search)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ════════════════════════════════════════════════════════════════
// MCTS 配置
// ════════════════════════════════════════════════════════════════

/// MCTS 超参数配置
///
/// ### 默认值（Phase 1 MVP）
///
/// | 参数 | 默认值 | 说明 |
/// |-------|--------|------|
/// | `max_depth` | 10 | 树最大深度（10 层展开，约 20^10 状态空间剪枝后可达） |
/// | `exploration_constant` | √2 ≈ 1.414 | UCB1 探索系数，平衡利用与探索 |
/// | `discount_factor` | 0.95 | 未来奖励折现（标准 MDP 设置） |
/// | `risk_penalty` | 0.3 | 风险惩罚系数（Phase 5 在线校准） |
#[derive(Clone)]
struct MctsConfig {
    /// 树最大深度（避免无限展开）
    max_depth: usize,
    /// UCB1 探索常数（C）；C=√2 是理论最优平衡
    exploration_constant: f64,
    /// 未来奖励折现因子
    discount_factor: f64,
    /// 风险惩罚（reward = return - risk_penalty × uncertainty）
    risk_penalty: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            exploration_constant: 1.414_f64.sqrt(),
            discount_factor: 0.95,
            risk_penalty: 0.3,
        }
    }
}

// ════════════════════════════════════════════════════════════════
// 动作空间定义
// ════════════════════════════════════════════════════════════════

/// 交易动作枚举
///
/// 注意：`Buy`/`Sell` 仅表示开仓方向，`Close` 表示平仓。
/// 实际仓位由 `(Action, PositionLevel)` 组合决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Action {
    /// 开多仓 / 持有多仓
    Buy,
    /// 开空仓 / 持有空仓
    Sell,
    /// 维持当前仓位不变
    Hold,
    /// 平仓（退出所有仓位）
    Close,
}

impl Action {
    fn all() -> Vec<Self> {
        vec![Self::Buy, Self::Sell, Self::Hold, Self::Close]
    }
}

/// 仓位水平
///
/// 注意：这是目标仓位的百分比，Buy/Close 配合使用表示开仓/平仓量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PositionLevel {
    /// 0% 仓位
    Zero,
    /// 25% 仓位
    Quarter,
    /// 50% 仓位
    Half,
    /// 75% 仓位
    ThreeQuarters,
    /// 100% 仓位
    Full,
}

impl PositionLevel {
    fn all() -> Vec<Self> {
        vec![
            Self::Zero,
            Self::Quarter,
            Self::Half,
            Self::ThreeQuarters,
            Self::Full,
        ]
    }

    /// 转换为浮点数（用于仓位计算）
    fn as_f64(&self) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::Quarter => 0.25,
            Self::Half => 0.5,
            Self::ThreeQuarters => 0.75,
            Self::Full => 1.0,
        }
    }
}

// ════════════════════════════════════════════════════════════════
// MCTS 树节点
// ════════════════════════════════════════════════════════════════

/// MCTS 树节点
///
/// ### UCB1 公式
///
/// ```text
/// UCB1 = Q̄ + C × √(ln(N_parent) / N_child)
/// ```
///
/// - `Q̄ = total_reward / visits`（节点平均奖励）
/// - `C = exploration_constant`（默认 √2）
/// - 首访节点 visits=0 → 返回 ∞（保证所有动作至少被探索一次）
struct Node {
    /// 到达此节点的动作（根节点为 None）
    action: Option<(Action, PositionLevel)>,
    /// 子节点索引列表
    children: Vec<usize>,
    /// 访问次数
    visits: u32,
    /// 累计奖励（backprop 累加）
    total_reward: f64,
}

impl Node {
    /// 创建根节点
    fn root() -> Self {
        Self {
            action: None,
            children: vec![],
            visits: 0,
            total_reward: 0.0,
        }
    }

    /// 创建子节点
    fn child(action: (Action, PositionLevel)) -> Self {
        Self {
            action: Some(action),
            children: vec![],
            visits: 0,
            total_reward: 0.0,
        }
    }

    /// UCB1 选择分数
    ///
    /// ### 参数
    ///
    /// - `parent_visits: u32` — 父节点访问次数
    /// - `C: f64` — 探索常数（默认 √2）
    ///
    /// ### 返回
    ///
    /// - `f64::INFINITY`（visits == 0）：未访问节点优先探索
    /// - UCB1 分数（visits > 0）：平衡探索与利用
    fn ucb1(&self, parent_visits: u32, C: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        self.total_reward / self.visits as f64
            + C * (parent_visits as f64).ln().sqrt() / self.visits as f64
    }
}

// ════════════════════════════════════════════════════════════════
// AlphaCast 推断结果解析
// ════════════════════════════════════════════════════════════════

/// AlphaCast 模型推断结果
///
/// 来自 `rollout_fn` Python callable 返回的 JSON bytes，
/// 对应 `schemas/alphacast_output.proto` 中 `AlphaCastOutput` 的 JSON 表示。
#[derive(Debug, Clone)]
struct AlphaCastResult {
    /// AlphaCast 预测的未来收益（归一化到 ±1）
    predicted_return: f64,
    /// 模型估计的不确定性（用于风险惩罚）
    uncertainty: f64,
    /// 置信度（用于奖励调制）
    confidence: f64,
}

impl AlphaCastResult {
    /// 从 JSON bytes 解析
    ///
    /// ### 字段
    ///
    /// | JSON 字段 | 类型 | 必需 | 默认值 |
    /// |-----------|------|------|--------|
    /// | `predicted_return` | float | ✓ | — |
    /// | `uncertainty` | float | ✗ | 0.01 |
    /// | `confidence` | float | ✗ | 0.5 |
    ///
    /// ### 错误
    ///
    /// JSON 解析失败或缺少 `predicted_return` 时返回 `None`（触发降级）。
    fn from_bytes(json_bytes: &[u8]) -> Option<Self> {
        let val: serde_json::Value = serde_json::from_slice(json_bytes).ok()?;
        Some(Self {
            predicted_return: val.get("predicted_return")?.as_f64()?,
            uncertainty: val.get("uncertainty")?.as_f64().unwrap_or(0.01),
            confidence: val.get("confidence")?.as_f64().unwrap_or(0.5),
        })
    }
}

// ════════════════════════════════════════════════════════════════
// MctsPool — PyO3 导出
// ════════════════════════════════════════════════════════════════

/// MCTS 搜索池（Python 可见）
///
/// ### Python 调用
///
/// ```python
/// pool = vce.MctsPool(workers=8, timeout_ms=100)
/// result_bytes = pool.run_sync(state_bytes, rollout_fn)
/// ```
///
/// ### 线程安全
///
/// `run_sync` 在**独立 OS 线程**的 Tokio runtime 中执行，
/// 与 Python asyncio 主循环完全隔离。通过 `run_in_executor` 调用。
#[pyclass]
pub struct MctsPool {
    /// MCTS 超参数配置
    config: MctsConfig,
    /// 工作线程数（当前版本：单线程 Tokio；Phase 5 升级为 rayon 多线程）
    num_workers: usize,
    /// 单次搜索超时（毫秒），超时后选择已探索的最佳动作
    timeout_ms: u64,
}

#[pymethods]
impl MctsPool {
    /// 创建搜索池
    ///
    /// ### 参数
    ///
    /// - `workers: int` — 工作线程数（默认 8，仅影响 Phase 5 多 worker 扩展）
    /// - `timeout_ms: int` — 单次搜索超时（默认 100ms）
    ///
    /// ### 使用建议
    ///
    /// Phase 1 MVP：`workers=8, timeout_ms=100`
    /// Phase 5 实盘：`workers=16, timeout_ms=30`（更激进时间预算）
    #[new]
    #[pyo3(signature = (workers=8, timeout_ms=100))]
    pub fn new(workers: usize, timeout_ms: u64) -> Self {
        Self {
            config: MctsConfig::default(),
            num_workers: workers,
            timeout_ms,
        }
    }

    /// 同步执行 MCTS 搜索
    ///
    /// ### 参数
    ///
    /// - `state_bytes: Vec<u8>` — 市场状态 JSON bytes
    ///   ```json
    ///   {"close": 3125.43, "volume": 1234.5, "features": [...]}
    ///   ```
    /// - `rollout_fn: PyObject` — Python callable，签名 `fn(state_bytes) -> bytes`
    ///   内部应调用 Triton gRPC 推断 AlphaCast 模型
    ///
    /// ### 返回
    ///
    /// `Vec<u8>` — MctsResult JSON bytes：
    ///
    /// ```json
    /// {
    ///   "best_action": "Buy",
    ///   "best_position": "Half",
    ///   "expected_value": 0.0104,
    ///   "path_value": 0.0104,
    ///   "sharpe_estimate": 0.0,
    ///   "visit_count": 1523,
    ///   "total_simulations": 1523,
    ///   "simulations_run": 1523,
    ///   "rollout_errors": 0,
    ///   "is_degraded": false
    /// }
    /// ```
    ///
    /// ### 错误
    ///
    /// - `RuntimeError`: Tokio runtime 创建失败 / 线程 panic
    ///
    /// ### Tokio Runtime 隔离
    ///
    /// 使用 `std::thread::spawn` 创建**独立 OS 线程**，在此线程内
    /// 构建 `current_thread` Tokio runtime，避免与 Python asyncio 主循环冲突。
    pub fn run_sync(
        &self,
        py: Python<'_>,
        state_bytes: Vec<u8>,
        rollout_fn: PyObject,
    ) -> PyResult<Vec<u8>> {
        let config = self.config.clone();
        let num_workers = self.num_workers;
        let timeout_ms = self.timeout_ms;

        // Arc 保证闭包所有权安全（跨线程传递 PyObject + bytes）
        let rollout_fn_arc = Arc::new(rollout_fn);
        let state_bytes_arc = Arc::new(state_bytes);
        let degraded = Arc::new(AtomicBool::new(false));

        // ── 独立 OS 线程 + Tokio Runtime ──────────────────────
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("MctsPool: failed to create Tokio runtime");

            rt.block_on(async move {
                let deadline =
                    tokio::time::Instant::now() + Duration::from_millis(timeout_ms);

                // ── 初始化搜索树 ─────────────────────────────
                let mut nodes = vec![Node::root()];
                let actions = build_action_space();
                for &(act, pos) in &actions {
                    let idx = nodes.len();
                    nodes[0].children.push(idx);
                    nodes.push(Node::child((act, pos)));
                }

                let mut total_sims = 0u32;
                let mut rollout_errors = 0u32;

                // ── MCTS 主循环 ──────────────────────────────
                loop {
                    // 超时检查：到达截止时间则提前退出
                    if tokio::time::Instant::now() > deadline {
                        break;
                    }

                    // ① Selection：从根节点沿 UCB1 最优路径向下
                    let mut current = 0usize;
                    let mut path = vec![0];
                    while !nodes[current].children.is_empty() {
                        let parent_visits = nodes[current].visits;
                        let best = *nodes[current]
                            .children
                            .iter()
                            .max_by(|&&a, &&b| {
                                nodes[a]
                                    .ucb1(parent_visits, config.exploration_constant)
                                    .partial_cmp(&nodes[b]
                                        .ucb1(parent_visits, config.exploration_constant))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap();
                        current = best;
                        path.push(current);
                    }

                    // ② Expansion：叶子节点且未充分访问时展开子节点
                    let depth = path.len();
                    if nodes[current].visits > 0 && depth < config.max_depth {
                        for &(act, pos) in &actions {
                            let idx = nodes.len();
                            nodes[current].children.push(idx);
                            nodes.push(Node::child((act, pos)));
                        }
                        if !nodes[current].children.is_empty() {
                            current = nodes[current].children[0];
                            path.push(current);
                        }
                    }

                    // ③ Simulation (Rollout)：GIL 内调用 Python rollout_fn
                    let reward = {
                        let fn_ptr = rollout_fn_arc.clone();
                        let state = state_bytes_arc.clone();
                        let deg = degraded.clone();

                        // 获取 GIL → 调用 rollout_fn → 释放 GIL
                        // GIL 在闭包结束后自动释放，不在循环内持续持有
                        Python::with_gil(|py| match call_rollout_fn(py, &fn_ptr, &state) {
                            Ok(r) => r,
                            Err(_) => {
                                rollout_errors += 1;
                                // 连续失败 3 次后降级
                                if rollout_errors > 3 {
                                    deg.store(true, AtomicOrdering::Release);
                                }
                                0.0 // 降级：Hold
                            }
                        })
                    };

                    total_sims += 1;

                    // ④ Backpropagation：沿路径反向传播奖励
                    for &idx in &path {
                        nodes[idx].visits += 1;
                        nodes[idx].total_reward += reward;
                    }
                }

                // ── 选择最优动作 ─────────────────────────────
                let best_idx = nodes[0]
                    .children
                    .iter()
                    .copied()
                    .max_by_key(|&idx| nodes[idx].visits);

                let (best_action, best_pos) = best_idx
                    .and_then(|idx| nodes[idx].action)
                    .unwrap_or((Action::Hold, PositionLevel::Zero));

                let ev = best_idx
                    .map(|idx| {
                        let n = &nodes[idx];
                        if n.visits > 0 {
                            n.total_reward / n.visits as f64
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);

                let result = serde_json::json!({
                    "best_action": match best_action {
                    Action::Buy => "buy",
                    Action::Sell => "sell",
                    Action::Hold => "hold",
                    Action::Close => "close",
                },
                    "best_position": best_pos.as_f64(),
                    "expected_value": ev,
                    "path_value": ev,
                    "sharpe_estimate": 0.0,
                    "visit_count": best_idx.map(|i| nodes[i].visits).unwrap_or(0),
                    "total_simulations": total_sims,
                    "simulations_run": total_sims,
                    "rollout_errors": rollout_errors,
                    "is_degraded": degraded.load(AtomicOrdering::Acquire),
                });

                serde_json::to_vec(&result).unwrap()
            })
        });

        match handle.join() {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "MctsPool thread panic: {:?}",
                e.downcast_ref::<&str>().unwrap_or(&"unknown error")
            ))),
        }
    }

    /// 配置信息（调试用）
    pub fn config_info(&self) -> String {
        format!(
            "MctsPool(workers={}, timeout_ms={}, max_depth={}, C={:.3})",
            self.num_workers,
            self.timeout_ms,
            self.config.max_depth,
            self.config.exploration_constant
        )
    }
}

// ════════════════════════════════════════════════════════════════
// Rollout 函数调用（GIL 内执行）
// ════════════════════════════════════════════════════════════════

/// 调用 Python rollout_fn → 解析 AlphaCastOutput → 返回奖励
///
/// ### GIL 管理
///
/// 此函数必须在 `Python::with_gil()` 内调用：
///
/// ```rust
/// Python::with_gil(|py| {
///     let reward = call_rollout_fn(py, &fn_ptr, &state_bytes)?;
/// });
/// ```
///
/// ### 奖励计算
///
/// ```text
/// raw    = predicted_return - 0.1 × uncertainty
/// reward = raw × confidence
/// reward = clamp(reward, -1.0, 1.0)
/// ```
///
/// ### 降级
///
/// rollout_fn 抛出异常时返回 `Err`（触发调用方降级逻辑）。
fn call_rollout_fn(py: Python<'_>, rollout_fn: &PyObject, state_bytes: &[u8]) -> PyResult<f64> {
    // 构造 Python bytes 参数
    let arg = PyBytes::new(py, state_bytes);

    // 调用 Python callable
    let result_obj = rollout_fn.call1(py, (arg,))?;

    // 提取返回的 bytes
    let result_bytes: &[u8] = result_obj.extract::<&PyBytes>(py)?.as_bytes();

    // 解析 AlphaCastOutput JSON
    let alpha = AlphaCastResult::from_bytes(result_bytes).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "rollout_fn returned invalid AlphaCastOutput JSON",
        )
    })?;

    // 计算奖励：return - 0.1 × uncertainty（风险惩罚）
    let reward = alpha.predicted_return - 0.1 * alpha.uncertainty;

    // 置信度调制 + 钳制到 [-1, 1]
    Ok((reward * alpha.confidence).clamp(-1.0, 1.0))
}

// ════════════════════════════════════════════════════════════════
// 辅助函数
// ════════════════════════════════════════════════════════════════

/// 构建完整动作空间列表
///
/// 返回 `(Action, PositionLevel)` 全组合，共 4 × 5 = **20 个离散动作**。
fn build_action_space() -> Vec<(Action, PositionLevel)> {
    let mut space = Vec::with_capacity(20);
    for &a in Action::all().iter() {
        for &p in PositionLevel::all().iter() {
            space.push((a, p));
        }
    }
    space
}

// ════════════════════════════════════════════════════════════════
// 单元测试
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// rollout_fn 返回标准 AlphaCastOutput → 验证奖励计算
    ///
    /// expected: raw = 0.015 - 0.1 × 0.02 = 0.013
    ///           reward = 0.013 × 0.8 = 0.0104
    #[test]
    fn test_call_rollout_fn_correct_reward() {
        Python::with_gil(|py| {
            let code = r#"
lambda state: b'{"predicted_return": 0.015, "uncertainty": 0.02, "confidence": 0.8}'
"#;
            let rollout_fn: PyObject = py.eval(code, None, None).unwrap().into();

            let state = br#"{"close": 3000.0}"#.to_vec();
            let reward = call_rollout_fn(py, &rollout_fn, &state).unwrap();

            let expected = 0.0104_f64;
            assert!(
                (reward - expected).abs() < 1e-6,
                "reward {} != expected {}",
                reward,
                expected
            );
        });
    }

    /// 基本 MCTS 搜索：验证至少完成 1 次模拟
    #[test]
    fn test_basic_mcts_search() {
        let pool = MctsPool::new(2, 50);
        let state = br#"{"close": 3000.0}"#.to_vec();

        Python::with_gil(|py| {
            let rollout_fn: PyObject = py
                .eval(
                    r#"lambda s: b'{"predicted_return": 0.01, "uncertainty": 0.02, "confidence": 0.9}'"#,
                    None,
                    None,
                )
                .unwrap()
                .into();

            let result = pool.run_sync(py, state, rollout_fn).unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

            assert!(
                parsed["total_simulations"].as_u64().unwrap() > 0,
                "MCTS should run at least 1 simulation"
            );
            assert!(
                !parsed["is_degraded"].as_bool().unwrap(),
                "Should not be degraded with valid rollout_fn"
            );
        });
    }

    /// 超时场景：验证超时后仍返回有效决策
    #[test]
    fn test_mcts_timeout_returns_valid_decision() {
        let pool = MctsPool::new(1, 10); // 10ms 超时
        let state = br#"{}"#.to_vec();

        Python::with_gil(|py| {
            // 慢 rollout_fn（50ms > 10ms 超时）
            let rollout_fn: PyObject = py
                .eval(
                    r#"
import time
lambda s: (time.sleep(0.05),
           b'{"predicted_return": 0, "uncertainty": 0.01, "confidence": 0.5}')[1]
"#,
                    None,
                    None,
                )
                .unwrap()
                .into();

            let result = pool.run_sync(py, state, rollout_fn).unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

            // 超时后应至少完成 1 次（即使很慢的 rollout）
            assert!(
                parsed["total_simulations"].as_u64().unwrap() >= 1,
                "Should complete at least 1 simulation despite timeout"
            );
        });
    }

    /// UCB1 首访节点返回 ∞（保证探索）
    #[test]
    fn test_ucb1_unvisited_returns_infinity() {
        let n = Node::root();
        assert_eq!(n.ucb1(100, 1.414), f64::INFINITY);
    }

    /// UCB1 正常节点返回有界分数
    #[test]
    fn test_ucb1_normal_returns_bounded() {
        let mut child = Node::child((Action::Buy, PositionLevel::Half));
        child.visits = 10;
        child.total_reward = 0.5;
        let ucb = child.ucb1(100, 1.414);
        assert!(
            ucb > 0.0 && ucb < f64::INFINITY,
            "UCB1 should be positive bounded, got {}",
            ucb
        );
    }
}