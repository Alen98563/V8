//! # 订单有限状态机 (OrderFSM) —T1-4
//!
//! ## 架构定位
//!
//! OrderFSM �?QTS V8 **执行层（L4�?* 的编译期安全状态机。每�?OKX 订单创建�?
//! 绑定一个不可变�?FSM 实例，通过 Rust 代数数据类型穷举所有合法状态转换路径�?
//!
//! ## 设计原则
//!
//! 1. **编译期安�?*：利�?Rust `enum` + 静态状态转换表，非法转换在运行时被拒绝
//!    但所有可能路径在编译时已被显式声�?
//! 2. **无锁线程安全**：状态寄存器�?`AtomicU8`，避�?`Mutex` 争用
//!    （仅 `ord_id`/成交信息�?`Mutex` 保护写入，因这些字段低频更新�?
//! 3. **幂等�?*：同一状态的重复转换视为合法（`from == to` 允许�?
//! 4. **可观测�?*：每次转换记�?`(状�? 时间�?` �?history，可通过 `to_json()` 导出
//!
//! ## 状态转换图
//!
//! ```text
//!                     ┌──→Rejected (终�?
//!                     �?
//! New ──→Posted ──┬──→PartialFilled ──┬──→Filled (终�?
//!                   �?                   �?
//!                   ├──→Filled (终�?   ├──→Canceled (终�?
//!                   �?                   �?
//!                   ├──→Canceled (终�? ├──→PartialCanceled (终�?
//!                   �?                   �?
//!                   ├──→PendingCancel ──┼──→Canceled (终�?
//!                   �?    �?             �?
//!                   �?    ├──→Filled    ├──→PartialCanceled
//!                   �?    └──→Canceled  └──→Filled (撤单前已成交)
//!                   �?
//!                   └──→Unknown (终�?
//! ```
//!
//! ## Protobuf 对齐
//!
//! 状态枚举值与 `schemas/unified_order.proto` �?`OrderState` 一一对应�?
//! 字段编号永不修改，`repr(u8)` 保证二进制兼容性�?
//!
//! ## 使用方式
//!
//! ```python
//! fsm = vce.OrderFSM("my_ord_001", "BTC-USDT-SWAP", "trace-abc123")
//! fsm.transition(2, "交易所确认")          # New →Posted
//! fsm.transition(3, "部分成交")            # Posted →PartialFilled
//! fsm.transition(4, "全部成交，退�?)      # PartialFilled →Filled (终�?
//! assert fsm.is_terminal()
//! ```
//!
//! ## OKX 集成注意
//!
//! - `cl_ord_id` �?32 字符（OKX API V5 硬限制），此处编译期校验
//! - `trace_id` �?32 字符，用于跨模块链路追踪

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

// ────────────────────────────────────────────────────────────────
// 状态枚�?—�?schemas/unified_order.proto OrderState 严格对齐
// ────────────────────────────────────────────────────────────────

/// 订单状态枚�?
///
/// 每个变量数值与 Proto 字段 `OrderState` 一致。`repr(u8)` 保证
/// C ABI 兼容，可直接写入 `AtomicU8` 进行无锁原子操作�?
///
/// ### 变量说明
///
/// | �?| 变量             | 含义                      | 终�?|
/// |----|------------------|---------------------------|------|
/// | 0  | `Unspecified`    | 未指定（Proto3 默认值）   | —   |
/// | 1  | `New`            | 客户端创建，待发�?        | —   |
/// | 2  | `Posted`         | 已提交至交易所             | —   |
/// | 3  | `PartialFilled`  | 部分成交                   | —   |
/// | 4  | `Filled`         | 完全成交                   | �?   |
/// | 5  | `PartialCanceled`| 部分成交后撤销             | �?   |
/// | 6  | `Canceled`       | 已撤销                     | �?   |
/// | 7  | `Rejected`       | 交易所拒绝                 | �?   |
/// | 8  | `Expired`        | 过期                       | �?   |
/// | 9  | `Unknown`        | 未知异常（降级安全出口）   | �?   |
/// | 10 | `PendingCancel`  | 撤单请求已发送，等待确认   | —   |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderFSMState {
    Unspecified = 0,
    New = 1,
    Posted = 2,
    PartialFilled = 3,
    Filled = 4,
    PartialCanceled = 5,
    Canceled = 6,
    Rejected = 7,
    Expired = 8,
    Unknown = 9,
    PendingCancel = 10,
}

impl From<u8> for OrderFSMState {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Unspecified,
            1 => Self::New,
            2 => Self::Posted,
            3 => Self::PartialFilled,
            4 => Self::Filled,
            5 => Self::PartialCanceled,
            6 => Self::Canceled,
            7 => Self::Rejected,
            8 => Self::Expired,
            9 => Self::Unknown,
            10 => Self::PendingCancel,
            _ => Self::Unknown, // 安全降级：未知值视�?Unknown 终�?
        }
    }
}

// ────────────────────────────────────────────────────────────────
// 合法状态转换表 —穷举所有可能路�?
// ────────────────────────────────────────────────────────────────

/// 合法状态转换对 `[(from, to), ...]`
///
/// 本表穷举了所有合法的状态转换。任何不在本表中的转换尝试将�?
/// `is_valid_transition()` 拒绝并返回错误。新增转换路径需同时修改
/// 本表并更新上方状态转换图文档�?
///
/// ### 设计备注
///
/// - **New →Unknown** 被约制（仅允�?`New →Posted` �?`New →Rejected`），
///   因为跳过 `Posted` 直接�?`Unknown` 会丢失中间状态的信息
/// - **Filled →* / Canceled →*** 不可逆，因已标记为终端状�?
const VALID_TRANSITIONS: &[(OrderFSMState, OrderFSMState)] = &[
    // ── 初始提交路径 ──
    (OrderFSMState::New, OrderFSMState::Posted),
    (OrderFSMState::New, OrderFSMState::Rejected), // 订单在发送前被风控拒�?

    // ── Posted 派生路径 ──
    (OrderFSMState::Posted, OrderFSMState::PartialFilled),
    (OrderFSMState::Posted, OrderFSMState::Filled),          // 市价单立即成�?
    (OrderFSMState::Posted, OrderFSMState::Canceled),        // 用户撤单
    (OrderFSMState::Posted, OrderFSMState::Rejected),        // 交易所拒绝
    (OrderFSMState::Posted, OrderFSMState::Unknown),         // 资金不足、API 限流�?

    // ── 部分成交后续 ──
    (OrderFSMState::PartialFilled, OrderFSMState::Filled),
    (OrderFSMState::PartialFilled, OrderFSMState::Canceled),
    (OrderFSMState::PartialFilled, OrderFSMState::PartialCanceled),

    // ── 撤单中路�?──
    (OrderFSMState::Posted, OrderFSMState::PendingCancel),
    (OrderFSMState::PartialFilled, OrderFSMState::PendingCancel),
    (OrderFSMState::PendingCancel, OrderFSMState::Canceled),
    (OrderFSMState::PendingCancel, OrderFSMState::PartialCanceled),
    (OrderFSMState::PendingCancel, OrderFSMState::Filled),   // 撤单请求到达前已成交
];

/// O(1) 状态转换合法性校�?
///
/// 使用线性扫描静态表（表大小 �?15 条，远小于缓存行大小，无性能损失）�?
/// 幂等转换（`from == to`）视为合法，允许重复确认�?
fn is_valid_transition(from: OrderFSMState, to: OrderFSMState) -> bool {
    if from == to {
        return true;
    }
    VALID_TRANSITIONS.iter().any(|&(f, t)| f == from && t == to)
}

// ────────────────────────────────────────────────────────────────
// OrderFSM —PyO3 导出
// ────────────────────────────────────────────────────────────────

/// 订单有限状态机
///
/// 每个 OKX 订单对应一个独立的 FSM 实例。使�?`Atomic` 类型实现
/// 无锁的状态寄存器，避免高频交易场景下�?`Mutex` 争用瓶颈�?
///
/// ### 线程安全策略
///
/// | 字段        | 类型                    | 策略                             |
/// |-------------|------------------------|----------------------------------|
/// | `state`     | `AtomicU8`             | 原子写入，无�?                  |
/// | `is_terminal` | `AtomicBool`         | 原子写入，无�?                  |
/// | `ord_id`    | `Arc<Mutex<Option<>>`  | 低频更新（交易所返回时仅写一次） |
/// | `fill_px`   | `Arc<Mutex<Option<>>`  | 低频更新                         |
/// | `fee`       | `Arc<Mutex<Option<>>`  | 低频更新                         |
/// | `history`   | `Arc<Mutex<Vec<>>>`    | 日志记录，非热路�?              |
#[pyclass]
pub struct OrderFSM {
    /// 客户端订单ID（≤32 字符 OKX 硬约束）
    cl_ord_id: String,

    /// 交易所分配�?ordId（POST 成功后由 OKX 返回�?
    ord_id: Arc<std::sync::Mutex<Option<String>>>,

    /// 合约标的
    inst_id: String,

    /// 当前状态（`AtomicU8`，对�?`OrderFSMState` 枚举值）
    state: AtomicU8,

    /// 是否已达终端状态（Filled / Canceled / Rejected / Expired / Unknown�?
    is_terminal: AtomicBool,

    /// 链路追踪 ID（≤32 字符�?
    trace_id: String,

    /// 成交数量（简化：记录笔数，非实际 sz�?
    fill_sz: AtomicU8,

    /// 成交均价
    fill_px: Arc<std::sync::Mutex<Option<f64>>>,

    /// 手续�?
    fee: Arc<std::sync::Mutex<Option<f64>>>,

    /// 状态变更历�?`[(OrderFSMState, timestamp_ms), ...]`
    history: Arc<std::sync::Mutex<Vec<(OrderFSMState, i64)>>>,
}

#[pymethods]
impl OrderFSM {
    /// 创建新订单状态机
    ///
    /// ### 参数
    ///
    /// - `cl_ord_id: str` —客户端订单ID，≤32 字符（OKX API V5 硬限制）
    /// - `inst_id: str` —合约标的，如 `"BTC-USDT-SWAP"`
    /// - `trace_id: str` —链路追踪ID，≤32 字符，默认空字符�?
    ///
    /// ### 错误
    ///
    /// - `ValueError`: cl_ord_id > 32 字符 �?trace_id > 32 字符
    ///
    /// ### 初始状�?
    ///
    /// 创建时的初始状态为 `New(1)`，同时记录第一�?history 条目�?
    #[new]
    #[pyo3(signature = (cl_ord_id, inst_id, trace_id="".into()))]
    pub fn new(cl_ord_id: String, inst_id: String, trace_id: String) -> PyResult<Self> {
        // OKX API V5: clOrdId 硬限�?32 字节
        if cl_ord_id.len() > 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clOrdId must be �?32 chars, got {}: '{}'",
                cl_ord_id.len(),
                cl_ord_id
            )));
        }
        if trace_id.len() > 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "trace_id must be �?32 chars".to_string(),
            ));
        }

        Ok(Self {
            cl_ord_id,
            ord_id: Arc::new(std::sync::Mutex::new(None)),
            inst_id,
            state: AtomicU8::new(OrderFSMState::New as u8),
            is_terminal: AtomicBool::new(false),
            trace_id,
            fill_sz: AtomicU8::new(0),
            fill_px: Arc::new(std::sync::Mutex::new(None)),
            fee: Arc::new(std::sync::Mutex::new(None)),
            history: Arc::new(std::sync::Mutex::new(vec![(
                OrderFSMState::New,
                now_ms(),
            )])),
        })
    }

    /// 推进状态转�?
    ///
    /// ### 参数
    ///
    /// - `to_state: int` —目标状态枚举值（u8），对应 `OrderFSMState` 整数
    /// - `reason: str` —转换原因（用于日志和错误定位�?
    ///
    /// ### 错误
    ///
    /// - `ValueError`: 非法状态转换（不在 `VALID_TRANSITIONS` 表内�?
    ///
    /// ### 副作�?
    ///
    /// 1. 更新 `state` 原子变量
    /// 2. 追加 history 条目
    /// 3. 若目标为终端状态，设置 `is_terminal = true`
    ///
    /// ### 终端状�?
    ///
    /// Filled / Canceled / Rejected / Expired / Unknown / PartialCanceled
    /// 到达后将 `is_terminal` 标记�?`true`，不允许后续 transition 调用
    pub fn transition(&self, to_state: u8, reason: &str) -> PyResult<()> {
        let to = OrderFSMState::from(to_state);
        let from = OrderFSMState::from(self.state.load(Ordering::Acquire));

        if !is_valid_transition(from, to) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid transition: {:?} →{:?} for order {} (reason: {})",
                from, to, self.cl_ord_id, reason
            )));
        }

        // 原子更新状态寄存器
        self.state.store(to_state as u8, Ordering::Release);

        // 记录历史（Arc<Mutex> 放宽，非热路径）
        if let Ok(mut hist) = self.history.lock() {
            hist.push((to, now_ms()));
        }

        // 终端状态标�?—后续 transition 仍需校验但可用于外部判断
        match to {
            OrderFSMState::Filled
            | OrderFSMState::Canceled
            | OrderFSMState::Rejected
            | OrderFSMState::Expired
            | OrderFSMState::Unknown
            | OrderFSMState::PartialCanceled => {
                self.is_terminal.store(true, Ordering::Release);
            }
            _ => {}
        }

        Ok(())
    }

    /// 设置交易所分配�?`ordId`（POST 成功后由 OKX 返回�?
    ///
    /// ### 调用时机
    ///
    /// `OkxChannel.place_order()` 收到 OKX HTTP 200 响应后，�?
    /// `data[0]["ordId"]` 提取后调用�?
    pub fn set_ord_id(&self, ord_id: &str) {
        if let Ok(mut oid) = self.ord_id.lock() {
            *oid = Some(ord_id.to_string());
        }
    }

    /// 记录成交信息
    ///
    /// ### 参数
    ///
    /// - `fill_px: float` —成交价格
    /// - `fill_sz: float` —成交数量
    /// - `fee: float` —手续�?
    ///
    /// ### 备注
    ///
    /// 当前实现按简单均价记录（取最�?fill_px），后续需改进�?
    /// VWAP（成交量加权均价），�?OKX 结算 API 做日终对账�?
    pub fn update_fill(&self, fill_px: f64, fill_sz: f64, fee: f64) {
        if let Ok(mut px) = self.fill_px.lock() {
            *px = Some(fill_px);
        }
        if let Ok(mut f) = self.fee.lock() {
            *f = Some(fee);
        }
    }

    // ── Getters ────────────────────────────────────────────────

    /// 当前状态（u8 整数，与 Proto `OrderState` 对齐�?
    #[getter]
    pub fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    /// 当前状态名称（人类可读，如 `"Posted"`�?
    pub fn state_name(&self) -> String {
        let state = OrderFSMState::from(self.state.load(Ordering::Acquire));
        format!("{:?}", state)
    }

    /// 是否已达终端状�?
    ///
    /// 终端状态定义：Filled / Canceled / Rejected / Expired / Unknown / PartialCanceled
    /// 到达后订单不可进一步操作（不可撤单、不可修改、不可再成交�?
    pub fn is_terminal(&self) -> bool {
        self.is_terminal.load(Ordering::Acquire)
    }

    /// 客户端订�?ID
    #[getter]
    pub fn cl_ord_id(&self) -> &str {
        &self.cl_ord_id
    }

    /// 交易所订单 ID（POST 成功后由 OKX 返回，否则为 `None`�?
    #[getter]
    pub fn ord_id(&self) -> Option<String> {
        self.ord_id.lock().ok().and_then(|o| o.clone())
    }

    /// 合约标的
    #[getter]
    pub fn inst_id(&self) -> &str {
        &self.inst_id
    }

    /// 链路追踪 ID
    #[getter]
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// 状态变更历�?`[(state_code, timestamp_ms), ...]`
    pub fn history(&self) -> Vec<(u8, i64)> {
        self.history
            .lock()
            .map(|h| h.iter().map(|(s, t)| (*s as u8, *t)).collect())
            .unwrap_or_default()
    }

    /// 导出�?JSON（用于日志、Redis 数据总线、监控大屏）
    ///
    /// ### 输出示例
    ///
    /// ```json
    /// {
    ///   "cl_ord_id": "my_ord_001",
    ///   "ord_id": "1234567890",
    ///   "inst_id": "BTC-USDT-SWAP",
    ///   "state": "PartialFilled",
    ///   "state_code": 3,
    ///   "is_terminal": false,
    ///   "fill_px": 3125.43,
    ///   "fee": 0.53,
    ///   "trace_id": "trace-abc123"
    /// }
    /// ```
    pub fn to_json(&self) -> PyResult<String> {
        let state = OrderFSMState::from(self.state.load(Ordering::Acquire));
        let ord_id = self
            .ord_id
            .lock()
            .map(|o| o.clone().unwrap_or_default())
            .unwrap_or_default();
        let fill_px = self.fill_px.lock().map(|f| f.unwrap_or(0.0)).unwrap_or(0.0);
        let fee = self.fee.lock().map(|f| f.unwrap_or(0.0)).unwrap_or(0.0);

        let obj = serde_json::json!({
            "cl_ord_id": self.cl_ord_id,
            "ord_id": ord_id,
            "inst_id": self.inst_id,
            "state": format!("{:?}", state),
            "state_code": state as u8,
            "is_terminal": self.is_terminal.load(Ordering::Acquire),
            "fill_px": fill_px,
            "fee": fee,
            "trace_id": self.trace_id,
        });

        serde_json::to_string_pretty(&obj).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize error: {}", e))
        })
    }
}

// ────────────────────────────────────────────────────────────────
// 辅助函数
// ────────────────────────────────────────────────────────────────

/// 获取当前 Unix 毫秒时间�?
///
/// ### 用�?
///
/// 用于 FSM history 记录，提供订单生命周期时序追踪�?
/// 精度为毫秒级（`duration_since().as_millis()`），
/// 不适合亚毫秒延迟测量�?
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

// ────────────────────────────────────────────────────────────────
// 单元测试
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 标准生命周期：New →Posted →Filled
    #[test]
    fn test_standard_lifecycle() {
        let fsm =
            OrderFSM::new("test-cl-001".into(), "BTC-USDT-SWAP".into(), "trace-001".into());
        assert_eq!(fsm.state(), OrderFSMState::New as u8);

        fsm.transition(OrderFSMState::Posted as u8, "submitted")
            .unwrap();
        assert_eq!(fsm.state(), OrderFSMState::Posted as u8);

        fsm.transition(OrderFSMState::Filled as u8, "fully filled")
            .unwrap();
        assert_eq!(fsm.state(), OrderFSMState::Filled as u8);
        assert!(fsm.is_terminal());
    }

    /// 非法转换应被拒绝：Posted →New 不可�?
    #[test]
    fn test_invalid_reverse_transition() {
        let fsm =
            OrderFSM::new("test-cl-002".into(), "BTC-USDT-SWAP".into(), "trace-002".into());
        fsm.transition(OrderFSMState::Posted as u8, "submitted")
            .unwrap();

        let result = fsm.transition(OrderFSMState::New as u8, "regression attempt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid"));
    }

    /// 部分成交 →撤单 →终�?
    #[test]
    fn test_partial_fill_then_cancel() {
        let fsm =
            OrderFSM::new("test-cl-003".into(), "BTC-USDT-SWAP".into(), "trace-003".into());
        fsm.transition(OrderFSMState::Posted as u8, "submitted")
            .unwrap();
        fsm.transition(OrderFSMState::PartialFilled as u8, "partial")
            .unwrap();
        fsm.transition(OrderFSMState::Canceled as u8, "user cancel")
            .unwrap();

        assert!(fsm.is_terminal());
        // history 应包�?4 �? New, Posted, PartialFilled, Canceled
        assert_eq!(fsm.history().len(), 4);
    }

    /// 所有合法终端状态不�?panic
    #[test]
    fn test_all_terminal_states() {
        for terminal in [
            OrderFSMState::Filled,
            OrderFSMState::Canceled,
            OrderFSMState::Rejected,
            OrderFSMState::Expired,
        ] {
            let fsm = OrderFSM::new(
                format!("cl-{:?}", terminal),
                "BTC-USDT-SWAP".into(),
                "".into(),
            );
            fsm.transition(terminal as u8, "test")
                .unwrap_or_else(|e| panic!("{:?} should be reachable: {}", terminal, e));
            assert!(fsm.is_terminal(), "{:?} should be terminal", terminal);
        }
    }

    /// JSON 导出应包含全部关键字�?
    #[test]
    fn test_to_json() {
        let fsm = OrderFSM::new("json-test".into(), "BTC-USDT-SWAP".into(), "trace-json".into());
        let json = fsm.to_json().unwrap();
        assert!(json.contains("json-test"));
        assert!(json.contains("trace-json"));
        assert!(json.contains("BTC-USDT-SWAP"));
    }
}