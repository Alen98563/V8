//! # Protobuf 生成类型 — prost-build 编译时产物
//!
//! ## 用途
//!
//! 本模块提供由 `build.rs` 通过 `prost-build` 从 `schemas/*.proto` 自动生成的
//! Rust 类型定义。所有类型自动实现 `prost::Message`（序列化/反序列化）和
//! `serde::Serialize + Deserialize`（JSON 互操作）。
//!
//! ## 编译机制
//!
//! 1. `cargo build` → 触发 `build.rs`
//! 2. `build.rs` → 调用 `prost_build::compile_protos()` 编译 proto 文件
//! 3. 生成代码写入 `$OUT_DIR/v8.*.rs`
//! 4. 本模块通过 `include!()` 宏在编译时内联生成的代码
//!
//! ## 包含的类型
//!
//! | 生成文件 | Proto 源 | 主要类型 | 用途 |
//! |----------|----------|----------|------|
//! | `v8.market.rs` | `market_snapshot.proto` | `MarketSnapshot`, `OrderbookSnapshot` | 行情快照（Tick/OHLCV/OrderBook） |
//! | `v8.order.rs` | `unified_order.proto` | `UnifiedOrder`, `CancelRequest` | 标准化订单报文 |
//! | `v8.alpha.rs` | `alpha_signal.proto` | `AlphaSignal` | Alpha 信号（含 trace_id） |
//! | `v8.alphacast.rs` | `alphacast_output.proto` | `AlphaCastOutput` | AlphaCast 模型推断输出 |
//!
//! ## Feature Gate
//!
//! 本模块仅在 `--features protobuf` 启用时编译。无此 feature 时，
//! `FeatureEngine.on_tick()` 退化为 JSON 解析路径。
//!
//! ## 字段兼容性
//!
//! Protobuf 字段编号 **永不修改**。新增字段只能追加新编号，保持向后兼容。

// prost 生成的 Rust 类型需要此 trait 在作用域内
use prost::Message;

// ── 编译时 include（由 build.rs 生成到 $OUT_DIR） ─────────────
// 注意：这些文件在 cargo check 时不可见，仅在 cargo build 时生成

/// 来自 schemas/market_snapshot.proto
///
/// 包含：MarketSnapshot（主结构体）、OrderbookSnapshot（嵌套结构体）、
///       双轨 Tick 字段（trade / book_ticker）、市场状态枚举
include!(concat!(env!("OUT_DIR"), "/v8.market.rs"));

/// 来自 schemas/unified_order.proto
///
/// 包含：UnifiedOrder（下单请求，对齐 OKX V5 API 字段）、
///       CancelRequest（撤单请求）、订单类型枚举
include!(concat!(env!("OUT_DIR"), "/v8.order.rs"));

/// 来自 schemas/alpha_signal.proto
///
/// 包含：AlphaSignal（信号载体，含 trace_id / signal_value / confidence）
include!(concat!(env!("OUT_DIR"), "/v8.alpha.rs"));

/// 来自 schemas/alphacast_output.proto
///
/// 包含：AlphaCastOutput（多步收益预测、置信度、模型版本号）
include!(concat!(env!("OUT_DIR"), "/v8.alphacast.rs"));