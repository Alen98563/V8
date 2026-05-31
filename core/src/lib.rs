use pyo3::prelude::*;

// ── 模块声明 ──────────────────────────────────────────────────
// 每个模块对应 phase1 计划中的一个 T-x 任务
// 模块间禁止直接循环依赖：buffer → exec_channels 通过 JSON bytes 解耦

mod buffer;          // T1-1: ShmBridge — mmap2 共享内存 + DLPack 零拷贝
mod feature_engine;  // T1-2: FeatureEngine — 50 维微观市场特征引擎
#[cfg(feature = "protobuf")]
mod exec_channels;   // T1-3: OkxChannel — HMAC-SHA256 签名 + Token Bucket 限流
mod order_state;     // T1-4: OrderFSM — 编译期安全订单状态机
mod mcts_core;       // T4-1: MctsPool — Tokio 异步蒙特卡洛树搜索池
#[cfg(feature = "protobuf")]
mod okx_ws;          // T0-2: OkxWsClient — OKX WebSocket 实时行情接入

/// Protobuf 生成类型（需 `--features protobuf` 编译）
///
/// 包含 prost-build 从 schemas/*.proto 自动生成的 Rust 结构体：
///   - MarketSnapshot (market_snapshot.proto)
///   - UnifiedOrder (unified_order.proto)
///   - AlphaSignal (alpha_signal.proto)
///   - AlphaCastOutput (alphacast_output.proto)
///
/// 启用此 feature 后，FeatureEngine.on_tick() 将直接解析 Protobuf bytes
/// 而非 JSON 降级路径，实现零反序列化拷贝的热路径性能。
#[cfg(feature = "protobuf")]
mod proto_types;

// ── PyO3 模块注册 ─────────────────────────────────────────────

/// QTS V8 Core Engine Python 模块入口
///
/// 注册所有 PyO3 导出的类，Python 侧通过 `import v8_core_engine as vce`
/// 使用。注册顺序不影响功能，但按 T-x 编号排列便于维护。
///
/// ### 注册清单
///
/// | T-x   | 类名          | 源文件            | 用途                         | Feature Gate |
/// |-------|---------------|-------------------|------------------------------|-------------|
/// | T0-2  | `OkxWsClient` | okx_ws.rs         | OKX WebSocket 实时行情接入     | protobuf    |
/// | T1-1  | `ShmBridge`   | buffer.rs         | SHM 共享内存桥接 (DLPack)     | (always)    |
/// | T1-2  | `FeatureEngine` | feature_engine.rs | 50 维实时特征引擎            | (always)    |
/// | T1-3  | `OkxChannel`  | exec_channels.rs  | OKX REST 执行通道            | protobuf    |
/// | T1-4  | `OrderFSM`    | order_state.rs    | 编译期安全订单状态机         | (always)    |
/// | T4-1  | `MctsPool`    | mcts_core.rs      | Tokio 异步 MCTS 搜索池       | (always)    |
#[pymodule]
fn v8_core_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<buffer::ShmBridge>()?;
    m.add_class::<feature_engine::FeatureEngine>()?;
    m.add_class::<order_state::OrderFSM>()?;
    m.add_class::<mcts_core::MctsPool>()?;

    // Protobuf-gated modules (need `cargo build --features protobuf`)
    #[cfg(feature = "protobuf")]
    {
        m.add_class::<okx_ws::OkxWsClient>()?;
        m.add_class::<exec_channels::OkxChannel>()?;
    }

    Ok(())
}
