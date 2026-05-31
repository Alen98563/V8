//! # Prost 构建脚本 — Protobuf → Rust 类型生成
//!
//! ## 功能
//!
//! `cargo build` 时自动执行，将 `schemas/` 目录下的 `.proto` 文件编译为
//! Rust 类型，生成代码写入 `$OUT_DIR/v8.*.rs`，由 `proto_types.rs` 通过
//! `include!()` 宏导入。
//!
//! ## 生成配置
//!
//! - **serde 支持**：所有生成类型自动派生 `Serialize + Deserialize`
//! - **prost 版本**：0.12（通过 `[build-dependencies]` 指定）
//! - **输出前缀**：`v8.{market|order|alpha|alphacast}.rs`
//!
//! ## 自动重编译
//!
//! 通过 `cargo:rerun-if-changed` 指令，仅在 proto 文件实际变更时触发
//! 重新生成，避免无意义的增量编译。
//!
//! ## Proto 文件清单
//!
//! | 文件 | 大小 | 用途 |
//! |------|------|------|
//! | `market_snapshot.proto` | — | 行情快照（Tick/OHLCV/OrderBook） |
//! | `unified_order.proto` | — | OKX V5 标准化订单报文 |
//! | `alpha_signal.proto` | — | Alpha 信号载体 |
//! | `alphacast_output.proto` | — | AlphaCast 模型输出 |
//!
//! ## 字段兼容性约定
//!
//! Protobuf 字段编号 **永不删除或修改语义**。如需废弃字段：
//! 使用 `reserved` 关键字保留编号，新增字段从下一个可用编号开始。

use std::io::Result;

fn main() -> Result<()> {
    // Proto 文件根目录（相对于 Cargo.toml）
    let proto_dir = "schemas";

    // ── Prost 编译配置 ─────────────────────────────────────────
    prost_build::Config::new()
        // 所有生成类型自动实现 serde 序列化，方便 JSON 互操作
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile_protos(
            &[
                &format!("{}/market_snapshot.proto", proto_dir),
                &format!("{}/unified_order.proto", proto_dir),
                &format!("{}/alpha_signal.proto", proto_dir),
                &format!("{}/alphacast_output.proto", proto_dir),
            ],
            &[proto_dir], // include 搜索路径
        )?;

    // ── 增量编译优化 ──────────────────────────────────────────
    // 仅在 proto 文件变更时重新触发 build.rs
    println!("cargo:rerun-if-changed=schemas/market_snapshot.proto");
    println!("cargo:rerun-if-changed=schemas/unified_order.proto");
    println!("cargo:rerun-if-changed=schemas/alpha_signal.proto");
    println!("cargo:rerun-if-changed=schemas/alphacast_output.proto");

    Ok(())
}