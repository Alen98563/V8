# V8 开发目录搭建 — DeepSeek-V4-Pro Phase 0

## 目标
根据 QTS V8 三份计划文档（phase1.md / quant_system_v8_structure.md / qts_v8_okx_devplan.html），搭建 `C:\Users\jerry\Desktop\V8\quant_system_v8` 完整开发目录，落位 DeepSeek-V4-Pro 专属的核心源文件。

## 交付物

### 目录结构 (30 个目录)
- `schemas/` — Protobuf 契约
- `core/src/` — Rust 核心模块 (PyO3 导出)
- `models/resnet/`, `models/alphacast/`, `models/mcts/` — AI 模型
- `ml/` — 机器学习重训
- 以及全部支撑目录: adapters, data, features, alpha/*, gating, risk, execution/*, orchestrator, harness, infra/*, scripts, tests/*

### DeepSeek 专属文件 (18 个源文件, 128KB)

**契约层 (Protobuf)**
| 文件 | 大小 | 说明 |
|------|------|------|
| `schemas/okx_order.proto` | 4.3KB | OKX V5 订单契约 (UnifiedOrder, OrderFSM 枚举, V8 扩展字段) |
| `schemas/market_data.proto` | 4.9KB | 行情数据契约 (TradeTick, OrderBookSnapshot, Bar5m, FeatureVector, FusedFeature) |
| `schemas/alpha_signal.proto` | 1.7KB | Alpha 信号 + AlphaCast 输出契约 |

**Rust 核心模块 (PyO3)**
| 文件 | 大小 | 说明 |
|------|------|------|
| `Cargo.toml` | 733B | Rust 依赖: pyo3(abi3-py312), tokio, ring, memmap2, prost, rayon |
| `core/src/lib.rs` | 698B | PyO3 模块注册入口 |
| `core/build.rs` | 828B | Prost 编译脚本 |
| `core/src/buffer.rs` | 10.2KB | SHM 环形缓冲区 (mmap2 + DLPack 零拷贝) |
| `core/src/feature_engine.rs` | 24.5KB | 50 维特征引擎 (预计算 + O(1) 闭盘修正) |
| `core/src/okx_router.rs` | 10.1KB | OKX HMAC-SHA256 签名 + Token Bucket 限流 |
| `core/src/order_state.rs` | 10.2KB | 订单 FSM (编译期安全枚举转换) |
| `core/src/mcts_core.rs` | 13.4KB | Tokio 并行 MCTS (UCB1 + Sharpe 奖励) |

**Python AI 模型**
| 文件 | 大小 | 说明 |
|------|------|------|
| `pyproject.toml` | 941B | uv/maturin 构建配置 |
| `models/resnet/resnet_encoder.py` | 8.9KB | ResNet 6残差块 + MultiScaleConv1D + FeatureFusion |
| `models/alphacast/alphacast_model.py` | 13.6KB | AlphaCast Transformer 6L8H + Temperature Scaling |
| `ml/meta_labeler.py` | 15.0KB | LightGBM MetaLabeler (97.4% 多数类防御) |
| `models/mcts/mcts_config.py` | 916B | MCTS 超参配置 |

## 关键设计决策

1. **Rust 模块位于 `core/src/`** — 符合 Cargo 标准，maturin 编译为 `v8_core_engine.pyd`
2. **Protobuf 字段全部 string 存储 px/sz** — 防止 float 精度损失 (OKX API 规范要求)
3. **特征引擎 50 维 MVP** — Phase 2 扩展到 178d 融合向量
4. **MCTS 独立 Tokio runtime** — 避免与 Python asyncio 嵌套死锁
5. **MetaLabeler 自动 scale_pos_weight** — 动态平衡 97.4% 负样本偏置

## 后续 (Phase 1+)
- Qwen 负责写入的文件 (adapters, data/shm_bridge.py, gating, harness, orchestrator, execution Python 侧)
- `maturin develop --release` 编译验证
- Python 侧 `import v8_core_engine` 集成测试
