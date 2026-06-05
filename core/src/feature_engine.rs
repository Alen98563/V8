//! # 5m 特征引擎 (FeatureEngine) — T1-2
//!
//! ## 架构定位
//!
//! FeatureEngine 是 QTS V8 **数据层（L2-L3）** 的核心计算引擎。
//! 每接收一个 Tick（来自 OkxWsClient），实时更新滑动窗口缓冲区，
//! 在 5m K线闭盘时计算 **50 维微观市场特征**，通过 ShmBridge 写入
//! 共享内存供 Python AI 层（ResNet / AlphaCast）直接读取。
//!
//! ## 数据结构流
//!
//! ```text
//! OkxWsClient (adapters/okx_ws.rs)
//!   │
//!   │ 每 tick → MarketSnapshot Protobuf bytes
//!   │
//!   ├── FeatureEngine.on_tick(snapshot_bytes)
//!   │     │
//!   │     ├── parse_snapshot()     # Protobuf / JSON 双模解析
//!   │     │
//!   │     └── ingest_tick()        # 更新 price / vol / ts 环形缓冲
//!   │           │                  # 更新当前 5m Bar (OHLCV + bid/ask)
//!   │           │
//!   │           └── (累积 ticks 直到 5m 边界)
//!   │
//!   ├── FeatureEngine.on_5m_close()
//!   │     │
//!   │     ├── 关闭当前 Bar → 推入 bars VecDeque (容量 288 = 24h)
//!   │     │
//!   │     ├── compute_all_features()     # 计算 50 维特征
//!   │     │
//!   │     ├── 返回 BarCloseEvent JSON bytes (→ Redis Stream bar:close:events)
//!   │     │
//!   │     └── 下游: ShmBridge.push_snapshot(ts_ms, features[50])
//!   │
//!   └── FeatureEngine.get_features_50d() # 按需查询当前特征
//!         │
//!         └── → bytes (f32 × 50 = 200B) → ShmBridge 写入
//! ```
//!
//! ## 50 维特征体系
//!
//! | 维度 | 特征名 | 类型 | 窗口 | 说明 |
//! |------|--------|------|------|------|
//! | 0-4 | OHLCV | 当前Bar | 5m | 开盘/最高/最低/收盘/成交量 |
//! | 5 | OBI | 实时 | 全窗口 | 订单簿不平衡 |
//! | 6 | OFI | 实时 | 5m | 订单流不平衡 |
//! | 7 | Spread_Z | 实时 | 5m | 价差标准化 |
//! | 8 | Depth Imbalance | 最细Bar | 5m | 挂单深度非对称 |
//! | 9-12 | Momentum(1m,5m,15m,1h) | 收益率 | 多窗口 | 多周期价格动量 |
//! | 13-15 | HistoricalVol(5m,15m,1h) | 波动率 | 多窗口 | 年化波动率 (sqrt(252×288)) |
//! | 16 | Hurst | 分形 | 5m | 市场效率/持续性 (0,1) |
//! | 17 | VWAP Deviation | 偏离 | 5m | VWAP 相对偏离 |
//! | 18 | Realized Skewness | 矩 | 5m | 收益分布偏度 |
//! | 19 | Realized Kurtosis | 矩 | 5m | 收益分布峰度(超额) |
//! | 20 | Bid/Ask Ratio | 流动性 | 当时 | 最优买卖价比 |
//! | 21 | Vol Profile | 成交量 | 5m | 窗口成交量占比 |
//! | 22-27 | Velocity(10s,30s,1m,2m,5m,15m) | 速度 | 6窗口 | 多粒度价格速度 |
//! | 28 | Trade Intensity | 频率 | 实时 | 每秒交易次数 |
//! | 29 | Vol-Price Corr | 相关 | 5m | 量价相关性 |
//! | 30 | Price Acceleration | 加速度 | 60tick | 动量变化率 |
//! | 31-34 | LogReturn(1m,5m,15m,1h) | 对数收益率 | 多窗口 | 多周期对数收益 |
//! | 35 | Funding Rate | 资金费率 | 当前 | OKX 8h 资金费率 |
//! | 36 | Amihud | 非流动性 | 5m | Amihud 非流动性指标 |
//! | 37 | Vol Mean Reversion | 均值回归 | 交叉 | 短期vs长期波动率差异 |
//! | 38 | Bollinger Position | 渠道 | 5m | 价格在布林带中的位置 |
//! | 39 | Funding Rate 8h Cum | 累积 | 8h | 过去 8h 累计资金费率 |
//! | 40 | ATR | 波动范围 | 5m | 平均真实波幅 |
//! | 41-49 | reserved | — | — | Phase 2 开放给 ResNet 特征工厂 |
//!
//! ## 双模解析策略
//!
//! FeatureEngine 支持两种 Tick 数据源：
//!
//! | 模式 | 判断方式 | 解析器 | 适用场景 |
//! |------|---------|--------|----------|
//! | Protobuf | 首字节 ≠ `{` | `prost::Message::decode(MarketSnapshot)` | 生产（WebSocket → Protobuf） |
//! | JSON | 首字节 = `{` | `serde_json::from_slice(Value)` | 开发 / CI 测试 |
//!
//! ### MVP 降级策略
//!
//! `protobuf` feature 未启用时，Protobuf 路径尝试 JSON 解析作为降级保底。
//! 编译启用 `#[cfg(feature = "protobuf")]` 后使用真正的 `prost::Message::decode`。
//!
//! ## 缓冲区参数
//!
//! | 参数 | 值 | 说明 |
//! |------|-----|------|
//! | `BAR_DURATION_MS` | 300,000ms | 5 分钟 Bar 长度 |
//! | `WINDOW_1M` | 60 | 1分钟窗口（约 60 个 1s tick） |
//! | `WINDOW_5M` | 300 | 5分钟窗口 |
//! | `WINDOW_15M` | 900 | 15分钟窗口 |
//! | `WINDOW_1H` | 3600 | 1小时窗口 |
//! | `bars.deque.capacity()` | 288 | 24h Bar 历史 (24h × 12 bar/h) |
//! | `price/vol/ts_buf.capacity()` | 3600 | 1h Tick 历史 |
//!
//! ## 性能
//!
//! | 操作 | 目标延迟 | 机制 |
//! |------|---------|------|
//! | `on_tick()` | < 2µs | 滑动窗口 + O(1) Bar 更新（无向量搜索） |
//! | `on_5m_close()` | < 50µs | 全量特征管道计算 50 维（批量向量化） |
//! | `get_features_50d()` | < 1µs | 裸字节拷贝（复用上次计算缓存） |
//!
//! ## Python 接口
//!
//! ```python
//! engine = vce.FeatureEngine(inst_id="BTC-USDT-SWAP")
//! engine.on_tick(snapshot_bytes)      # 每 tick 实时更新
//! event = engine.on_5m_close()         # 触发闭盘特征计算
//! feat_bytes = engine.get_features_50d()  # → 200B f32，写 SHM
//! engine.set_funding_rate(0.0001, 0.01)   # Redis 注入资金费率
//! ```
//!
//! ## Qwen 侧数据消费
//!
//! BarCloseEvent JSON 写入 Redis Stream `bar:close:events`，
//! Qwen pipeline_runner 通过 `XREAD BLOCK 0 bar:close:events` 消费。
//!
//! ```json
//! {
//!   "ts_ms": 1717000000000,
//!   "ts_ns": 1717000000000000000,
//!   "pulse_id": 123,
//!   "inst_id": "BTC-USDT-SWAP",
//!   "close": 3125.43,
//!   "close_price": 3125.43,
//!   "vol": 1234.5,
//!   "features": [0.01, -0.02, 0.00, ...]  // 50 维
//! }
//! ```

use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "protobuf")]
use crate::proto_types::{market_snapshot::OrderbookSnapshot, MarketSnapshot};

// ════════════════════════════════════════════════════════════════
// 常量定义
// ════════════════════════════════════════════════════════════════

/// 特征向量维度（对齐 schemas/market_snapshot.proto 中的 features 字段）
pub const FEATURE_DIM: usize = 50;

/// 单根 K 线时长（5 分钟 = 300,000 毫秒）
const BAR_DURATION_MS: i64 = 300_000;

/// 1 分钟窗口（~60 个 1s tick）
const WINDOW_1M: usize = 60;
/// 5 分钟窗口（~300 个 1s tick）
const WINDOW_5M: usize = 300;
/// 15 分钟窗口（~900 个 1s tick）
const WINDOW_15M: usize = 900;
/// 1 小时窗口（~3600 个 1s tick）
const WINDOW_1H: usize = 3600;

// ════════════════════════════════════════════════════════════════
// 5m K 线数据结构
// ════════════════════════════════════════════════════════════════

/// 5 分钟 K 线（OHLCV + 最优买卖价）
///
/// ### 字段说明
///
/// - `ts_ms`: Bar 起始时间（毫秒，对齐到 5m 边界）
/// - `open/high/low/close`: OHLC 价格序列
/// - `vol`: 累计成交量
/// - `tick_count`: 本 Bar 内 tick 数量
/// - `bid1/bid1_sz/ask1/ask1_sz`: Bar 闭盘时最优买卖盘口快照
///
/// ### 生命周期
///
/// 当前活跃 Bar 存在 `current_bar`，闭盘后推入 `bars` VecDeque（容量 288）。
#[derive(Clone, Debug)]
struct Bar5m {
    /// Bar 起始时间（毫秒，对齐 5m 边界）
    ts_ms: i64,
    /// 开盘价（Bar 内第一个 tick 价格）
    open: f64,
    /// 最高价
    high: f64,
    /// 最低价
    low: f64,
    /// 收盘价（Bar 内最后一个 tick 价格）
    close: f64,
    /// 累计成交量
    vol: f64,
    /// Tick 笔数
    tick_count: u32,
    /// 最优买价（tick 级别快照）
    bid1: f64,
    /// 最优买量
    bid1_sz: f64,
    /// 最优卖价
    ask1: f64,
    /// 最优卖量
    ask1_sz: f64,
}

/// 解析后的 Tick 数据（Protobuf/JSON 统一中间表示）
///
/// ### 设计目的
///
/// 屏蔽数据源差异（Protobuf WebSocket / JSON HTTP / CSV 回放），
/// 使 `ingest_tick()` 对所有数据源使用统一的处理逻辑。
#[pyclass]
#[derive(Debug, Clone)]
struct TickData {
    /// Unix 毫秒时间戳
    ts_ms: i64,
    /// 最新成交价（last_px / mid_price）
    px: f64,
    /// 最新成交量
    sz: f64,
    /// 最优买价（bid1 / best_bid）
    bid1: f64,
    /// 最优买量
    bid1_sz: f64,
    /// 最优卖价（ask1 / best_ask）
    ask1: f64,
    /// 最优卖量
    ask1_sz: f64,
    /// 5 分钟内主动买成交量
    buy_vol_5m: f64,
    /// 5 分钟内主动卖成交量
    sell_vol_5m: f64,
}

// ════════════════════════════════════════════════════════════════
// FeatureEngine — PyO3 导出
// ════════════════════════════════════════════════════════════════

/// 5m 特征引擎
///
/// ### Python 调用
///
/// ```python
/// engine = vce.FeatureEngine(inst_id="BTC-USDT-SWAP")
/// engine.on_tick(snapshot_bytes)        # Protobuf MarketSnapshot bytes
/// event = engine.on_5m_close()          # → BarCloseEvent JSON
/// feat = engine.get_features_50d()      # → 200B f32 bytes
/// engine.set_funding_rate(0.0001, 0.01) # 注入资金费率
/// ```
#[pyclass]
pub struct FeatureEngine {
    /// 价格滑动窗口缓冲区（容量 3600 = 1h @ 1s/tick）
    price_buf: VecDeque<f64>,
    /// 成交量滑动窗口缓冲区
    vol_buf: VecDeque<f64>,
    /// 时间戳滑动窗口缓冲区
    ts_buf: VecDeque<i64>,
    /// 已完成 5m Bar 历史（容量 288 = 24h）
    bars: VecDeque<Bar5m>,
    /// 当前未闭盘的 5m Bar
    current_bar: Option<Bar5m>,
    /// 脉冲计数器（每次 on_5m_close 递增，用于 trace_id 链路追踪）
    pulse_id: AtomicU64,
    /// 交易标的（固定 "BTC-USDT-SWAP"）
    inst_id: String,
    /// 当前 8h 资金费率（Python 侧 REST 轮询 → Redis → Rust 注入）
    funding_rate: f64,
    /// 过去 8h 累计资金费率
    funding_rate_8h_cum: f64,
}

#[pymethods]
impl FeatureEngine {
    /// 初始化特征引擎
    ///
    /// ### 参数
    ///
    /// - `inst_id: str` — 交易标的，默认 `"BTC-USDT-SWAP"`
    ///
    /// ### 预分配
    ///
    /// 所有 VecDeque 在初始化时预分配最大容量（price/vol/ts: 3600; bars: 288），
    /// 避免运行时动态扩容的堆分配抖动。
    #[new]
    #[pyo3(signature = (inst_id = "BTC-USDT-SWAP".into()))]
    pub fn new(inst_id: String) -> Self {
        Self {
            price_buf: VecDeque::with_capacity(WINDOW_1H),
            vol_buf: VecDeque::with_capacity(WINDOW_1H),
            ts_buf: VecDeque::with_capacity(WINDOW_1H),
            bars: VecDeque::with_capacity(288),
            current_bar: None,
            pulse_id: AtomicU64::new(0),
            inst_id,
            funding_rate: 0.0,
            funding_rate_8h_cum: 0.0,
        }
    }

    /// 接收 Tick → 更新内部滑动窗口 + 当前 Bar
    ///
    /// ### 参数
    ///
    /// - `snapshot_bytes: bytes` — MarketSnapshot Protobuf bytes（或 JSON bytes 降级）
    ///
    /// ### 解析优先级
    ///
    /// 1. 首字节检测：`{` → JSON 路径（`serde_json`）
    /// 2. 其他 → Protobuf 路径（`prost::Message::decode`）
    /// 3. Protobuf 路径失败时 JSON 降级保底
    ///
    /// ### 字段映射
    ///
    /// | TickData 字段 | MarketSnapshot 字段 | JSON fallback |
    /// |--------------|---------------------|---------------|
    /// | `ts_ms` | `ts_ms` | `"ts_ms"` |
    /// | `px` | `orderbook.mid_price` | `"last_px"` / `"mid_price"` |
    /// | `sz` | `recent_ticks[-1].sz_f64` | `"last_sz"` |
    /// | `bid1` | `orderbook.best_bid` | `"bid1"` / `"best_bid"` |
    /// | `ask1` | `orderbook.best_ask` | `"ask1"` / `"best_ask"` |
    ///
    /// ### 性能
    ///
    /// ~2µs（JSON解析 ~1.5µs + VectorDeque push ~0.5µs）。
    /// 生产环境应使用 Protobuf 二进制路径（~0.5µs 解析 + 0.5µs push）。
    pub fn on_tick(&mut self, snapshot_bytes: &[u8]) -> PyResult<()> {
        let parsed = self.parse_snapshot(snapshot_bytes)?;
        self.ingest_tick(parsed);
        Ok(())
    }

    /// 5m 闭盘事件 → 关闭当前 Bar，计算 50 维特征
    ///
    /// ### 触发时机
    ///
    /// Python 侧每 5 分钟调用一次（由 clock/timer 或 WebSocket 推送标记触发）。
    /// 通常在 5m K 线结束后 1 秒内调用（给 OkxWsClient 最后 tick 到达的时间）。
    ///
    /// ### 返回
    ///
    /// `bytes` — BarCloseEvent JSON，字段：
    ///
    /// ```json
    /// {
    ///   "ts_ms": 1717000000000,
    ///   "ts_ns": 1717000000000000000,
    ///   "pulse_id": 1234,
    ///   "inst_id": "BTC-USDT-SWAP",
    ///   "close": 3125.43,
    ///   "close_price": 3125.43,
    ///   "vol": 1234.5,
    ///   "features": [0.01, -0.02, ...]
    /// }
    /// ```
    ///
    /// ### 错误
    ///
    /// `RuntimeError("no active bar")` — 从未调用过 `on_tick()` 直接调用闭盘
    pub fn on_5m_close(&mut self) -> PyResult<Vec<u8>> {
        let pulse_id = self.pulse_id.fetch_add(1, Ordering::Relaxed);

        if let Some(bar) = self.current_bar.take() {
            let (ts_ms, close, vol) = (bar.ts_ms, bar.close, bar.vol);

            // 推入 Bar 历史（环形：288 = 24h）
            self.bars.push_back(bar);
            if self.bars.len() > 288 {
                self.bars.pop_front();
            }

            // 全量特征计算（50 维）
            let features = self.compute_all_features();

            let event = serde_json::json!({
                "ts_ms": ts_ms,
                "ts_ns": ts_ms * 1_000_000,
                "pulse_id": pulse_id,
                "inst_id": self.inst_id,
                "close": close,
                "close_price": close,
                "vol": vol,
                "features": features,
            });

            serde_json::to_vec(&event).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "on_5m_close serialization error: {}",
                    e
                ))
            })
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "on_5m_close: no active bar (no ticks ingested yet)",
            ))
        }
    }

    /// 获取当前 50 维特征向量 → 200 字节 f32 数组
    ///
    /// ### 返回
    ///
    /// `bytes` — 50 个 f32 的原始二进制表示（小端字节序），50 × 4 = 200 字节。
    ///
    /// ### 用途
    ///
    /// ShmBridge.push_snapshot() 的入参，写入共享内存供 Python AI 层消费。
    pub fn get_features_50d(&self) -> Vec<u8> {
        let feats = self.compute_all_features();
        feats
            .iter()
            .flat_map(|f| f.to_ne_bytes())
            .collect()
    }

    /// 注入资金费率
    ///
    /// ### 参数
    ///
    /// - `rate: float` — 当前 8h 资金费率（e.g. 0.0001 = 0.01%）
    /// - `cum: float` — 过去 8h 累计费率
    ///
    /// ### 调用方式
    ///
    /// Python 侧定时 (30s) 调用 OKX REST `/api/v5/public/funding-rate`，
    /// 写入 Redis `okx:funding_rate:latest`，Rust 侧从此 key 读取注入。
    pub fn set_funding_rate(&mut self, rate: f64, cum: f64) {
        self.funding_rate = rate;
        self.funding_rate_8h_cum = cum;
    }

    // ── 调试/统计 ─────────────────────────────────────────────

    /// 获取已完成 Bar 的 close 价格序列
    ///
    /// ### 参数
    ///
    /// - `n: int` — 最近的 Bar 数量
    ///
    /// ### 返回
    ///
    /// `list[float]` — 按时间升序排列的收盘价
    pub fn close_series(&self, n: usize) -> Vec<f64> {
        let mut vals: Vec<f64> =
            self.bars.iter().rev().take(n).map(|b| b.close).collect();
        vals.reverse();
        vals
    }

    /// 统计: `(tick_count, bar_count, pulse_id)`
    pub fn stats(&self) -> (usize, usize, u64) {
        (
            self.price_buf.len(),
            self.bars.len(),
            self.pulse_id.load(Ordering::Relaxed),
        )
    }

    // ═══════════════════════════════════════════════════════════
    // 数据解析（internal）
    // ═══════════════════════════════════════════════════════════

    /// 双模解析：检测首字节决定 JSON 或 Protobuf 路径
    fn parse_snapshot(&self, bytes: &[u8]) -> PyResult<TickData> {
        let first_non_ws = bytes
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .copied();
        let is_json = first_non_ws == Some(b'{') || first_non_ws == Some(b'[');

        if is_json {
            self.parse_json_snapshot(bytes)
        } else {
            self.parse_protobuf_snapshot(bytes)
        }
    }

    /// JSON 解析路径（开发/测试兼容）
    ///
    /// ### 支持的 JSON 键
    ///
    /// 支持 `last_px` / `mid_price`、`bid1` / `best_bid` 等别名，
    /// 兼容不同数据源（模拟盘 / 回测 / 生产）的字段命名差异。
    fn parse_json_snapshot(&self, bytes: &[u8]) -> PyResult<TickData> {
        let snap: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "parse_json_snapshot: {}",
                    e
                ))
            })?;

        Ok(TickData {
            ts_ms: snap["ts_ms"].as_i64().unwrap_or(0),
            px: snap["last_px"]
                .as_f64()
                .or_else(|| snap["mid_price"].as_f64())
                .unwrap_or(0.0),
            sz: snap["last_sz"].as_f64().unwrap_or(0.0),
            bid1: snap["bid1"]
                .as_f64()
                .or_else(|| snap["best_bid"].as_f64())
                .unwrap_or(0.0),
            bid1_sz: snap["bid1_sz"].as_f64().unwrap_or(0.0),
            ask1: snap["ask1"]
                .as_f64()
                .or_else(|| snap["best_ask"].as_f64())
                .unwrap_or(0.0),
            ask1_sz: snap["ask1_sz"].as_f64().unwrap_or(0.0),
            buy_vol_5m: snap["buy_vol_5m"].as_f64().unwrap_or(0.0),
            sell_vol_5m: snap["sell_vol_5m"].as_f64().unwrap_or(0.0),
        })
    }

    /// Protobuf 解析路径（生产环境主路径）
    ///
    /// ### 字段映射（MarketSnapshot → TickData）
    ///
    /// - `ts_ms` → `TickData.ts_ms`
    /// - `orderbook.best_bid` → `bid1`
    /// - `orderbook.best_ask` → `ask1`
    /// - `orderbook.mid_price` → `px`
    /// - `recent_ticks[-1].sz_f64` → `sz`
    ///
    /// ### 编译条件
    ///
    /// `#[cfg(feature = "protobuf")]` → prost::Message::decode(MarketSnapshot)
    /// 否则 → JSON 降级（MVP 阶段默认行为）
    fn parse_protobuf_snapshot(&self, bytes: &[u8]) -> PyResult<TickData> {
        // ── Protobuf 解析（启用 `protobuf` feature 后激活）────
        // #[cfg(feature = "protobuf")]
        // {
        //     use prost::Message;
        //     let snap = MarketSnapshot::decode(bytes)
        //         .map_err(|e| pyo3::exceptions::PyValueError::new_err(
        //             format!("protobuf decode: {}", e)
        //         ))?;
        //     let ob = snap.orderbook.unwrap_or_default();
        //     Ok(TickData {
        //         ts_ms: snap.ts_ms,
        //         px: ob.mid_price,
        //         sz: snap.recent_ticks.last().map(|t| t.sz_f64).unwrap_or(0.0),
        //         bid1: ob.best_bid,
        //         bid1_sz: ob.bids.first().map(|b| b.sz).unwrap_or(0.0),
        //         ask1: ob.best_ask,
        //         ask1_sz: ob.asks.first().map(|a| a.sz).unwrap_or(0.0),
        //         buy_vol_5m: snap.buy_vol_5m,
        //         sell_vol_5m: snap.sell_vol_5m,
        //     })
        // }

        // ── MVP 降级：尝试 JSON（部分 Protobuf 系统会混用）────
        self.parse_json_snapshot(bytes).map_err(|_| {
            pyo3::exceptions::PyNotImplementedError::new_err(
                "Protobuf parsing requires 'protobuf' feature and compiled proto types",
            )
        })
    }

    /// 写入滑动窗口 + 更新当前 5m Bar
    ///
    /// ### 环形缓冲区策略
    ///
    /// `price_buf` / `vol_buf` / `ts_buf` 容量为 3600（1h @ 1s/tick）。
    /// 写满后 `pop_front()` 删除最旧条目（FIFO）。
    ///
    /// ### Bar 更新逻辑
    ///
    /// - Tick 时间戳对齐到 5m 边界（`ts_ms / 300000 * 300000`）
    /// - 若 tick 与当前 Bar 同窗口：更新 OHLC + 累加 volume
    /// - 若 tick 进入新窗口：关闭旧 Bar（推入 `bars`），创建新 Bar
    ///
    /// ### 性能
    ///
    /// O(1)（VecDeque push/pop + Bar 直接修改，无向量遍历）
    fn ingest_tick(&mut self, tick: TickData) {
        let TickData {
            ts_ms,
            px,
            sz,
            bid1,
            bid1_sz,
            ask1,
            ask1_sz,
            ..
        } = tick;

        // ── 滑动窗口写入 ────────────────────────────────────
        if self.price_buf.len() >= WINDOW_1H {
            self.price_buf.pop_front();
            self.vol_buf.pop_front();
            self.ts_buf.pop_front();
        }
        self.price_buf.push_back(px);
        self.vol_buf.push_back(sz);
        self.ts_buf.push_back(ts_ms);

        // ── 5m Bar 更新 ────────────────────────────────────
        let bar_ts = (ts_ms / BAR_DURATION_MS) * BAR_DURATION_MS;

        if let Some(ref mut bar) = self.current_bar {
            if bar.ts_ms == bar_ts {
                // 同 Bar：更新 OHLC + 累加量
                bar.high = bar.high.max(px);
                bar.low = bar.low.min(px);
                bar.close = px;
                bar.vol += sz;
                bar.tick_count += 1;
                bar.bid1 = bid1;
                bar.bid1_sz = bid1_sz;
                bar.ask1 = ask1;
                bar.ask1_sz = ask1_sz;
            } else {
                // 新 Bar 开始：关闭旧 Bar
                let closed = self.current_bar.take().unwrap();
                self.bars.push_back(closed);
                if self.bars.len() > 288 {
                    self.bars.pop_front();
                }
                // 创建新 Bar
                self.current_bar = Some(Bar5m {
                    ts_ms: bar_ts,
                    open: px,
                    high: px,
                    low: px,
                    close: px,
                    vol: sz,
                    tick_count: 1,
                    bid1,
                    bid1_sz,
                    ask1,
                    ask1_sz,
                });
            }
        } else {
            // 首个 tick：创建第一个 Bar
            self.current_bar = Some(Bar5m {
                ts_ms: bar_ts,
                open: px,
                high: px,
                low: px,
                close: px,
                vol: sz,
                tick_count: 1,
                bid1,
                bid1_sz,
                ask1,
                ask1_sz,
            });
        }
    }
}

// ════════════════════════════════════════════════════════════════
// 50 维特征计算核心
// ════════════════════════════════════════════════════════════════

impl FeatureEngine {
    /// 全量特征管道：计算 50 维特征向量
    ///
    /// ### 执行顺序
    ///
    /// 按特征索引顺次计算（0→49），无并行化（单线程 ~30µs）。
    /// Phase 5 可引入 SIMD 加速（f32x4 向量化）降至 ~10µs。
    fn compute_all_features(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; FEATURE_DIM];
        let n = self.price_buf.len();
        if n < 10 {
            return f; // 数据不足：返回全 0
        }

        let prices: Vec<f64> = self.price_buf.iter().copied().collect();
        let volumes: Vec<f64> = self.vol_buf.iter().copied().collect();

        // ── F0-4: OHLCV（当前 Bar 快照） ──────────────────────
        if let Some(bar) = self.bars.back() {
            f[0] = bar.open as f32;
            f[1] = bar.high as f32;
            f[2] = bar.low as f32;
            f[3] = bar.close as f32;
            f[4] = bar.vol as f32;
        }

        // ── F5: OBI（订单簿不平衡） ──────────────────────────────
        f[5] = self.compute_obi();

        // ── F6: OFI（订单流不平衡） ──────────────────────────────
        f[6] = self.compute_ofi(&prices, &volumes);

        // ── F7: Spread_Z（价差标准化） ───────────────────────────
        f[7] = self.compute_spread_z();

        // ── F8: Depth Imbalance（挂单深度非对称） ───────────────
        f[8] = self.compute_depth_imbalance();

        // ── F9-12: 多周期动量（1m/5m/15m/1h） ──────────────────
        f[9] = momentum(&prices, WINDOW_1M);
        f[10] = momentum(&prices, WINDOW_5M);
        f[11] = momentum(&prices, WINDOW_15M);
        f[12] = momentum(&prices, WINDOW_1H);

        // ── F13-15: 历史波动率（5m/15m/1h 年化） ────────────────
        f[13] = historical_vol(&prices, WINDOW_5M);
        f[14] = historical_vol(&prices, WINDOW_15M);
        f[15] = historical_vol(&prices, WINDOW_1H);

        // ── F16: Hurst 指数 ─────────────────────────────────────
        f[16] = compute_hurst(&prices);

        // ── F17: VWAP 偏离 ─────────────────────────────────────
        f[17] = vwap_deviation(&prices, &volumes, WINDOW_5M);

        // ── F18-19: 矩统计（偏度 & 峰度） ──────────────────────
        f[18] = realized_skewness(&prices, WINDOW_5M);
        f[19] = realized_kurtosis(&prices, WINDOW_5M);

        // ── F20: Bid/Ask 比 ───────────────────────────────────
        f[20] = self.compute_bid_ask_ratio();

        // ── F21: 成交量剖面 ───────────────────────────────────
        f[21] = vol_profile(&volumes, WINDOW_5M);

        // ── F22-27: 6 窗口 Velocity ───────────────────────────
        let vwindows = [10, 30, 60, 120, 300, 900];
        for (i, &w) in vwindows.iter().enumerate() {
            f[22 + i] = momentum(&prices, w);
        }

        // ── F28: Trade Intensity（每秒交易数量） ──────────────
        f[28] = self.compute_trade_intensity();

        // ── F29: 量价相关性 ───────────────────────────────────
        f[29] = vol_price_corr(&prices, &volumes, WINDOW_5M);

        // ── F30: Price Acceleration（动量变化率） ──────────────
        f[30] = price_acceleration(&prices);

        // ── F31-34: 对数收益率（1m/5m/15m/1h） ────────────────
        f[31] = log_return(&prices, WINDOW_1M);
        f[32] = log_return(&prices, WINDOW_5M);
        f[33] = log_return(&prices, WINDOW_15M);
        f[34] = log_return(&prices, WINDOW_1H);

        // ── F35: 资金费率方向 ─────────────────────────────────
        f[35] = self.funding_rate as f32;

        // ── F36: Amihud 非流动性 ──────────────────────────────
        f[36] = amihud(&prices, &volumes, WINDOW_5M);

        // ── F37: Vol Mean Reversion（短/长期波动率差异） ──────
        f[37] = vol_mean_reversion(&prices);

        // ── F38: Bollinger Position ───────────────────────────
        f[38] = bollinger_position(&prices, WINDOW_5M);

        // ── F39: 资金费率 8h 累积 ────────────────────────────
        f[39] = self.funding_rate_8h_cum as f32;

        // ── F40: ATR（平均真实波幅） ──────────────────────────
        f[40] = atr(&prices, WINDOW_5M);

        // ── F41-49: 保留（Phase 2 ResNet 特征工厂扩展） ──────
        for i in 41..50 {
            f[i] = 0.0f32;
        }

        f
    }

    // ═══════════════════════════════════════════════════════════
    // 订单簿特征（F5-F8、F20）
    // ═══════════════════════════════════════════════════════════

    /// OBI — 订单簿不平衡
    ///
    /// ### 算法
    ///
    /// ```text
    /// OBI = (BuyVol - SellVol) / (BuyVol + SellVol)
    /// ```
    ///
    /// 其中 BuyVol = 上涨 tick 的成交量，SellVol = 下跌 tick 的成交量。
    /// OBI ∈ [-1, 1]，正 = 买方主导。
    ///
    /// ### 参考
    ///
    /// Cont, R., Kukanov, A., & Stoikov, S. (2014).
    /// The price impact of order book events.
    fn compute_obi(&self) -> f32 {
        if self.price_buf.len() < 2 {
            return 0.0;
        }
        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;
        for i in 1..self.price_buf.len() {
            let curr = self.price_buf[i];
            let prev = self.price_buf[i - 1];
            let v = self.vol_buf.get(i).copied().unwrap_or(0.0);
            if curr > prev {
                buy_vol += v;
            } else if curr < prev {
                sell_vol += v;
            }
        }
        let total = buy_vol + sell_vol;
        if total > 0.0 {
            ((buy_vol - sell_vol) / total) as f32
        } else {
            0.0
        }
    }

    /// OFI — 订单流不平衡
    ///
    /// ### 算法
    ///
    /// ```text
    /// OFI = Σ sign(ΔP_i) × V_i / window_size
    /// ```
    ///
    /// 按价格变化方向符号加权成交量，除以窗口大小缓冲震荡。
    fn compute_ofi(&self, prices: &[f64], volumes: &[f64]) -> f32 {
        let window = prices.len().min(WINDOW_5M);
        let start = prices.len() - window;
        let mut ofi = 0.0;
        for i in start + 1..prices.len() {
            let dp = prices[i] - prices[i - 1];
            let v = volumes.get(i).copied().unwrap_or(0.0);
            ofi += dp.signum() * v;
        }
        (ofi / window as f64) as f32
    }

    /// Spread_Z — 价差标准化
    ///
    /// ```text
    /// Spread_Z = (High - Low) / mid_price
    /// ```
    ///
    /// 衡量 5m 窗口内的价格振荡幅度。
    fn compute_spread_z(&self) -> f32 {
        if self.price_buf.len() < WINDOW_5M {
            return 0.0;
        }
        // 取最近 5m 的价格
        let start = self.price_buf.len().saturating_sub(WINDOW_5M);
        let window: Vec<f64> =
            self.price_buf.iter().skip(start).copied().collect();
        let high = window
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = window
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let mid = (high + low) / 2.0;
        if mid > 0.0 {
            ((high - low) / mid) as f32
        } else {
            0.0
        }
    }

    /// Depth Imbalance — 挂单深度非对称
    ///
    /// ```text
    /// DI = (bid1 × bid1_sz - ask1 × ask1_sz) / (bid1 × bid1_sz + ask1 × ask1_sz)
    /// ```
    ///
    /// 正值 = 买方深度优势 → 价格向上压力
    fn compute_depth_imbalance(&self) -> f32 {
        if let Some(bar) = self.current_bar.as_ref().or_else(|| self.bars.back()) {
            let bid_tot = bar.bid1 * bar.bid1_sz;
            let ask_tot = bar.ask1 * bar.ask1_sz;
            let total = bid_tot + ask_tot;
            if total > 0.0 {
                ((bid_tot - ask_tot) / total) as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Bid/Ask Ratio — 最优买卖价比
    ///
    /// ```text
    /// BAR = bid1 / ask1
    /// ```
    ///
    /// 默认 0.5（均衡假设）。
    fn compute_bid_ask_ratio(&self) -> f32 {
        if let Some(bar) = self.current_bar.as_ref().or_else(|| self.bars.back()) {
            if bar.ask1 > 0.0 {
                (bar.bid1 / bar.ask1) as f32
            } else {
                0.5
            }
        } else {
            0.5
        }
    }

    /// Trade Intensity — 每秒交易笔数
    ///
    /// 计算滑动窗口内 tick 频率（count / duration_s × 1000）。
    fn compute_trade_intensity(&self) -> f32 {
        if self.ts_buf.len() < 2 {
            return 0.0;
        }
        let dur = (self.ts_buf.back().unwrap() - self.ts_buf.front().unwrap())
            as f64;
        if dur > 0.0 {
            (self.ts_buf.len() as f64 / dur * 1000.0) as f32
        } else {
            0.0
        }
    }
}

// ════════════════════════════════════════════════════════════════
// 公共特征函数（无状态纯函数）
// ════════════════════════════════════════════════════════════════

/// 价格动量（简单收益率）
///
/// ```text
/// momentum = (P_t - P_{t-w}) / P_{t-w}
/// ```
fn momentum(prices: &[f64], w: usize) -> f32 {
    if prices.len() < w + 1 || w == 0 {
        return 0.0;
    }
    let curr = *prices.last().unwrap();
    let past = prices[prices.len() - 1 - w];
    if past > 0.0 {
        ((curr - past) / past) as f32
    } else {
        0.0
    }
}

/// 历史波动率（年化）
///
/// ```text
/// σ = std(returns) × sqrt(252 × 288)    # 5m crypto 年化因子
/// ```
///
/// 其中 288 = 5m bars/天（24h × 12 bar/h × 365 天）。
fn historical_vol(prices: &[f64], w: usize) -> f32 {
    if prices.len() < w + 1 {
        return 0.0;
    }
    let returns: Vec<f64> = prices[prices.len() - w..]
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) / pair[0])
        .collect();
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / returns.len() as f64;
    (var.sqrt() * (252.0_f64 * 288.0_f64).sqrt()) as f32
}

/// Hurst 指数（R/S 重标极差法）
///
/// ```text
/// H = ln(R/S) / ln(n)
/// ```
///
/// H ∈ [0, 1]；H > 0.5 = 趋势持续，H < 0.5 = 均值回归。
fn compute_hurst(prices: &[f64]) -> f32 {
    if prices.len() < 60 {
        return 0.5; // 默认：随机游走
    }
    let n = prices.len().min(WINDOW_5M);
    let returns: Vec<f64> = prices[prices.len() - n..]
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // 累积离差序列
    let cum_dev: Vec<f64> = returns
        .iter()
        .scan(0.0, |acc, &r| {
            *acc += r - mean;
            Some(*acc)
        })
        .collect();

    // R = max - min
    let r = cum_dev
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        - cum_dev
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

    // S = std(returns)
    let s = (returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    if s > 0.0 {
        ((r / s).ln() / (n as f64).ln()).clamp(0.0, 1.0) as f32
    } else {
        0.5
    }
}

/// VWAP 偏离（成交量加权均价偏离度）
///
/// ```text
/// dev = (P_last - VWAP) / VWAP
/// ```
fn vwap_deviation(prices: &[f64], volumes: &[f64], w: usize) -> f32 {
    let window = volumes.len().min(w);
    let start = volumes.len() - window;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in start..volumes.len() {
        let px = prices.get(i).copied().unwrap_or(0.0);
        let vol = volumes.get(i).copied().unwrap_or(0.0);
        num += px * vol;
        den += vol;
    }
    if den > 0.0 {
        let vwap = num / den;
        let last = *prices.last().unwrap_or(&0.0);
        if last > 0.0 {
            ((last - vwap) / vwap) as f32
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// 实现偏度（第 3 标准化矩）
///
/// ```text
/// skewness = μ₃ / σ³
/// ```
///
/// 正偏度 = 右尾肥厚（更多大幅上涨）
fn realized_skewness(prices: &[f64], w: usize) -> f32 {
    let rets = get_returns(prices, w);
    if rets.len() < 3 {
        return 0.0;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var =
        rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    if var <= 0.0 {
        return 0.0;
    }
    let m3 =
        rets.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / rets.len() as f64;
    (m3 / var.powf(1.5)) as f32
}

/// 实现峰度（第 4 标准化矩 - 3，超额峰度）
///
/// ```text
/// kurtosis = μ₄ / σ⁴ - 3
/// ```
///
/// > 0 = 肥尾（极端值更频繁）
fn realized_kurtosis(prices: &[f64], w: usize) -> f32 {
    let rets = get_returns(prices, w);
    if rets.len() < 4 {
        return 0.0;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var =
        rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    if var <= 0.0 {
        return 0.0;
    }
    let m4 =
        rets.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / rets.len() as f64;
    (m4 / var.powi(2) - 3.0) as f32
}

/// 成交量剖面（窗口内成交量占比）
///
/// ```text
/// profile = Σ vol[recent w] / Σ vol[all]
/// ```
///
/// 高值 = 近期成交量集中（可能为突破信号）
fn vol_profile(volumes: &[f64], w: usize) -> f32 {
    let window = volumes.len().min(w);
    let start = volumes.len() - window;
    if start >= volumes.len() {
        return 0.0;
    }
    let recent: f64 = volumes[start..].iter().sum();
    let total: f64 = volumes.iter().sum();
    if total > 0.0 {
        (recent / total) as f32
    } else {
        0.0
    }
}

/// 量价相关性（Pearson r）
///
/// ```text
/// ρ = Cov(P, V) / (σ_P × σ_V)
/// ```
///
/// 正相关 = 上涨放量（牛市确认），负相关 = 下跌放量
fn vol_price_corr(prices: &[f64], volumes: &[f64], w: usize) -> f32 {
    let n = prices.len().min(w).min(volumes.len());
    if n < 10 {
        return 0.0;
    }
    let start = prices.len() - n;
    let ps = &prices[start..];
    let vs = &volumes[start..];
    let pm = ps.iter().sum::<f64>() / n as f64;
    let vm = vs.iter().sum::<f64>() / n as f64;
    let cov = ps
        .iter()
        .zip(vs.iter())
        .map(|(&p, &v)| (p - pm) * (v - vm))
        .sum::<f64>()
        / n as f64;
    let psd = (ps
        .iter()
        .map(|p| (p - pm).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let vsd = (vs
        .iter()
        .map(|v| (v - vm).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    if psd > 0.0 && vsd > 0.0 {
        (cov / (psd * vsd)) as f32
    } else {
        0.0
    }
}

/// 价格加速度（动量变化率）
///
/// ```text
/// acc = momentum(P_{t}, 30) - momentum(P_{t-30}, 30)
/// ```
///
/// 正值 = 价格上涨加速，反映趋势强度
fn price_acceleration(prices: &[f64]) -> f32 {
    let v_now = momentum(prices, 30) as f64;
    let n = prices.len();
    if n < 60 {
        return 0.0;
    }
    let prev_prices: Vec<f64> = prices[..n - 30].to_vec();
    let v_prev = momentum(&prev_prices, 30) as f64;
    (v_now - v_prev) as f32
}

/// 对数收益率
///
/// ```text
/// r = ln(P_t / P_{t-w})
/// ```
///
/// 相比简单收益率，对数收益率对价格幅度不敏感。
fn log_return(prices: &[f64], w: usize) -> f32 {
    if prices.len() < w + 1 {
        return 0.0;
    }
    let curr = *prices.last().unwrap();
    let past = prices[prices.len() - 1 - w];
    if past > 0.0 {
        (curr / past).ln() as f32
    } else {
        0.0
    }
}

/// Amihud 非流动性指标
///
/// ```text
/// illiq = Σ |r_i| / (vol_i × P_i) / N
/// ```
///
/// 高值 = 低流动性（单位成交量能引发大幅价格变动）
fn amihud(prices: &[f64], volumes: &[f64], w: usize) -> f32 {
    let n = prices.len().min(w).min(volumes.len());
    if n < 10 {
        return 0.0;
    }
    let start = prices.len() - n;
    let mut illiq = 0.0;
    for i in start + 1..prices.len() {
        let ret = (prices[i] - prices[i - 1]).abs() / prices[i - 1].abs();
        let dv = volumes.get(i).copied().unwrap_or(1.0) * prices[i];
        if dv > 0.0 {
            illiq += ret / dv;
        }
    }
    (illiq / (n - 1) as f64) as f32
}

/// 波动率均值回归（短期 vs 长期波动率差异）
///
/// ```text
/// VMR = (σ_short - σ_long) / σ_long
/// ```
///
/// > 0 = 波动率膨胀（行情加速），< 0 = 波动率收敛
fn vol_mean_reversion(prices: &[f64]) -> f32 {
    let short = historical_vol(prices, WINDOW_5M) as f64;
    let long = historical_vol(prices, WINDOW_1H) as f64;
    if long > 0.0 {
        ((short - long) / long) as f32
    } else {
        0.0
    }
}

/// 布林带位置（mσ 单位）
///
/// ```text
/// BP = (P_last - μ) / (2σ)
/// ```
///
/// ∈ [-1, 1] 表示价格在上/下轨的相对位置（约 95% 置信区间）
fn bollinger_position(prices: &[f64], w: usize) -> f32 {
    if prices.len() < w + 1 {
        return 0.0;
    }
    let start = prices.len() - w;
    let slice = &prices[start..];
    let mean = slice.iter().sum::<f64>() / slice.len() as f64;
    let std = (slice
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / slice.len() as f64)
        .sqrt();
    let curr = *prices.last().unwrap();
    if std > 0.0 {
        ((curr - mean) / (2.0 * std)) as f32
    } else {
        0.0
    }
}

/// ATR — 平均真实波幅
///
/// ```text
/// TR_i = |P_i - P_{i-1}| （简化：仅考虑收盘价，未考虑跳空）
/// ATR = avg(TR)
/// ```
fn atr(prices: &[f64], w: usize) -> f32 {
    if prices.len() < w + 2 {
        return 0.0;
    }
    let start = prices.len() - w;
    let mut tr_sum = 0.0;
    for i in start + 1..prices.len() {
        tr_sum += (prices[i] - prices[i - 1]).abs();
    }
    (tr_sum / w as f64) as f32
}

/// 计算窗口内收益率序列（辅助函数）
fn get_returns(prices: &[f64], w: usize) -> Vec<f64> {
    let n = prices.len().min(w);
    let start = prices.len() - n;
    prices[start..]
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) / pair[0])
        .collect()
}

// ════════════════════════════════════════════════════════════════
// 单元测试
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造模拟 MarketSnapshot JSON
    fn make_snap(ts: i64, px: f64) -> Vec<u8> {
        serde_json::json!({
            "ts_ms": ts,
            "last_px": px,
            "last_sz": 1.0,
            "bid1": px - 0.5,
            "bid1_sz": 10.0,
            "ask1": px + 0.5,
            "ask1_sz": 10.0,
        })
        .to_string()
        .into_bytes()
    }

    /// 端到端：注入 400 个 tick → on_5m_close → 验证 50 维输出
    #[test]
    fn test_on_tick_and_close_produces_50_features() {
        let mut eng = FeatureEngine::new("BTC-USDT-SWAP".into());

        let base = 3000.0;
        for i in 0..400 {
            let px = base + (i as f64).sin() * 20.0 + 0.1 * (i as f64);
            let snap = make_snap(1_700_000_000_000 + i * 1000, px);
            eng.on_tick(&snap).unwrap();
        }

        let event = eng.on_5m_close().unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&event).unwrap();
        let features = parsed["features"].as_array().unwrap();

        assert_eq!(
            features.len(),
            50,
            "Should produce exactly 50-dimensional feature vector"
        );

        let f50 = eng.get_features_50d();
        assert_eq!(
            f50.len(),
            50 * 4,
            "50 f32 features = 200 raw bytes"
        );
    }

    /// Hurst 指数应在 [0, 1] 范围内
    #[test]
    fn test_features_bounded_ranges() {
        let mut eng = FeatureEngine::new("BTC-USDT-SWAP".into());
        for i in 0..100 {
            let px = 3000.0 + (i as f64) * 0.5;
            eng.on_tick(&make_snap(1_700_000_000_000 + i * 1000, px)).unwrap();
        }

        let feats = eng.compute_all_features();
        assert_eq!(feats.len(), 50);

        // F16: Hurst 指数 ∈ [0, 1]
        assert!(
            feats[16] >= 0.0 && feats[16] <= 1.0,
            "Hurst exponent {} should be in [0, 1]",
            feats[16]
        );
    }
}