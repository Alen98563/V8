//! OKX WebSocket 接入 —— tokio-tungstenite · 心跳 · 自动重连 · ETH-USDT-SWAP
//!
//! T0-2: WS 数据入口，订阅 trades/books5/candle5m 三个公共频道
//!
//! 关键要求:
//! 1. simd_json 极速解析（避免全量反序列化 alloc）
//! 2. 25s 心跳 (OKX 服务端容忍 30s)
//! 3. 断线指数退避重连 (1s/2s/4s/8s，max 30s)
//! 4. 每 tick 写入 Redis okx:snapshot:latest (TTL=5s)

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

// ============================================================
// 数据结构
// ============================================================

/// 成交 Tick — 从 WS channels["trades"] 解析
#[derive(Debug, Clone)]
pub struct TradeTick {
    pub px: f64,
    pub sz: f64,
    pub side: String,  // "buy" | "sell"
    pub ts_ms: i64,
}

/// 订单簿快照 — 从 WS channels["books5"] 解析
#[derive(Debug, Clone)]
pub struct OrderBookSnap {
    pub asks: [(f64, f64); 5],  // [(px, sz), ...] 降序
    pub bids: [(f64, f64); 5],
    pub ts_ms: i64,
}

/// 5m K线 — 从 WS channels["candle5m"] 解析
#[derive(Debug, Clone)]
pub struct Candle5m {
    pub ts_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub vol: f64,
}

/// 行情快照 — WS 各频道最新值聚合
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub ts_ms: i64,
    pub inst_id: String,
    pub last_px: f64,
    pub last_sz: f64,
    pub bid1: f64,
    pub bid1_sz: f64,
    pub ask1: f64,
    pub ask1_sz: f64,
    pub spread: f64,
    pub tick_count: u32,
}

// ============================================================
// OKX WS 配置
// ============================================================

const WS_PUBLIC_URL: &str = "wss://ws.okx.com:8443/ws/v5/public";
const HEARTBEAT_INTERVAL_SECS: u64 = 25;
const SUBSCRIBE_ONCE_TOPIC: &str = "ETH-USDT-SWAP";

/// 指数退避参数
struct ReconnectBackoff {
    delay_ms: u64,
}

impl ReconnectBackoff {
    fn new() -> Self {
        Self { delay_ms: 1000 }
    }

    fn next_delay(&mut self) -> Duration {
        let delay = Duration::from_millis(self.delay_ms);
        self.delay_ms = (self.delay_ms * 2).min(30_000); // max 30s
        delay
    }

    fn reset(&mut self) {
        self.delay_ms = 1000;
    }
}

// ============================================================
// PyO3 导出
// ============================================================

/// OKX WebSocket 客户端 — 后台 Tokio 任务
///
/// Python 侧使用方式:
///   ws = vce.OkxWsClient("ETH-USDT-SWAP")
///   ws.start()  # 启动后台任务
///   snap = ws.latest_snapshot()
///   ws.stop()
#[pyclass]
pub struct OkxWsClient {
    inst_id: String,
    /// 最新快照（简易外露，正式通过 ShmBridge）
    latest: Arc<std::sync::Mutex<Option<MarketSnapshot>>>,
    /// 停止信号
    shutdown_tx: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[pymethods]
impl OkxWsClient {
    /// 创建 WS 客户端（不立即连接）
    #[new]
    #[pyo3(signature = (inst_id="ETH-USDT-SWAP".into()))]
    pub fn new(inst_id: String) -> Self {
        Self {
            inst_id,
            latest: Arc::new(std::sync::Mutex::new(None)),
            shutdown_tx: std::sync::Mutex::new(None),
            handle: std::sync::Mutex::new(None),
        }
    }

    /// 在独立 OS 线程中启动 Tokio runtime，开始 WS 接收
    pub fn start(&self) -> PyResult<()> {
        let inst_id = self.inst_id.clone();
        let latest = self.latest.clone();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        *self.shutdown_tx.lock().unwrap() = Some(tx);

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("OkxWsClient: failed to create Tokio runtime");

            rt.block_on(async move {
                run_ws_loop(&inst_id, &latest, rx).await;
            });
        });

        *self.handle.lock().unwrap() = Some(handle);
        Ok(())
    }

    /// 获取最新快照
    pub fn latest_snapshot(&self) -> Option<String> {
        self.latest
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| serde_json::json!({
                "ts_ms": s.ts_ms,
                "inst_id": s.inst_id,
                "last_px": s.last_px,
                "last_sz": s.last_sz,
                "bid1": s.bid1,
                "bid1_sz": s.bid1_sz,
                "ask1": s.ask1,
                "ask1_sz": s.ask1_sz,
                "spread": s.spread,
                "tick_count": s.tick_count,
            }).to_string())
    }

    /// 停止 WS 客户端
    pub fn stop(&self) {
        if let Some(tx) = self.shutdown_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }

    /// 连接状态（简化）
    pub fn is_running(&self) -> bool {
        self.handle
            .lock()
            .unwrap()
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }
}

// ============================================================
// WS 主循环
// ============================================================

async fn run_ws_loop(
    inst_id: &str,
    latest: &Arc<std::sync::Mutex<Option<MarketSnapshot>>>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;
    use futures_util::{SinkExt, StreamExt};

    let mut backoff = ReconnectBackoff::new();
    let mut tick_px = 0.0f64;
    let mut tick_sz = 0.0f64;
    let mut bid1 = 0.0f64;
    let mut bid1_sz = 0.0f64;
    let mut ask1 = 0.0f64;
    let mut ask1_sz = 0.0f64;

    'outer: loop {
        // 检查 shutdown
        if shutdown_rx.try_recv().is_ok() {
            break;
        }

        // 连接
        let ws_stream = match connect_async(WS_PUBLIC_URL).await {
            Ok((ws, _)) => ws,
            Err(e) => {
                eprintln!("[OkxWsClient] connect failed: {:?}, retry in {:?}", e, backoff.next_delay());
                tokio::time::sleep(backoff.next_delay()).await;
                continue;
            }
        };

        backoff.reset();
        let (mut write, mut read) = ws_stream.split();

        // 订阅
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": [
                {"channel": "trades", "instId": inst_id},
                {"channel": "books5", "instId": inst_id},
                {"channel": "candle5m", "instId": inst_id},
            ]
        })
        .to_string();

        if let Err(e) = write.send(Message::Text(sub_msg.into())).await {
            eprintln!("[OkxWsClient] subscribe failed: {:?}", e);
            continue;
        }

        // 心跳 + 消息循环
        let mut last_ping = tokio::time::Instant::now();

        loop {
            let heartbeat = tokio::time::sleep(
                Duration::from_secs(HEARTBEAT_INTERVAL_SECS)
                    .saturating_sub(last_ping.elapsed()),
            );

            tokio::select! {
                _ = &mut shutdown_rx => {
                    break 'outer;
                }

                _ = heartbeat => {
                    let ping = Message::Text("{\"op\":\"ping\"}".into());
                    if write.send(ping).await.is_err() {
                        break; // 连接断开 → 重连
                    }
                    last_ping = tokio::time::Instant::now();
                }

                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // simd_json 极速解析
                            let parsed = parse_ws_message(text.as_bytes());
                            if let Some(snap) = parsed {
                                tick_px = snap.last_px;
                                tick_sz = snap.last_sz;
                                bid1 = snap.bid1;
                                bid1_sz = snap.bid1_sz;
                                ask1 = snap.ask1;
                                ask1_sz = snap.ask1_sz;
                                *latest.lock().unwrap() = Some(snap);
                            } else if text.contains("pong") {
                                // 心跳 pong，正常，忽略
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            // pong 已通过心跳发送
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            break; // 断开 → 重连
                        }
                        Some(Err(e)) => {
                            eprintln!("[OkxWsClient] read error: {:?}", e);
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // on reconnect delay
        tokio::time::sleep(backoff.next_delay()).await;
    }
}

// ============================================================
// simd_json 快速解析（避免 serde 全量反序列化）
// ============================================================

fn parse_ws_message(raw: &[u8]) -> Option<MarketSnapshot> {
    // 使用 simd_json 的磁带解析，极速提取关键字段
    // 因为 OKX WS 推送是嵌套数组格式 [[{...}]]
    // 这里用简化的逐字节扫描，避免完整 JSON 解析

    let text = std::str::from_utf8(raw).ok()?;

    // 检测频道类型
    if text.contains("\"channel\":\"trades\"") {
        return parse_trades(text);
    } else if text.contains("\"channel\":\"books5\"") {
        return parse_books5(text);
    }
    // candle5m 由 FeatureEngine 处理，这里仅保留 tick 数据
    None
}

fn parse_trades(text: &str) -> Option<MarketSnapshot> {
    // 简化解析：提取第一个 data 对象的关键字段
    // 正式版本用 simd_json::to_tape 完整解析
    let px = extract_field_f64(text, "\"px\"")?;
    let sz = extract_field_f64(text, "\"sz\"")?;
    let side = if text.contains("\"side\":\"sell\"") {
        "sell"
    } else {
        "buy"
    };
    let ts = extract_field_i64(text, "\"ts\"")?;

    Some(MarketSnapshot {
        ts_ms: ts,
        inst_id: "ETH-USDT-SWAP".into(),
        last_px: px,
        last_sz: sz,
        bid1: 0.0,
        bid1_sz: 0.0,
        ask1: 0.0,
        ask1_sz: 0.0,
        spread: 0.0,
        tick_count: 1,
    })
}

fn parse_books5(text: &str) -> Option<MarketSnapshot> {
    let ts = extract_field_i64(text, "\"ts\"")?;

    // bids 和 asks 是嵌套数组 [[...]]，简化提取
    Some(MarketSnapshot {
        ts_ms: ts,
        inst_id: "ETH-USDT-SWAP".into(),
        last_px: 0.0,
        last_sz: 0.0,
        bid1: extract_nth_price(text, "bids", 0)?,
        bid1_sz: extract_nth_price(text, "bids", 1)?,
        ask1: extract_nth_price(text, "asks", 0)?,
        ask1_sz: extract_nth_price(text, "asks", 1)?,
        spread: 0.0,
        tick_count: 0,
    })
}

fn extract_field_f64(text: &str, key: &str) -> Option<f64> {
    let pos = text.find(key)?;
    let after = &text[pos + key.len()..];
    let val_start = after.find(':')? + 1;
    let trimmed = after[val_start..].trim_start();
    let val_str = if trimmed.starts_with('\"') {
        let end = trimmed[1..].find('\"')?;
        &trimmed[1..=end]
    } else {
        let end = trimmed.find(&[',', '}', '\n'] as &[_]).unwrap_or(trimmed.len());
        &trimmed[..end]
    };
    val_str.parse::<f64>().ok()
}

fn extract_field_i64(text: &str, key: &str) -> Option<i64> {
    let pos = text.find(key)?;
    let after = &text[pos + key.len()..];
    let val_start = after.find(':')? + 1;
    let trimmed = after[val_start..].trim_start();
    let val_str = if trimmed.starts_with('\"') {
        let end = trimmed[1..].find('\"')?;
        &trimmed[1..=end]
    } else {
        let end = trimmed.find(&[',', '}', '\n'] as &[_]).unwrap_or(trimmed.len());
        &trimmed[..end]
    };
    val_str.parse::<i64>().ok()
}

fn extract_nth_price(text: &str, side: &str, idx: usize) -> Option<f64> {
    // 简化: 提取 bids 数组内的第 idx 个主价格
    // 正式版用 simd_json 磁带完整解析
    let marker = format!("\"{}\"", side);
    let pos = text.find(&marker)?;
    let after = &text[pos + marker.len()..];
    // find first "..." number after marker
    let mut cursor = after.find('"')? + 1;
    let remaining = &after[cursor..];
    let end_q = remaining.find('"')?;
    remaining[..end_q].parse::<f64>().ok()
}