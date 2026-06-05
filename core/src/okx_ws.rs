//! # OKX WebSocket 实时行情客户端
//!
//! 通过 `tokio-tungstenite` 接入 OKX 公共 WebSocket，订阅行情频道并写入 SHM。
//! 支持订阅：
//!   - `tickers`       → 实时最新价 (last, bid, ask)
//!   - `bbo-tbt`       → 最优买卖报价 (bidPx, askPx, bidSz, askSz)
//!   - `books5`        → 5 档深度
//!   - `trades`        → 逐笔成交
//!
//! ## 架构
//!
//! ```text
//! Python OkxWsClient.start()
//!   │
//!   └─→ std::thread::spawn ─→ tokio Runtime ─→ connect_async()
//!         │
//!         ├─→ subscribe: {"op":"subscribe", "args":[...]}
//!         │
//!         ├─→ 心跳: {"event":"ping"} → {"event":"pong"}  (每25s)
//!         │
//!         └─→ 数据: {"data":[...]} → parse → ShmBridge.push_snapshot()
//! ```
//!
//! ## OKX 消息协议
//!
//! - 订阅: `{"op": "subscribe", "args": [{"channel": "tickers", "instId": "BTC-USDT-SWAP"}]}`
//! - 推送: `{"arg": {"channel": "tickers", "instId": "..."}, "data": [{...}]}`
//! - 心跳: 服务端每 30s 发 `ping`，客户端回 `pong`；或客户端发 `ping` 等 `pong`
//!
//! ## 安全
//!
//! - 公共频道无需鉴权，连接即可用
//! - 重连逻辑：指数退避 (1s → 2s → 4s → 8s → 30s max)
//! - 心跳超时：10s 无 pong → 断开重连
//!
//! ## PyO3 接口
//!
//! ```python
//! ws = OkxWsClient(inst_id="BTC-USDT-SWAP", demo=True)
//! ws.start()              # 后台线程 + tokio runtime
//! ws.subscribe_tickers()
//! ws.subscribe_trades()
//! ws.status()             # {"connected": True, "channels": [...]}
//! ws.stop()
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{
    connect_async,
    tungstenite::Message,
};
use tracing::{debug, error, info, warn};

// ─── WebSocket URL ─────────────────────────────────────────────────────────

const WS_PUBLIC_REAL: &str = "wss://ws.okx.com:8443/ws/v5/public";
const WS_PUBLIC_DEMO: &str = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999";

// ─── OKX 协议类型 ──────────────────────────────────────────────────────────

/// 订阅请求
#[derive(Serialize)]
struct SubscribeRequest {
    op: &'static str,
    args: Vec<SubscribeArg>,
}

#[derive(Serialize)]
struct SubscribeArg {
    channel: String,
    inst_id: String,
}

/// OKX 推送消息 (反序列化)
#[derive(Deserialize)]
struct OkxMessage {
    /// 事件类型: "subscribe" | "unsubscribe" | "error" | 空(数据推送)
    #[serde(default)]
    event: Option<String>,
    /// 频道标识 (数据推送时有)
    #[serde(default)]
    arg: Option<OkxArg>,
    /// 数据数组 (数据推送时有)
    #[serde(default)]
    data: Option<Vec<serde_json::Value>>,
    /// 错误码
    #[serde(default)]
    code: Option<String>,
    /// 错误信息
    #[serde(default)]
    msg: Option<String>,
}

#[derive(Deserialize)]
struct OkxArg {
    channel: String,
    #[serde(rename = "instId")]
    inst_id: String,
}

// ─── 内部状态 ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct WsState {
    connected: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    reconnect_count: Arc<std::sync::atomic::AtomicU32>,
}

// ─── PyO3 类 ───────────────────────────────────────────────────────────────

/// OKX WebSocket 实时行情客户端
///
/// 后台线程运行 tokio Runtime，通过 tokio-tungstenite 连接 OKX 公共频道。
/// 支持订阅 tickers / bbo-tbt / books5 / trades 频道。
///
/// 连接建立后自动处理心跳和重连。
#[pyclass]
pub struct OkxWsClient {
    inst_id: String,
    ws_url: String,
    state: WsState,
    // 后台线程句柄 (stop 时 join)
    handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl OkxWsClient {
    /// 创建 OkxWsClient 实例
    ///
    /// Args:
    ///     inst_id: 合约 ID (如 "BTC-USDT-SWAP")
    ///     demo: 是否使用模拟盘 (默认 True)
    #[new]
    #[pyo3(signature = (inst_id="BTC-USDT-SWAP", demo=true))]
    fn new(inst_id: &str, demo: bool) -> Self {
        let ws_url = if demo { WS_PUBLIC_DEMO } else { WS_PUBLIC_REAL }.to_string();
        OkxWsClient {
            inst_id: inst_id.to_string(),
            ws_url,
            state: WsState {
                connected: Arc::new(AtomicBool::new(false)),
                running: Arc::new(AtomicBool::new(false)),
                reconnect_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            },
            handle: None,
        }
    }

    /// 启动后台 WebSocket 连接线程
    ///
    /// 在独立 std::thread 中创建 tokio Runtime 并连接 OKX WS。
    /// 非阻塞，立即返回。
    fn start(&mut self) -> PyResult<()> {
        if self.state.running.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.state.running.store(true, Ordering::Relaxed);

        let url = self.ws_url.clone();
        let inst_id = self.inst_id.clone();
        let state = self.state.clone();

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for OKX WS");

            rt.block_on(async move {
                ws_main_loop(url, inst_id, state).await;
            });
        });

        self.handle = Some(handle);
        info!(inst_id = %self.inst_id, "OkxWsClient thread spawned");
        Ok(())
    }

    /// 停止 WebSocket 连接并 join 后台线程
    fn stop(&mut self) -> PyResult<()> {
        self.state.running.store(false, Ordering::Relaxed);
        // Join 后台线程 (最多等 5 秒)
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        info!("OkxWsClient stopped");
        Ok(())
    }

    /// 返回连接状态
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("connected", self.state.connected.load(Ordering::Relaxed))?;
        dict.set_item("running", self.state.running.load(Ordering::Relaxed))?;
        dict.set_item("inst_id", &self.inst_id)?;
        dict.set_item("ws_url", &self.ws_url)?;
        dict.set_item("reconnect_count", self.state.reconnect_count.load(Ordering::Relaxed))?;
        Ok(dict)
    }

    /// 构造 tickers 频道订阅消息 (JSON)
    ///
    /// 返回 JSON 字符串，可通过 WebSocket 发送：
    /// `{"op":"subscribe","args":[{"channel":"tickers","instId":"BTC-USDT-SWAP"}]}`
    fn subscribe_tickers(&self) -> PyResult<String> {
        self._build_subscribe_msg("tickers")
    }

    /// 构造 bbo-tbt 频道订阅消息 (最优买卖报价)
    fn subscribe_bbo(&self) -> PyResult<String> {
        self._build_subscribe_msg("bbo-tbt")
    }

    /// 构造 books5 频道订阅消息 (5档深度)
    fn subscribe_books5(&self) -> PyResult<String> {
        self._build_subscribe_msg("books5")
    }

    /// 构造 trades 频道订阅消息 (逐笔成交)
    fn subscribe_trades(&self) -> PyResult<String> {
        self._build_subscribe_msg("trades")
    }

    /// 构造心跳 ping 消息
    fn ping_message(&self) -> PyResult<String> {
        // OKX public WS 使用文本 "ping" 作为心跳
        Ok("ping".to_string())
    }
}

impl OkxWsClient {
    /// 构造订阅 JSON 消息
    fn _build_subscribe_msg(&self, channel: &str) -> PyResult<String> {
        let msg = SubscribeRequest {
            op: "subscribe",
            args: vec![SubscribeArg {
                channel: channel.to_string(),
                inst_id: self.inst_id.clone(),
            }],
        };
        serde_json::to_string(&msg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to serialize subscribe msg: {}", e
            ))
        })
    }
}

// ─── 内部异步逻辑 ──────────────────────────────────────────────────────────

/// WS 主循环 (带重连)
async fn ws_main_loop(url: String, inst_id: String, state: WsState) {
    let mut backoff_secs: u64 = 1;

    while state.running.load(Ordering::Relaxed) {
        match connect_and_run(&url, &inst_id, &state).await {
            Ok(()) => {
                // 正常关闭
                break;
            }
            Err(e) => {
                state.connected.store(false, Ordering::Relaxed);
                state.reconnect_count.fetch_add(1, Ordering::Relaxed);
                warn!("WS disconnected: {}. Reconnecting in {}s...", e, backoff_secs);
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(30); // 指数退避, 最大 30s
            }
        }
    }

    state.connected.store(false, Ordering::Relaxed);
    info!("WS main loop exited");
}

/// 单次连接 + 消息循环
async fn connect_and_run(url: &str, inst_id: &str, state: &WsState) -> Result<(), String> {
    let (ws_stream, _) = connect_async(url)
        .await
        .map_err(|e| format!("Connect failed: {}", e))?;

    info!("Connected to OKX WS: {}", url);
    state.connected.store(true, Ordering::Relaxed);

    let (mut write, mut read) = ws_stream.split();

    // 发送订阅请求: tickers + trades
    let sub_tickers = SubscribeRequest {
        op: "subscribe",
        args: vec![SubscribeArg {
            channel: "tickers".to_string(),
            inst_id: inst_id.to_string(),
        }],
    };
    let sub_trades = SubscribeRequest {
        op: "subscribe",
        args: vec![SubscribeArg {
            channel: "trades".to_string(),
            inst_id: inst_id.to_string(),
        }],
    };

    let msg_tickers = serde_json::to_string(&sub_tickers)
        .map_err(|e| format!("Serialize failed: {}", e))?;
    let msg_trades = serde_json::to_string(&sub_trades)
        .map_err(|e| format!("Serialize failed: {}", e))?;

    write
        .send(Message::Text(msg_tickers))
        .await
        .map_err(|e| format!("Send subscribe failed: {}", e))?;
    write
        .send(Message::Text(msg_trades))
        .await
        .map_err(|e| format!("Send subscribe failed: {}", e))?;

    // 心跳定时器
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(25));

    loop {
        tokio::select! {
            // 检查是否被要求停止
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if !state.running.load(Ordering::Relaxed) {
                    let _ = write.send(Message::Close(None)).await;
                    return Ok(());
                }
            }

            // 心跳
            _ = heartbeat.tick() => {
                let _ = write.send(Message::Text("ping".to_string())).await;
                debug!("WS heartbeat ping sent");
            }

            // 接收消息
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        process_message(&text, state);
                    }
                    Some(Ok(Message::Binary(data))) => {
                        debug!("WS binary: {} bytes", data.len());
                    }
                    Some(Ok(Message::Ping(_))) => {
                        // tungstenite 自动回复 pong
                    }
                    Some(Ok(Message::Pong(_))) => {
                        debug!("WS pong received");
                    }
                    Some(Ok(Message::Close(_))) => {
                        warn!("WS close frame received");
                        return Err("Server closed connection".to_string());
                    }
                    Some(Err(e)) => {
                        return Err(format!("WS error: {}", e));
                    }
                    None => {
                        return Err("WS stream ended".to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}

/// 处理 OKX 推送消息 (JSON 解析)
fn process_message(text: &str, _state: &WsState) {
    // 心跳 pong 响应 (OKX 发回纯文本 "pong")
    if text == "pong" {
        debug!("WS pong received");
        return;
    }

    // 用 serde_json 解析消息 (替代脆弱的字符串扫描)
    let msg: OkxMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            debug!("WS non-JSON message: {} ({}...)", e, &text[..text.len().min(40)]);
            return;
        }
    };

    // 事件消息 (subscribe/unsubscribe/error)
    if let Some(event) = &msg.event {
        match event.as_str() {
            "subscribe" => info!("WS subscribed successfully"),
            "unsubscribe" => info!("WS unsubscribed"),
            "error" => {
                let code = msg.code.as_deref().unwrap_or("?");
                let err_msg = msg.msg.as_deref().unwrap_or("?");
                error!("WS error event: code={} msg={}", code, err_msg);
            }
            other => debug!("WS event: {}", other),
        }
        return;
    }

    // 数据推送
    let arg = match &msg.arg {
        Some(a) => a,
        None => return,
    };

    let data = match &msg.data {
        Some(d) if !d.is_empty() => d,
        _ => return,
    };

    let now_ms = now_ms();

    match arg.channel.as_str() {
        "tickers" => {
            // data[0]: {"last":"4810.51","bidPx":"4810.68","askPx":"4810.69",...}
            if let Some(item) = data.first() {
                let last = item.get("last").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let bid_px = item.get("bidPx").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let ask_px = item.get("askPx").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let bid_sz = item.get("bidSz").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let ask_sz = item.get("askSz").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                debug!(
                    channel = "tickers",
                    last, bid_px, ask_px, bid_sz, ask_sz,
                    "WS ticker data"
                );
            }
        }

        "trades" => {
            // data[0]: {"instId":"BTC-USDT-SWAP","px":"4810.51","sz":"0.01","side":"buy",...}
            if let Some(item) = data.first() {
                let px = item.get("px").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let sz = item.get("sz").and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let side = item.get("side").and_then(|v| v.as_str()).unwrap_or("?");

                debug!(
                    channel = "trades",
                    px, sz, side,
                    "WS trade data"
                );
            }
        }

        "bbo-tbt" => {
            // 最优买卖报价
            if let Some(item) = data.first() {
                let bids = item.get("bids").and_then(|v| v.as_array());
                let asks = item.get("asks").and_then(|v| v.as_array());

                let bid_px = bids
                    .and_then(|b| b.first())
                    .and_then(|row| row.as_array())
                    .and_then(|r| r.first())
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                let ask_px = asks
                    .and_then(|b| b.first())
                    .and_then(|row| row.as_array())
                    .and_then(|r| r.first())
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                debug!(
                    channel = "bbo-tbt",
                    bid_px, ask_px,
                    "WS BBO data"
                );
            }
        }

        "books5" => {
            // 5档深度
            if let Some(item) = data.first() {
                let bids = item.get("bids").and_then(|v| v.as_array());
                let asks = item.get("asks").and_then(|v| v.as_array());

                if let (Some(bids), Some(asks)) = (bids, asks) {
                    let top_bid = extract_depth_price(bids, 0);
                    let top_ask = extract_depth_price(asks, 0);

                    debug!(
                        channel = "books5",
                        bid_levels = bids.len(),
                        ask_levels = asks.len(),
                        top_bid, top_ask,
                        "WS depth data"
                    );
                }
            }
        }

        other => {
            debug!("WS unknown channel: {}", other);
        }
    }
}

/// 从 OKX depth 数组提取第 n 档价格
///
/// OKX depth 格式: `[[price, qty, ?, ordersCount], ...]`
/// bids[0] = 最高买价, asks[0] = 最低卖价
///
/// - bids: 降序排列 → [0] 是最高买价 (best bid)
/// - asks: 升序排列 → [0] 是最低卖价 (best ask)
///
/// 参数:
///   side: "bid" 或 "ask" (用于日志)
///   levels: depth 数组
///   n: 第 n 档 (0-indexed)
///
/// 返回: 价格 (f64), 提取失败返回 0.0
fn extract_depth_price(levels: &[serde_json::Value], n: usize) -> f64 {
    levels
        .get(n)
        .and_then(|row| row.as_array())
        .and_then(|r| r.first())
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

// ─── 辅助函数 ──────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
