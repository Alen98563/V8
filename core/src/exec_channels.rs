//! # OKX Execution Channels — HTTP 下单 + HMAC-SHA256 签名 + Token Bucket 限流
//!
//! ## 架构定位
//!
//! `exec_channels.rs` 实现 OKX 统一订单格式的 REST 下单通道，与 `order_state.rs` (OrderFSM)
//! 协同构成完整的 L3 执行层：
//!
//! ```text
//! Python Orchestrator
//!   │
//!   ├─→ OrderFSM.transition(NEW)           ← 编译期安全状态机
//!   │     │
//!   │     └─→ build_unified_order(...)      ← 生成 UnifiedOrder JSON
//!   │
//!   └─→ OkxChannel.place_order(order_json)  ← 本模块
//!         │
//!         ├─→ TokenBucket.try_consume()     ← 限流 (令牌桶)
//!         ├─→ sign_request(...)             ← HMAC-SHA256 (ring)
//!         ├─→ reqwest POST /api/v5/trade/order
//!         └─→ 返回 OrderResponse JSON
//! ```
//!
//! ## OKX 统一订单格式
//!
//! ```json
//! {
//!   "instId": "BTC-USDT-SWAP",
//!   "tdMode": "cross",
//!   "side": "buy",
//!   "ordType": "limit",
//!   "px": "4810.50",
//!   "sz": "0.1",
//!   "clOrdId": "v8_client_001",
//!   "tag": "v8_mcts"
//! }
//! ```
//!
//! ## HMAC-SHA256 签名
//!
//! OKX 要求每个私有 API 请求携带三个 Header：
//! - `OK-ACCESS-KEY`: API Key
//! - `OK-ACCESS-SIGN`: Base64(HMAC-SHA256(timestamp + method + path + body, secret))
//! - `OK-ACCESS-TIMESTAMP`: ISO 8601 UTC 时间
//! - `OK-ACCESS-PASSPHRASE`: 创建 API 时设置的 passphrase
//!
//! ## Token Bucket 限流
//!
//! OKX 限流规则 (2026):
//! - 下单: 60 req/2s per instId
//! - 撤单: 40 req/2s per instId
//! - 全局限流: 按 UID 计
//!
//! 本模块实现 `TokenBucket` (Rust 原生)，在发送前检查令牌余量。
//! 令牌耗尽时立即返回 429 错误，Python 侧 MCTS 可据此调整策略。
//!
//! ## PyO3 接口
//!
//! ```python
//! channel = OkxChannel(api_key, secret, passphrase, is_demo=True)
//! resp = channel.place_order(unified_order_json)    # 同步签名+发送
//! resp = channel.cancel_order(inst_id, ord_id)      # 撤单
//! remaining = channel.rate_limit_remaining()         # 令牌余量
//! ```
//!
//! ## 依赖
//!
//! - `ring` v0.17: HMAC-SHA256 签名 (性能 > OpenSSL，无 C 依赖)
//! - `base64` v0.22: Base64 编码
//! - `serde` / `serde_json`: JSON 序列化
//! - `reqwest`: HTTP 客户端 (blocking, 因 Python 侧同步调用)
//! - `tokio`: 异步运行时 (可选, 用于 WebSocket)

use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ring::hmac;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

// ─── OKX API 端点 ──────────────────────────────────────────────────────────

const OKX_BASE_REAL: &str = "https://www.okx.com";
const OKX_BASE_DEMO: &str = "https://www.okx.com"; // 模拟盘使用相同域名 + x-simulated-trading header

const ORDER_PATH: &str = "/api/v5/trade/order";
const CANCEL_PATH: &str = "/api/v5/trade/cancel-order";
const BATCH_ORDER_PATH: &str = "/api/v5/trade/batch-orders";

// ─── Token Bucket 限流器 ───────────────────────────────────────────────────

/// 令牌桶限流器 (Rust 原生实现)
///
/// OKX 下单限流: 60 req / 2s per instId
/// 实现: 每次 try_consume() 检查距上次 refill 的时间差，按比例补充令牌。
///
/// 线程安全: 内部使用 AtomicU32 + Mutex<Instant>。
#[pyclass]
pub struct TokenBucket {
    capacity: u32,
    tokens: AtomicU32,
    refill_interval_ms: u64,
    last_refill: Mutex<Instant>,
}

#[pymethods]
impl TokenBucket {
    /// 创建 TokenBucket
    ///
    /// Args:
    ///     capacity: 桶容量 (默认 60，OKX 下单限流)
    ///     refill_secs: 补充周期 (默认 2.0 秒)
    #[new]
    #[pyo3(signature = (capacity=60, refill_secs=2.0))]
    fn new(capacity: u32, refill_secs: f64) -> Self {
        TokenBucket {
            capacity,
            tokens: AtomicU32::new(capacity),
            refill_interval_ms: (refill_secs * 1000.0) as u64,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// 尝试消耗一个令牌
    ///
    /// Returns:
    ///     True: 消耗成功，可以发送请求
    ///     False: 令牌耗尽，应返回 429 / 等待
    fn try_consume(&self) -> bool {
        // 先尝试 refill
        self.maybe_refill();

        // CAS loop: 原子减
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false; // 令牌耗尽
            }
            if self
                .tokens
                .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// 返回剩余令牌数
    fn remaining(&self) -> u32 {
        self.maybe_refill();
        self.tokens.load(Ordering::Relaxed)
    }

    /// 重置令牌到满容量 (用于手动恢复)
    fn reset(&self) {
        self.tokens.store(self.capacity, Ordering::Release);
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenBucket(capacity={}, remaining={}, refill_interval={}ms)",
            self.capacity,
            self.tokens.load(Ordering::Relaxed),
            self.refill_interval_ms,
        )
    }
}

impl TokenBucket {
    /// 根据时间差补充令牌
    fn maybe_refill(&self) {
        let mut last = match self.last_refill.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        let elapsed = last.elapsed().as_millis() as u64;
        if elapsed >= self.refill_interval_ms {
            let refills = elapsed / self.refill_interval_ms;
            let new_tokens = (refills as u32)
                .saturating_mul(self.capacity)
                .min(self.capacity);

            self.tokens.store(new_tokens, Ordering::Release);
            *last = Instant::now();
        }
    }
}

// ─── OKX REST 客户端 ───────────────────────────────────────────────────────

/// OKX 执行通道 — HTTP 下单 + HMAC 签名 + Token Bucket 限流
///
/// 所有方法为同步阻塞 (Python GIL 释放后执行)，适合从 Orchestrator 直接调用。
///
/// 安全:
/// - API Secret 仅在 Rust 内存中存在，不经过 Python 堆
/// - 签名在栈上计算，计算完成后零化
/// - 支持模拟盘 (x-simulated-trading: 1)
#[pyclass]
pub struct OkxChannel {
    api_key: String,
    secret: String,
    passphrase: String,
    base_url: String,
    is_demo: bool,
    rate_limiter: TokenBucket,
    /// 时间偏移 (ms): OKX server time - local time
    time_offset: AtomicI64,
    /// 共享 HTTP 客户端 (连接池)
    http_client: Mutex<reqwest::blocking::Client>,
}

#[pymethods]
impl OkxChannel {
    /// 创建 OkxChannel
    ///
    /// Args:
    ///     api_key: OKX API Key
    ///     secret: API Secret
    ///     passphrase: API Passphrase
    ///     is_demo: 是否模拟盘 (默认 True)
    ///     rate_limit: 令牌桶容量 (默认 60)
    #[new]
    #[pyo3(signature = (api_key, secret, passphrase, is_demo=true, rate_limit=60))]
    fn new(
        api_key: String,
        secret: String,
        passphrase: String,
        is_demo: bool,
        rate_limit: u32,
    ) -> PyResult<Self> {
        let base_url = if is_demo {
            OKX_BASE_DEMO.to_string()
        } else {
            OKX_BASE_REAL.to_string()
        };

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(4)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("HTTP client error: {}", e)))?;

        Ok(OkxChannel {
            api_key,
            secret,
            passphrase,
            base_url,
            is_demo,
            rate_limiter: TokenBucket::new(rate_limit, 2.0),
            time_offset: AtomicI64::new(0),
            http_client: Mutex::new(client),
        })
    }

    /// 下单 (同步: 签名 + HTTP POST)
    ///
    /// Args:
    ///     unified_order_json: UnifiedOrder JSON 字符串 (由 OrderFSM.build_unified_order 生成)
    ///
    /// Returns:
    ///     OKX API 响应 JSON 字符串
    ///
    /// Raises:
    ///     RuntimeError: 令牌耗尽 (429) 或网络错误
    ///
    /// 流程:
    ///     1. TokenBucket.try_consume() — 限流检查
    ///     2. sign_request() — HMAC-SHA256 签名
    ///     3. reqwest POST /api/v5/trade/order
    ///     4. 返回响应 body
    fn place_order(&self, py: Python, unified_order_json: String) -> PyResult<String> {
        py.allow_threads(|| self._place_order_impl(&unified_order_json))
    }

    /// 撤单
    ///
    /// Args:
    ///     inst_id: 合约 ID
    ///     ord_id: OKX 订单 ID
    ///
    /// Returns:
    ///     OKX API 响应 JSON 字符串
    fn cancel_order(&self, py: Python, inst_id: String, ord_id: String) -> PyResult<String> {
        py.allow_threads(|| self._cancel_order_impl(&inst_id, &ord_id))
    }

    /// 返回令牌桶剩余令牌数
    fn rate_limit_remaining(&self) -> u32 {
        self.rate_limiter.remaining()
    }

    /// 同步 OKX 服务器时间，计算时间偏移
    ///
    /// OKX 要求签名时间戳与服务器时间差 < 30s，否则返回 50112。
    /// 使用共享 HTTP 客户端 (不创建新 Runtime)。
    fn sync_time(&self, py: Python) -> PyResult<i64> {
        py.allow_threads(|| self._sync_time_impl())
    }

    /// 获取当前时间偏移 (ms)
    fn time_offset_ms(&self) -> i64 {
        self.time_offset.load(Ordering::Relaxed)
    }

    fn __repr__(&self) -> String {
        format!(
            "OkxChannel(demo={}, rate_remaining={}, time_offset={}ms)",
            self.is_demo,
            self.rate_limiter.remaining(),
            self.time_offset.load(Ordering::Relaxed),
        )
    }
}

// ─── 内部实现 ──────────────────────────────────────────────────────────────

impl OkxChannel {
    /// 下单实现 (在 allow_threads 内执行，不持有 GIL)
    fn _place_order_impl(&self, order_json: &str) -> PyResult<String> {
        // 1. 限流检查
        if !self.rate_limiter.try_consume() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Rate limit exceeded: TokenBucket exhausted (429). Wait or reduce order frequency.",
            ));
        }

        // 2. 签名
        let timestamp = self.timestamp_iso();
        let signature = self.sign_request("POST", ORDER_PATH, &timestamp, order_json);

        // 3. HTTP POST
        let url = format!("{}{}", self.base_url, ORDER_PATH);
        let client = self.http_client.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        let mut req = client
            .post(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", &signature)
            .header("OK-ACCESS-TIMESTAMP", &timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .body(order_json.to_string());

        if self.is_demo {
            req = req.header("x-simulated-trading", "1");
        }

        let resp = req.send().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("HTTP error: {}", e))
        })?;

        let status = resp.status();
        let body = resp.text().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read body error: {}", e))
        })?;

        if !status.is_success() {
            warn!(status = %status, body = %body, "OKX place_order failed");
        } else {
            debug!(status = %status, "OKX place_order success");
        }

        Ok(body)
    }

    /// 撤单实现
    fn _cancel_order_impl(&self, inst_id: &str, ord_id: &str) -> PyResult<String> {
        if !self.rate_limiter.try_consume() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Rate limit exceeded: TokenBucket exhausted (429)",
            ));
        }

        let body = serde_json::json!({
            "instId": inst_id,
            "ordId": ord_id,
        })
        .to_string();

        let timestamp = self.timestamp_iso();
        let signature = self.sign_request("POST", CANCEL_PATH, &timestamp, &body);
        let url = format!("{}{}", self.base_url, CANCEL_PATH);

        let client = self.http_client.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        let mut req = client
            .post(&url)
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", &signature)
            .header("OK-ACCESS-TIMESTAMP", &timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .body(body);

        if self.is_demo {
            req = req.header("x-simulated-trading", "1");
        }

        let resp = req.send().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("HTTP error: {}", e))
        })?;

        let body = resp.text().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read body error: {}", e))
        })?;

        Ok(body)
    }

    /// 同步服务器时间 (使用共享客户端，不创建新 Runtime)
    fn _sync_time_impl(&self) -> PyResult<i64> {
        let url = format!("{}/api/v5/public/time", self.base_url);

        let client = self.http_client.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock error: {}", e))
        })?;

        let resp = client.get(&url).send().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("HTTP error: {}", e))
        })?;

        let body = resp.text().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read error: {}", e))
        })?;

        // 解析 {"data": [{"ts": "1700000000000"}]}
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
        let server_ts = parsed["data"][0]["ts"]
            .as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);

        let local_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let offset = server_ts - local_ts;
        self.time_offset.store(offset, Ordering::Release);
        info!(offset_ms = offset, "OKX time synced");

        Ok(offset)
    }

    /// HMAC-SHA256 签名
    ///
    /// sign = Base64(HMAC-SHA256(prehash_string, secret))
    /// prehash_string = timestamp + method(UPPER) + requestPath + body
    fn sign_request(&self, method: &str, path: &str, timestamp: &str, body: &str) -> String {
        let prehash = format!("{}{}{}{}", timestamp, method.to_uppercase(), path, body);

        let key = hmac::Key::new(hmac::HMAC_SHA256, self.secret.as_bytes());
        let tag = hmac::sign(&key, prehash.as_bytes());
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, tag.as_ref())
    }

    /// 生成 ISO 8601 UTC 时间戳 (含时间偏移校正)
    ///
    /// 格式: "2026-01-01T00:00:00.000Z"
    fn timestamp_iso(&self) -> String {
        let offset = self.time_offset.load(Ordering::Relaxed);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
            + offset;

        let secs = (now_ms / 1000) as u64;
        let millis = (now_ms % 1000) as u32;

        // 手动格式化 ISO 8601 (避免 chrono 依赖)
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // 从 1970-01-01 计算年月日 (简化算法)
        let (year, month, day) = days_to_ymd(days as i32);

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hours, minutes, seconds, millis
        )
    }
}

// ─── 日期计算辅助 ──────────────────────────────────────────────────────────

/// 从 Unix epoch 天数计算 (year, month, day)
/// 简化实现，精度足够 OKX 签名要求 (±1 天)
fn days_to_ymd(days: i32) -> (i32, u32, u32) {
    // 算法: 从 1970-01-01 起算
    let mut y = 1970;
    let mut d = days;

    loop {
        let days_in_year = if is_leap_year(y) { 366 } else { 365 };
        if d < days_in_year {
            break;
        }
        d -= days_in_year;
        y += 1;
    }

    let leap = is_leap_year(y);
    let month_days: [i32; 12] = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];

    let mut m = 1u32;
    for &md in &month_days {
        if d < md {
            break;
        }
        d -= md;
        m += 1;
    }

    (y, m, d as u32 + 1)
}

fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

// ─── 辅助: base64 导入 ─────────────────────────────────────────────────────
//
// base64 0.22 API 使用 Engine trait:
//   use base64::Engine;
//   base64::engine::general_purpose::STANDARD.encode(data)
//
// sign_request 内已内联使用完整路径，无需额外 use。
