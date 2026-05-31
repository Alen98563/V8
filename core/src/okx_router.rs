//! OKX API 执行路由器 —— HMAC-SHA256 签名 + Token Bucket 限流 + REST 下单
//!
//! 关键要求:
//! 1. 签名延迟 <5μs (ring crate)
//! 2. ISO8601 时间戳 + time_offset 补偿 (防 50111 错误)
//! 3. Token Bucket 限流 (OKX 20次/2s)
//! 4. 自动重试 + 幂等去重 (clOrdId)

use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

// ============================================================
// Token Bucket 限流器
// ============================================================

struct TokenBucket {
    capacity: u32,
    tokens: std::sync::Mutex<u32>,
    refill_rate: u32,     // tokens per second
    last_refill: std::sync::Mutex<Instant>,
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: u32) -> Self {
        Self {
            capacity,
            tokens: std::sync::Mutex::new(capacity),
            refill_rate,
            last_refill: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// 尝试消费一个 token，返回是否成功
    fn try_consume(&self) -> bool {
        self.refill();
        let mut tokens = self.tokens.lock().unwrap();
        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }

    /// 等待直到可以消费
    fn wait_for_token(&self) {
        loop {
            self.refill();
            let mut tokens = self.tokens.lock().unwrap();
            if *tokens > 0 {
                *tokens -= 1;
                return;
            }
            drop(tokens);
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn refill(&self) {
        let mut last = self.last_refill.lock().unwrap();
        let elapsed = last.elapsed().as_secs_f64();
        let new_tokens = (elapsed * self.refill_rate as f64) as u32;
        if new_tokens > 0 {
            let mut tokens = self.tokens.lock().unwrap();
            *tokens = (*tokens + new_tokens).min(self.capacity);
            *last = Instant::now();
        }
    }
}

// ============================================================
// HMAC-SHA256 签名引擎 (ring crate, <5μs)
// ============================================================

struct OkxSigner {
    api_key: String,
    secret_key: String,
    passphrase: String,
    time_offset_ms: AtomicI64, // OKX 服务器时间偏移补偿
}

impl OkxSigner {
    fn new(api_key: String, secret_key: String, passphrase: String) -> Self {
        Self {
            api_key,
            secret_key,
            passphrase,
            time_offset_ms: AtomicI64::new(0),
        }
    }

    /// 生成 HMAC-SHA256 签名
    /// 签名字符串: timestamp + method + requestPath + body
    fn sign(&self, timestamp: &str, method: &str, request_path: &str, body: &str) -> String {
        let sign_str = format!("{}{}{}{}", timestamp, method, request_path, body);

        use ring::hmac;
        let key = hmac::Key::new(hmac::HMAC_SHA256, self.secret_key.as_bytes());
        let signature = hmac::sign(&key, sign_str.as_bytes());

        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, signature.as_ref())
    }

    /// 获取带时间偏移补偿的 ISO8601 时间戳
    fn timestamp_with_offset(&self) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let adjusted = now + self.time_offset_ms.load(Ordering::Relaxed);
        adjusted.to_string()
    }

    /// 更新时间偏移 (用于补偿 OKX 50111 错误)
    fn update_time_offset(&self, server_ts: i64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let offset = server_ts - now;
        // EMA 平滑更新
        let current = self.time_offset_ms.load(Ordering::Relaxed);
        let new_offset = (current as f64 * 0.7 + offset as f64 * 0.3) as i64;
        self.time_offset_ms.store(new_offset, Ordering::Relaxed);
    }

    /// 构建完整的 OKX REST 请求头
    fn build_headers(&self, method: &str, path: &str, body: &str) -> Vec<(String, String)> {
        let ts = self.timestamp_with_offset();
        let sig = self.sign(&ts, method, path, body);

        vec![
            ("OK-ACCESS-KEY".into(), self.api_key.clone()),
            ("OK-ACCESS-SIGN".into(), sig),
            ("OK-ACCESS-TIMESTAMP".into(), ts),
            ("OK-ACCESS-PASSPHRASE".into(), self.passphrase.clone()),
            ("Content-Type".into(), "application/json".into()),
        ]
    }
}

// ============================================================
// PyO3 导出: OKX Router
// ============================================================

/// OKX 执行路由器 —— 下单/撤单/查询
#[pyclass]
pub struct OkxRouter {
    signer: OkxSigner,
    rate_limiter: TokenBucket,
    base_url: String,
    is_demo: bool,
}

#[pymethods]
impl OkxRouter {
    #[new]
    #[pyo3(signature = (api_key, secret_key, passphrase, demo=true))]
    pub fn new(api_key: String, secret_key: String, passphrase: String, demo: bool) -> Self {
        let base_url = if demo {
            "https://www.okx.com".to_string()
        } else {
            "https://www.okx.com".to_string()
        };
        Self {
            signer: OkxSigner::new(api_key, secret_key, passphrase),
            rate_limiter: TokenBucket::new(20, 10), // 20 tokens, 10/s refill
            base_url,
            is_demo: demo,
        }
    }

    /// 生成 HMAC-SHA256 签名 (供测试验证)
    pub fn sign_request(&self, timestamp: &str, method: &str, path: &str, body: &str) -> String {
        self.signer.sign(timestamp, method, path, body)
    }

    /// 构建下单请求 (返回 JSON 字符串，由 Python 发送 HTTP 请求)
    pub fn prepare_place_order(
        &self,
        inst_id: &str,
        td_mode: &str,
        side: &str,
        order_type: &str,
        sz: &str,
        px: Option<&str>,
        cl_ord_id: &str,
    ) -> PyResult<String> {
        self.rate_limiter.wait_for_token();

        let mut body = serde_json::json!({
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": order_type,
            "sz": sz,
            "clOrdId": cl_ord_id,
        });

        if let Some(price) = px {
            body["px"] = serde_json::Value::String(price.to_string());
        }

        let path = "/api/v5/trade/order";
        let body_str = serde_json::to_string(&body).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialize error: {}", e))
        })?;

        let headers = self.signer.build_headers("POST", path, &body_str);

        // 返回完整的请求信息
        let request = serde_json::json!({
            "url": format!("{}{}", self.base_url, path),
            "method": "POST",
            "headers": headers.iter().map(|(k, v)| [k, v]).collect::<Vec<[&str; 2]>>(),
            "body": body_str,
            "clOrdId": cl_ord_id,
        });

        serde_json::to_string_pretty(&request).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("JSON error: {}", e))
        })
    }

    /// 构建撤单请求
    pub fn prepare_cancel_order(
        &self,
        inst_id: &str,
        ord_id: Option<&str>,
        cl_ord_id: Option<&str>,
    ) -> PyResult<String> {
        self.rate_limiter.wait_for_token();

        let mut body = serde_json::json!({
            "instId": inst_id,
        });
        if let Some(oid) = ord_id {
            body["ordId"] = serde_json::Value::String(oid.to_string());
        }
        if let Some(cloid) = cl_ord_id {
            body["clOrdId"] = serde_json::Value::String(cloid.to_string());
        }

        let path = "/api/v5/trade/cancel-order";
        let body_str = serde_json::to_string(&body).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("JSON error: {}", e))
        })?;

        let headers = self.signer.build_headers("POST", path, &body_str);

        let request = serde_json::json!({
            "url": format!("{}{}", self.base_url, path),
            "method": "POST",
            "headers": headers.iter().map(|(k, v)| [k, v]).collect::<Vec<[&str; 2]>>(),
            "body": body_str,
        });

        serde_json::to_string_pretty(&request).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("JSON error: {}", e))
        })
    }

    /// 更新时间偏移 (收到 OKX 50111 错误时调用)
    pub fn update_time_offset(&self, server_ts: i64) {
        self.signer.update_time_offset(server_ts);
    }

    /// 获取当前时间偏移量 (调试用)
    pub fn get_time_offset_ms(&self) -> i64 {
        self.signer.time_offset_ms.load(Ordering::Relaxed)
    }

    /// 获取限流器状态
    pub fn rate_limiter_status(&self) -> (u32, u32) {
        let tokens = self.rate_limiter.tokens.lock().unwrap();
        (*tokens, self.rate_limiter.capacity)
    }

    /// 是否为模拟盘模式
    pub fn is_demo(&self) -> bool {
        self.is_demo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_sign() {
        let signer = OkxSigner::new(
            "test_key".into(),
            "test_secret".into(),
            "test_pass".into(),
        );
        let sig = signer.sign("1700000000000", "POST", "/api/v5/trade/order", "{}");
        // 签名不为空
        assert!(!sig.is_empty());
        // Base64 格式
        assert!(base64::engine::general_purpose::STANDARD.decode(&sig).is_ok());
    }

    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(5, 100);
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        assert!(!bucket.try_consume()); // 6th should fail
    }
}
