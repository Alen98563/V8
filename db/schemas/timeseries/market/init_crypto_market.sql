-- =============================================================
-- QTS V8 · TimescaleDB 实时市场数据库 - Crypto
-- 数据库: qts_market
-- 覆盖: OKX / Binance ETH-USDT-SWAP 等永续合约全量行情
-- =============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE SCHEMA IF NOT EXISTS crypto;
CREATE SCHEMA IF NOT EXISTS meta;

-- =============================================================
-- ██████ 1. 逐笔成交 (Trade Tick) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.tick (
    ts          TIMESTAMPTZ     NOT NULL,
    ts_ns       BIGINT          NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    exchange    VARCHAR(32)     NOT NULL,               -- 'OKX' | 'Binance'
    trade_id    VARCHAR(64)     NOT NULL,
    price       NUMERIC(24, 8)  NOT NULL,
    size        NUMERIC(24, 8)  NOT NULL,
    side        CHAR(4)         NOT NULL,
    trade_mode  VARCHAR(16),
    source      VARCHAR(16)     DEFAULT 'ws'
);

SELECT create_hypertable('crypto.tick', 'ts',
    chunk_time_interval => INTERVAL '4 hours', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ctick_ticker_ts ON crypto.tick (ticker, ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uidx_ctick_trade ON crypto.tick (ticker, trade_id, ts);

SELECT add_retention_policy('crypto.tick', INTERVAL '14 days', if_not_exists => TRUE);

ALTER TABLE crypto.tick SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('crypto.tick', INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 2. 盘口快照 (OrderBook - books5) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.orderbook (
    ts      TIMESTAMPTZ     NOT NULL,
    ts_ns   BIGINT          NOT NULL,
    ticker  VARCHAR(32)     NOT NULL,
    side    CHAR(4)         NOT NULL,
    level   SMALLINT        NOT NULL,
    price   NUMERIC(24, 8)  NOT NULL,
    size    NUMERIC(24, 8)  NOT NULL,
    count   INTEGER,
    seq_id  BIGINT,
    action  VARCHAR(16)     DEFAULT 'snapshot'
);

SELECT create_hypertable('crypto.orderbook', 'ts',
    chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cob_ticker_ts ON crypto.orderbook (ticker, ts DESC);
SELECT add_retention_policy('crypto.orderbook', INTERVAL '3 days', if_not_exists => TRUE);

ALTER TABLE crypto.orderbook SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker, side',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('crypto.orderbook', INTERVAL '1 day', if_not_exists => TRUE);

-- =============================================================
-- ██████ 3. OHLCV K 线 ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.ohlcv (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    bar         VARCHAR(8)      NOT NULL,               -- '1m'|'3m'|'5m'|'15m'|'1H'|'4H'|'1D'
    open        NUMERIC(24, 8)  NOT NULL,
    high        NUMERIC(24, 8)  NOT NULL,
    low         NUMERIC(24, 8)  NOT NULL,
    close       NUMERIC(24, 8)  NOT NULL,
    vol         NUMERIC(32, 8)  NOT NULL,
    vol_ccy     NUMERIC(32, 4),
    vol_ccy_quote NUMERIC(32, 4),
    confirm     BOOLEAN         DEFAULT TRUE,
    source      VARCHAR(16)     DEFAULT 'ws'
);

SELECT create_hypertable('crypto.ohlcv', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_cohlcv ON crypto.ohlcv (ticker, bar, ts);

ALTER TABLE crypto.ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker, bar',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('crypto.ohlcv', INTERVAL '30 days', if_not_exists => TRUE);

-- 连续聚合：1m → 5m
CREATE MATERIALIZED VIEW IF NOT EXISTS crypto.ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', ts)   AS bucket,
    ticker,
    FIRST(open,  ts)               AS open,
    MAX(high)                      AS high,
    MIN(low)                       AS low,
    LAST(close,  ts)               AS close,
    SUM(vol)                       AS vol,
    SUM(vol_ccy)                   AS vol_ccy
FROM crypto.ohlcv
WHERE bar = '1m'
GROUP BY bucket, ticker
WITH NO DATA;

SELECT add_continuous_aggregate_policy('crypto.ohlcv_5m',
    start_offset      => INTERVAL '30 minutes',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE
);

-- =============================================================
-- ██████ 4. 资金费率 (Funding Rate) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.funding_rate (
    ts                  TIMESTAMPTZ     NOT NULL,
    ticker              VARCHAR(32)     NOT NULL,
    funding_rate        NUMERIC(16, 10) NOT NULL,
    next_funding_rate   NUMERIC(16, 10),
    next_funding_time   TIMESTAMPTZ,
    method              VARCHAR(32)     DEFAULT 'next_period_min',
    realized_rate       NUMERIC(16, 10),
    source              VARCHAR(16)     DEFAULT 'rest'
);

SELECT create_hypertable('crypto.funding_rate', 'ts',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_cfunding ON crypto.funding_rate (ticker, ts);

-- =============================================================
-- ██████ 5. 标记价格 & 指数价格 ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.mark_price (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    mark_px     NUMERIC(24, 8)  NOT NULL,
    index_px    NUMERIC(24, 8),
    basis       NUMERIC(16, 8)  GENERATED ALWAYS AS (mark_px - index_px) STORED,
    basis_rate  NUMERIC(16, 10) GENERATED ALWAYS AS
                (CASE WHEN index_px > 0 THEN (mark_px - index_px) / index_px ELSE NULL END) STORED,
    source      VARCHAR(16)     DEFAULT 'ws'
);

SELECT create_hypertable('crypto.mark_price', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cmark_ticker_ts ON crypto.mark_price (ticker, ts DESC);
SELECT add_retention_policy('crypto.mark_price', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 6. 持仓量 (Open Interest) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.open_interest (
    ts      TIMESTAMPTZ     NOT NULL,
    ticker  VARCHAR(32)     NOT NULL,
    oi      NUMERIC(32, 4)  NOT NULL,
    oi_ccy  NUMERIC(32, 8),
    oi_usd  NUMERIC(32, 4),
    source  VARCHAR(16)     DEFAULT 'rest'
);

SELECT create_hypertable('crypto.open_interest', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_coi_ticker_ts ON crypto.open_interest (ticker, ts DESC);
SELECT add_retention_policy('crypto.open_interest', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 7. 强平数据 (Liquidations) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.liquidations (
    ts      TIMESTAMPTZ     NOT NULL,
    ticker  VARCHAR(32)     NOT NULL,
    side    CHAR(4)         NOT NULL,
    bk_px   NUMERIC(24, 8)  NOT NULL,
    sz      NUMERIC(24, 8)  NOT NULL,
    bk_loss NUMERIC(24, 8),
    source  VARCHAR(16)     DEFAULT 'ws'
);

SELECT create_hypertable('crypto.liquidations', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cliq_ticker_ts ON crypto.liquidations (ticker, ts DESC);
SELECT add_retention_policy('crypto.liquidations', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 8. 多空比 (L/S Ratio) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.ls_ratio (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    ratio_type  VARCHAR(32)     NOT NULL,
    long_ratio  NUMERIC(10, 6)  NOT NULL,
    short_ratio NUMERIC(10, 6)  NOT NULL,
    ls_ratio    NUMERIC(10, 6)  GENERATED ALWAYS AS
                (long_ratio / NULLIF(short_ratio, 0)) STORED,
    source      VARCHAR(16)     DEFAULT 'rest'
);

SELECT create_hypertable('crypto.ls_ratio', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_clsr_ticker_ts ON crypto.ls_ratio (ticker, ts DESC);
SELECT add_retention_policy('crypto.ls_ratio', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 9. 市场状态 (Regime Detection) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS crypto.regime (
    ts              TIMESTAMPTZ     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    regime          VARCHAR(16)     NOT NULL,           -- 'trending'|'ranging'|'volatile'|'chaotic'
    regime_score    FLOAT4,
    hurst           FLOAT4,
    vol_regime      VARCHAR(16),                        -- 'low'|'normal'|'high'|'extreme'
    vol_percentile  FLOAT4,
    detection_model VARCHAR(32)     DEFAULT 'hmm'
);

SELECT create_hypertable('crypto.regime', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_creg_ticker_ts ON crypto.regime (ticker, ts DESC);

-- =============================================================
-- META: 数据采集质量审计
-- =============================================================
CREATE TABLE IF NOT EXISTS meta.ingestion_log (
    id          BIGSERIAL       PRIMARY KEY,
    ts          TIMESTAMPTZ     DEFAULT NOW(),
    source      VARCHAR(32)     NOT NULL,
    table_name  VARCHAR(64)     NOT NULL,
    ticker      VARCHAR(32),
    records_in  INTEGER,
    records_ok  INTEGER,
    records_err INTEGER,
    latency_ms  FLOAT,
    error_msg   TEXT
);

CREATE TABLE IF NOT EXISTS meta.ws_health (
    ts              TIMESTAMPTZ     DEFAULT NOW() PRIMARY KEY,
    exchange        VARCHAR(32)     NOT NULL,
    channel         VARCHAR(64)     NOT NULL,
    status          VARCHAR(16)     NOT NULL,
    last_msg_ts     TIMESTAMPTZ,
    lag_ms          FLOAT,
    reconnect_count INTEGER DEFAULT 0
);

-- =============================================================
-- 常用查询视图
-- =============================================================
CREATE OR REPLACE VIEW crypto.latest_snapshot AS
SELECT DISTINCT ON (ticker)
    ts, ticker, price AS last_price, side AS last_side, size AS last_size
FROM crypto.tick
ORDER BY ticker, ts DESC;

CREATE OR REPLACE VIEW crypto.current_funding AS
SELECT DISTINCT ON (ticker)
    ts, ticker, funding_rate, next_funding_rate, next_funding_time,
    EXTRACT(EPOCH FROM (next_funding_time - NOW())) / 60 AS minutes_to_settlement
FROM crypto.funding_rate
ORDER BY ticker, ts DESC;

CREATE OR REPLACE VIEW crypto.current_obi AS
SELECT
    ticker,
    MAX(ts) AS ts,
    SUM(CASE WHEN side='bid' AND level=1 THEN size ELSE 0 END) AS bid1_sz,
    SUM(CASE WHEN side='ask' AND level=1 THEN size ELSE 0 END) AS ask1_sz,
    SUM(CASE WHEN side='bid' THEN size ELSE 0 END) AS total_bid_sz,
    SUM(CASE WHEN side='ask' THEN size ELSE 0 END) AS total_ask_sz,
    (SUM(CASE WHEN side='bid' THEN size ELSE 0 END) -
     SUM(CASE WHEN side='ask' THEN size ELSE 0 END)) /
    NULLIF(SUM(size), 0) AS obi
FROM crypto.orderbook
WHERE ts >= NOW() - INTERVAL '1 minute'
GROUP BY ticker;
