-- =============================================================
-- QTS V8 · TimescaleDB 实时市场数据库 - Prediction Market
-- 数据库: qts_market
-- Schema: prediction
-- 覆盖: Polymarket CLOB 二元期权 — Tick / OrderBook / Resolution
--        + Parity Arbitrage / Spread Tightening / Liquidity Pulse
-- =============================================================

CREATE SCHEMA IF NOT EXISTS prediction;

-- =============================================================
-- ██████ 1. 市场元数据 ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.markets (
    id              VARCHAR(128)    PRIMARY KEY,           -- Polymarket token_id / condition_id
    slug            VARCHAR(256)    NOT NULL UNIQUE,       -- URL slug
    question        TEXT            NOT NULL,
    description     TEXT,
    tags            TEXT[],
    category        VARCHAR(128),
    -- 二元期权参数
    outcomes        TEXT[]          DEFAULT '{YES,NO}',
    outcome_prices  TEXT[]          DEFAULT '{"0.50","0.50"}',
    volume          NUMERIC(20, 2)  DEFAULT 0,
    liquidity       NUMERIC(20, 2)  DEFAULT 0,
    -- 时间
    created_at      TIMESTAMPTZ,
    start_date      TIMESTAMPTZ,
    end_date        TIMESTAMPTZ,                            -- 事件发生时间
    resolution_time TIMESTAMPTZ,                            -- 实际解决时间
    -- 解决
    resolved        BOOLEAN         DEFAULT FALSE,
    resolution      VARCHAR(32),                            -- 'YES' | 'NO' | 'INVALID' | NULL
    resolution_source VARCHAR(256),
    -- 状态
    active          BOOLEAN         DEFAULT TRUE,
    closed          BOOLEAN         DEFAULT FALSE,
    archived        BOOLEAN         DEFAULT FALSE,
    -- 源
    source          VARCHAR(32)     DEFAULT 'polymarket',
    last_updated    TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX idx_pm_slug      ON prediction.markets (slug);
CREATE INDEX idx_pm_category   ON prediction.markets (category, active);
CREATE INDEX idx_pm_end_date   ON prediction.markets (end_date) WHERE active = TRUE;
CREATE INDEX idx_pm_resolved   ON prediction.markets (resolved, resolution_time DESC);

-- =============================================================
-- ██████ 2. 逐笔成交 (CLOB Tick) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.tick (
    ts_ns       BIGINT          NOT NULL,
    ts          TIMESTAMPTZ     GENERATED ALWAYS AS
                (to_timestamp(ts_ns::float8 / 1e9)) STORED,
    market_id   VARCHAR(128)    NOT NULL,
    trade_id    VARCHAR(128)    NOT NULL,
    price       NUMERIC(12, 6)  NOT NULL,              -- 0.00 ~ 1.00 美元
    size         NUMERIC(18, 6)  NOT NULL,              -- 成交数量 (token)
    notional    NUMERIC(18, 6)  GENERATED ALWAYS AS (price * size) STORED,
    side        VARCHAR(4)      NOT NULL,               -- 'BUY' | 'SELL'
    taker_side  VARCHAR(4),                             -- TAKER 方向
    outcome     VARCHAR(16),                            -- 'YES' | 'NO' (from trade)
    -- Parity
    yes_price   NUMERIC(12, 6),                         -- 同一时刻 YES token 价格
    no_price    NUMERIC(12, 6),                         -- 同一时刻 NO token 价格
    parity_dev  NUMERIC(12, 6)  GENERATED ALWAYS AS
                (yes_price + no_price - 1.0) STORED,   -- 偏离值 → 套利信号
    -- 链上
    transaction_hash VARCHAR(66),
    block_number BIGINT,
    -- 来源
    source      VARCHAR(32)     DEFAULT 'clob'
);

SELECT create_hypertable('prediction.tick', 'ts',
    chunk_time_interval => INTERVAL '4 hours', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ptick_market_ts ON prediction.tick (market_id, ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uidx_ptick_trade ON prediction.tick (market_id, trade_id, ts);

SELECT add_retention_policy('prediction.tick', INTERVAL '90 days', if_not_exists => TRUE);

ALTER TABLE prediction.tick SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market_id',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('prediction.tick', INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 3. 盘口快照 (CLOB OrderBook) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.orderbook (
    ts_ns       BIGINT          NOT NULL,
    ts          TIMESTAMPTZ     GENERATED ALWAYS AS
                (to_timestamp(ts_ns::float8 / 1e9)) STORED,
    market_id   VARCHAR(128)    NOT NULL,
    token_id    VARCHAR(128),                            -- YES/NO token-specific orderbook
    outcome     VARCHAR(16)     NOT NULL,                -- 'YES' | 'NO'
    side        VARCHAR(4)      NOT NULL,                -- 'bid' | 'ask'
    level       SMALLINT        NOT NULL,
    price       NUMERIC(12, 6)  NOT NULL,
    size         NUMERIC(18, 6)  NOT NULL,
    -- 衍生
    best_bid    NUMERIC(12, 6),
    best_ask    NUMERIC(12, 6),
    mid         NUMERIC(12, 6)  GENERATED ALWAYS AS
                ((COALESCE(best_bid, price) + COALESCE(best_ask, price)) / 2) STORED,
    spread      NUMERIC(12, 6)  GENERATED ALWAYS AS
                (COALESCE(best_ask, price) - COALESCE(best_bid, price)) STORED,
    obi         NUMERIC(10, 6),                         -- OBI = (bid_depth - ask_depth) / total_depth
    seq_id      BIGINT,
    source      VARCHAR(32)     DEFAULT 'clob'
);

SELECT create_hypertable('prediction.orderbook', 'ts',
    chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_pob_market_ts ON prediction.orderbook (market_id, ts DESC);
SELECT add_retention_policy('prediction.orderbook', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE prediction.orderbook SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market_id',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('prediction.orderbook', INTERVAL '1 day', if_not_exists => TRUE);

-- =============================================================
-- ██████ 4. 分钟级聚合快照 (1m 粒度) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.minute_snapshot (
    ts          TIMESTAMPTZ     NOT NULL,
    market_id   VARCHAR(128)    NOT NULL,
    -- YES token
    yes_price   NUMERIC(12, 6),
    yes_volume  NUMERIC(18, 6),
    yes_vwap    NUMERIC(12, 6),
    -- NO token
    no_price    NUMERIC(12, 6),
    no_volume   NUMERIC(18, 6),
    no_vwap     NUMERIC(12, 6),
    -- 衍生指标
    parity      NUMERIC(12, 6)  GENERATED ALWAYS AS (yes_price + no_price - 1.0) STORED,
    volume_usd  NUMERIC(18, 6)  GENERATED ALWAYS AS (yes_volume * yes_vwap + no_volume * no_vwap) STORED,
    -- 深度
    best_bid    NUMERIC(12, 6),
    best_ask    NUMERIC(12, 6),
    spread      NUMERIC(12, 6),
    obi         NUMERIC(10, 6),
    -- 链上
    transactions INTEGER,
    unique_traders INTEGER,
    -- 来源
    source      VARCHAR(32)     DEFAULT 'clob'
);

SELECT create_hypertable('prediction.minute_snapshot', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_pms_market_ts ON prediction.minute_snapshot (market_id, ts DESC);
SELECT add_retention_policy('prediction.minute_snapshot', INTERVAL '180 days', if_not_exists => TRUE);

ALTER TABLE prediction.minute_snapshot SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market_id',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('prediction.minute_snapshot', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ 5. 事件解决记录 ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.resolutions (
    id              BIGSERIAL       PRIMARY KEY,
    market_id       VARCHAR(128)    NOT NULL,
    resolved_at     TIMESTAMPTZ     NOT NULL,
    outcome         VARCHAR(32)     NOT NULL,            -- 'YES' | 'NO' | 'INVALID'
    yes_payout      NUMERIC(6, 4)  DEFAULT 1.0,         -- 1.0 表示全额赔付
    no_payout       NUMERIC(6, 4)  DEFAULT 0.0,
    source          VARCHAR(256),
    transaction_hash VARCHAR(66),
    block_number    BIGINT,
    oracle          VARCHAR(128)    DEFAULT 'UMA',
    disputed        BOOLEAN         DEFAULT FALSE,
    dispute_rounds  INTEGER         DEFAULT 0
);

CREATE INDEX idx_pres_market ON prediction.resolutions (market_id);
CREATE INDEX idx_pres_ts     ON prediction.resolutions (resolved_at DESC);

-- =============================================================
-- ██████ 6. 链上事件日志 (Polygon) ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.onchain_events (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    market_id       VARCHAR(128)    NOT NULL,
    event_type      VARCHAR(64)     NOT NULL,            -- 'Split' | 'Merge' | 'Redeem' | 'Mint' | 'Transfer'
    transaction_hash VARCHAR(66)    NOT NULL,
    block_number    BIGINT          NOT NULL,
    log_index       INTEGER,
    -- 事件参数
    sender          VARCHAR(42),
    recipient       VARCHAR(42),
    token_id        VARCHAR(128),
    amount          NUMERIC(38, 0),
    collateral      NUMERIC(38, 0),
    -- 解析后
    parsed_data     JSONB,
    source          VARCHAR(32)     DEFAULT 'polygon_rpc'
);

SELECT create_hypertable('prediction.onchain_events', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_poe_market_ts ON prediction.onchain_events (market_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_poe_tx ON prediction.onchain_events (transaction_hash);

-- =============================================================
-- ██████ 7. 跨市场平价套利追踪 ██████
-- =============================================================
CREATE TABLE IF NOT EXISTS prediction.parity_arbitrage (
    ts          TIMESTAMPTZ     NOT NULL,
    market_id   VARCHAR(128)    NOT NULL,
    yes_price   NUMERIC(12, 6)  NOT NULL,
    no_price    NUMERIC(12, 6)  NOT NULL,
    parity      NUMERIC(12, 6)  GENERATED ALWAYS AS (yes_price + no_price - 1.0) STORED,
    abs_parity   NUMERIC(12, 6)  GENERATED ALWAYS AS (ABS(yes_price + no_price - 1.0)) STORED,
    -- 套利信号
    arb_signal  VARCHAR(16),                             -- 'buy_yes_sell_no' | 'buy_no_sell_yes' | 'none'
    arb_spread   NUMERIC(12, 6),                         -- 扣除手续费后套利空间
    net_profit   NUMERIC(12, 6),                         -- 预期净收益
    -- 执行
    executed    BOOLEAN         DEFAULT FALSE,
    fill_ts     TIMESTAMPTZ,
    pnl         NUMERIC(12, 6)
);

SELECT create_hypertable('prediction.parity_arbitrage', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ppa_market_ts ON prediction.parity_arbitrage (market_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ppa_signal ON prediction.parity_arbitrage (arb_signal, ts DESC)
    WHERE arb_signal IS NOT NULL AND arb_signal != 'none';

-- =============================================================
-- ██████ 8. 市场类别聚合视图 ██████
-- =============================================================
CREATE OR REPLACE VIEW prediction.active_markets AS
SELECT
    m.id, m.slug, m.question, m.category,
    m.volume, m.liquidity,
    m.end_date,
    EXTRACT(EPOCH FROM (m.end_date - NOW())) / 3600 AS hours_to_resolution,
    m.outcome_prices[1] AS yes_last,
    m.outcome_prices[2] AS no_last,
    CAST(m.outcome_prices[1] AS NUMERIC) + CAST(m.outcome_prices[2] AS NUMERIC) - 1.0 AS parity
FROM prediction.markets m
WHERE m.active = TRUE AND m.closed = FALSE;

CREATE OR REPLACE VIEW prediction.parity_alerts AS
SELECT
    ts, market_id, yes_price, no_price,
    parity, arb_signal, net_profit
FROM prediction.parity_arbitrage
WHERE abs_parity > 0.02
  AND ts >= NOW() - INTERVAL '5 minutes'
ORDER BY abs_parity DESC;

CREATE OR REPLACE VIEW prediction.volume_leaderboard AS
SELECT
    market_id,
    COUNT(*) AS tick_count,
    SUM(notional) FILTER (WHERE side = 'BUY') AS buy_volume,
    SUM(notional) FILTER (WHERE side = 'SELL') AS sell_volume,
    SUM(notional) AS total_volume,
    AVG(price) AS avg_price,
    MAX(ts) AS last_trade
FROM prediction.tick
WHERE ts >= NOW() - INTERVAL '1 hour'
GROUP BY market_id
ORDER BY total_volume DESC;
