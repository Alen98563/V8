-- =============================================================
-- QTS V8 · TimescaleDB 期货行情 Schema
-- 数据库: qts_market
-- 数据源: yfinance / IBKR (实时 Tick) + CFTC COT (免费周报)
-- =============================================================

CREATE SCHEMA IF NOT EXISTS futures;

-- =============================================================
-- 1. 期货逐笔成交 + 报价 (热数据, 90 天)
-- =============================================================
CREATE TABLE IF NOT EXISTS futures.tick (
    ts              TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(8)      NOT NULL,
    contract_month  CHAR(6)         NOT NULL,
    expiry          DATE            NOT NULL,
    bid             NUMERIC(18, 8),
    ask             NUMERIC(18, 8),
    last            NUMERIC(18, 8)  NOT NULL,
    volume          INTEGER         DEFAULT 0,
    open_interest   INTEGER,
    settlement      NUMERIC(18, 8),
    tick_size       NUMERIC(12, 8),
    tick_value      NUMERIC(12, 4),
    session         VARCHAR(8)     DEFAULT 'RTH',
    source          VARCHAR(16)    DEFAULT 'ibkr'
);

SELECT create_hypertable(
    'futures.tick', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fut_tick_symbol  ON futures.tick (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_fut_tick_contract ON futures.tick (symbol, contract_month, ts DESC);
CREATE INDEX IF NOT EXISTS idx_fut_tick_oi       ON futures.tick (symbol, contract_month, open_interest DESC);

SELECT add_retention_policy('futures.tick', INTERVAL '90 days', if_not_exists => TRUE);

COMMENT ON TABLE futures.tick IS '期货逐笔成交+报价热数据，90天全精度保留，OI为换月判断核心';

-- =============================================================
-- 2. 期货 OHLCV K线
-- =============================================================
CREATE TABLE IF NOT EXISTS futures.ohlcv (
    ts              TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(8)      NOT NULL,
    contract_month  CHAR(6)         NOT NULL,
    timeframe       VARCHAR(4)      NOT NULL,
    open            NUMERIC(18, 8)  NOT NULL,
    high            NUMERIC(18, 8)  NOT NULL,
    low             NUMERIC(18, 8)  NOT NULL,
    close           NUMERIC(18, 8)  NOT NULL,
    volume          INTEGER         DEFAULT 0,
    open_interest   INTEGER,
    vwap            NUMERIC(18, 8),
    trades          INTEGER,
    source          VARCHAR(16)     DEFAULT 'ibkr'
);

SELECT create_hypertable(
    'futures.ohlcv', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_fut_ohlcv ON futures.ohlcv (symbol, contract_month, timeframe, ts);
CREATE INDEX       IF NOT EXISTS idx_fut_ohlcv_sym ON futures.ohlcv (symbol, ts DESC);

CREATE MATERIALIZED VIEW IF NOT EXISTS futures.ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS bucket,
    symbol,
    contract_month,
    FIRST(open,  ts) AS open,
    MAX(high)        AS high,
    MIN(low)         AS low,
    LAST(close,  ts) AS close,
    SUM(volume)      AS volume,
    LAST(open_interest, ts) AS open_interest
FROM futures.ohlcv
WHERE timeframe = '1m'
GROUP BY bucket, symbol, contract_month
WITH NO DATA;

SELECT add_continuous_aggregate_policy('futures.ohlcv_1h',
    start_offset     => INTERVAL '3 hours',
    end_offset       => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists    => TRUE
);

-- =============================================================
-- 3. 期限结构 — Contango/Backwardation 判断
-- =============================================================
CREATE TABLE IF NOT EXISTS futures.term_structure (
    ts              TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(8)      NOT NULL,
    contract_month  CHAR(6)         NOT NULL,
    expiry          DATE            NOT NULL,
    dte             SMALLINT        NOT NULL,
    price           NUMERIC(18, 8)  NOT NULL,
    spot_price      NUMERIC(18, 8),
    basis           NUMERIC(18, 8),
    basis_pct       NUMERIC(10, 6),
    roll_spread     NUMERIC(18, 8),
    regime          VARCHAR(16),
    open_interest   INTEGER,
    volume          INTEGER
);

SELECT create_hypertable(
    'futures.term_structure', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fut_ts_symbol ON futures.term_structure (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_fut_ts_dte    ON futures.term_structure (symbol, dte, ts DESC);

COMMENT ON TABLE futures.term_structure IS '期货期限结构 | contango/backwardation | 展期价差 | 曲线形态分析';

-- =============================================================
-- 4. CFTC 每周持仓报告 — 免费, 周五更新
-- =============================================================
CREATE TABLE IF NOT EXISTS futures.cot (
    report_date     DATE            NOT NULL,
    symbol          VARCHAR(8)      NOT NULL,
    exchange        VARCHAR(16)     NOT NULL,
    contract_name   VARCHAR(128),
    comm_long       INTEGER,
    comm_short      INTEGER,
    managed_long    INTEGER,
    managed_short   INTEGER,
    nonrep_long     INTEGER,
    nonrep_short    INTEGER,
    net_spec_pos    INTEGER,
    spec_long_pct   NUMERIC(6, 4),
    spec_short_pct  NUMERIC(6, 4),
    cot_index       NUMERIC(6, 4),
    oi_total        INTEGER,
    spread_pos      NUMERIC(12, 2),
    PRIMARY KEY (report_date, symbol)
);

COMMENT ON TABLE futures.cot IS 'CFTC 每周持仓报告 — 聪明钱/散户仓位判断，完全免费';

-- =============================================================
-- 5. 便捷视图
-- =============================================================

CREATE OR REPLACE VIEW futures.latest_quotes AS
SELECT DISTINCT ON (symbol, contract_month)
    ts, symbol, contract_month, expiry, bid, ask, last, volume, open_interest
FROM futures.tick
ORDER BY symbol, contract_month, ts DESC;

CREATE OR REPLACE VIEW futures.active_contracts AS
SELECT DISTINCT ON (symbol)
    symbol, contract_month, expiry, open_interest, last, ts
FROM futures.tick
WHERE open_interest IS NOT NULL
ORDER BY symbol, open_interest DESC;

CREATE OR REPLACE VIEW futures.curve AS
SELECT
    ts, symbol, contract_month, dte, price, basis_pct, regime
FROM futures.term_structure
WHERE ts >= NOW() - INTERVAL '1 day';
