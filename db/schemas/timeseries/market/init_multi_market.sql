-- =============================================================
-- QTS V8 · TimescaleDB 实时市场数据库 - Multi-Asset
-- 数据库: qts_market
-- 覆盖: 美股 Tick/OHLCV | 期权报价 Greeks | 外汇 Tick/盘口
-- =============================================================

CREATE SCHEMA IF NOT EXISTS equity;
CREATE SCHEMA IF NOT EXISTS options;
CREATE SCHEMA IF NOT EXISTS fx;

-- =============================================================
-- ██████ EQUITY — 美股 ██████
-- =============================================================

-- 逐笔成交
CREATE TABLE IF NOT EXISTS equity.tick (
    ts          TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(16)     NOT NULL,
    exchange    CHAR(4)         NOT NULL,
    price       NUMERIC(18, 6)  NOT NULL,
    size        BIGINT          NOT NULL,
    bid         NUMERIC(18, 6),
    ask         NUMERIC(18, 6),
    bid_size    INTEGER,
    ask_size    INTEGER,
    conditions  VARCHAR(32),
    tape        CHAR(1),
    source      VARCHAR(16)     DEFAULT 'polygon'
);

SELECT create_hypertable('equity.tick', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_etick_sym_ts ON equity.tick (symbol, ts DESC);
SELECT add_retention_policy('equity.tick', INTERVAL '90 days', if_not_exists => TRUE);

-- OHLCV
CREATE TABLE IF NOT EXISTS equity.ohlcv (
    ts          TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(16)     NOT NULL,
    timeframe   VARCHAR(4)      NOT NULL,
    open        NUMERIC(18, 6)  NOT NULL,
    high        NUMERIC(18, 6)  NOT NULL,
    low         NUMERIC(18, 6)  NOT NULL,
    close       NUMERIC(18, 6)  NOT NULL,
    volume      BIGINT          NOT NULL,
    vwap        NUMERIC(18, 6),
    trades      INTEGER,
    source      VARCHAR(16)     DEFAULT 'polygon'
);

SELECT create_hypertable('equity.ohlcv', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_eohlcv ON equity.ohlcv (symbol, timeframe, ts);

-- 盘口快照
CREATE TABLE IF NOT EXISTS equity.orderbook (
    ts      TIMESTAMPTZ     NOT NULL,
    symbol  VARCHAR(16)     NOT NULL,
    side    CHAR(1)         NOT NULL,
    price   NUMERIC(18, 6)  NOT NULL,
    size    BIGINT          NOT NULL,
    level   SMALLINT        NOT NULL
);

SELECT create_hypertable('equity.orderbook', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

SELECT add_retention_policy('equity.orderbook', INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ OPTIONS — 期权 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS options.quote (
    ts            TIMESTAMPTZ     NOT NULL,
    underlying    VARCHAR(16)     NOT NULL,
    expiry        DATE            NOT NULL,
    strike        NUMERIC(12, 4)  NOT NULL,
    option_type   CHAR(1)         NOT NULL,
    bid           NUMERIC(12, 6),
    ask           NUMERIC(12, 6),
    last          NUMERIC(12, 6),
    volume        INTEGER         DEFAULT 0,
    open_interest INTEGER,
    iv            NUMERIC(8, 6),
    delta         NUMERIC(8, 6),
    gamma         NUMERIC(10, 8),
    theta         NUMERIC(10, 6),
    vega          NUMERIC(10, 6),
    rho           NUMERIC(10, 6),
    model         VARCHAR(16)     DEFAULT 'BSM',
    source        VARCHAR(16)     DEFAULT 'cboe'
);

SELECT create_hypertable('options.quote', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_oq_underlying   ON options.quote (underlying, ts DESC);
CREATE INDEX IF NOT EXISTS idx_oq_contract     ON options.quote (underlying, expiry, strike, option_type, ts DESC);
SELECT add_retention_policy('options.quote', INTERVAL '90 days', if_not_exists => TRUE);

-- 波动率曲面
CREATE TABLE IF NOT EXISTS options.vol_surface (
    ts            TIMESTAMPTZ     NOT NULL,
    underlying    VARCHAR(16)     NOT NULL,
    expiry        DATE            NOT NULL,
    moneyness     NUMERIC(8, 4)   NOT NULL,
    iv            NUMERIC(8, 6)   NOT NULL,
    surface_type  VARCHAR(16)     DEFAULT 'market'
);

SELECT create_hypertable('options.vol_surface', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- 期权链日终快照
CREATE TABLE IF NOT EXISTS options.eod_chain (
    trade_date    DATE            NOT NULL,
    underlying    VARCHAR(16)     NOT NULL,
    expiry        DATE            NOT NULL,
    strike        NUMERIC(12, 4)  NOT NULL,
    option_type   CHAR(1)         NOT NULL,
    open          NUMERIC(12, 6),
    high          NUMERIC(12, 6),
    low           NUMERIC(12, 6),
    close         NUMERIC(12, 6),
    volume        INTEGER,
    open_interest INTEGER,
    iv_close      NUMERIC(8, 6),
    delta         NUMERIC(8, 6),
    gamma         NUMERIC(10, 8),
    theta         NUMERIC(10, 6),
    vega          NUMERIC(10, 6),
    PRIMARY KEY (trade_date, underlying, expiry, strike, option_type)
);

-- =============================================================
-- ██████ FX — 外汇 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS fx.tick (
    ts          TIMESTAMPTZ     NOT NULL,
    pair        VARCHAR(8)      NOT NULL,
    bid         NUMERIC(18, 8)  NOT NULL,
    ask         NUMERIC(18, 8)  NOT NULL,
    mid         NUMERIC(18, 8)  GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    spread      NUMERIC(10, 6)  GENERATED ALWAYS AS (ask - bid) STORED,
    bid_size    NUMERIC(16, 2),
    ask_size    NUMERIC(16, 2),
    venue       VARCHAR(16)     DEFAULT 'oanda',
    source      VARCHAR(16)     DEFAULT 'oanda'
);

SELECT create_hypertable('fx.tick', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ftick_pair ON fx.tick (pair, ts DESC);
SELECT add_retention_policy('fx.tick', INTERVAL '90 days', if_not_exists => TRUE);

-- OHLCV
CREATE TABLE IF NOT EXISTS fx.ohlcv (
    ts          TIMESTAMPTZ     NOT NULL,
    pair        VARCHAR(8)      NOT NULL,
    timeframe   VARCHAR(4)      NOT NULL,
    open        NUMERIC(18, 8)  NOT NULL,
    high        NUMERIC(18, 8)  NOT NULL,
    low         NUMERIC(18, 8)  NOT NULL,
    close       NUMERIC(18, 8)  NOT NULL,
    volume      NUMERIC(20, 2),
    source      VARCHAR(16)     DEFAULT 'oanda'
);

SELECT create_hypertable('fx.ohlcv', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_fohlcv ON fx.ohlcv (pair, timeframe, ts);

-- 远期汇率 & 掉期点
CREATE TABLE IF NOT EXISTS fx.forward (
    ts          TIMESTAMPTZ     NOT NULL,
    pair        VARCHAR(8)      NOT NULL,
    tenor       VARCHAR(8)      NOT NULL,
    bid         NUMERIC(18, 8),
    ask         NUMERIC(18, 8),
    swap_points NUMERIC(12, 6),
    source      VARCHAR(16)     DEFAULT 'bloomberg'
);

SELECT create_hypertable('fx.forward', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================
-- 视图
-- =============================================================
CREATE OR REPLACE VIEW equity.latest_quotes AS
SELECT DISTINCT ON (symbol)
    ts, symbol, price, bid, ask, size
FROM equity.tick ORDER BY symbol, ts DESC;

CREATE OR REPLACE VIEW fx.latest_quotes AS
SELECT DISTINCT ON (pair)
    ts, pair, bid, ask, mid, spread, venue
FROM fx.tick ORDER BY pair, ts DESC;

CREATE OR REPLACE VIEW options.iv_summary AS
SELECT
    underlying, expiry, option_type,
    COUNT(*) AS contracts,
    AVG(iv) AS avg_iv, MIN(iv) AS min_iv, MAX(iv) AS max_iv,
    MAX(ts) AS last_update
FROM options.quote
WHERE ts >= NOW() - INTERVAL '1 day'
GROUP BY underlying, expiry, option_type;
