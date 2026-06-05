-- =============================================================
-- QTS V8 · PostgreSQL 策略运营数据库 (Enhanced)
-- 数据库: qts_ops
-- 覆盖: Crypto + Multi-Asset (Equity / Options / FX / Futures) + Prediction
-- 融合: crypto_qts_db + quant_db_kit + ORACLE-FORGE 全市场架构
-- =============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================
-- SCHEMA 划分
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ref;           -- 交易所 & 合约参考数据
CREATE SCHEMA IF NOT EXISTS strategy;      -- 策略管理 & 信号记录
CREATE SCHEMA IF NOT EXISTS portfolio;     -- 账户 & 组合 & 绩效
CREATE SCHEMA IF NOT EXISTS risk;          -- 风控规则 & 告警
CREATE SCHEMA IF NOT EXISTS calibration;   -- 模型校准 & 在线训练
CREATE SCHEMA IF NOT EXISTS audit;         -- 操作审计

-- =============================================================
-- ██████ REF — 参考数据 ██████
-- =============================================================

-- ────── 交易所配置 (统一 Crypto + 传统交易所) ──────
CREATE TABLE IF NOT EXISTS ref.exchanges (
    id              SMALLSERIAL     PRIMARY KEY,
    mic             CHAR(4)         UNIQUE,              -- ISO 10383 MIC (传统交易所)
    name            VARCHAR(128)    NOT NULL UNIQUE,
    market_type     VARCHAR(16)     NOT NULL,            -- 'crypto' | 'equity' | 'options' | 'fx'
    env             VARCHAR(16)     NOT NULL DEFAULT 'live', -- 'live' | 'demo' | 'simulated'
    country         CHAR(2),
    timezone        VARCHAR(64)     DEFAULT 'UTC',
    currency        CHAR(3),
    ws_url          VARCHAR(256),
    rest_url        VARCHAR(256),
    open_time       TIME,
    close_time      TIME,
    is_active       BOOLEAN         DEFAULT TRUE,
    notes           TEXT
);

-- Crypto 交易所种子数据
INSERT INTO ref.exchanges (mic, name, market_type, env, ws_url, rest_url, timezone) VALUES
    ('OKXX', 'OKX',            'crypto', 'live', 'wss://ws.okx.com:8443/ws/v5/public', 'https://www.okx.com', 'UTC'),
    ('OKXD', 'OKX_DEMO',       'crypto', 'demo', 'wss://wspap.okx.com:8443/ws/v5/public', 'https://www.okx.com', 'UTC'),
    ('BNBX', 'Binance',        'crypto', 'live', 'wss://stream.binance.com:9443/ws', 'https://api.binance.com', 'UTC'),
    ('BNBD', 'Binance_DEMO',   'crypto', 'demo', 'wss://testnet.binance.vision/ws', 'https://testnet.binance.vision', 'UTC')
ON CONFLICT (name) DO NOTHING;

-- 传统交易所种子数据
INSERT INTO ref.exchanges (mic, name, market_type, country, timezone, currency, open_time, close_time) VALUES
    ('XNAS', 'NASDAQ',                    'equity',  'US', 'America/New_York',  'USD', '09:30', '16:00'),
    ('XNYS', 'New York Stock Exchange',   'equity',  'US', 'America/New_York',  'USD', '09:30', '16:00'),
    ('XCBO', 'CBOE Global Markets',       'options', 'US', 'America/New_York',  'USD', '09:30', '16:15'),
    ('XLON', 'London Stock Exchange',     'equity',  'GB', 'Europe/London',     'GBP', '08:00', '16:30'),
    ('XTSE', 'Tokyo Stock Exchange',      'equity',  'JP', 'Asia/Tokyo',        'JPY', '09:00', '15:30'),
    ('XHKG', 'Hong Kong Stock Exchange',  'equity',  'HK', 'Asia/Hong_Kong',    'HKD', '09:30', '16:00')
ON CONFLICT (mic) DO NOTHING;

-- FX 数据源（无 MIC，name 替代）
INSERT INTO ref.exchanges (name, market_type, env, timezone, currency) VALUES
    ('OANDA',       'fx', 'live', 'America/New_York', 'USD'),
    ('Reuters_FX',  'fx', 'live', 'Europe/London',    'USD')
ON CONFLICT (name) DO NOTHING;

-- ────── 合约统一维度表 (Unified Instruments) ──────
CREATE TABLE IF NOT EXISTS ref.instruments (
    id              BIGSERIAL       PRIMARY KEY,
    ticker          VARCHAR(32)     NOT NULL,           -- 'ETH-USDT-SWAP' | 'AAPL' | 'EURUSD'
    market_type     VARCHAR(16)     NOT NULL,           -- 'crypto' | 'equity' | 'option' | 'fx'
    exchange_id     SMALLINT        REFERENCES ref.exchanges(id),

    -- Crypto 专属
    inst_type       VARCHAR(16),                        -- 'SWAP'|'FUTURES'|'SPOT'
    base_ccy        VARCHAR(16),                        -- 'ETH'
    quote_ccy       VARCHAR(16),                        -- 'USDT'
    settle_ccy      VARCHAR(16),                        -- 'USDT'（U本位）| 'ETH'（币本位）
    ct_val          NUMERIC(20, 8),                     -- 合约面值
    ct_mult         NUMERIC(10, 4)  DEFAULT 1,
    lot_sz          NUMERIC(20, 8),                     -- 最小下单量
    min_sz          NUMERIC(20, 8),
    tick_sz         NUMERIC(20, 8),                     -- 最小价格精度
    max_lev         NUMERIC(6, 2),                      -- 最大杠杆倍数
    funding_interval_h SMALLINT     DEFAULT 8,          -- 资金费率结算间隔

    -- Equity 专属
    name            VARCHAR(256),                       -- 公司全名
    isin            CHAR(12),
    sedol           CHAR(7),
    figi            VARCHAR(12),
    cusip           CHAR(9),
    sector          VARCHAR(64),
    industry        VARCHAR(128),
    market_cap      BIGINT,
    shares_out      BIGINT,

    -- FX 专属
    pip_decimal     SMALLINT        DEFAULT 4,
    typical_spread  NUMERIC(8, 4),
    margin_rate     NUMERIC(6, 4),

    -- 通用
    is_active       BOOLEAN         DEFAULT TRUE,
    is_tradeable    BOOLEAN         DEFAULT TRUE,
    list_date       DATE,
    expire_date     DATE,                               -- NULL = 永续/永不到期
    delist_date     DATE,
    last_updated    TIMESTAMPTZ     DEFAULT NOW(),

    UNIQUE (ticker, exchange_id)
);

CREATE INDEX idx_instr_symbol    ON ref.instruments (ticker);
CREATE INDEX idx_instr_market    ON ref.instruments (market_type, is_active);
CREATE INDEX idx_instr_sector    ON ref.instruments (sector, market_type) WHERE sector IS NOT NULL;
CREATE INDEX idx_instr_isin      ON ref.instruments (isin) WHERE isin IS NOT NULL;

-- -- Crypto 种子
-- INSERT INTO ref.instruments
--     (ticker, market_type, exchange_id, inst_type, base_ccy, quote_ccy, settle_ccy,
--      ct_val, lot_sz, min_sz, tick_sz, max_lev)
-- VALUES
--     ('ETH-USDT-SWAP', 'crypto', 1, 'SWAP', 'ETH', 'USDT', 'USDT', 0.1, 1, 1, 0.01, 100),
--     ('BTC-USDT-SWAP', 'crypto', 1, 'SWAP', 'BTC', 'USDT', 'USDT', 0.01, 1, 1, 0.1, 100);

-- ────── 期权合约详情 ──────
CREATE TABLE IF NOT EXISTS ref.option_contracts (
    id              BIGSERIAL       PRIMARY KEY,
    occ_symbol      VARCHAR(21)     UNIQUE,            -- OCC 标准合约代码
    underlying_id   BIGINT          REFERENCES ref.instruments(id),
    underlying      VARCHAR(16)     NOT NULL,
    expiry          DATE            NOT NULL,
    strike          NUMERIC(12, 4)  NOT NULL,
    option_type     CHAR(1)         NOT NULL CHECK (option_type IN ('C', 'P')),
    style           CHAR(1)         DEFAULT 'A' CHECK (style IN ('A', 'E')), -- A=美式 E=欧式
    multiplier      INTEGER         DEFAULT 100,
    settlement      VARCHAR(16)     DEFAULT 'physical',
    exchange_id     SMALLINT        REFERENCES ref.exchanges(id),
    list_date       DATE,
    expiry_type     VARCHAR(16),                       -- standard / weekly / monthly / quarterly
    is_active       BOOLEAN         DEFAULT TRUE
);

CREATE INDEX idx_opt_underlying_expiry ON ref.option_contracts (underlying, expiry);
CREATE INDEX idx_opt_expiry            ON ref.option_contracts (expiry) WHERE is_active = TRUE;

-- ────── 期货合约规格 ──────
CREATE TABLE IF NOT EXISTS ref.futures_contracts (
    id              BIGSERIAL       PRIMARY KEY,
    symbol          VARCHAR(8)      NOT NULL,
    exchange        CHAR(4)         NOT NULL REFERENCES ref.exchanges(mic),
    name            VARCHAR(256),
    group_name      VARCHAR(64),
    currency        CHAR(3)         DEFAULT 'USD',
    contract_size   NUMERIC(12, 4)  NOT NULL,
    tick_size       NUMERIC(12, 8)  NOT NULL,
    tick_value      NUMERIC(12, 4)  NOT NULL,
    margin_initial  NUMERIC(12, 2),
    margin_maintain NUMERIC(12, 2),
    trading_hours   VARCHAR(128),
    months_active   VARCHAR(64),
    expiry_rule     VARCHAR(256),
    settlement_type VARCHAR(16)     DEFAULT 'physical',
    price_format    VARCHAR(16)     DEFAULT 'decimal',
    quote_unit      NUMERIC(12, 8),
    daily_limit     NUMERIC(12, 8),
    is_mini         BOOLEAN         DEFAULT FALSE,
    is_micro        BOOLEAN         DEFAULT FALSE,
    parent_symbol   VARCHAR(8),
    is_active       BOOLEAN         DEFAULT TRUE,
    notes           TEXT
);

CREATE INDEX idx_fut_instr_symbol ON ref.futures_contracts (symbol);
CREATE INDEX idx_fut_instr_group  ON ref.futures_contracts (group_name);

INSERT INTO ref.futures_contracts (symbol, exchange, name, group_name, currency, contract_size, tick_size, tick_value, months_active, settlement_type, is_mini, is_micro)
VALUES
    ('ES', 'XCME', 'E-mini S&P 500',         'Equity Index',  'USD', 50,     0.25,     12.50, 'H,M,U,Z',   'cash',       TRUE,  FALSE),
    ('MES','XCME', 'Micro E-mini S&P 500',    'Equity Index',  'USD', 5,      0.25,      1.25, 'H,M,U,Z',   'cash',       FALSE, TRUE),
    ('NQ', 'XCME', 'E-mini NASDAQ-100',       'Equity Index',  'USD', 20,     0.25,      5.00, 'H,M,U,Z',   'cash',       TRUE,  FALSE),
    ('MNQ','XCME', 'Micro E-mini NASDAQ',     'Equity Index',  'USD', 2,      0.25,      0.50, 'H,M,U,Z',   'cash',       FALSE, TRUE),
    ('RTY','XCME', 'E-mini Russell 2000',     'Equity Index',  'USD', 50,     0.10,      5.00, 'H,M,U,Z',   'cash',       TRUE,  FALSE),
    ('CL', 'XNYM','Crude Oil',                'Energy',        'USD', 1000,   0.01,     10.00, 'F,G,H,J,K,M,N,Q,U,V,X,Z', 'physical', TRUE, FALSE),
    ('MCL','XNYM','Micro Crude Oil',          'Energy',        'USD', 100,    0.01,      1.00, 'F,G,H,J,K,M,N,Q,U,V,X,Z', 'cash',     FALSE, TRUE),
    ('NG', 'XNYM','Natural Gas',              'Energy',        'USD', 10000,  0.001,    10.00, 'F,G,H,J,K,M,N,Q,U,V,X,Z', 'physical', TRUE, FALSE),
    ('GC', 'XCOM','Gold',                     'Metals',        'USD', 100,    0.10,     10.00, 'G,J,M,Q,V,Z', 'physical', TRUE,  FALSE),
    ('MGC','XCOM','Micro Gold',               'Metals',        'USD', 10,     0.10,      1.00, 'G,J,M,Q,V,Z', 'cash',     FALSE, TRUE),
    ('SI', 'XCOM','Silver',                   'Metals',        'USD', 5000,   0.005,    25.00, 'H,K,N,U,Z',   'physical', TRUE,  FALSE),
    ('HG', 'XCOM','Copper',                   'Metals',        'USD', 25000,  0.0005,   12.50, 'H,K,N,U,Z',   'physical', TRUE,  FALSE),
    ('ZN', 'XCBT','10-Year T-Note',           'Interest Rate', 'USD', 100000, 0.015625, 15.625,'H,M,U,Z',  'physical', TRUE,  FALSE),
    ('ZB', 'XCBT','30-Year T-Bond',           'Interest Rate', 'USD', 100000, 0.03125,  31.25, 'H,M,U,Z',   'physical', TRUE,  FALSE),
    ('ZF', 'XCBT','5-Year T-Note',            'Interest Rate', 'USD', 100000, 0.0078125,7.8125,'H,M,U,Z',  'physical', TRUE,  FALSE),
    ('ZC', 'XCBT','Corn',                     'Grains',        'USD', 5000,   0.25,     12.50, 'H,K,N,U,Z',   'physical', TRUE,  FALSE),
    ('ZS', 'XCBT','Soybean',                  'Grains',        'USD', 5000,   0.25,     12.50, 'F,H,K,N,Q,U,X','physical',TRUE,  FALSE),
    ('ZW', 'XCBT','Wheat',                    'Grains',        'USD', 5000,   0.25,     12.50, 'H,K,N,U,Z',   'physical', TRUE,  FALSE)
ON CONFLICT DO NOTHING;

-- ────── 交易日历 ──────
CREATE TABLE IF NOT EXISTS ref.trading_calendar (
    exchange_id     SMALLINT        REFERENCES ref.exchanges(id),
    date            DATE            NOT NULL,
    is_trading      BOOLEAN         NOT NULL,
    open_time       TIME,
    close_time      TIME,
    notes           VARCHAR(128),
    PRIMARY KEY (exchange_id, date)
);

-- ────── Crypto 资金费率结算时间表 (G5 时间门控) ──────
CREATE TABLE IF NOT EXISTS ref.funding_schedule (
    ticker          VARCHAR(32)     NOT NULL,
    settlement_hour SMALLINT        NOT NULL,           -- 0 | 8 | 16（UTC）
    gate_before_min SMALLINT        DEFAULT 30,
    gate_after_min  SMALLINT        DEFAULT 5,
    PRIMARY KEY (ticker, settlement_hour)
);

-- ────── 外汇利率基准 ──────
CREATE TABLE IF NOT EXISTS ref.interest_rates (
    id              BIGSERIAL       PRIMARY KEY,
    currency        CHAR(3)         NOT NULL,
    tenor           VARCHAR(8)      NOT NULL,           -- ON, 1W, 1M, 3M, 6M, 1Y
    rate            NUMERIC(10, 6)  NOT NULL,
    rate_type       VARCHAR(16)     DEFAULT 'SOFR',     -- SOFR / EURIBOR / LIBOR
    effective_date  DATE            NOT NULL,
    source          VARCHAR(64),
    UNIQUE (currency, tenor, effective_date)
);

-- =============================================================
-- ██████ STRATEGY — 策略管理 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS strategy.strategies (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(128)    NOT NULL UNIQUE,
    version         VARCHAR(16)     DEFAULT '1.0.0',
    market          VARCHAR(16)     NOT NULL,            -- 'crypto' | 'equity' | 'options' | 'fx' | 'multi'
    strategy_type   VARCHAR(32),                         -- 'momentum' | 'mean_revert' | 'arb' | 'vol' | 'ml'
    universe        VARCHAR(256),                        -- 交易标的集合描述
    ticker          VARCHAR(32),                         -- 主交易品种 (多品种策略按行拆分)
    timeframe       VARCHAR(8)      DEFAULT '5m',
    phase           SMALLINT        DEFAULT 0,           -- QTS V8 Phase 0~5

    -- 风控参数 (策略级)
    max_pos_usd     NUMERIC(12, 2)  DEFAULT 500,
    max_daily_loss  NUMERIC(12, 2)  DEFAULT 100,
    max_leverage    NUMERIC(6, 2)   DEFAULT 3.0,

    -- QTS V8 五门控阈值 (G1~G5 快速热更新)
    g1_min_vol      NUMERIC(12, 4),                     -- G1: 最小成交量
    g2_regime_mode  VARCHAR(16)     DEFAULT 'any',       -- G2: 允许的市场状态
    g3_min_conf     NUMERIC(6, 4)   DEFAULT 0.55,        -- G3: 最小 AlphaCast 置信度
    g4_active       BOOLEAN         DEFAULT FALSE,       -- G4: MetaLabeler 是否激活
    g5_time_gate    BOOLEAN         DEFAULT TRUE,        -- G5: 资金费率时间门控

    -- 通用
    code_path       VARCHAR(256),
    description     TEXT,
    is_active       BOOLEAN         DEFAULT FALSE,
    is_live         BOOLEAN         DEFAULT FALSE,
    created_by      VARCHAR(64),
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- 策略参数版本历史 (JSONB 灵活存储)
CREATE TABLE IF NOT EXISTS strategy.param_history (
    id              BIGSERIAL       PRIMARY KEY,
    strategy_id     UUID            NOT NULL REFERENCES strategy.strategies(id),
    version         VARCHAR(16)     NOT NULL,
    params          JSONB           NOT NULL,
    is_current      BOOLEAN         DEFAULT FALSE,
    changed_by      VARCHAR(64),
    changed_at      TIMESTAMPTZ     DEFAULT NOW(),
    reason          TEXT,
    UNIQUE (strategy_id, version)
);

-- 策略信号记录 (统一 Crypto + Multi-Asset)
CREATE TABLE IF NOT EXISTS strategy.signals (
    id              BIGSERIAL       PRIMARY KEY,
    strategy_id     UUID            NOT NULL REFERENCES strategy.strategies(id),
    ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    market          VARCHAR(16)     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    bar_index       BIGINT,
    -- 信号内容
    direction       SMALLINT        NOT NULL CHECK (direction IN (-1, 0, 1)),
    action          VARCHAR(8)      NOT NULL,            -- 'buy'|'sell'|'hold'|'close'
    target_pos_level SMALLINT,                          -- 目标仓位档位 0~5
    strength        NUMERIC(5, 4),                       -- 信号强度 0~1
    target_price    NUMERIC(18, 8),
    stop_price      NUMERIC(18, 8),
    take_profit     NUMERIC(18, 8),
    -- 决策链快照 (Crypto: AlphaCast+MCTS+Regime / Equity: Factor Model)
    alphacast_conf  FLOAT4,
    alphacast_y_hat FLOAT4,
    alphacast_sigma FLOAT4,
    mcts_action     VARCHAR(8),
    mcts_value      FLOAT4,
    mcts_simulations INTEGER,
    regime          VARCHAR(16),
    model_version   VARCHAR(32),
    -- 门控结果 (QTS V8 G1~G5)
    g1_pass         BOOLEAN,
    g1_reason       VARCHAR(64),
    g2_pass         BOOLEAN,
    g2_reason       VARCHAR(64),
    g3_pass         BOOLEAN,
    g4_pass         BOOLEAN,
    g5_pass         BOOLEAN,
    final_pass      BOOLEAN,
    reject_reason   VARCHAR(64),
    -- 因子值快照 (Equity/FX 用)
    factor_values   JSONB,
    -- 执行状态
    executed        BOOLEAN         DEFAULT FALSE,
    order_id        VARCHAR(64),
    executed_at     TIMESTAMPTZ,
    signal_data     JSONB                               -- 完整信号上下文
);

CREATE INDEX idx_signals_strat_ts   ON strategy.signals (strategy_id, ts DESC);
CREATE INDEX idx_signals_ticker_ts  ON strategy.signals (ticker, ts DESC);
CREATE INDEX idx_signals_final_pass ON strategy.signals (strategy_id, final_pass, ts DESC);

-- =============================================================
-- ██████ PORTFOLIO — 账户 & 组合管理 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS portfolio.accounts (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(128)    NOT NULL UNIQUE,
    exchange_id     SMALLINT        REFERENCES ref.exchanges(id),
    broker          VARCHAR(64),
    account_type    VARCHAR(16)     DEFAULT 'paper',     -- 'paper' | 'live' | 'backtest'
    market          VARCHAR(16)     DEFAULT 'crypto',
    base_currency   CHAR(3)         DEFAULT 'USDT',
    margin_mode     VARCHAR(16)     DEFAULT 'cross',     -- 'cross' | 'isolated'
    initial_capital NUMERIC(18, 4)  NOT NULL,
    leverage        NUMERIC(6, 2)   DEFAULT 1.0,
    is_active       BOOLEAN         DEFAULT TRUE,
    api_key_ref     VARCHAR(64),
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- 持仓快照 (统一 Crypto / Equity / Options / FX)
CREATE TABLE IF NOT EXISTS portfolio.positions (
    id              BIGSERIAL       PRIMARY KEY,
    account_id      UUID            NOT NULL REFERENCES portfolio.accounts(id),
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    market          VARCHAR(16)     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    pos_side        VARCHAR(8)      NOT NULL,            -- 'long' | 'short' | 'net'
    margin_mode     VARCHAR(16)     DEFAULT 'cross',
    -- 仓位数据
    pos_qty         NUMERIC(20, 8)  NOT NULL,            -- 数量（正=多 负=空）
    avg_px          NUMERIC(24, 8)  NOT NULL,            -- 开仓均价
    mark_px         NUMERIC(24, 8),                      -- 最新标记价格
    liq_px          NUMERIC(24, 8),                      -- 强平价
    imr             NUMERIC(12, 4),                      -- 初始保证金
    mmr             NUMERIC(12, 4),                      -- 维持保证金率
    lever           NUMERIC(6, 2),
    -- P&L
    upl             NUMERIC(16, 4),                      -- 未实现盈亏
    upl_ratio       NUMERIC(10, 8),
    pnl             NUMERIC(16, 4)  DEFAULT 0,           -- 已实现盈亏
    -- 期权专属 Greeks
    delta           NUMERIC(10, 6),
    gamma           NUMERIC(12, 8),
    theta           NUMERIC(12, 6),
    vega            NUMERIC(12, 6),
    rho             NUMERIC(12, 6),
    iv              NUMERIC(8, 6),
    -- 时间
    open_ts         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    close_ts        TIMESTAMPTZ,
    last_update_ts  TIMESTAMPTZ     DEFAULT NOW(),
    status          VARCHAR(16)     DEFAULT 'open'       -- 'open' | 'closed' | 'partial'
);

CREATE INDEX idx_pos_account   ON portfolio.positions (account_id, status);
CREATE INDEX idx_pos_ticker    ON portfolio.positions (ticker, status);
CREATE INDEX idx_pos_strategy  ON portfolio.positions (strategy_id, status);

-- 成交记录 (Fill)
CREATE TABLE IF NOT EXISTS portfolio.fills (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id      UUID            NOT NULL REFERENCES portfolio.accounts(id),
    position_id     BIGINT          REFERENCES portfolio.positions(id),
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    signal_id       BIGINT          REFERENCES strategy.signals(id),
    -- 订单信息
    ord_id          VARCHAR(64)     NOT NULL,
    cl_ord_id       VARCHAR(64),                        -- 客户端 clOrdId (trace)
    trade_id        VARCHAR(64),
    market          VARCHAR(16)     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    fill_ts         TIMESTAMPTZ     NOT NULL,
    -- 成交详情
    side            VARCHAR(8)      NOT NULL CHECK (side IN ('buy', 'sell', 'short', 'cover')),
    ord_type        VARCHAR(16)     NOT NULL,            -- 'limit' | 'market' | 'post_only'
    fill_px         NUMERIC(24, 8)  NOT NULL,
    fill_sz         NUMERIC(20, 8)  NOT NULL,
    fill_notional   NUMERIC(24, 4) GENERATED ALWAYS AS (fill_px * fill_sz) STORED,
    fee             NUMERIC(16, 8)  DEFAULT 0,
    fee_ccy         VARCHAR(16)     DEFAULT 'USDT',
    -- 滑点分析
    signal_px       NUMERIC(24, 8),
    slippage_bps    FLOAT4,
    -- 决策链追踪
    alphacast_conf  FLOAT4,
    mcts_value      FLOAT4,
    regime          VARCHAR(16),
    elapsed_ms      FLOAT4,
    status          VARCHAR(16)     DEFAULT 'filled',
    notes           TEXT
);

CREATE INDEX idx_fills_account_ts ON portfolio.fills (account_id, fill_ts DESC);
CREATE INDEX idx_fills_strategy   ON portfolio.fills (strategy_id, fill_ts DESC);
CREATE INDEX idx_fills_ticker     ON portfolio.fills (ticker, fill_ts DESC);
CREATE INDEX idx_fills_ord_id     ON portfolio.fills (ord_id);

-- 每日 P&L 快照
CREATE TABLE IF NOT EXISTS portfolio.daily_performance (
    account_id      UUID            NOT NULL REFERENCES portfolio.accounts(id),
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    trade_date      DATE            NOT NULL,
    -- 绩效
    nav             NUMERIC(18, 4)  NOT NULL,
    cash            NUMERIC(18, 4),
    gross_exposure  NUMERIC(18, 4),
    net_exposure    NUMERIC(18, 4),
    equity          NUMERIC(18, 4),
    daily_pnl       NUMERIC(16, 4),
    daily_ret       NUMERIC(12, 8),
    cum_ret         NUMERIC(12, 8),
    drawdown        NUMERIC(10, 8),
    -- 风险指标
    sharpe_7d       NUMERIC(8, 4),
    sharpe_30d      NUMERIC(8, 4),
    sortino_30d     NUMERIC(8, 4),
    max_dd_30d      NUMERIC(10, 8),
    -- 成交统计
    trade_count     INTEGER         DEFAULT 0,
    win_count       INTEGER         DEFAULT 0,
    total_fee       NUMERIC(14, 6)  DEFAULT 0,
    avg_slippage_bps FLOAT4,
    -- AlphaCast 质量 (Crypto)
    avg_conf        FLOAT4,
    pred_err_pct    FLOAT4,
    ic              FLOAT4,
    PRIMARY KEY (account_id, trade_date)
);

-- =============================================================
-- ██████ RISK — 风控规则 & 告警 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS risk.limits (
    id              BIGSERIAL       PRIMARY KEY,
    account_id      UUID            REFERENCES portfolio.accounts(id),
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    limit_type      VARCHAR(64)     NOT NULL,            -- 'max_position_pct' | 'max_drawdown' | 'max_daily_loss' | 'max_leverage' | 'max_sector_pct'
    limit_value     NUMERIC(18, 8)  NOT NULL,
    limit_unit      VARCHAR(16)     DEFAULT 'pct',       -- 'pct' | 'absolute' | 'ratio'
    scope           VARCHAR(16)     DEFAULT 'global',    -- 'global' | 'per_symbol' | 'per_sector'
    is_active       BOOLEAN         DEFAULT TRUE,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk.alerts (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    account_id      UUID            REFERENCES portfolio.accounts(id),
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    limit_id        BIGINT          REFERENCES risk.limits(id),
    alert_type      VARCHAR(64)     NOT NULL,
    severity        VARCHAR(16)     NOT NULL CHECK (severity IN ('INFO','WARN','ERROR','CRITICAL')),
    message         TEXT,
    actual_value    NUMERIC(18, 8),
    limit_value     NUMERIC(18, 8),
    is_resolved     BOOLEAN         DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX idx_alerts_ts        ON risk.alerts (ts DESC);
CREATE INDEX idx_alerts_severity  ON risk.alerts (severity, is_resolved);
CREATE INDEX idx_alerts_account   ON risk.alerts (account_id, ts DESC);

-- =============================================================
-- ██████ CALIBRATION — 模型校准 & 在线训练 ██████
-- =============================================================

-- Temperature Scaling 在线更新记录
CREATE TABLE IF NOT EXISTS calibration.temp_scaling_log (
    id                  BIGSERIAL       PRIMARY KEY,
    ts                  TIMESTAMPTZ     DEFAULT NOW(),
    strategy_id         UUID            NOT NULL REFERENCES strategy.strategies(id),
    ticker              VARCHAR(32)     NOT NULL,
    trigger_fill_count  INTEGER,
    t_before            FLOAT4          NOT NULL,
    t_after             FLOAT4          NOT NULL,
    nll_loss            FLOAT4,
    ece_before          FLOAT4,
    ece_after           FLOAT4,
    elapsed_ms          FLOAT4,
    is_paused           BOOLEAN         DEFAULT FALSE,
    notes               TEXT
);

CREATE INDEX idx_tsl_strat_ts ON calibration.temp_scaling_log (strategy_id, ts DESC);

-- 在线模型性能追踪
CREATE TABLE IF NOT EXISTS calibration.model_performance (
    ts                  TIMESTAMPTZ     PRIMARY KEY DEFAULT NOW(),
    strategy_id         UUID            NOT NULL REFERENCES strategy.strategies(id),
    ticker              VARCHAR(32)     NOT NULL,
    model_version       VARCHAR(32),
    current_t           FLOAT4          NOT NULL,
    ece                 FLOAT4,
    ic_7d               FLOAT4,
    ic_30d              FLOAT4,
    mcts_acc_7d         FLOAT4,
    pred_err_pct        FLOAT4,
    label_count         INTEGER,
    metalabeler_active  BOOLEAN         DEFAULT FALSE,
    metalabeler_auc     FLOAT4,
    alert_triggered     BOOLEAN         DEFAULT FALSE,
    alert_msg           TEXT
);

-- MetaLabeler 训练运行记录
CREATE TABLE IF NOT EXISTS calibration.metalabeler_runs (
    id              BIGSERIAL       PRIMARY KEY,
    run_ts          TIMESTAMPTZ     DEFAULT NOW(),
    ticker          VARCHAR(32)     NOT NULL,
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    label_count     INTEGER         NOT NULL,
    train_start     TIMESTAMPTZ,
    train_end       TIMESTAMPTZ,
    auc             FLOAT4,
    lift_top_decile FLOAT4,
    threshold       FLOAT4,
    model_path      VARCHAR(256),
    features_used   TEXT[],
    is_active       BOOLEAN         DEFAULT FALSE,
    notes           TEXT
);

-- =============================================================
-- ██████ EVOLUTION — 策略进化 & 遗传算法 ██████
-- =============================================================
CREATE SCHEMA IF NOT EXISTS evolution;

-- 基因组档案（遗传编程个体）
CREATE TABLE IF NOT EXISTS evolution.genomes (
    gene_id         UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name   VARCHAR(128)    NOT NULL,
    version         VARCHAR(16)     NOT NULL,
    parent_gene_id  UUID            REFERENCES evolution.genomes(gene_id),
    market          VARCHAR(16)     NOT NULL,
    category        VARCHAR(64)     DEFAULT 'genetic_discovery',
    -- 遗传编程树（表达式字符串）
    entry_tree      TEXT,
    exit_tree       TEXT,
    stop_loss       NUMERIC(10, 6),
    take_profit     NUMERIC(10, 6),
    position_pct    NUMERIC(10, 6),
    raw_config      JSONB,
    -- 模型（允许 ML 模型路径）
    model_path      VARCHAR(256),
    model_hash      VARCHAR(64),
    -- 允许的市场状态
    allowed_regimes TEXT[],
    features_used   TEXT[],
    -- 性能指标
    fitness_score   FLOAT4          DEFAULT 0,
    sharpe_ratio    FLOAT4,
    stability_score FLOAT4,
    win_rate        FLOAT4,
    max_drawdown    FLOAT4,
    total_pnl       FLOAT4,
    total_trades    INTEGER,
    avg_return      FLOAT4,
    -- 生命周期
    status          VARCHAR(16)     DEFAULT 'sandbox',   -- 'sandbox'|'arena'|'promoted'|'retired'
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    retirement_reason VARCHAR(256),
    generation      INTEGER         DEFAULT 1
);

CREATE INDEX idx_evo_gene_status ON evolution.genomes (status, fitness_score DESC);
CREATE INDEX idx_evo_gene_parent ON evolution.genomes (parent_gene_id);

-- 变异记录
CREATE TABLE IF NOT EXISTS evolution.mutations (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    parent_gene_id  UUID            REFERENCES evolution.genomes(gene_id),
    child_gene_id   UUID            REFERENCES evolution.genomes(gene_id),
    mutation_type   VARCHAR(64)     NOT NULL,            -- 'genetic_programming'|'crossover'|'param_tweak'|'regime_expand'
    changes         JSONB           NOT NULL,            -- {"entry_tree": "...", "stop_loss": "0.15"}
    mutated_by      VARCHAR(64)     DEFAULT 'GeneticStrategyEngine',
    fitness_delta   FLOAT4,
    notes           TEXT
);

CREATE INDEX idx_evo_mut_ts ON evolution.mutations (ts DESC);

-- Arena 竞技记录
CREATE TABLE IF NOT EXISTS evolution.arena (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    gene_id         UUID            REFERENCES evolution.genomes(gene_id),
    opponent_id     UUID            REFERENCES evolution.genomes(gene_id),
    period_start    TIMESTAMPTZ     NOT NULL,
    period_end      TIMESTAMPTZ,
    score           FLOAT4,                             -- 对战得分
    wins            INTEGER         DEFAULT 0,
    losses          INTEGER         DEFAULT 0,
    draws           INTEGER         DEFAULT 0,
    pnl_vs_buyhold  FLOAT4,
    is_winner       BOOLEAN,
    notes           TEXT
);

CREATE INDEX idx_evo_arena_gene ON evolution.arena (gene_id, ts DESC);

-- 晋升记录
CREATE TABLE IF NOT EXISTS evolution.promotions (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    gene_id         UUID            REFERENCES evolution.genomes(gene_id),
    from_status     VARCHAR(16)     NOT NULL,
    to_status       VARCHAR(16)     NOT NULL,
    arena_score     FLOAT4,
    stability_check BOOLEAN         DEFAULT TRUE,
    approved_by     VARCHAR(64),
    notes           TEXT
);

-- 退役记录
CREATE TABLE IF NOT EXISTS evolution.retirements (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    gene_id         UUID            REFERENCES evolution.genomes(gene_id),
    reason          VARCHAR(256)    NOT NULL,
    reason_category VARCHAR(64),                        -- 'degradation'|'drawdown'|'correlation'|'budget'|'age'
    fitness_at_retire FLOAT4,
    lifetime_days   INTEGER,
    total_pnl       FLOAT4
);

CREATE INDEX idx_evo_ret_ts ON evolution.retirements (ts DESC);

-- =============================================================
-- ██████ STRATEGY — 扩展：AB 测试 & 稳定性 ██████
-- =============================================================

-- AB 测试实验
CREATE TABLE IF NOT EXISTS strategy.ab_tests (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(128)    NOT NULL,
    control_strat   UUID            REFERENCES strategy.strategies(id),
    treatment_strat UUID            REFERENCES strategy.strategies(id),
    metric          VARCHAR(64)     NOT NULL,            -- 'sharpe'|'win_rate'|'pnl'|'stability'
    start_ts        TIMESTAMPTZ     NOT NULL,
    end_ts          TIMESTAMPTZ,
    confidence      FLOAT4,                             -- 统计置信度
    winner          VARCHAR(16),                         -- 'control'|'treatment'|'inconclusive'
    status          VARCHAR(16)     DEFAULT 'running',
    notes           TEXT
);

-- 策略稳定性追踪
CREATE TABLE IF NOT EXISTS strategy.stability (
    ts              TIMESTAMPTZ     DEFAULT NOW() PRIMARY KEY,
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    gene_id         UUID            REFERENCES evolution.genomes(gene_id),
    stability_score FLOAT4          NOT NULL,
    regime_breakdown JSONB,                             -- 各 regime 下性能分布
    correlation_exposure FLOAT4,                        -- 与其他策略的相关性
    drawdown_depth  FLOAT4,
    ret_std_30d     FLOAT4,
    calmar_30d      FLOAT4,
    is_degrading    BOOLEAN         DEFAULT FALSE,
    degradation_score FLOAT4,
    notes           TEXT
);

CREATE INDEX idx_stab_strat_ts ON strategy.stability (strategy_id, ts DESC);

-- 截面排名快照
CREATE TABLE IF NOT EXISTS strategy.cross_section (
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    market          VARCHAR(16)     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    ranking         JSONB           NOT NULL,           -- {cs_rank_obi_mean_60s: 0.78, cs_composite: 0.65, ...}
    cs_composite    FLOAT4          NOT NULL,
    cs_tier         SMALLINT        NOT NULL CHECK (cs_tier BETWEEN 0 AND 3),
    n_markets       INTEGER         NOT NULL,
    PRIMARY KEY (ts, market, ticker)
);

-- Regime ← 策略映射
CREATE TABLE IF NOT EXISTS strategy.regime_map (
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    regime          VARCHAR(32)     NOT NULL,
    weight          FLOAT4          DEFAULT 1.0,
    is_preferred    BOOLEAN         DEFAULT FALSE,
    PRIMARY KEY (strategy_id, regime)
);

-- 策略间相关性矩阵
CREATE TABLE IF NOT EXISTS strategy.correlation_matrix (
    ts              TIMESTAMPTZ     NOT NULL,
    strategy_a      UUID            REFERENCES strategy.strategies(id),
    strategy_b      UUID            REFERENCES strategy.strategies(id),
    pearson         FLOAT4,
    spearman        FLOAT4,
    window_days     SMALLINT        DEFAULT 30,
    n_samples       INTEGER,
    PRIMARY KEY (ts, strategy_a, strategy_b)
);
CREATE INDEX idx_corr_ts ON strategy.correlation_matrix (ts DESC);

-- 执行质量聚合统计
CREATE TABLE IF NOT EXISTS portfolio.exec_quality_agg (
    ts              TIMESTAMPTZ     DEFAULT NOW() PRIMARY KEY,
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    market          VARCHAR(16)     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    window          VARCHAR(8)      NOT NULL,
    total_orders    INTEGER         DEFAULT 0,
    fill_count      INTEGER         DEFAULT 0,
    fill_rate       FLOAT4          DEFAULT 0,
    avg_slippage_bps FLOAT4         DEFAULT 0,
    p95_slippage_bps FLOAT4         DEFAULT 0,
    avg_latency_ms  FLOAT4          DEFAULT 0,
    total_commission NUMERIC(18,6)  DEFAULT 0,
    total_cost      NUMERIC(18,6)   DEFAULT 0,
    win_rate        FLOAT4          DEFAULT 0,
    sharpe_rolling  FLOAT4
);

-- 失败模式学习 (CFL 负样本聚类)
CREATE TABLE IF NOT EXISTS risk.failure_patterns (
    id              BIGSERIAL       PRIMARY KEY,
    discovered_at   TIMESTAMPTZ     DEFAULT NOW(),
    pattern_name    VARCHAR(128),
    pattern_hash    VARCHAR(64)     UNIQUE,
    n_samples       INTEGER,
    avg_loss        FLOAT4,
    feature_centroid JSONB,
    top_contributors TEXT[],
    regime          VARCHAR(32),
    gate_breakdown  JSONB,
    last_seen_at    TIMESTAMPTZ,
    is_active       BOOLEAN         DEFAULT TRUE
);
CREATE INDEX idx_fp_active ON risk.failure_patterns (is_active, n_samples DESC);

-- =============================================================
-- ██████ AUDIT — 操作审计 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS audit.event_log (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     DEFAULT NOW(),
    user_name       VARCHAR(64),
    event_type      VARCHAR(64)     NOT NULL,            -- 'order_placed'|'param_change'|'mode_switch'|'api_call'
    strategy_id     UUID            REFERENCES strategy.strategies(id),
    account_id      UUID            REFERENCES portfolio.accounts(id),
    table_name      VARCHAR(128),
    record_id       VARCHAR(128),
    old_value       JSONB,
    new_value       JSONB,
    ip_address      INET,
    result          VARCHAR(16),                         -- 'success'|'failed'|'rollback'
    notes           TEXT
);

CREATE INDEX idx_audit_ts         ON audit.event_log (ts DESC);
CREATE INDEX idx_audit_event_type ON audit.event_log (event_type, ts DESC);
CREATE INDEX idx_audit_strategy   ON audit.event_log (strategy_id, ts DESC);

-- =============================================================
-- 触发器
-- =============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_strategies_updated_at
    BEFORE UPDATE ON strategy.strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_positions_updated_at
    BEFORE UPDATE ON portfolio.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_limitis_updated_at
    BEFORE UPDATE ON risk.limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================
-- 视图：实盘监控快查
-- =============================================================

-- 当前持仓汇总
CREATE OR REPLACE VIEW portfolio.current_positions AS
SELECT
    p.account_id, p.market, p.ticker, p.pos_side,
    p.pos_qty, p.avg_px, p.mark_px,
    p.upl, p.upl_ratio, p.pnl,
    p.liq_px, p.lever, p.status,
    COALESCE(p.delta, 0) AS delta_exposure,
    EXTRACT(EPOCH FROM (NOW() - p.open_ts)) / 3600 AS hold_hours
FROM portfolio.positions p
WHERE p.status = 'open';

-- 今日成交汇总
CREATE OR REPLACE VIEW portfolio.today_fills AS
SELECT
    account_id, market, ticker,
    COUNT(*)                    AS fill_count,
    SUM(fill_notional)          AS total_notional,
    SUM(fee)                    AS total_fee,
    AVG(slippage_bps)           AS avg_slippage_bps,
    AVG(elapsed_ms)             AS avg_latency_ms,
    MIN(fill_ts)                AS first_fill,
    MAX(fill_ts)                AS last_fill
FROM portfolio.fills
WHERE fill_ts >= CURRENT_DATE
GROUP BY account_id, market, ticker;

-- 今日信号通过率
CREATE OR REPLACE VIEW strategy.signal_pass_rate AS
SELECT
    strategy_id, ticker,
    COUNT(*)                                                AS total_signals,
    COUNT(*) FILTER (WHERE final_pass = TRUE)                AS passed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE final_pass = TRUE) / NULLIF(COUNT(*), 0), 2) AS pass_rate_pct,
    COUNT(*) FILTER (WHERE g1_pass = FALSE)                  AS blocked_g1,
    COUNT(*) FILTER (WHERE g2_pass = FALSE)                  AS blocked_g2,
    COUNT(*) FILTER (WHERE g3_pass = FALSE)                  AS blocked_g3,
    COUNT(*) FILTER (WHERE g4_pass = FALSE)                  AS blocked_g4,
    COUNT(*) FILTER (WHERE g5_pass = FALSE)                  AS blocked_g5
FROM strategy.signals
WHERE ts >= CURRENT_DATE
GROUP BY strategy_id, ticker;
