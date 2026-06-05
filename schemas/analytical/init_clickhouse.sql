-- =============================================================
-- QTS V8 · ClickHouse 历史分析数据库 (Enhanced)
-- 数据库: qts_hist
-- 覆盖: Crypto / Equity / Options / FX / Futures / Prediction 全市场
-- 用途: 历史回测 / 因子分析 / Alpha Research / 模型训练
-- =============================================================

CREATE DATABASE IF NOT EXISTS qts_hist;

-- =============================================================
-- ██████ CRYPTO 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.crypto_ohlcv_1m (
    ts          DateTime64(3, 'UTC'),
    ticker      LowCardinality(String),
    open        Float64, high Float64, low Float64, close Float64,
    vol         Float64, vol_ccy Float64,
    ret         Float32, log_ret Float32, hl_range Float32,
    source      LowCardinality(String) DEFAULT 'okx'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (ticker, ts)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.crypto_ohlcv_5m (
    ts          DateTime64(3, 'UTC'),
    ticker      LowCardinality(String),
    open        Float64, high Float64, low Float64, close Float64,
    vol         Float64, vol_ccy Float64,
    ret         Float32, log_ret Float32, hl_range Float32,
    rsi_14      Float32, atr_14 Float32, hv_20 Float32,
    source      LowCardinality(String) DEFAULT 'okx'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (ticker, ts)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.crypto_funding_rate (
    ts              DateTime64(3, 'UTC'),
    ticker          LowCardinality(String),
    funding_rate    Float32,
    next_funding_rate Float32,
    method          LowCardinality(String),
    annualized_rate Float32 MATERIALIZED funding_rate * 3 * 365
) ENGINE = MergeTree()
PARTITION BY toYear(ts)
ORDER BY (ticker, ts)
SETTINGS index_granularity = 512;

CREATE TABLE IF NOT EXISTS qts_hist.crypto_open_interest (
    ts      DateTime64(3, 'UTC'),
    ticker  LowCardinality(String),
    oi      Float64, oi_ccy Float64, oi_usd Float64,
    oi_delta_1h Float32, oi_delta_1d Float32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (ticker, ts)
SETTINGS index_granularity = 4096;

-- =============================================================
-- ██████ EQUITY 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.equity_daily (
    trade_date  Date,
    symbol      LowCardinality(String),
    exchange    LowCardinality(String),
    open        Float64, high Float64, low Float64, close Float64,
    adj_close   Float64,
    volume      UInt64, vwap Float64, trades UInt32,
    ret_1d      Float32, ret_5d Float32, log_ret Float32, range_pct Float32,
    source      LowCardinality(String) DEFAULT 'polygon'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (symbol, trade_date)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.equity_minute (
    ts          DateTime,
    symbol      LowCardinality(String),
    timeframe   LowCardinality(String),
    open        Float64, high Float64, low Float64, close Float64,
    volume      UInt64, vwap Float64, trades UInt32
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(ts), symbol)
ORDER BY (symbol, timeframe, ts)
TTL ts + INTERVAL 3 YEAR
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ OPTIONS 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.options_eod (
    trade_date    Date,
    underlying    LowCardinality(String),
    expiry        Date, strike Float64,
    option_type   LowCardinality(String),
    open          Float64, high Float64, low Float64, close Float64,
    volume        UInt32, open_interest UInt32,
    iv            Float32, delta Float32, gamma Float32,
    theta         Float32, vega Float32, rho Float32,
    dte           Int16, moneyness Float32,
    source        LowCardinality(String) DEFAULT 'cboe'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (underlying, trade_date, expiry, strike, option_type)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.vol_surface_hist (
    trade_date    Date,
    underlying    LowCardinality(String),
    expiry        Date, moneyness Float32,
    iv            Float32,
    surface_type  LowCardinality(String) DEFAULT 'market'
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (underlying, trade_date, expiry, moneyness)
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ FX 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.fx_ohlcv (
    ts          DateTime,
    pair        LowCardinality(String),
    timeframe   LowCardinality(String),
    open        Float64, high Float64, low Float64, close Float64,
    volume      Float64, spread_avg Float32
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(ts), pair)
ORDER BY (pair, timeframe, ts)
TTL ts + INTERVAL 10 YEAR
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.fx_economic_calendar (
    event_ts    DateTime,
    country     LowCardinality(String),
    currency    LowCardinality(String),
    event_name  String,
    impact      LowCardinality(String),
    actual      Float32, forecast Float32, previous Float32,
    surprise    Float32 MATERIALIZED actual - forecast
) ENGINE = MergeTree()
PARTITION BY toYear(event_ts)
ORDER BY (event_ts, country)
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ ALPHA FACTORS (统一因子表) ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.alpha_factors (
    trade_date  Date,
    symbol      LowCardinality(String),
    factor_name LowCardinality(String),
    factor_val  Float64,
    percentile  Float32,
    z_score     Float32,
    universe    LowCardinality(String) DEFAULT 'SP500'
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(trade_date), factor_name)
ORDER BY (factor_name, trade_date, symbol)
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ CFL 标签历史归档 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.cfl_labels (
    ts              DateTime64(3, 'UTC'),
    ticker          LowCardinality(String),
    bar_index       UInt64,
    triple_barrier  Int8,
    ret_1bar        Float32, ret_3bar Float32, ret_5bar Float32,
    actual_ret      Float32,
    alphacast_conf  Float32, alphacast_y_hat Float32, mcts_value Float32,
    regime          LowCardinality(String),
    g4_pass         UInt8,
    is_finalized    UInt8,
    feat_version    UInt8, label_version UInt8
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (ticker, ts, bar_index)
SETTINGS index_granularity = 4096;

-- Crypto 特征向量归档
CREATE TABLE IF NOT EXISTS qts_hist.crypto_features_5m (
    ts              DateTime64(3, 'UTC'),
    ticker          LowCardinality(String),
    bar_index       UInt64,
    feat_version    UInt8,
    f_open Float32, f_high Float32, f_low Float32, f_close Float32,
    f_vol Float32, f_vol_ccy Float32,
    f_obi_1 Float32, f_obi_5 Float32, f_bid_depth Float32,
    f_ask_depth Float32, f_depth_imb Float32,
    f_ofi_1m Float32, f_ofi_5m Float32, f_ofi_15m Float32,
    f_spread_abs Float32, f_spread_bps Float32, f_spread_z Float32,
    f_mom_1m Float32, f_mom_5m Float32, f_mom_15m Float32,
    f_mom_1h Float32, f_mom_4h Float32, f_mom_1d Float32,
    f_hv_5m Float32, f_hv_1h Float32, f_hv_4h Float32,
    f_hv_1d Float32, f_atm_rv Float32,
    f_hurst_5m Float32, f_hurst_1h Float32,
    f_rsi_14 Float32, f_rsi_6 Float32,
    f_macd Float32, f_macd_sig Float32, f_macd_hist Float32,
    f_bb_upper Float32, f_bb_lower Float32, f_bb_pct Float32,
    f_atr_14 Float32, f_adx_14 Float32,
    f_ema_9 Float32, f_ema_21 Float32,
    f_funding_r Float32, f_funding_pred Float32,
    f_funding_cum8h Float32, f_funding_z Float32,
    f_oi_delta_5m Float32, f_oi_delta_1h Float32,
    f_oi_z Float32, f_oi_pv_corr Float32,
    f_ls_elite_acc Float32, f_ls_elite_pos Float32, f_ls_all_acc Float32,
    f_basis Float32, f_basis_rate Float32,
    f_mark_spot_ratio Float32, f_index_vol Float32,
    f_liq_buy_5m Float32, f_liq_sell_5m Float32,
    f_liq_imb_5m Float32, f_liq_z Float32,
    f_bar_close_event Float32, f_precomputed Float32,
    f_ext_1 Float32, f_ext_2 Float32, f_ext_3 Float32, f_ext_4 Float32
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(ts), ticker)
ORDER BY (ticker, ts)
TTL ts + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ PREDICTION MARKET 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.prediction_tick (
    ts              DateTime64(3, 'UTC'),
    market_id       LowCardinality(String),
    trade_id        String,
    price           Float32,
    size            Float64,
    notional        Float64 MATERIALIZED price * size,
    side            LowCardinality(String),
    outcome         LowCardinality(String),
    yes_price       Float32,
    no_price        Float32,
    parity_dev      Float32 MATERIALIZED yes_price + no_price - 1.0,
    transaction_hash String,
    block_number    UInt64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (market_id, ts)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.prediction_minute (
    ts              DateTime,
    market_id       LowCardinality(String),
    yes_price       Float32, yes_vwap Float32, yes_volume Float64,
    no_price        Float32, no_vwap Float32, no_volume Float64,
    parity          Float32 MATERIALIZED yes_price + no_price - 1.0,
    spread          Float32, obi Float32,
    transactions    UInt16, unique_traders UInt16
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (market_id, ts)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.prediction_resolutions (
    resolved_at     DateTime,
    market_id       LowCardinality(String),
    slug            String,
    question        String,
    category        LowCardinality(String),
    outcome         LowCardinality(String),
    yes_payout      Float32, no_payout Float32,
    total_volume    Float64,
    disputed        UInt8,
    dispute_rounds  UInt8
) ENGINE = MergeTree()
PARTITION BY toYear(resolved_at)
ORDER BY (market_id, resolved_at)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.prediction_parity (
    ts              DateTime64(3, 'UTC'),
    market_id       LowCardinality(String),
    yes_price       Float32, no_price Float32,
    parity          Float32, abs_parity Float32,
    arb_signal      LowCardinality(String),
    arb_spread      Float32,
    net_profit      Float32,
    executed        UInt8,
    pnl             Float32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (market_id, ts)
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ EVOLUTION 历史 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.evolution_genomes (
    gene_id         UUID,
    created_at      DateTime,
    strategy_name   String,
    version         String,
    parent_gene_id  UUID,
    market          LowCardinality(String),
    category        LowCardinality(String),
    status          LowCardinality(String),
    generation      UInt16,
    fitness_score   Float32,
    sharpe_ratio    Float32,
    stability_score Float32,
    win_rate        Float32,
    max_drawdown    Float32,
    total_pnl       Float32,
    total_trades    UInt32,
    entry_tree      String,
    exit_tree       String,
    promoted_at     DateTime,
    retired_at      DateTime,
    retirement_reason String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (gene_id, created_at)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.evolution_arena (
    ts          DateTime,
    gene_id     UUID,
    opponent_id UUID,
    score       Float32,
    wins        UInt16, losses UInt16, draws UInt16,
    is_winner   UInt8
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (gene_id, ts)
SETTINGS index_granularity = 8192;

-- =============================================================
-- ██████ 期货历史 — yfinance / CFTC COT ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.futures_ohlcv (
    ts              DateTime,
    symbol          LowCardinality(String),
    contract_month  LowCardinality(String),
    is_continuous   UInt8           DEFAULT 0,
    timeframe       LowCardinality(String),
    open            Float64,
    high            Float64,
    low             Float64,
    close           Float64,
    volume          UInt64,
    open_interest   UInt32,
    vwap            Float64,
    trades          UInt32
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(ts), symbol)
ORDER BY (symbol, contract_month, timeframe, ts)
TTL ts + INTERVAL 10 YEAR
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.futures_term_structure (
    trade_date      Date,
    symbol          LowCardinality(String),
    contract_month  LowCardinality(String),
    expiry          Date,
    dte             Int16,
    price           Float64,
    spot_price      Float64,
    basis           Float64,
    basis_pct       Float32,
    roll_spread     Float64,
    regime          LowCardinality(String),
    open_interest   UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (symbol, trade_date, contract_month)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS qts_hist.futures_cot (
    report_date     Date,
    symbol          LowCardinality(String),
    exchange        LowCardinality(String),
    contract_name   String,
    comm_long       UInt32,
    comm_short      UInt32,
    managed_long    UInt32,
    managed_short   UInt32,
    nonrep_long     UInt32,
    nonrep_short    UInt32,
    net_spec_pos    Int64,
    spec_long_pct   Float32,
    spec_short_pct  Float32,
    cot_index       Float32,
    oi_total        UInt64,
    spread_pos      UInt32
)
ENGINE = MergeTree()
PARTITION BY toYear(report_date)
ORDER BY (symbol, report_date)
SETTINGS index_granularity = 8192;

-- COT 情绪指标物化视图
CREATE MATERIALIZED VIEW IF NOT EXISTS qts_hist.futures_cot_sentiment_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYear(report_date)
ORDER BY (symbol, report_date)
AS SELECT
    report_date,
    symbol,
    net_spec_pos,
    spec_long_pct,
    cot_index,
    oi_total
FROM qts_hist.futures_cot;

-- =============================================================
-- ██████ 回测结果 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS qts_hist.backtest_results (
    run_id          UUID,
    strategy_name   String,
    strategy_id     String,
    run_ts          DateTime DEFAULT now(),
    market          LowCardinality(String),
    ticker          String,
    bar             LowCardinality(String),
    start_date      Date, end_date Date,
    phase           UInt8,
    -- 绩效
    total_ret       Float32, annual_ret Float32,
    sharpe          Float32, sortino Float32,
    max_dd          Float32, calmar Float32,
    win_rate        Float32, profit_factor Float32,
    ic              Float32,
    -- 交易统计
    total_trades    UInt32, avg_hold_bars Float32,
    avg_slippage_bps Float32, total_fee_usd Float32,
    -- 参数快照
    params          String, notes String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(run_ts)
ORDER BY (strategy_name, run_ts)
SETTINGS index_granularity = 512;

-- =============================================================
-- ██████ 物化视图 ██████
-- =============================================================

-- Crypto 5m → 日度汇总
CREATE MATERIALIZED VIEW IF NOT EXISTS qts_hist.crypto_daily_summary_mv
ENGINE = SummingMergeTree()
PARTITION BY toYear(trade_date)
ORDER BY (ticker, trade_date)
AS SELECT
    toDate(ts) AS trade_date, ticker,
    argMin(open, ts) AS open, max(high) AS high,
    min(low) AS low, argMax(close, ts) AS close,
    sum(vol) AS vol, sum(vol_ccy) AS vol_ccy
FROM qts_hist.crypto_ohlcv_5m
GROUP BY trade_date, ticker;

-- Equity 月度汇总
CREATE MATERIALIZED VIEW IF NOT EXISTS qts_hist.equity_monthly_mv
ENGINE = SummingMergeTree()
PARTITION BY toYear(month)
ORDER BY (symbol, month)
AS SELECT
    toStartOfMonth(trade_date) AS month, symbol,
    argMin(open, trade_date) AS open, max(high) AS high,
    min(low) AS low, argMax(close, trade_date) AS close,
    sum(volume) AS volume, sum(trades) AS trades
FROM qts_hist.equity_daily
GROUP BY month, symbol;

-- CFL 标签日度统计
CREATE MATERIALIZED VIEW IF NOT EXISTS qts_hist.cfl_daily_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYear(trade_date)
ORDER BY (ticker, trade_date)
AS SELECT
    toDate(ts) AS trade_date, ticker,
    count() AS total,
    countIf(triple_barrier = 1) AS up_hits,
    countIf(triple_barrier = -1) AS down_hits,
    countIf(triple_barrier = 0) AS timeouts,
    avgIf(alphacast_conf, is_finalized = 1) AS avg_conf
FROM qts_hist.cfl_labels
GROUP BY trade_date, ticker;

-- Options IV 日度汇总
CREATE MATERIALIZED VIEW IF NOT EXISTS qts_hist.options_iv_daily_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (underlying, trade_date)
AS SELECT
    trade_date, underlying,
    avgState(iv) AS avg_iv_state,
    minState(iv) AS min_iv_state,
    maxState(iv) AS max_iv_state
FROM qts_hist.options_eod
GROUP BY trade_date, underlying;
