-- =============================================================
-- QTS V8 · TimescaleDB 特征工程 & CFL 标签库 (Enhanced)
-- 数据库: qts_market
-- Schema: features / labels
-- 覆盖: Crypto 178维特征 + Multi-Asset Alpha Factors
-- =============================================================

CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS labels;

-- =============================================================
-- ██████ FEATURES — Crypto 5m 实时特征快照 ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS features.snapshot_5m (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    bar_index   BIGINT          NOT NULL,

    -- 价格与成交量 (5 维)
    f_open      FLOAT4, f_high     FLOAT4, f_low       FLOAT4,
    f_close     FLOAT4, f_vol      FLOAT4, f_vol_ccy   FLOAT4,

    -- 订单簿不平衡 OBI (5 维)
    f_obi_1     FLOAT4, f_obi_5    FLOAT4, f_bid_depth  FLOAT4,
    f_ask_depth FLOAT4, f_depth_imb FLOAT4,

    -- 订单流不平衡 OFI (3 维)
    f_ofi_1m    FLOAT4, f_ofi_5m   FLOAT4, f_ofi_15m   FLOAT4,

    -- 点差 (3 维)
    f_spread_abs FLOAT4, f_spread_bps FLOAT4, f_spread_z FLOAT4,

    -- 动量 (6 维)
    f_mom_1m    FLOAT4, f_mom_5m   FLOAT4, f_mom_15m   FLOAT4,
    f_mom_1h    FLOAT4, f_mom_4h   FLOAT4, f_mom_1d    FLOAT4,

    -- 波动率 (5 维)
    f_hv_5m     FLOAT4, f_hv_1h    FLOAT4, f_hv_4h     FLOAT4,
    f_hv_1d     FLOAT4, f_atm_rv   FLOAT4,

    -- Hurst 指数 (2 维)
    f_hurst_5m  FLOAT4, f_hurst_1h FLOAT4,

    -- 技术指标 (12 维)
    f_rsi_14    FLOAT4, f_rsi_6    FLOAT4,
    f_macd      FLOAT4, f_macd_sig FLOAT4, f_macd_hist  FLOAT4,
    f_bb_upper  FLOAT4, f_bb_lower FLOAT4, f_bb_pct      FLOAT4,
    f_atr_14    FLOAT4, f_adx_14   FLOAT4,
    f_ema_9     FLOAT4, f_ema_21   FLOAT4,

    -- 资金费率衍生 (4 维)
    f_funding_r FLOAT4, f_funding_pred FLOAT4,
    f_funding_cum8h FLOAT4, f_funding_z FLOAT4,

    -- 持仓量变化 (4 维)
    f_oi_delta_5m FLOAT4, f_oi_delta_1h FLOAT4,
    f_oi_z      FLOAT4, f_oi_pv_corr FLOAT4,

    -- 多空比 (3 维)
    f_ls_elite_acc FLOAT4, f_ls_elite_pos FLOAT4, f_ls_all_acc FLOAT4,

    -- 标记价格 / 基差 (4 维)
    f_basis     FLOAT4, f_basis_rate FLOAT4,
    f_mark_spot_ratio FLOAT4, f_index_vol FLOAT4,

    -- 强平信号 (4 维)
    f_liq_buy_5m FLOAT4, f_liq_sell_5m FLOAT4,
    f_liq_imb_5m FLOAT4, f_liq_z    FLOAT4,

    -- K 线事件 (2 维)
    f_bar_close_event FLOAT4, f_precomputed FLOAT4,

    -- Prediction Market 专项 (PolyMarket 独有的 3 维)
    f_parity_dev FLOAT4,  f_time_to_res FLOAT4,  f_log_ttr FLOAT4,
    -- 扩展槽 (4 维 → 总计 73 维单步)
    f_ext_1     FLOAT4, f_ext_2    FLOAT4,
    f_ext_3     FLOAT4, f_ext_4    FLOAT4,

    -- 元信息
    window_size     SMALLINT        DEFAULT 60,
    feat_version    SMALLINT        DEFAULT 1
);

SELECT create_hypertable('features.snapshot_5m', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_feat_5m ON features.snapshot_5m (ticker, ts);
CREATE INDEX       IF NOT EXISTS idx_feat_bar  ON features.snapshot_5m (ticker, bar_index);

SELECT add_retention_policy('features.snapshot_5m', INTERVAL '90 days', if_not_exists => TRUE);

ALTER TABLE features.snapshot_5m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('features.snapshot_5m', INTERVAL '14 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ FEATURES — Multi-Asset Alpha Factors ██████
-- =============================================================

-- Equity 因子快照
CREATE TABLE IF NOT EXISTS features.equity_factors (
    ts          TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(16)     NOT NULL,
    bar         VARCHAR(4)      NOT NULL DEFAULT '1d',
    -- 动量因子
    mom_1m      FLOAT4, mom_3m  FLOAT4, mom_6m  FLOAT4, mom_12m FLOAT4,
    -- 波动率因子
    hv_20d      FLOAT4, hv_60d  FLOAT4,
    -- 价值因子
    pe_ratio    FLOAT4, pb_ratio FLOAT4, ps_ratio FLOAT4,
    ev_ebitda   FLOAT4,
    -- 质量因子
    roe         FLOAT4, roa     FLOAT4,
    debt_equity FLOAT4, current_ratio FLOAT4,
    -- 规模/流动性
    market_cap  FLOAT4, avg_dollar_vol_20d FLOAT4, turnover_20d FLOAT4,
    -- 技术指标
    rsi_14      FLOAT4, ma_50d  FLOAT4, ma_200d FLOAT4,
    -- Beta
    beta_60d    FLOAT4, alpha_60d FLOAT4,
    -- 截面信息
    sector      VARCHAR(64),
    z_mom_1m    FLOAT4, z_hv_20d FLOAT4, z_pe_ratio FLOAT4
);

SELECT create_hypertable('features.equity_factors', 'ts',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_efact_sym_ts ON features.equity_factors (symbol, ts DESC);

-- FX 因子快照
CREATE TABLE IF NOT EXISTS features.fx_factors (
    ts          TIMESTAMPTZ     NOT NULL,
    pair        VARCHAR(8)      NOT NULL,
    bar         VARCHAR(4)      NOT NULL DEFAULT '1h',
    -- 动量
    mom_1h      FLOAT4, mom_4h  FLOAT4, mom_1d  FLOAT4,
    -- 波动率
    hv_1h       FLOAT4, hv_1d   FLOAT4,
    -- 利差
    ir_diff_3m  FLOAT4, ir_diff_1y FLOAT4,
    -- 持仓
    cot_long    FLOAT4, cot_short FLOAT4, cot_net FLOAT4,
    -- 技术指标
    rsi_14      FLOAT4, atr_14  FLOAT4,
    ma_20       FLOAT4, ma_50   FLOAT4,
    bb_pct      FLOAT4,
    -- 宏观
    vix         FLOAT4, dxy     FLOAT4
);

SELECT create_hypertable('features.fx_factors', 'ts',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ffact_pair_ts ON features.fx_factors (pair, ts DESC);

-- =============================================================
-- ██████ FEATURES — 模型推断输出 ██████
-- =============================================================

-- AlphaCast 推断输出 (Crypto 核心)
CREATE TABLE IF NOT EXISTS features.alphacast_output (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    bar_index   BIGINT          NOT NULL,
    y_hat       FLOAT4          NOT NULL,
    sigma       FLOAT4          NOT NULL,
    conf        FLOAT4          NOT NULL,
    regime      VARCHAR(16),
    raw_logit   FLOAT4,
    temp_t      FLOAT4          DEFAULT 1.0,
    model_ver   VARCHAR(32),
    elapsed_ms  FLOAT4
);

SELECT create_hypertable('features.alphacast_output', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_ac_output ON features.alphacast_output (ticker, ts);
SELECT add_retention_policy('features.alphacast_output', INTERVAL '180 days', if_not_exists => TRUE);

-- MCTS 决策记录
CREATE TABLE IF NOT EXISTS features.mcts_decision (
    ts              TIMESTAMPTZ     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    bar_index       BIGINT          NOT NULL,
    action          VARCHAR(8)      NOT NULL,
    position_level  SMALLINT,
    path_value      FLOAT4,
    simulations     INTEGER,
    elapsed_ms      FLOAT4,
    was_degraded    BOOLEAN         DEFAULT FALSE
);

SELECT create_hypertable('features.mcts_decision', 'ts',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_mcts_ticker_ts ON features.mcts_decision (ticker, ts DESC);
SELECT add_retention_policy('features.mcts_decision', INTERVAL '180 days', if_not_exists => TRUE);

-- 连续聚合：5m 截面排名 (MATERIALIZED 快照，CrossSectionEngine 写入)
CREATE TABLE IF NOT EXISTS features.cross_section_5m (
    ts          TIMESTAMPTZ     NOT NULL,
    market      VARCHAR(16)     NOT NULL,
    ticker      VARCHAR(32)     NOT NULL,
    cs_rank_obi_mean_60s    FLOAT4, cs_rank_obi_vel_60s FLOAT4,
    cs_rank_price_vel_60s   FLOAT4, cs_rank_price_vel_300s FLOAT4,
    cs_rank_net_flow_120s   FLOAT4, cs_rank_spread_current FLOAT4,
    cs_rank_spread_z_5m     FLOAT4, cs_rank_total_depth FLOAT4,
    cs_rank_realized_vol_5m FLOAT4, cs_rank_trade_rate_120s FLOAT4,
    cs_composite            FLOAT4, cs_tier SMALLINT,
    n_markets               INTEGER,
    PRIMARY KEY (ts, market, ticker)
);
SELECT create_hypertable('features.cross_section_5m', 'ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- =============================================================
-- ██████ LABELS — CFL 标签 (ML 训练数据核心) ██████
-- =============================================================

CREATE TABLE IF NOT EXISTS labels.cfl (
    ts              TIMESTAMPTZ     NOT NULL,
    ticker          VARCHAR(32)     NOT NULL,
    bar_index       BIGINT          NOT NULL,
    bar_close_ts    TIMESTAMPTZ     NOT NULL,

    -- 收益标签
    ret_1bar        FLOAT4, ret_3bar FLOAT4,
    ret_5bar        FLOAT4, ret_10bar FLOAT4,

    -- 三重障碍标签
    triple_barrier  SMALLINT,
    barrier_up      FLOAT4,
    barrier_down    FLOAT4,
    barrier_timeout SMALLINT        DEFAULT 5,
    hit_ts          TIMESTAMPTZ,

    -- 模型输出快照
    alphacast_conf  FLOAT4,
    alphacast_y_hat FLOAT4,
    alphacast_sigma FLOAT4,
    mcts_value      FLOAT4          DEFAULT 0,
    mcts_action     VARCHAR(8),
    regime          VARCHAR(16),
    temp_t          FLOAT4,

    -- 门控状态
    g1_pass         BOOLEAN, g2_pass BOOLEAN,
    g3_pass         BOOLEAN, g4_pass BOOLEAN,

    -- 实现状态
    realized_ts     TIMESTAMPTZ,
    is_finalized    BOOLEAN         DEFAULT FALSE,
    actual_ret      FLOAT4,

    -- 元信息
    feat_version    SMALLINT        DEFAULT 1,
    label_version   SMALLINT        DEFAULT 1
);

SELECT create_hypertable('labels.cfl', 'ts',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_cfl        ON labels.cfl (ticker, bar_index);
CREATE INDEX       IF NOT EXISTS idx_cfl_finalized ON labels.cfl (ticker, is_finalized, ts DESC);
CREATE INDEX       IF NOT EXISTS idx_cfl_regime   ON labels.cfl (regime, triple_barrier, ts DESC);

-- =============================================================
-- 聚合视图
-- =============================================================
CREATE OR REPLACE VIEW labels.cfl_stats AS
SELECT
    ticker,
    COUNT(*)                                            AS total_labels,
    COUNT(*) FILTER (WHERE is_finalized)                AS finalized_count,
    COUNT(*) FILTER (WHERE triple_barrier = 1)          AS up_hits,
    COUNT(*) FILTER (WHERE triple_barrier = -1)         AS down_hits,
    COUNT(*) FILTER (WHERE triple_barrier = 0)          AS timeouts,
    AVG(alphacast_conf)                                 AS avg_conf,
    AVG(mcts_value)                                     AS avg_mcts_val,
    MIN(ts) AS first_label, MAX(ts) AS last_label
FROM labels.cfl GROUP BY ticker;

CREATE OR REPLACE VIEW labels.training_set AS
SELECT
    c.ts, c.ticker, c.bar_index,
    c.triple_barrier                                    AS label,
    c.alphacast_conf, c.alphacast_y_hat, c.mcts_value,
    c.regime, c.actual_ret,
    f.f_obi_5, f.f_ofi_5m, f.f_spread_z,
    f.f_mom_5m, f.f_hv_5m, f.f_hurst_5m,
    f.f_funding_r, f.f_funding_z,
    f.f_oi_delta_5m, f.f_ls_elite_acc,
    f.f_liq_imb_5m, f.f_rsi_14, f.f_adx_14
FROM labels.cfl c
JOIN features.snapshot_5m f
    ON c.ticker = f.ticker AND c.bar_index = f.bar_index
WHERE c.is_finalized = TRUE;
