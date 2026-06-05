# QTS V8 · 量化交易数据库资源包 (Enhanced)

> Crypto + Multi-Asset (Equity / Futures / Options / FX) + Prediction Market（Polymarket）全市场覆盖的数据基础设施
> 融合 crypto_qts_db（QTS V8 ML Pipeline + 五门控）+ quant_db_kit（多市场架构）+ ORACLE-FORGE（策略进化 + 元学习）

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        QTS V8 数据架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数据源 (OKX/Polygon/OANDA/CBOE/yfinance/CFTC)                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐                                                    │
│  │  Kafka  │ ← 消息解耦 & 缓冲                                   │
│  └────┬────┘                                                    │
│       │                                                         │
│       ├──► TimescaleDB (qts_market) ← 实时行情，热数据 7-90天     │
│       │    ├── crypto.*      ← Crypto Tick/OHLCV/OrderBook/Funding│
│       │    ├── equity.*      ← 美股 Tick/OHLCV/OrderBook         │
│       │    ├── futures.*     ← 期货 Tick/OHLCV/TermStructure/COT │
│       │    ├── options.*     ← 期权 Quote Greeks/VolSurface      │
│       │    ├── fx.*          ← 外汇 Tick/OHLCV/Forward           │
│       │    ├── prediction.*  ← Polymarket CLOB Tick/OB/Parity    │
│       │    ├── features.*    ← 统一91维特征 / AlphaCast / MCTS     │
│       │    └── labels.*      ← CFL 三元标签 & MetaLabeler        │
│       │                                                         │
│       ├──► ClickHouse (qts_hist) ← 历史分析，列存永久归档         │
│       │    ├── crypto_*  ← Crypto OHLCV/Funding/OI/Features     │
│       │    ├── equity_*  ← 美股 日线/分钟线/Alpha Factors         │
│       │    ├── options_* ← 期权 EOD/VolSurface/Greeks TS        │
│       │    ├── fx_*      ← 外汇 OHLCV/经济日历                    │
│       │    ├── prediction_* ← Prediction Market 历史/Minute/Parity│
│       │    ├── evolution_* ← 策略进化基因组/竞技场归档            │
│       │    ├── cfl_labels ← CFL 标签历史                         │
│       │    └── backtest_results ← 回测结果归档                    │
│       │                                                         │
│       ├──► PostgreSQL (qts_ops) ← 策略/账户/风控/进化/模型校准    │
│       │    ├── ref.*          ← 交易所/合约/交易日历               │
│       │    ├── strategy.*     ← 策略定义/参数/信号/AB测试/截面    │
│       │    ├── evolution.*    ← 基因组/变异/竞技/晋升/退役         │
│       │    ├── portfolio.*    ← 账户/持仓/成交/绩效                │
│       │    ├── risk.*         ← 风控规则 & 告警                   │
│       │    ├── calibration.*  ← Temperature Scaling / MetaLabeler │
│       │    └── audit.*        ← 操作审计                          │
│       │                                                         │
│       └──► Redis ← 热缓存 (SHM替代/特征/门控/MCTS/限速/进化/期货)       │
│            ├── shm:*    ← 跨语言行情快照 (TTL: 3s)                │
│            ├── feat:*   ← 在线特征向量 (TTL: 60s)                 │
│            ├── mcts:*   ← MCTS决策缓存 (TTL: 30s)                │
│            ├── gate:*   ← 五门控状态 (TTL: 10s)                  │
│            └── stream:* ← 事件管道 (K线闭合/成交回执/告警)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
V8_db/
├── README.md
├── docker-compose.yml
├── protos/
│   └── qts_v8.proto               # 跨语言数据契约 (Rust/Python/Go)
├── schemas/
│   ├── relational/
│   │   └── init_postgres.sql       # PostgreSQL: 策略/账户/风控/校准
│   ├── timeseries/
│   │   ├── market/
│   │   │   ├── init_crypto_market.sql   # Crypto 行情 (9 张表)
│   │   │   └── init_multi_market.sql    # Equity/Options/FX 行情
│   │   └── features/
│   │       └── init_features_labels.sql  # 统一91维特征 + CFL 标签 + 截面排名
│   │   └── market/
│   │       ├── init_crypto_market.sql     # Crypto 行情
│   │       ├── init_multi_market.sql      # Equity/Options/FX 行情
│   │       └── init_prediction_market.sql # Polymarket 行情 + Parity
│   ├── analytical/
│   │   └── init_clickhouse.sql     # ClickHouse 历史分析 (含 Prediction/Evolution)
│   └── cache/
│       └── redis_design.md         # Redis 17 种数据结构设计
├── config/
│   └── settings.yaml               # 全局配置
├── proto/
│   └── qts_v8.proto                # 19 条 Proto 消息 (含 Prediction/Genome/CS)
├── infra/
│   ├── kafka/
│   │   └── init_topics.sh          # 22 个 Kafka Topic
│   └── grafana/
│       └── dashboards.yaml         # 监控看板
└── scripts/
    └── init_all.sh                 # 一键初始化全栈
```

## 快速开始

### 1. 启动基础设施
```bash
docker-compose up -d
```

### 2. 一键初始化数据库
```bash
bash scripts/init_all.sh
```

### 3. 分步初始化（可选）
```bash
# PostgreSQL 表结构
psql -h localhost -p 5432 -U quant -d qts_ops \
     -f schemas/relational/init_postgres.sql

# TimescaleDB: Crypto 行情
psql -h localhost -p 5433 -U quant -d qts_market \
     -f schemas/timeseries/market/init_crypto_market.sql

# TimescaleDB: Multi-Asset 行情
psql -h localhost -p 5433 -U quant -d qts_market \
     -f schemas/timeseries/market/init_multi_market.sql

# TimescaleDB: 特征 & 标签
psql -h localhost -p 5433 -U quant -d qts_market \
     -f schemas/timeseries/features/init_features_labels.sql

# ClickHouse 历史分析
clickhouse-client --host localhost --port 9000 \
     --multiquery < schemas/analytical/init_clickhouse.sql

# Kafka Topics
bash infra/kafka/init_topics.sh
```

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| PostgreSQL | 5432 | 策略/账户/风控 |
| TimescaleDB | 5433 | 实时行情 & 特征 |
| ClickHouse HTTP | 8123 | 分析查询 |
| ClickHouse Native | 9000 | 客户端连接 |
| Redis | 6379 | 热缓存 |
| Kafka | 9092 | 消息队列 |
| Kafka UI | 8080 | 可视化管理 |
| Grafana | 3000 | 监控看板 |

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| TimescaleDB | 2.16+ | 实时时序行情 + 自动聚合 |
| ClickHouse | 24.8+ | 列存历史分析，压缩比 10:1 |
| PostgreSQL | 16 | 元数据，策略运营，模型校准 |
| Redis | 7.2+ | SHM 替代，在线特征，限速器 |
| Apache Kafka | 3.7+ | 消息解耦 & 缓冲 |
| Grafana | 10.4+ | 实时监控 |

## 数据保留策略

| 数据 | 保留期 | 引擎 | 压缩 |
|------|--------|------|------|
| Crypto Tick | 14天 | TimescaleDB | 7天后压缩 |
| Crypto OrderBook | 3天 | TimescaleDB | 1天后压缩 |
| Crypto OHLCV | 永久 | TimescaleDB→ClickHouse | 30天后压缩 |
| Prediction Market Tick | 90天 | TimescaleDB | 7天后压缩 |
| Prediction OrderBook | 7天 | TimescaleDB | 1天后压缩 |
| Prediction Minute | 180天 | TimescaleDB | 30天后压缩 |
| Crypto 特征 | 90天 | TimescaleDB | 14天后压缩 |
| CFL 标签 | 永久 | TimescaleDB+ClickHouse | — |
| Equity Tick | 90天 | TimescaleDB | — |
| Equity 日线 | 永久 | ClickHouse | MergeTree |
| Options Quote | 90天 | TimescaleDB | — |
| Options EOD | 永久 | ClickHouse | MergeTree |
| FX OHLCV | 10年 | ClickHouse | MergeTree |
| Evolution Genome | 永久 | PostgreSQL+ClickHouse | — |

## 数据源

| 市场 | 数据源 | 协议 | 覆盖 |
|------|--------|------|------|
| Crypto | OKX | WebSocket / REST | Tick/OHLCV/OB/Funding/OI |
| Prediction Market | Polymarket CLOB | WebSocket / Polygon RPC | CLOB Tick/OB/Parity/Resolution/Onchain |
| Equity | Polygon.io | REST / WebSocket | Tick/OHLCV/OrderBook |
| Options | CBOE / OCC | REST | Quote/Greeks/VolSurface |
| FX | OANDA | REST | Tick/OHLCV/Forward |

## QTS V8 核心概念

### Phase 进化
| Phase | 标签数 | G4 | 执行 | 说明 |
|-------|--------|-----|------|------|
| 0 | 0 | ❌ | 模拟 | 数据采集 + CFL 标签 + 特征工程 |
| 1 | ~1K | ❌ | 模拟 | AlphaCast 训练 + G1-G3 门控 + 回测 |
| 2 | ~2K | ❌ | 模拟 | MCTS 搜索 (Phase 4) + G1-G4 门控 |
| 3 | ~5K | ⚠️ | 模拟 | MetaLabeler 训练中 |
| 4 | ~5K+ | ✅ | Demo | 完整 G1-G4 门控 + Demo 模拟盘 |
| 5 | 持续积累 | ✅ | Live | 完整 G1-G5 门控 + 实盘 |

### 五门控体系
| 门控 | 名称 | 含义 | 数据源 |
|------|------|------|--------|
| G1 | 流动性 | 交易量充足 | Tick Volume |
| G2 | Regime | 市场状态匹配 | Regime Detection |
| G3 | 置信度 | AlphaCast conf ≥ 阈值 | AlphaCast |
| G4 | MetaLabeler | 二次验证 | MetaLabeler |
| G5 | 时间 | 避开资金费率结算 | Funding Schedule |

## 许可证

MIT License — 仅供学习与研究，实盘使用请自行评估合规风险。
