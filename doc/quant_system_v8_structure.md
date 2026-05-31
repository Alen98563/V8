# 量化交易系统 V8 · 项目开发目录结构与说明

> 基于蓝图：QTS V8 FUSION — ResNet · AlphaCast · MCTS · Empirical Alpha · Harness

---

## 一、总体目录树

```
quant_system_v8/
│
├── README.md
├── pyproject.toml                  # 依赖管理 (Poetry / uv)
├── .env.example                    # 环境变量模板
├── docker-compose.yml              # Redis + 推断服务编排
│
├── config/
│   ├── settings.py                 # 全局参数（VaR 阈值、Kelly 上限、日亏停等）
│   ├── markets.yaml                # 每市场开关与参数
│   └── logging.yaml                # 结构化日志配置
│
├── adapters/                       # L1 · 市场适配器层
│   ├── base.py                     # IMarketAdapter 抽象接口
│   ├── poly_adapter.py             # Polymarket CLOB 签名 / 代理出口
│   ├── crypto_adapter.py           # Binance/OKX/Bybit 永续 · 资金费率 · 强平
│   ├── equity_adapter.py           # Alpaca REST / IB TWS · PDT · T+2
│   └── forex_adapter.py            # MT4/5 ZMQ Bridge · DWX EA
│
├── data/                           # L0–L2 · 数据源 & 标准化
│   ├── sources/
│   │   ├── poly_scanner.py         # REST 500+ 市场 Hot/Active/Warm 三级池
│   │   ├── crypto_scanner.py       # 资金费率排行 · OI 变化 · 多所对比
│   │   ├── equity_scanner.py       # IV rank · 期权流 · 异常成交量
│   │   ├── forex_scanner.py        # 主要货币对价差 · 新闻事件 · COT
│   │   └── news_nlp.py             # Twitter · NewsAPI · 事件 NLP
│   ├── normalizer.py               # 统一 Tick/OHLCV · 全 P&L 换算 USD
│   └── redis_bus.py                # Redis 总线命名空间管理
│                                   #   poly:* / crypto:* / equity:* / forex:*
│
├── features/                       # Phase 1 · 特征基础设施
│   ├── market_state_buffer.py      # 循环时序缓冲 deque(3600) · MarketSnapshot
│   ├── feature_engine.py           # 50+ velocity/regime/micro/flow 特征
│   └── cross_section.py            # 全市场百分位 · cs_composite · cs_tier · 5s 刷新
│
├── models/                         # V8 AI 推演引擎
│   │
│   ├── resnet/                     # 🧬 ResNet 深度特征提取（V8 NEW）
│   │   ├── resnet_encoder.py       # 6 残差块 · 128d 嵌入 · 多尺度 Conv1D
│   │   ├── feature_fusion.py       # 注意力门控融合 · EmpAlpha(50d) ⊕ ResNet(128d) → 178d
│   │   ├── train_resnet.py         # 离线训练脚本 (RTX3090)
│   │   └── serve_resnet.py         # ONNX 推断服务 localhost:8001 · <5ms
│   │
│   ├── alphacast/                  # 🔮 AlphaCast 时序预测（V8 NEW）
│   │   ├── alphacast_model.py      # Transformer 6L8H · 多任务头 · 收益/σ/conf/state
│   │   ├── alphacast_recalib.py    # MCTS 后二次校准 · 置信度过滤 · 仓位精炼
│   │   ├── train_alphacast.py      # 离线训练脚本
│   │   └── serve_alphacast.py      # TorchScript 推断服务 localhost:8002 · <10ms
│   │
│   └── mcts/                       # ♟ MCTS 蒙特卡洛树搜索（V8 NEW）
│       ├── mcts_planner.py         # UCB1 树搜索 · AlphaCast rollout · Sharpe 奖励
│       ├── mcts_worker.py          # 异步并行任务池 · Redis 结果缓存 30s · 降级模式
│       └── mcts_config.py          # N=5–20步 · 模拟=200–1000 · γ=0.95 · λ=0.3
│
├── alpha/                          # L3 · Alpha 引擎层（12 引擎并行）
│   ├── base_alpha.py               # AlphaSignal 基类
│   │
│   ├── poly/                       # Polymarket Alpha
│   │   ├── spread_capture.py       # 点差套利 · ~40–80笔/天
│   │   ├── obi_v2.py               # 订单簿不平衡 · ~20–40笔/天
│   │   ├── ofi_engine.py           # 订单流不平衡 · ~10–30笔/天
│   │   ├── prob_surface.py         # 概率曲面套利 · ~10–30笔/天
│   │   ├── temporal_arb.py         # 时间滞后套利 · ~10–40笔/天
│   │   ├── cluster_scanner.py      # 市场聚类 · ~10–30笔/天
│   │   ├── event_shock_nlp.py      # 事件冲击 · ~5–10笔/天
│   │   └── momentum.py             # 动量跟踪 · ~10–20笔/天
│   │
│   ├── crypto/
│   │   └── funding_rate_arb.py     # 资金费率套利 · 现货/永续对冲
│   │
│   ├── equity/
│   │   └── iv_surface_engine.py    # IV skew 套利 · Calendar spread
│   │
│   ├── forex/
│   │   └── fx_carry_engine.py      # 利差套利 · 新闻事件驱动
│   │
│   └── cross_market/
│       └── cross_market_momentum.py # BTC→Poly · NASDAQ 情绪传导
│
├── gating/                         # Phase 2 · 硬门控 & 反事实标签
│   ├── hard_gating.py              # G1–G4 分层门控 · V8: AlphaCast 动态调节 G3
│   └── counterfactual_labeler.py   # 100% 信号记录 · V8: 额外记录 AlphaCast+MCTS 字段
│
├── ml/                             # Phase 3 · 机器学习激活（5K labels 后）
│   ├── meta_labeler.py             # LightGBM · 输入含 AlphaCast.conf + MCTS.value
│   └── model_lifecycle.py          # 自动训练/重训/IC 衰减告警
│
├── risk/                           # L4 · 投资组合决策与风控
│   ├── portfolio_risk.py           # VaR · Kelly · V8: AlphaCast σ 入 VaR
│   ├── signal_fusion.py            # 多因子信号融合器 · 动态权重 · AlphaCast 校正
│   └── position_limits.py          # 每市场仓位上限 · 日亏损硬停 $300
│
├── execution/                      # L5–L7 · 路由 & 执行 & 结算
│   ├── smart_router.py             # UnifiedOrder 路由 · MCTS 最优执行时机
│   ├── order_state_machine.py      # NEW→POSTED→PARTIAL→FILLED/CANCELED
│   ├── behavior_humanizer.py       # 随机延迟 0.1–0.5s · Poisson 节奏 · ±0.002 价格微扰
│   │
│   ├── channels/                   # L6 · 多路并行执行
│   │   ├── poly_exec.py            # 住宅 IP 池 · REST POST /order · Polygon RPC
│   │   ├── crypto_exec.py          # 交易所 REST · 强平预警减仓
│   │   ├── equity_exec.py          # Alpaca REST / IB TWS port 7497
│   │   └── forex_exec.py           # pyzmq PUSH → DWX EA · ZMQ PULL 回执
│   │
│   └── settlement/                 # L7 · 多市场结算 & P&L
│       ├── poly_settlement.py      # Polygon RPC · Gas 优化 · 即时确认
│       ├── equity_settlement.py    # T+2 DTCC 清算 · 可用余额实时
│       ├── forex_settlement.py     # 保证金实时扣减 · swap 入账
│       └── pnl_aggregator.py       # 统一 P&L · Sharpe/Sortino/MaxDD · 写 Redis portfolio:*
│
├── orchestrator/                   # 全局编排
│   ├── orchestrator_patch.py       # 信号评估串联 · V8: 触发 ResNet/AlphaCast 在线校准
│   └── pipeline_runner.py          # 主循环入口 · 启动所有子系统
│
├── harness/                        # ⚙️ Harness · L5–L7 统一封装（V8 NEW）
│   ├── harness_pipeline.py         # L5-L7 统一编排 · 日志/告警/监控/回测 Wrapper
│   ├── logging_handler.py          # 结构化 JSON 日志 · 全链路追踪 ID
│   ├── alerting.py                 # Telegram / 钉钉 · IC 衰减/强平/PDT 告警
│   ├── monitoring.py               # FastAPI + WebSocket SPA · 秒级刷新
│   └── backtesting.py              # 多市场联合回测 · AlphaCast 历史推演 · MCTS 路径离线验证
│
├── infra/                          # 基础设施配置
│   ├── redis/
│   │   └── redis.conf              # Redis 7.x · 16G 内存配置
│   ├── proxy/
│   │   └── residential_ip.py       # Bright Data 住宅 IP · Session 绑定 · 5–15min 冷却
│   └── deploy/
│       ├── vps_setup.sh            # US East VPS 初始化 (8核/16G/SSD)
│       └── mt4_windows.md          # Windows VPS MT4 终端配置说明
│
├── scripts/                        # 运维 & 实用脚本
│   ├── start_all.sh                # 启动全系统
│   ├── stop_all.sh                 # 安全停止
│   ├── health_check.py             # 各服务存活检查
│   └── flush_redis.py              # 清空 Redis 指定命名空间
│
└── tests/
    ├── unit/
    │   ├── test_resnet_encoder.py
    │   ├── test_alphacast_model.py
    │   ├── test_mcts_planner.py
    │   ├── test_feature_engine.py
    │   ├── test_hard_gating.py
    │   └── test_portfolio_risk.py
    └── integration/
        ├── test_pipeline_e2e.py    # 全链路集成测试（含 V8 AI 模块）
        └── test_harness_wrapper.py
```

---

## 二、核心模块说明

### 2.1 L0–L2：数据层

| 目录/文件 | 职责 | 说明 |
|---|---|---|
| `adapters/base.py` | 统一接口 `IMarketAdapter` | 上层永远只与此通信：`connect()` / `place_order()` / `get_position()` 等 |
| `adapters/poly_adapter.py` | Polymarket 接入 | CLOB 签名、链上 Gas、Polygon RPC、住宅代理出口 |
| `adapters/crypto_adapter.py` | 加密永续 | Binance/OKX/Bybit WS、资金费率监控、强平预警 |
| `adapters/equity_adapter.py` | 美股/期权 | PDT 计数器、T+2 追踪、期权链、Alpaca/IB 双模式 |
| `adapters/forex_adapter.py` | 外汇 MT4/5 | 手数换算、pip 值、过夜利息、ZMQ→MT4 桥 |
| `data/redis_bus.py` | Redis 总线 | 命名空间：`poly:*` / `crypto:*` / `equity:*` / `forex:*`，存储 Orderbook、Trades、Signals、Features、Labels |

### 2.2 Phase 1：特征基础设施

| 文件 | 职责 | 关键参数 |
|---|---|---|
| `features/market_state_buffer.py` | 循环时序缓冲 | 每市场独立 `deque(3600)`，O(1) push，`get_window(seconds)` |
| `features/feature_engine.py` | 特征工程 | 50+ 特征：velocity(6窗口)、regime(Hurst/vol)、microstructure、order-flow |
| `features/cross_section.py` | 跨截面分析 | 全市场百分位排名、`cs_composite [0,1]`、`cs_tier 0–3`，每 5s 刷新 |

### 2.3 V8 AI 推演引擎

#### ResNet 深度特征提取
| 文件 | 职责 |
|---|---|
| `models/resnet/resnet_encoder.py` | 6 残差块，BatchNorm + LeakyReLU，多尺度 1D-Conv 时序编码 [30/60/120/240s]，输出 **128 维**深度嵌入 |
| `models/resnet/feature_fusion.py` | 注意力门控融合：**Empirical Alpha 50维 ⊕ ResNet 128维 → 178维**融合向量 |
| `models/resnet/serve_resnet.py` | ONNX 推断服务，`localhost:8001`，批量推断，**<5ms** |

#### AlphaCast 时序预测
| 文件 | 职责 |
|---|---|
| `models/alphacast/alphacast_model.py` | Transformer 6层8头，多任务输出：预测收益 ŷ、不确定性 σ、置信度 conf、市场状态；4个市场独立头 |
| `models/alphacast/alphacast_recalib.py` | MCTS 最优路径后二次校准；过滤规则：conf < 0.55 → 拒绝，σ 过大 → 降仓，收益/风险比 < 1.0 → 放弃 |
| `models/alphacast/serve_alphacast.py` | TorchScript 推断服务，`localhost:8002`，**<10ms** |

#### MCTS 蒙特卡洛树搜索
| 文件 | 职责 |
|---|---|
| `models/mcts/mcts_planner.py` | UCB1 树搜索，4 类动作（买入/卖出/持仓/平仓）× 仓位离散化，AlphaCast 快速 rollout，Sharpe 加权奖励 |
| `models/mcts/mcts_worker.py` | Celery/asyncio 异步任务池（8 workers），Redis 结果缓存 30s，降级模式（AlphaCast 直接输出） |
| `models/mcts/mcts_config.py` | N=5–20步，模拟=200–1000次，折现因子 γ=0.95，风险惩罚 λ=0.3 |

### 2.4 Alpha 引擎层（12 引擎）

| 引擎 | 市场 | 频率 |
|---|---|---|
| Spread Capture | Poly | ~40–80笔/天 |
| OBI v2 | Poly | ~20–40笔/天 |
| OFI Engine | Poly | ~10–30笔/天 |
| Prob. Surface | Poly | ~10–30笔/天 |
| Temporal Arb | Poly | ~10–40笔/天 |
| Cluster Scanner | Poly | ~10–30笔/天 |
| Event Shock NLP | Poly | ~5–10笔/天 |
| Momentum | Poly | ~10–20笔/天 |
| Funding Rate Arb | Crypto | 持续 |
| IV Surface Engine | Equity | 日内 |
| FX Carry Engine | Forex | 持续 |
| Cross-Market Mom | 跨市场 | ~5–15笔/天 |

### 2.5 Phase 2：门控与标签

| 文件 | 职责 |
|---|---|
| `gating/hard_gating.py` | **G1** 流动性门：depth < 50 / spread_z > 2.5σ；**G2** Regime 门：vol > 4% / Hurst < 0.5；**G3** 跨截面门：cs_composite < 0.35（V8: AlphaCast 动态调节）；**G4** Meta-Label 门（Phase 3 后激活） |
| `gating/counterfactual_labeler.py` | 100% 信号记录（通过+拒绝），50+ 特征快照，多窗口标注 5m/30m/2h/8h，V8 新增 AlphaCast+MCTS 字段 |

### 2.6 Phase 3：ML 激活（≥5K labels）

| 文件 | 职责 |
|---|---|
| `ml/meta_labeler.py` | LightGBM Binary，输入：50+ features + cs_* + AlphaCast.conf + MCTS.value；Top-Decile Filter G4，threshold = percentile(val_preds, 90)；预期 lift 1.3–1.8x，IC 0.06–0.12 |
| `ml/model_lifecycle.py` | n_complete ≥ 5K 触发首次训练，每 7 天滚动 30 天窗口重训，IC 衰减 → 告警 + 重训 |

### 2.7 L4：风控层

| 文件 | 职责 |
|---|---|
| `risk/signal_fusion.py` | 多因子权重：Poly-Spread 18% / OBI 16% / OFI 12% / Prob 16% / Temporal 16% 等；AlphaCast 置信度作为乘数加权最终仓位 |
| `risk/portfolio_risk.py` | 组合 VaR 95%（BTC-NASDAQ ρ≈0.7），AlphaCast σ 入 VaR 计算，Kelly ≤ 25% |
| `risk/position_limits.py` | Poly ≤$100 / Crypto ≤$500 / Equity ≤$1000 / Forex ≤1手；日亏损硬停 $300 |

### 2.8 L5–L7：执行与结算

| 文件 | 职责 |
|---|---|
| `execution/smart_router.py` | UnifiedOrder 按 market 字段路由；Limit Order 优先；MCTS 最优执行时机 |
| `execution/behavior_humanizer.py` | 风控反侦察：随机延迟 0.1–0.5s，Poisson 节奏，±0.002 价格微扰 |
| `execution/settlement/pnl_aggregator.py` | 全市场 P&L 汇总 USD，实时 Sharpe/Sortino/MaxDD，AlphaCast 预测误差追踪，MCTS 路径准确率统计 |

### 2.9 Harness 统一封装（V8 NEW）

| 文件 | 职责 |
|---|---|
| `harness/harness_pipeline.py` | L5–L7 统一编排，无功能回归 Wrapper |
| `harness/logging_handler.py` | 结构化 JSON 日志，全链路追踪 ID，信号→下单→成交全记录 |
| `harness/alerting.py` | Telegram/钉钉推送；触发条件：风控/IC 衰减/强平/PDT/AlphaCast 置信度骤降 |
| `harness/monitoring.py` | FastAPI + WebSocket SPA，秒级刷新，指标：gate_pass_rate / MCTS 仿真次数 / AlphaCast IC |
| `harness/backtesting.py` | 多市场联合回测，AlphaCast 历史推演，MCTS 路径离线验证 |

---

## 三、V8 分阶段上线计划

| 阶段 | 周期 | 关键操作 | 验收标准 | 回滚方案 |
|---|---|---|---|---|
| **P1 特征（原）** | Week 1–2 | BufferRegistry · FeatureEngine · CrossSection | 50+特征 · <5ms · 无内存泄漏 | 删除 `on_ws_snapshot()` 调用 |
| **ResNet 接入** | Week 3 | `resnet_encoder.py` + `feature_fusion.py` 接入特征流 | 178d 融合向量稳定 · 余弦相似 >0.85 · <5ms | 禁用 ResNet，回退 50d 纯经验特征 |
| **P2 门控（原）** | Week 3–4 | HardGating G1–G3 · CFL 标签工厂 | pass_rate 20–50% · 分布合理 | `evaluate_signal()` 直通 |
| **AlphaCast 接入** | Week 4–5 | `alphacast_model.py` · 初次预测接入信号流 | IC > 0.05 · conf 分布合理 · <10ms | conf 固定 0.5，回退原始逻辑 |
| **MCTS 接入** | Week 5–6 | `mcts_planner.py` · 异步任务池 · MCTS→校准串联 | 路径准确率 >55% · 全链路 <250ms · 无死锁 | 跳过 MCTS，AlphaCast 直接输出 |
| **Harness 封装** | Week 2+ | `harness_pipeline.py` 包装 L5–L7 | 无功能回归 · 日志结构化 · 告警正常 | 各子模块独立运行 |
| **P3 ML 激活** | Week 8+ | n_complete ≥ 5K → MetaLabeler 训练 · G4 启用 | AUC > 0.55 · lift > 1.2x · IC > 0.05 | `is_active=False` |

---

## 四、部署配置（最小化）

| 组件 | 规格 | 用途 |
|---|---|---|
| **US East VPS** | 8核/16G/SSD | 策略 + Redis + AI 推断服务 + MCTS 池 |
| **GPU 训练服务器** | RTX3090×1 | ResNet + AlphaCast 离线训练 |
| **住宅 IP × 3–5** | Bright Data | Poly/Crypto 下单出口（Virginia/NY/Illinois） |
| **Windows VPS** | 1核/2G | MT4 终端 + DWX EA |
| **Redis 7.x** | 16G 内存 | 数据总线 + MCTS 结果缓存 |

**最优网络路径：** VPS WS 直连（1–10ms）→ 策略+ResNet+AlphaCast+MCTS（异步 <220ms）→ 住宅 IP + 行为伪装（+100–300ms）→ 市场

---

## 五、V8 闭环反馈架构

```
实盘结果
    ↓
成交均价误差 / 预测收益 vs 实际收益 / MCTS 路径成功率
    ↓  写入 Redis
    ├─→ AlphaCast 在线校准（温度缩放系数更新）
    ├─→ MCTS 奖励函数更新（Sharpe 基准调整）
    └─→ MetaLabeler 重训触发（IC 衰减检测）
```

**四路反馈：**
- **A** — 标签积累 → 模型精度提升
- **B** — feature_importance 审查 → 特征工程迭代
- **C** — gate.stats() 偏移 → 参数调整
- **D（V8 新增）** — MCTS 仿真路径回测 → 奖励函数校准

---

## 六、关键接口约定

```python
# UnifiedOrder 跨市场订单标准格式
@dataclass
class UnifiedOrder:
    market: Literal["poly", "crypto", "equity", "forex"]
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["limit", "market"]
    price: Optional[float]
    # V8 扩展字段
    alphacast_conf: float      # AlphaCast 置信度
    mcts_path_value: float     # MCTS 最优路径期望收益
    trace_id: str              # Harness 全链路追踪 ID

# IMarketAdapter 统一接口
class IMarketAdapter(ABC):
    def connect(self) -> None: ...
    def place_order(self, order: UnifiedOrder) -> OrderResult: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_position(self, symbol: str) -> Position: ...
    def get_balance(self) -> float: ...
    def subscribe_data(self, callback: Callable) -> None: ...
    def is_session_open(self) -> bool: ...
```

---

*生成于 QTS V8 融合蓝图 · 文档版本 2026-05*
