# QTS V8 · Redis 热缓存数据结构设计 (Enhanced)

## 概述
Redis 在 QTS V8 中承担四层职责：
1. **SHM 替代层**：跨语言（Rust→Python→Go）零拷贝行情传递
2. **在线特征缓存**：策略引擎实时读取特征向量
3. **MCTS 决策缓存**：避免 K 线内重复计算
4. **风控计数器 + 系统状态**

建议 `maxmemory 8GB`，策略 `allkeys-lru`。

---

## Key 命名规范
```
{ns}:{market}:{category}:{ticker/identifier}[:{sub}]
ns: shm | feat | mcts | gate | pos | rl | calib | sys | stream
```

---

## 1. 最新市场快照（Hash, TTL: 3s）

### Crypto
```bash
HSET shm:crypto:snapshot:ETH-USDT-SWAP \
    last         "2456.78" \
    bid1         "2456.70" ask1 "2456.86" \
    bid1_sz      "12.3"   ask1_sz "8.7" \
    mark_px      "2456.82" index_px "2456.50" \
    funding_rate "0.000082" \
    oi           "1234567" \
    ts_ns        "1712345678901234567" seq_id "99887766"
EXPIRE shm:crypto:snapshot:ETH-USDT-SWAP 3
```

### Equity
```bash
HSET shm:equity:snapshot:AAPL \
    price "182.34" bid "182.32" ask "182.36" \
    size "100" ts "1712345678901"
EXPIRE shm:equity:snapshot:AAPL 5
```

### FX
```bash
HSET shm:fx:snapshot:EURUSD \
    bid "1.08432" ask "1.08435" mid "1.08433" \
    spread "0.00003" ts "1712345678901"
EXPIRE shm:fx:snapshot:EURUSD 2
```

### Options
```bash
HSET shm:options:snapshot:AAPL:20241220:180:C \
    bid "3.50" ask "3.60" iv "0.2534" \
    delta "0.5123" gamma "0.0234" theta "-0.0456" vega "0.1234"
EXPIRE shm:options:snapshot:AAPL:20241220:180:C 10
```

---

## 2. 在线特征向量（Hash, TTL: 60s）

### Crypto 5m Feature
```bash
HSET feat:crypto:5m:ETH-USDT-SWAP \
    bar_index "123456" feat_version "1" \
    f_obi_1 "0.0823" f_obi_5 "0.0612" f_depth_imb "0.0341" \
    f_ofi_1m "0.1234" f_ofi_5m "0.0876" \
    f_mom_5m "0.00234" f_mom_1h "0.00891" \
    f_hv_5m "0.4521" f_hurst_5m "0.5132" \
    f_funding_r "0.000082" f_funding_z "1.234" \
    f_ls_elite_acc "1.234" f_rsi_14 "58.34" f_adx_14 "32.1" \
    f_spread_z "0.823" \
    feat_json "[0.0823,0.0612,...]"  # 完整 66 维向量
EXPIRE feat:crypto:5m:ETH-USDT-SWAP 60
```

### Equity Factor Snapshot
```bash
HSET feat:equity:1d:AAPL \
    mom_1m "3.21" mom_3m "12.34" hv_20d "0.234" \
    pe_ratio "28.5" pb_ratio "45.2" \
    rsi_14 "62.3" beta_60d "1.23" z_mom_1m "1.52"
EXPIRE feat:equity:1d:AAPL 60
```

### FX Factor Snapshot
```bash
HSET feat:fx:1h:EURUSD \
    mom_1h "0.0012" mom_4h "0.0034" hv_1h "0.0032" \
    ir_diff_3m "0.015" cot_net "0.34" \
    rsi_14 "55.2" atr_14 "0.0042" vix "15.2"
EXPIRE feat:fx:1h:EURUSD 60
```

---

## 3. AlphaCast 推断输出（Hash, TTL: 60s）
```bash
HSET feat:crypto:alphacast:ETH-USDT-SWAP \
    bar_index "123456" y_hat "0.00234" sigma "0.00087" \
    conf "0.712" regime "trending" \
    raw_logit "0.891" temp_t "1.23" \
    model_ver "v2.1.0-20240601" elapsed_ms "4.2"
EXPIRE feat:crypto:alphacast:ETH-USDT-SWAP 60
```

---

## 4. MCTS 决策缓存（Hash, TTL: 30s）
```bash
HSET mcts:crypto:result:ETH-USDT-SWAP \
    bar_index "123456" action "buy" position_level "2" \
    path_value "0.00189" simulations "800" \
    elapsed_ms "42.3" was_degraded "0"
EXPIRE mcts:crypto:result:ETH-USDT-SWAP 30
```

---

## 5. QTS V8 五门控状态（Hash, TTL: 10s）
```bash
HSET gate:crypto:status:ETH-USDT-SWAP \
    g1_pass "1" g1_reason "" \
    g2_pass "1" g2_regime "trending" \
    g3_pass "1" g3_conf "0.712" g3_threshold "0.55" \
    g4_pass "0" g4_active "0" g4_label_cnt "3420" \
    g5_pass "1" g5_mins_to_settle "234.5" \
    final_pass "1" ts_ns "1712345678000000000"
EXPIRE gate:crypto:status:ETH-USDT-SWAP 10
```

---

## 6. 实时持仓状态（Hash, 持久化）
```bash
HSET pos:crypto:live:ETH-USDT-SWAP \
    pos_qty "3.0" pos_side "long" avg_px "2441.23" \
    mark_px "2456.78" upl "46.65" upl_ratio "0.0064" \
    liq_px "1987.43" lever "3.0" margin_mode "cross" \
    open_ts_ns "1712340000000000000" last_update_ns "1712345678000000000"
```

---

## 7. API 限速计数器（String + INCR, TTL 按窗口）
```bash
# Crypto
INCR rl:crypto:okx:public:2s
EXPIRE rl:crypto:okx:public:2s 2
INCR rl:crypto:okx:private:trade:2s
EXPIRE rl:crypto:okx:private:trade:2s 2

# Equity
INCR rl:equity:polygon:1m
EXPIRE rl:equity:polygon:1m 60
```

### 原子令牌消耗 Lua 脚本
```lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local current = redis.call('INCR', key)
if current == 1 then redis.call('EXPIRE', key, tonumber(ARGV[2])) end
if current > limit then return 0 end
return 1
```

---

## 8. 风控实时计数器
```bash
# 单日成交次数
INCR risk:trades:daily:{account_id}:2024-06-01
EXPIRE risk:trades:daily:{account_id}:2024-06-01 86400

# 亏损连续计数
LPUSH risk:loss_streak:{account_id} "1"
LTRIM risk:loss_streak:{account_id} 0 9
```

---

## 9. Redis Streams（消息管道）
```bash
# K线闭合事件
XADD stream:crypto:bar_close:ETH-USDT-SWAP * \
    bar_index "123456" ts_ms "1712344800000" close "2456.78" vol "15234.0"
XTRIM stream:crypto:bar_close:ETH-USDT-SWAP MAXLEN ~ 1000

# 成交回执流（Rust→Python 标签工厂）
XADD stream:crypto:fills:ETH-USDT-SWAP * \
    ord_id "1234567890" fill_px "2456.78" fill_sz "1.0" \
    side "buy" ts_ns "1712345678000000000" conf "0.712" bar_index "123456"
XTRIM stream:crypto:fills:ETH-USDT-SWAP MAXLEN ~ 500

# 风控告警
XADD stream:risk_alert * \
    severity "WARN" type "drawdown_approaching" value "0.038" \
    threshold "0.04" ticker "ETH-USDT-SWAP"
XTRIM stream:risk_alert MAXLEN ~ 200
```

---

## 10. 系统健康状态（Hash, TTL: 30s）
```bash
HSET sys:health \
    okx_ws_status "connected" okx_ws_lag_ms "2.3" \
    feature_engine "running" last_feat_bar "123456" \
    triton_status "ok" triton_latency "4.2" \
    mcts_status "ok" db_write_lag_ms "8.1" \
    cfl_label_count "3420" daily_pnl_usdt "47.23" \
    daily_trade_cnt "12" phase "3"
EXPIRE sys:health 30
```

---

## 11. Temperature Scaling 校准（String + List）
```bash
SET calib:crypto:temp_t:ETH-USDT-SWAP "1.234"
RPUSH calib:crypto:temp_history:ETH-USDT-SWAP \
    '{"ts":"2024-06-01T08:00:00Z","t_before":1.1,"t_after":1.23,"ece_after":0.041}'
LTRIM calib:crypto:temp_history:ETH-USDT-SWAP 0 99
```

---

## 12. 活跃标的索引（Set）
```bash
SADD index:crypto:active ETH-USDT-SWAP BTC-USDT-SWAP SOL-USDT-SWAP
SADD index:equity:active AAPL MSFT TSLA NVDA AMZN
SADD index:fx:active EURUSD GBPUSD USDJPY AUDUSD USDCAD
SADD index:options:underlying SPY QQQ AAPL TSLA
```

---

## 13. 近期 K 线 Ring Buffer（List, 60 根）
```bash
LPUSH candle:crypto:ring:5m:ETH-USDT-SWAP \
    '{"ts_ms":1712344800000,"o":2450.12,"h":2460.0,"l":2448.5,"c":2456.78,"vol":15234}'
LTRIM candle:crypto:ring:5m:ETH-USDT-SWAP 0 59
```

---

## 14. 策略运行状态（Hash, 持久化）
```bash
HSET strategy:state:{strategy_id} \
    status "running" position_cnt "5" \
    gross_exp "150000" net_exp "50000" \
    daily_pnl "1234.56" last_signal "1712345678" last_trade "1712345600"
```

---

## 15. Pub/Sub 频道
```bash
# 策略引擎订阅
SUBSCRIBE channel:signals:crypto
SUBSCRIBE channel:signals:equity
SUBSCRIBE channel:signals:fx

# 采集器发布
PUBLISH channel:signals:crypto '{"ticker":"ETH-USDT-SWAP","price":2456.78,...}'
PUBLISH channel:alerts '{"severity":"WARN","type":"drawdown","value":0.08}'
```

---

## 内存估算（10 品种 Crypto + 500 Equity + 50 FX）

| 数据类型 | 数量 | 单条 | 合计 |
|----------|------|------|------|
| Crypto 快照 | 10 | ~1 KB | ~10 KB |
| Crypto 特征 | 10 | ~3 KB | ~30 KB |
| Equity 快照 | 500 | ~200 B | ~100 KB |
| Equity 因子 | 500 | ~500 B | ~250 KB |
| FX 快照 | 50 | ~200 B | ~10 KB |
| FX 因子 | 50 | ~300 B | ~15 KB |
| AlphaCast | 10 | ~500 B | ~5 KB |
| MCTS | 10 | ~500 B | ~5 KB |
| Gate | 10 | ~500 B | ~5 KB |
| Ring Buffer | 20×60 | ~150 B | ~180 KB |
| Streams | ~2000 | ~100 B | ~200 KB |
| **合计** | | | **~1 MB** |

建议 `maxmemory 4GB`，策略 `allkeys-lru`。

---

## 16. Prediction Market 队列

```bash
# 平价套利信号队列 (FIFO)
RPUSH parabolic_arb:{slug} '{"yes_price":0.58,"no_price":0.44,"parity":0.02,"arb_signal":"buy_no_sell_yes","ts_ns":"..."}'
LTRIM parabolic_arb:{slug} 0 99

# 活跃市场排名 (Sorted Set by volume)
ZADD index:prediction:volume_leaderboard 123456 "will-btc-hit-100k"
ZADD index:prediction:volume_leaderboard 98765 "fed-cut-rates-july"

# 即将到期市场
ZADD index:prediction:expiring "{{end_ts}}" "slug"

# 链上事件缓冲
RPUSH stream:prediction:onchain:{market_id} '{"event_type":"Split","tx_hash":"0x...","block":12345}'
LTRIM stream:prediction:onchain:{market_id} 0 499
```

## 17. 策略进化缓存

```bash
# 基因组热点
genome:{gene_id}  HASH  {strategy_name, version, status, fitness, sharpe, stability, created_at}
genome_trees:{gene_id}  HASH  {entry_tree, exit_tree, stop_loss, take_profit, position_pct}

# 进化排行 (Sorted Sets)
ZADD evo:leaderboard:fitness {{fitness_score}} {{gene_id}}
ZADD evo:leaderboard:sharpe {{sharpe_ratio}} {{gene_id}}

# 变异任务队列 (FIFO, 触发回测)
LPUSH evo:mutation_queue '{"parent_gene_id":"...","mutation_type":"genetic_programming","changes":{...}}'

# AB 测试实验
ab_test:{id}  HASH  {name, control_strat, treatment_strat, metric, status, confidence, winner}

# 截面排名 (CrossSection Cache)
cs_rank:{market}:{ticker}  HASH  {cs_composite, cs_tier, cs_rank_obi_vel_60s, ...}
ZADD cs_tier3:{market} {{cs_composite}} {{ticker}}  # 仅 tier>=3 的最优机会
```

> **Prediction + Evolution 增量估算**: ~50 MB for 500 active markets × 10 snapshots + 350 genomes

---

## 18. 期货特征缓存

```bash
# COT 情绪 (HASH, TTL: 7d — 每周更新即可)
HSET feat:fut:cot:ES \
    net_spec_pos    "28500" \
    spec_long_pct   "65.2" \
    spec_short_pct  "18.5" \
    cot_index       "72.3" \
    oi_total        "2405721"

# 期限结构 (ZSET, 按 DTE 排序)
ZADD feat:fut:term_structure:ES \
    30 "basis_pct:0.08|price:6012.50" \
    90 "basis_pct:0.02|price:6010.25" \
    180 "basis_pct:-0.15|price:5998.00"

# 市场 Regime 快照
SET feat:fut:regime:ES "contango" EX 60

# 活跃合约
HSET fut:active_contracts ES "202606" CL "202607" GC "202608" ZN "202609"

# 曲线数据
ZADD fut:curve:ES 30 "6002.50" 60 "6008.00" 90 "6010.25"
```

---

## 更新内存估算

| 数据类型 | 数量 | 单条 | 合计 |
|----------|------|------|------|
| Crypto/Eq/FX 原有 | (上述) | — | ~1 MB |
| Prediction Market | 500 | ~200 B | ~100 KB |
| Evolution Genome | 350 | ~2 KB | ~700 KB |
| CrossSection Cache | 500 | ~300 B | ~150 KB |
| AB Test / Parity Arb | 100 | ~500 B | ~50 KB |
| Futures COT/Term/Curve | 50 | ~500 B | ~25 KB |
| **合计** | | | **~2.1 MB** |

建议 `maxmemory 4GB`，策略 `allkeys-lru`。

---

## Python 连接示例
```python
import redis

r = redis.Redis(host='localhost', port=6379, password='quant2024',
                decode_responses=True, max_connections=50)

# 读取 Crypto 快照
snap = r.hgetall('shm:crypto:snapshot:ETH-USDT-SWAP')

# Pipeline 批量读
pipe = r.pipeline()
for ticker in ['ETH-USDT-SWAP', 'BTC-USDT-SWAP']:
    pipe.hgetall(f'shm:crypto:snapshot:{ticker}')
results = pipe.execute()

# 读取特征向量
feat = r.hgetall('feat:crypto:5m:ETH-USDT-SWAP')
```
