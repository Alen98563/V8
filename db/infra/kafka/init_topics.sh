#!/bin/bash
# =============================================================
# QTS V8 · Kafka Topic 初始化
# =============================================================
set -euo pipefail

KAFKA_BIN="${KAFKA_HOME:-/usr/local/kafka}/bin"
BOOTSTRAP="${BOOTSTRAP_SERVERS:-localhost:9092}"

TOPICS=(
    # Crypto 行情流
    "market.crypto.tick:12:1"           # 逐笔成交
    "market.crypto.orderbook:6:1"       # 盘口快照
    "market.crypto.ohlcv:6:1"           # K线
    "market.crypto.funding:3:1"         # 资金费率
    "market.crypto.mark_price:3:1"      # 标记价格
    "market.crypto.open_interest:3:1"   # 持仓量
    "market.crypto.liquidations:3:1"    # 强平
    "market.crypto.ls_ratio:3:1"        # 多空比

    # Multi-Asset 行情流
    "market.equity.tick:12:1"           # 美股逐笔
    "market.equity.ohlcv:6:1"           # 美股K线
    "market.options.quote:6:1"          # 期权报价
    "market.options.eod:3:1"            # 期权日终
    "market.futures.tick:6:1"           # 期货逐笔
    "market.futures.ohlcv:6:1"          # 期货K线
    "market.futures.cot:1:1"            # CFTC 持仓周报
    "market.futures.eod:3:1"            # 期货日终结算
    "market.fx.tick:6:1"                # 外汇逐笔
    "market.fx.ohlcv:3:1"               # 外汇K线

    # 特征 & ML 流
    "features.crypto.5m:6:1"            # 5m 特征向量
    "features.alphacast.output:3:1"     # AlphaCast 输出
    "features.mcts.decision:3:1"        # MCTS 决策

    # 标签 & 训练流
    "labels.cfl:6:1"                    # CFL 标签
    "labels.metalabeler.run:3:1"        # MetaLabeler 训练触发
    "calibration.temp_scaling:3:1"      # Temperature Scaling 校准

    # 风控 & 审计流
    "risk.alerts:3:1"                   # 风控告警
    "audit.events:3:1"                  # 操作审计

    # Prediction Market 行情流
    "market.prediction.tick:6:1"        # Polymarket 逐笔
    "market.prediction.orderbook:3:1"   # Polymarket OB
    "market.prediction.minute:3:1"      # 1m 聚合快照
    "market.prediction.resolution:1:1"  # 事件解决
    "market.prediction.onchain:3:1"     # 链上事件
    "market.prediction.parity:3:1"      # 平价套利信号

    # 策略进化流
    "evolution.genome:3:1"              # 基因组创建/更新
    "evolution.mutation:3:1"            # 变异事件
    "evolution.arena:3:1"               # 竞技结果
    "evolution.promotion:1:1"           # 晋升事件
    "evolution.retirement:1:1"          # 退役事件

    # 信号 & 订单流
    "signals.strategy:6:1"              # 策略信号
    "orders.execution:6:1"              # 订单执行
    "orders.fills:6:1"                  # 成交回执
)

echo "============================================"
echo " QTS V8 · Kafka Topic Initialization"
echo " Bootstrap: $BOOTSTRAP"
echo "============================================"

for topic_spec in "${TOPICS[@]}"; do
    IFS=':' read -r name partitions rf <<< "$topic_spec"

    if "$KAFKA_BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" \
        --describe --topic "$name" &>/dev/null; then
        echo "  [SKIP] $name (already exists)"
    else
        "$KAFKA_BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" \
            --create --topic "$name" \
            --partitions "$partitions" \
            --replication-factor "$rf" \
            --config retention.ms=604800000 \
            --config compression.type=lz4
        echo "  [CREATE] $name (p=$partitions, rf=$rf)"
    fi
done

echo ""
echo "============================================"
echo " Topics created successfully"
echo "============================================"

# List all topics
echo ""
"$KAFKA_BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --list | grep 'market\.\|features\.\|labels\.\|risk\.\|audit\.\|signals\.\|orders\.\|calibration\.'
