#!/bin/bash
# =============================================================
# QTS V8 · 一键初始化全栈数据库
# 用法: bash scripts/init_all.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCHEMAS_DIR="$PROJECT_DIR/schemas"

PG_HOST="${PG_HOST:-localhost}"
PG_USER="${PG_USER:-quant}"
PG_PASS="${PG_PASS:-quant2024}"

TS_HOST="${TS_HOST:-localhost}"
TS_PORT="${TS_PORT:-5433}"

CH_HOST="${CH_HOST:-localhost}"
CH_PORT="${CH_PORT:-9000}"
CH_USER="${CH_USER:-quant}"
CH_PASS="${CH_PASS:-quant2024}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASS="${REDIS_PASS:-quant2024}"

export PGPASSWORD="$PG_PASS"

echo "============================================"
echo " QTS V8 · Database Initialization"
echo "============================================"

# ─────────────────────────────────────────────
# 1. PostgreSQL: 策略/账户/风控/校准
# ─────────────────────────────────────────────
echo ""
echo "[1/6] Initializing PostgreSQL (qts_ops)..."
psql -h "$PG_HOST" -p 5432 -U "$PG_USER" -d qts_ops \
     -f "$SCHEMAS_DIR/relational/init_postgres.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ PostgreSQL done"

# ─────────────────────────────────────────────
# 2. TimescaleDB: Crypto 行情
# ─────────────────────────────────────────────
echo ""
echo "[2/6] Initializing TimescaleDB - Crypto Market..."
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market \
     -f "$SCHEMAS_DIR/timeseries/market/init_crypto_market.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ Crypto market done"

# ─────────────────────────────────────────────
# 3. TimescaleDB: Futures 行情
# ─────────────────────────────────────────────
echo ""
echo "[3/8] Initializing TimescaleDB - Futures Market..."
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market \
     -f "$SCHEMAS_DIR/timeseries/market/init_futures_market.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ Futures market done"

# ─────────────────────────────────────────────
# 4. TimescaleDB: Multi-Asset 行情
# ─────────────────────────────────────────────
echo ""
echo "[4/8] Initializing TimescaleDB - Multi-Asset Market..."
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market \
     -f "$SCHEMAS_DIR/timeseries/market/init_multi_market.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ Multi-asset market done"

# ─────────────────────────────────────────────
# 5. TimescaleDB: Prediction Market 行情
# ─────────────────────────────────────────────
echo ""
echo "[5/8] Initializing TimescaleDB - Prediction Market..."
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market \
     -f "$SCHEMAS_DIR/timeseries/market/init_prediction_market.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ Prediction market done"

# ─────────────────────────────────────────────
# 6. TimescaleDB: 特征 & 标签
# ─────────────────────────────────────────────
echo ""
echo "[6/8] Initializing TimescaleDB - Features & Labels..."
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market \
     -f "$SCHEMAS_DIR/timeseries/features/init_features_labels.sql" \
     -v ON_ERROR_STOP=1
echo "  ✓ Features & labels done"

# ─────────────────────────────────────────────
# 7. ClickHouse: 历史分析
# ─────────────────────────────────────────────
echo ""
echo "[7/8] Initializing ClickHouse (qts_hist)..."
clickhouse-client \
    --host "$CH_HOST" --port "$CH_PORT" \
    --user "$CH_USER" --password "$CH_PASS" \
    --multiquery < "$SCHEMAS_DIR/analytical/init_clickhouse.sql"
echo "  ✓ ClickHouse done"

# ─────────────────────────────────────────────
# 8. Kafka Topics
# ─────────────────────────────────────────────
echo ""
echo "[8/8] Initializing Kafka Topics..."
bash "$PROJECT_DIR/infra/kafka/init_topics.sh"
echo "  ✓ Kafka topics done"

# ─────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────
echo ""
echo "============================================"
echo " Verifying..."
echo "============================================"

# PostgreSQL
echo -n "  PostgreSQL: "
psql -h "$PG_HOST" -p 5432 -U "$PG_USER" -d qts_ops -t -c \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('ref','strategy','portfolio','risk','calibration','audit')" \
    2>/dev/null | tr -d ' ' && echo " tables" || echo " ✗ FAILED"

# TimescaleDB
echo -n "  TimescaleDB: "
psql -h "$TS_HOST" -p "$TS_PORT" -U "$PG_USER" -d qts_market -t -c \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('crypto','equity','futures','options','fx','prediction','features','labels','meta')" \
    2>/dev/null | tr -d ' ' && echo " tables" || echo " ✗ FAILED"

# ClickHouse
echo -n "  ClickHouse: "
clickhouse-client --host "$CH_HOST" --port "$CH_PORT" --user "$CH_USER" --password "$CH_PASS" \
    --query "SELECT COUNT() FROM system.tables WHERE database='qts_hist'" 2>/dev/null && echo " tables" || echo " ✗ FAILED"

# Redis
echo -n "  Redis: "
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASS" --no-auth-warning PING 2>/dev/null || echo " ✗ FAILED"

echo ""
echo "============================================"
echo " QTS V8 · Database Initialization Complete"
echo "============================================"
