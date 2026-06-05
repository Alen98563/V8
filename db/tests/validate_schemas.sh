#!/usr/bin/env bash
# ============================================================
# QTS V8 · Schema Validation Test Suite
# Usage: ./tests/validate_schemas.sh [--verbose]
# ============================================================
set -euo pipefail

VERBOSE=${1:-}
TIMESCALE_PORT=${TS_PORT:-5433}
PG_PORT=${PG_PORT:-5432}
CH_HTTP_PORT=${CH_HTTP_PORT:-8123}
CH_NATIVE_PORT=${CH_NATIVE_PORT:-9000}

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); [ -n "$VERBOSE" ] && echo "  PASS  $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL  $1 — $2"; }

echo "============================================"
echo " QTS V8 · Schema Validation"
echo "============================================"

# --- PostgreSQL ---
echo ""
echo "[1/6] PostgreSQL (qts_ops)"

pg_check() {
  psql "postgresql://${PG_USER:-quant}:${PG_PASSWORD:-quant2024}@${PG_HOST:-localhost}:$PG_PORT/${PG_DB:-qts_ops}" \
    -t -c "$1" 2>/dev/null
}

for schema in strategy account risk calibration evolution; do
  if pg_check "SELECT 1 FROM information_schema.schemata WHERE schema_name='$schema'" | grep -q 1; then
    pass "schema: $schema"
  else
    fail "schema: $schema" "missing"
  fi
done

for tbl in strategy.definitions strategy.metalabeler strategy.ab_tests strategy.stability strategy.cross_section; do
  if pg_check "SELECT 1 FROM information_schema.tables WHERE table_schema||'.'||table_name='$tbl'" | grep -q 1; then
    pass "table: $tbl"
  else
    fail "table: $tbl" "missing"
  fi
done

# --- TimescaleDB ---
echo ""
echo "[2/6] TimescaleDB (qts_market)"

ts_check() {
  psql "postgresql://${TS_USER:-quant}:${TS_PASSWORD:-quant2024}@${TS_HOST:-localhost}:$TIMESCALE_PORT/${TS_DB:-qts_market}" \
    -t -c "$1" 2>/dev/null
}

for schema in crypto_market futures features prediction_market multi_market meta; do
  if ts_check "SELECT 1 FROM information_schema.schemata WHERE schema_name='$schema'" | grep -q 1; then
    pass "schema: $schema"
  else
    fail "schema: $schema" "missing"
  fi
done

# Verify TimescaleDB extension
if ts_check "SELECT 1 FROM pg_extension WHERE extname='timescaledb'" | grep -q 1; then
  pass "extension: timescaledb"
else
  fail "extension: timescaledb" "not installed"
fi

# Verify hypertables
EXPECTED_HT=$(ts_check "SELECT count(*) FROM timescaledb_information.hypertables")
HT_COUNT=$(echo "$EXPECTED_HT" | tr -d ' ')
echo "  INFO  Hypertables: $HT_COUNT"

# --- ClickHouse ---
echo ""
echo "[3/6] ClickHouse (qts_hist)"

ch_check() {
  curl -s -u "${CH_USER:-quant}:${CH_PASSWORD:-quant2024}" \
    "http://${CH_HOST:-localhost}:$CH_HTTP_PORT" \
    --data-binary "$1" 2>/dev/null
}

if ch_check "SELECT 1" | grep -q "1"; then
  pass "connection"
else
  fail "connection" "unreachable"
fi

for db in qts_hist; do
  if ch_check "SELECT 1 FROM system.databases WHERE name='$db'" | grep -q "1"; then
    pass "database: $db"
  else
    fail "database: $db" "missing"
  fi
done

# --- Redis ---
echo ""
echo "[4/6] Redis Cache"

REDIS_AUTH="${REDIS_PASSWORD:-quant2024}"
if redis-cli -a "$REDIS_AUTH" ping 2>/dev/null | grep -q PONG; then
  pass "connection"
else
  fail "connection" "unreachable"
fi

# --- Kafka ---
echo ""
echo "[5/6] Kafka Topics"

KAFKA_BROKER="${KAFKA_BOOTSTRAP:-localhost:9092}"
MIN_TOPICS=30
TOPIC_COUNT=$(kafka-topics --bootstrap-server "$KAFKA_BROKER" --list 2>/dev/null | wc -l)
if [ "$TOPIC_COUNT" -ge "$MIN_TOPICS" ]; then
  pass "topics: $TOPIC_COUNT (min $MIN_TOPICS)"
else
  fail "topics: $TOPIC_COUNT" "expected >= $MIN_TOPICS"
fi

# --- Grafana ---
echo ""
echo "[6/6] Grafana"

if curl -s -o /dev/null -w "%{http_code}" "http://localhost:3000/api/health" | grep -q "200"; then
  pass "grafana endpoint"
else
  echo "  WARN  grafana endpoint unreachable (maybe not started?)"
fi

# --- Summary ---
echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
