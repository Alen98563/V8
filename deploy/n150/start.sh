#!/usr/bin/env bash
# start.sh — 启动单个市场 Alpha 引擎
# Usage: bash deploy/n150/start.sh btc_5m [--dry-run]

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MARKET="${1:-}"
DRY_RUN="${2:---dry-run}"

if [ -z "$MARKET" ]; then
    echo "Usage: bash deploy/n150/start.sh <btc_5m|eth_5m|sol_5m> [--dry-run|--live]"
    exit 1
fi

MARKET_DIR="$REPO_DIR/markets/$MARKET"
MARKET_CONFIG="$MARKET_DIR/config.yaml"

if [ ! -f "$MARKET_CONFIG" ]; then
    echo "✗ Config not found: $MARKET_CONFIG"
    echo "  Run: bash deploy/n150/setup.sh $MARKET"
    exit 1
fi

# Extract inst_id from config (reads the YAML directly)
INST_ID=$(grep '^inst_id:' "$MARKET_CONFIG" | awk '{print $2}')
PIDFILE="$MARKET_DIR/.pid"
LOGFILE="$MARKET_DIR/logs/engine_$(date +%Y%m%d).log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Check if already running ──
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log "✗ $MARKET already running (PID $OLD_PID)"
        exit 1
    else
        log "Stale PID file removed (PID $OLD_PID is dead)"
        rm -f "$PIDFILE"
    fi
fi

# ── Verify SHM ──
SHM_PATH="/dev/shm/qts_$(echo "$INST_ID" | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')5m"
log "SHM path: $SHM_PATH"

# ── Launch ──
mkdir -p "$MARKET_DIR/logs"
cd "$REPO_DIR"

log "Starting $MARKET (inst_id=$INST_ID $DRY_RUN)..."
nohup python3 -m orchestrator.main \
    --config "$MARKET_CONFIG" \
    --base-config config/v8.yaml \
    --inst-id "$INST_ID" \
    $DRY_RUN \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
log "✓ $MARKET started (PID $PID)"
log "  Log: $LOGFILE"
log "  Stop: bash deploy/n150/stop.sh $MARKET"