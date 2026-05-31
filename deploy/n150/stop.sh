#!/usr/bin/env bash
# stop.sh — 优雅停止单个市场引擎
# Usage: bash deploy/n150/stop.sh btc_5m

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MARKET="${1:-}"

if [ -z "$MARKET" ]; then
    echo "Usage: bash deploy/n150/stop.sh <btc_5m|eth_5m|sol_5m>"
    echo "  stop all: for m in btc_5m eth_5m sol_5m; do bash deploy/n150/stop.sh \$m; done"
    exit 1
fi

PIDFILE="$REPO_DIR/markets/$MARKET/.pid"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$PIDFILE" ]; then
    log "✗ $MARKET not running (no PID file)"
    exit 1
fi

PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then
    log "$MARKET already dead (PID $PID)"
    rm -f "$PIDFILE"
    exit 0
fi

# SIGTERM → grace period → SIGKILL
log "Sending SIGTERM to $MARKET (PID $PID)..."
kill "$PID"

# Wait up to 10s for graceful shutdown
for i in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then
        log "✓ $MARKET stopped gracefully"
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 0.5
done

log "$MARKET not responding, sending SIGKILL..."
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
log "✓ $MARKET force-killed"