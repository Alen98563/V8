#!/usr/bin/env bash
# deploy/n150/start_cross_section.sh
# 启动独立 CrossSectionEngine 进程
# Usage: bash deploy/n150/start_cross_section.sh [--tick-hz 10]

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TICK_HZ="${1:-10}"
PIDFILE="$REPO_DIR/.cross_section.pid"
LOGFILE="$REPO_DIR/logs/cross_section_$(date +%Y%m%d).log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Check if already running ──
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log "✗ CrossSectionEngine already running (PID $OLD_PID)"
        exit 1
    else
        log "Stale PID file removed (PID $OLD_PID is dead)"
        rm -f "$PIDFILE"
    fi
fi

# ── Which markets to cross-section ──
CROSS_INST_IDS="${V8_CROSS_INST_IDS:-BTC-USDT-SWAP,ETH-USDT-SWAP}"

# ── Launch ──
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

log "Starting CrossSectionEngine (inst_ids=$CROSS_INST_IDS tick_hz=$TICK_HZ)..."
nohup /home/jerry/V8/.venv/bin/python3 -m alpha.crypto.cross_engine \
    --inst-ids "$CROSS_INST_IDS" \
    --tick-hz "$TICK_HZ" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
log "✓ CrossSectionEngine started (PID $PID)"
log "  Log: $LOGFILE"
log "  Stop: kill $PID"