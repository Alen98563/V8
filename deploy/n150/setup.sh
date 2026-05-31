#!/usr/bin/env bash
# setup.sh — N150 初始部署
# Usage: bash deploy/n150/setup.sh [btc_5m|eth_5m|sol_5m|all]
# 1. cd 到 qts_v8 根目录
# 2. 安装 Python 依赖
# 3. 编译 Rust so

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MARKET="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_python() {
    log "Installing Python dependencies..."
    cd "$REPO_DIR"
    if command -v uv &>/dev/null; then
        uv pip install -e ".[n150]" 2>&1 | tail -5
    else
        pip install -e ".[n150]" 2>&1 | tail -5
    fi
    log "✓ Python deps installed"
}

compile_rust() {
    log "Compiling Rust core (release)..."
    cd "$REPO_DIR"
    maturin develop --release 2>&1 | tail -5
    log "✓ libv8_core_engine.so compiled"
}

setup_dirs() {
    for m in $(ls "$REPO_DIR/markets/" | grep -v '^_'); do
        if [ "$MARKET" != "all" ] && [ "$m" != "$MARKET" ]; then continue; fi
        mkdir -p "$REPO_DIR/markets/$m/logs"
        log "  Created markets/$m/logs/"
    done
}

log "=== N150 Setup for market: $MARKET ==="
compile_rust
setup_python
setup_dirs

log "=== Setup complete ==="
log "Next: bash deploy/n150/start.sh $MARKET"