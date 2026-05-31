#!/bin/bash
export ALL_PROXY=socks5h://127.0.0.1:1080
set -a
source /home/jerry/strategy.okx/.env
set +a
cd /home/jerry/V8
mkdir -p logs

# Multi-market config
export V8_INST_IDS="ETH-USDT-SWAP,BTC-USDT-SWAP"
export V8_BAR_SECS=300
export V8_MAX_BARS=0
export V8_LOG_INTERVAL=60

exec /home/jerry/.local/bin/python3.12 -u v8_ws_bridge.py > logs/bridge.log 2>&1
