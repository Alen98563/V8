#!/bin/bash
export ALL_PROXY=socks5h://127.0.0.1:1080
set -a
source /home/jerry/strategy.okx/.env
set +a
cd /home/jerry/V8
exec /home/jerry/.local/bin/python3.12 -u v8_ws_bridge.py
