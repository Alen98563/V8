"""
monitor/dashboard.py — P0: FastAPI 实时面板
=============================================

提供 V8 系统运行状态的 HTTP 可视化面板：

    - GET /               → HTML 仪表盘页面
    - GET /api/status     → 系统健康状态 JSON
    - GET /api/pnl        → P&L 汇总 JSON
    - GET /api/metrics    → Prometheus 格式指标
    - GET /api/performance → 性能统计 JSON
    - GET /api/alerts     → 最近告警列表
    - WS  /ws             → 实时推送 (每秒)

启动：
    uvicorn monitor.dashboard:app --host 0.0.0.0 --port 8080
    或
    python -m monitor.dashboard --port 8080
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.logging_setup import get_logger

_log = get_logger("monitor.dashboard")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    FastAPI = None  # type: ignore
    _log.error("FastAPI required; pip install fastapi uvicorn")

# ── 全局单例 (由 Orchestrator 注入) ──────────────────────────

_health_monitor = None   # SystemHealthMonitor 实例
_perf_tracker = None     # PerformanceTracker 实例
_pnl_aggregator = None   # PnLAggregator 实例
_orchestrator_state = None  # OrchestratorState 实例


def register_components(
    health_monitor=None,
    perf_tracker=None,
    pnl_aggregator=None,
    orchestrator_state=None,
):
    """由 Orchestrator 启动时注入组件引用"""
    global _health_monitor, _perf_tracker, _pnl_aggregator, _orchestrator_state
    _health_monitor = health_monitor
    _perf_tracker = perf_tracker
    _pnl_aggregator = pnl_aggregator
    _orchestrator_state = orchestrator_state


# ── FastAPI App ──────────────────────────────────────────────

if FastAPI is not None:
    app = FastAPI(
        title="QTS V8 Dashboard",
        description="Real-time monitoring dashboard for V8 quantitative trading system",
        version="0.1.0",
    )
else:
    app = None


# ── HTML Dashboard ───────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QTS V8 Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0f; color: #e0e0e0; }
.header { background: #111118; padding: 16px 24px; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 18px; color: #7c3aed; }
.header .status { font-size: 13px; }
.status-ok { color: #22c55e; }
.status-warn { color: #eab308; }
.status-error { color: #ef4444; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; padding: 24px; }
.card { background: #111118; border: 1px solid #222; border-radius: 8px; padding: 16px; }
.card h2 { font-size: 13px; color: #888; text-transform: uppercase; margin-bottom: 12px; }
.metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a24; }
.metric-label { color: #888; font-size: 13px; }
.metric-value { font-size: 14px; font-weight: 600; }
.pos { color: #22c55e; }
.neg { color: #ef4444; }
.neutral { color: #888; }
.alert-list { max-height: 200px; overflow-y: auto; }
.alert-item { padding: 6px 8px; margin: 4px 0; border-radius: 4px; font-size: 12px; background: #1a1a24; }
.ws-status { font-size: 12px; color: #666; }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ QTS V8</h1>
  <div>
    <span id="overall-status" class="status status-ok">●</span>
    <span id="ws-indicator" class="ws-status">WS: connecting</span>
    <span id="last-update" class="ws-status"></span>
  </div>
</div>
<div class="grid">
  <div class="card" id="card-system">
    <h2>System</h2>
    <div id="system-metrics"></div>
  </div>
  <div class="card" id="card-pnl">
    <h2>P&L</h2>
    <div id="pnl-metrics"></div>
  </div>
  <div class="card" id="card-perf">
    <h2>Performance</h2>
    <div id="perf-metrics"></div>
  </div>
  <div class="card" id="card-alerts">
    <h2>Alerts</h2>
    <div id="alert-list" class="alert-list"></div>
  </div>
</div>
<script>
const wsUrl = `ws://${location.host}/ws`;
let ws;
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { document.getElementById('ws-indicator').textContent = 'WS: connected'; };
  ws.onclose = () => { document.getElementById('ws-indicator').textContent = 'WS: disconnected'; setTimeout(connect, 3000); };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    updateAll(d);
  };
}
function mk(label, val, cls='') { return `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value ${cls}">${val}</span></div>`; }
function updateAll(d) {
  // Status
  const s = d.overall || 'unknown';
  const el = document.getElementById('overall-status');
  el.className = 'status status-' + s; el.textContent = '● ' + s.toUpperCase();
  // System
  let sys = '';
  if (d.system) { const c = d.system; sys += mk('Backend', c.backend||'-'); sys += mk('Dry Run', c.dry_run?'Yes':'No'); sys += mk('Running', c.running?'✓':'✗'); sys += mk('Pulses', c.pulses||0); sys += mk('Ticks', c.ticks||0); }
  document.getElementById('system-metrics').innerHTML = sys;
  // PnL
  let pnl = '';
  if (d.pnl) { const p = d.pnl; pnl += mk('Realized', (p.realized_pnl||0).toFixed(2)+' USDT', (p.realized_pnl||0)>=0?'pos':'neg'); pnl += mk('Unrealized', (p.unrealized_pnl||0).toFixed(2)+' USDT', (p.unrealized_pnl||0)>=0?'pos':'neg'); pnl += mk('Position', (p.position||0).toFixed(4)+' ETH'); pnl += mk('Fills', p.fills||0); pnl += mk('Sharpe', (p.sharpe||0).toFixed(2)); }
  document.getElementById('pnl-metrics').innerHTML = pnl;
  // Perf
  let perf = '';
  if (d.performance && d.performance._throughput) { const t = d.performance._throughput; perf += mk('Ticks/s', t.ticks_per_sec); perf += mk('Pulses/min', t.pulses_per_min); perf += mk('Uptime', t.uptime_sec+'s'); }
  document.getElementById('perf-metrics').innerHTML = perf;
  // Alerts
  let al = '';
  if (d.alerts) { d.alerts.slice(-10).reverse().forEach(a => { al += `<div class="alert-item">[${a.component}] ${a.message}</div>`; }); }
  document.getElementById('alert-list').innerHTML = al || '<div class="alert-item neutral">No alerts</div>';
}
connect();
// Fallback: poll REST API every 5s if WS fails
setInterval(async () => { if (!ws || ws.readyState !== 1) { try { const r = await fetch('/api/status'); const d = await r.json(); updateAll(d); } catch(e){} } }, 5000);
</script>
</body>
</html>"""


if app is not None:

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/api/status")
    async def api_status():
        data = _build_snapshot()
        return JSONResponse(data)

    @app.get("/api/pnl")
    async def api_pnl():
        pnl = _get_pnl()
        return JSONResponse(pnl)

    @app.get("/api/metrics")
    async def api_metrics():
        """Prometheus-compatible text format"""
        data = _build_snapshot()
        lines = []
        if data.get("system"):
            s = data["system"]
            lines.append(f'# HELP v8_pulses_total Total pulses')
            lines.append(f'# TYPE v8_pulses_total counter')
            lines.append(f'v8_pulses_total {s.get("pulses", 0)}')
            lines.append(f'# HELP v8_ticks_total Total ticks')
            lines.append(f'# TYPE v8_ticks_total counter')
            lines.append(f'v8_ticks_total {s.get("ticks", 0)}')
        if data.get("pnl"):
            p = data["pnl"]
            lines.append(f'# HELP v8_realized_pnl Realized PnL (USDT)')
            lines.append(f'# TYPE v8_realized_pnl gauge')
            lines.append(f'v8_realized_pnl {p.get("realized_pnl", 0)}')
            lines.append(f'# HELP v8_position Position (ETH)')
            lines.append(f'# TYPE v8_position gauge')
            lines.append(f'v8_position {p.get("position", 0)}')
        return "\n".join(lines) + "\n"

    @app.get("/api/performance")
    async def api_performance():
        if _perf_tracker:
            return JSONResponse(_perf_tracker.get_all_stats())
        return JSONResponse({"error": "performance tracker not registered"})

    @app.get("/api/alerts")
    async def api_alerts():
        if _health_monitor:
            return JSONResponse(_health_monitor.get_alerts())
        return JSONResponse([])

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                snapshot = _build_snapshot()
                await websocket.send_json(snapshot)
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass


def _build_snapshot() -> Dict[str, Any]:
    """构建当前系统状态快照"""
    snap: Dict[str, Any] = {"ts_ms": int(time.time() * 1000)}

    # Orchestrator state
    if _orchestrator_state:
        try:
            snap["system"] = _orchestrator_state.as_dict() if hasattr(_orchestrator_state, 'as_dict') else {}
        except Exception:
            snap["system"] = {}
    else:
        snap["system"] = {"backend": "unknown", "running": False}

    # P&L
    snap["pnl"] = _get_pnl()

    # Health
    if _health_monitor:
        snap["health"] = _health_monitor.get_status()
        snap["overall"] = snap["health"].get("overall", "unknown")
        snap["alerts"] = _health_monitor.get_alerts(limit=20)
    else:
        snap["overall"] = "unknown"
        snap["alerts"] = []

    # Performance
    if _perf_tracker:
        snap["performance"] = _perf_tracker.get_all_stats()

    return snap


def _get_pnl() -> Dict[str, Any]:
    if _pnl_aggregator:
        try:
            s = _pnl_aggregator.snapshot()
            return {
                "realized_pnl": s.realized_pnl,
                "unrealized_pnl": getattr(s, 'unrealized_pnl', 0.0),
                "position": s.position,
                "fills": s.fill_count,
                "sharpe": s.sharpe,
            }
        except Exception:
            pass
    if _orchestrator_state and hasattr(_orchestrator_state, 'as_dict'):
        d = _orchestrator_state.as_dict()
        return {k: d.get(k, 0) for k in ("realized_pnl", "unrealized_pnl", "position", "fills", "sharpe")}
    return {}


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V8 Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if app is None:
        print("ERROR: FastAPI not installed. pip install fastapi uvicorn")
        sys.exit(1)

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("ERROR: uvicorn not installed. pip install uvicorn")
        sys.exit(1)
