#!/usr/bin/env python3
"""
sqlite_price_recorder.py — SQLite-based tick recorder for V8 (v2: 20-field schema)

Writes 1s tick snapshots to per-day SQLite databases.
Schema mirrors full Redis v8:snapshot fields.

Usage:
    python3 sqlite_price_recorder.py --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP --interval 1

Storage estimate: ~15 MB/day for BTC+ETH combined (20 fields vs old 8).
"""

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

# — config —
REDIS_URL = os.environ.get("V8_REDIS_URL", "redis://localhost:6379/0")
DATA_DIR = Path(os.environ.get("V8_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))) / "ticks_db"

# 20-field tick schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    ts              INTEGER NOT NULL,   -- unix ms
    inst_id         TEXT    NOT NULL,
    last            REAL    NOT NULL,
    last_sz         REAL    NOT NULL,
    bid             REAL    NOT NULL,
    bid_sz          REAL    NOT NULL,
    ask             REAL    NOT NULL,
    ask_sz          REAL    NOT NULL,
    mid_px          REAL    NOT NULL,
    spread          REAL    NOT NULL,
    prev_last_px    REAL    NOT NULL,
    funding_rate    REAL    NOT NULL DEFAULT 0,
    next_funding_ts INTEGER NOT NULL DEFAULT 0,
    open_interest   REAL    NOT NULL DEFAULT 0,
    oi_ts           INTEGER NOT NULL DEFAULT 0,
    open_24h        REAL    NOT NULL DEFAULT 0,
    high_24h        REAL    NOT NULL DEFAULT 0,
    low_24h         REAL    NOT NULL DEFAULT 0,
    vol_24h         REAL    NOT NULL DEFAULT 0,
    vol_ccy_24h     REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (ts)
);
CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks (ts);
"""

# Redis → SQLite column mapping
COL_MAP = [
    ("inst_id",         "inst_id"),
    ("last_px",         "last"),
    ("last_sz",         "last_sz"),
    ("bid1",            "bid"),
    ("bid1_sz",         "bid_sz"),
    ("ask1",            "ask"),
    ("ask1_sz",         "ask_sz"),
    ("mid_px",          "mid_px"),
    ("spread",          "spread"),
    ("prev_last_px",    "prev_last_px"),
    ("funding_rate",    "funding_rate"),
    ("next_funding_ts", "next_funding_ts"),
    ("open_interest",   "open_interest"),
    ("oi_ts",           "oi_ts"),
    ("open_24h",        "open_24h"),
    ("high_24h",        "high_24h"),
    ("low_24h",         "low_24h"),
    ("vol_24h",         "vol_24h"),
    ("vol_ccy_24h",     "vol_ccy_24h"),
]

INSERT_COLS = ["ts"] + [c[1] for c in COL_MAP]
INSERT_SQL = f"INSERT OR IGNORE INTO ticks ({', '.join(INSERT_COLS)}) VALUES ({', '.join('?' * len(INSERT_COLS))})"


class SqliteRecorder:
    def __init__(self, inst_id: str, interval: float = 1.0):
        self.inst_id = inst_id
        self.interval = interval
        self._running = False
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None
        self._day: str = ""
        self._count = 0

    def _rotate(self) -> None:
        """Open (or create) today's DB and ensure schema."""
        today = time.strftime("%Y%m%d", time.localtime())
        if today == self._day and self._conn is not None:
            return

        if self._conn:
            self._conn.commit()
            self._conn.close()
            print(f"[db:{self.inst_id}] rotated {self._day} ({self._count} ticks)")

        self._day = today
        inst_dir = DATA_DIR / self.inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = inst_dir / f"{self.inst_id}_{today}.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        self._count = 0
        print(f"[db:{self.inst_id}] opened {self._db_path}")

    def start(self) -> None:
        from adapters.redis_feed import RedisFeedClient

        self._rotate()
        feed = RedisFeedClient(self.inst_id, redis_url=REDIS_URL)
        feed.start()
        self._running = True

        print(f"[db:{self.inst_id}] started, interval={self.interval}s")

        while self._running:
            try:
                snap = feed.latest_snapshot()
                if snap and snap != "{}":
                    self._write_tick(snap)
                elif self._count == 0:
                    print(f"[db:{self.inst_id}] waiting for first snapshot...", flush=True)
            except Exception as exc:
                print(f"[db:{self.inst_id}] error: {exc}", flush=True)

            time.sleep(self.interval)

        feed.stop()
        if self._conn:
            self._conn.commit()
            self._conn.close()
        print(f"[db:{self.inst_id}] stopped ({self._count} ticks total)")

    def _write_tick(self, snap: str) -> None:
        d = json.loads(snap)
        self._rotate()

        assert self._conn is not None

        values = [d.get("ts_ms", 0)]
        for redis_key, _ in COL_MAP:
            values.append(d.get(redis_key, 0))

        self._conn.execute(INSERT_SQL, values)
        self._count += 1

        if self._count % 60 == 0:
            self._conn.commit()


def run_multi(inst_ids: list[str], interval: float) -> None:
    import threading

    recorders = []
    threads = []
    for iid in inst_ids:
        rec = SqliteRecorder(iid, interval)
        recorders.append(rec)
        t = threading.Thread(target=rec.start, daemon=True, name=f"db-{iid}")
        threads.append(t)
        t.start()

    def _sig(signum, frame):
        print("\n[db] shutting down ...", flush=True)
        for rec in recorders:
            rec._running = False
        for t in threads:
            t.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        _sig(signal.SIGINT, None)


def main() -> None:
    p = argparse.ArgumentParser(description="SQLite tick recorder for V8")
    p.add_argument("--inst", action="append", dest="inst", required=True,
                   help="Instrument ID (repeatable)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Polling interval in seconds")
    args = p.parse_args()

    inst_ids = args.inst
    if len(inst_ids) == 1:
        rec = SqliteRecorder(inst_ids[0], args.interval)
        signal.signal(signal.SIGINT, lambda s, f: setattr(rec, '_running', False) or sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(rec, '_running', False) or sys.exit(0))
        rec.start()
    else:
        run_multi(inst_ids, args.interval)


if __name__ == "__main__":
    main()