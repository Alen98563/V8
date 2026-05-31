"""
orchestrator/main.py — CLI entry point
Usage: python -m orchestrator.main --config markets/btc_5m/config.yaml
"""

from __future__ import annotations

import argparse
import sys

from orchestrator.main_loop import run_main_loop


def main():
    parser = argparse.ArgumentParser(description="QTS V8 Alpha Engine")
    parser.add_argument("--config", required=True, help="Market config YAML")
    parser.add_argument("--base-config", default="config/v8.yaml")
    parser.add_argument("--inst-id", required=True, help="OKX instrument ID")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", dest="dry_run", action="store_false")
    parser.add_argument("--pulses", type=int, default=0, help="Bounded pulse count (0=unlimited)")
    parser.add_argument("--tick-hz", type=int, default=1, help="Tick cadence override")

    args = parser.parse_args()
    print(f"[orchestrator] Starting {args.inst_id} (dry_run={args.dry_run})")

    sys.exit(run_main_loop(
        config_path=args.config,
        base_config_path=args.base_config,
        inst_id=args.inst_id,
        dry_run=args.dry_run,
        max_pulses=args.pulses,
        tick_hz=args.tick_hz,
    ))


if __name__ == "__main__":
    main()
