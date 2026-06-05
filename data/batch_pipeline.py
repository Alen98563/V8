#!/usr/bin/env python3
"""
batch_pipeline.py — Multi-instrument Feature Engine + CFL Labeling Pipeline
Usage: python3 batch_pipeline.py --data-dir /home/jerry/V8/data
Processes ALL instruments in ticks_db/ directory.
"""

import argparse, os, sys, subprocess
from pathlib import Path

def run(cmd, desc):
    print(f"\n[{desc}]")
    print(f"  $ {cmd[:120]}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[:300]}")
    else:
        for line in r.stdout.strip().split('\n')[-5:]:
            print(f"  {line}")
    return r.returncode == 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='/home/jerry/V8/data')
    p.add_argument('--fe-script', default='/home/jerry/V8/features/offline_feature_engine.py')
    p.add_argument('--cfl-script', default='/home/jerry/V8/labeling/batch_cfl.py')
    p.add_argument('--skip-fe', action='store_true')
    p.add_argument('--skip-cfl', action='store_true')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    ticks_db = data_dir / 'ticks_db'
    
    # Discover instruments
    instruments = sorted([
        d.name for d in ticks_db.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])
    print(f"Found {len(instruments)} instruments: {instruments}")

    # Step 1: Feature Engine
    if not args.skip_fe:
        for inst in instruments:
            sym = inst.split('-')[0].lower()
            out = data_dir / f'features_{sym}.parquet'
            ok = run(
                f'cd /home/jerry/V8 && PYTHONPATH=. python3 {args.fe_script} '
                f'--inst {inst} --db-dir {ticks_db} --output {out}',
                f'FE {inst}'
            )
    
    # Step 2: CFL Labeling (in-process, simpler)
    parquets = sorted(data_dir.glob('features_*.parquet'))
    print(f"\nFound {len(parquets)} feature parquets for CFL labeling")
    
    # Step 3: Merge all CFL outputs
    import pandas as pd
    cfl_files = sorted(data_dir.glob('cfl_*.parquet'))
    if cfl_files:
        dfs = []
        for f in cfl_files:
            df = pd.read_parquet(f)
            dfs.append(df)
        merged = pd.concat(dfs, ignore_index=True)
        merged_path = data_dir / 'cfl_all.parquet'
        merged.to_parquet(merged_path, index=False)
        print(f"\nMerged: {merged_path} ({len(merged)} rows, {len(merged.columns)} cols)")
        
        # Report label distribution
        if 'label' in merged.columns:
            vc = merged['label'].value_counts().sort_index()
            total = len(merged)
            print(f"  Labels: +1={vc.get(1,0)} ({100*vc.get(1,0)/total:.1f}%), "
                  f"0={vc.get(0,0)} ({100*vc.get(0,0)/total:.1f}%), "
                  f"-1={vc.get(-1,0)} ({100*vc.get(-1,0)/total:.1f}%)")

if __name__ == '__main__':
    main()
