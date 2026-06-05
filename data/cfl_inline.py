#!/usr/bin/env python3
"""Inline CFL labeling for all features_*.parquet"""
import polars as pl, numpy as np
from pathlib import Path

data_dir = Path('/home/jerry/V8/data')
parquets = sorted(data_dir.glob('features_*.parquet'))
print(f"Processing {len(parquets)} feature files for CFL labeling")

total = 0
ldist = {1: 0, 0: 0, -1: 0}

for fp in parquets:
    df = pl.read_parquet(fp)
    inst = fp.stem.replace('features_', '').upper()
    
    if df.height < 10:
        print(f"  SKIP {inst}: only {df.height} bars")
        continue

    closes = df['close'].to_numpy()
    fwd_ret = np.zeros(len(closes))
    fwd_ret[:-1] = (closes[1:] - closes[:-1]) / closes[:-1]

    non_zero = fwd_ret[fwd_ret != 0]
    ret_std = float(np.std(non_zero)) if len(non_zero) > 0 else 0.001
    threshold = ret_std * 1.0

    labels = np.zeros(len(closes), dtype=int)
    labels[fwd_ret > threshold] = 1
    labels[fwd_ret < -threshold] = -1

    df = df.with_columns(pl.Series('label', labels, dtype=pl.Int64))
    out_path = data_dir / f'cfl_{inst.lower()}.parquet'
    df.write_parquet(out_path)

    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == -1))
    neu = int(np.sum(labels == 0))
    ldist[1] += pos; ldist[0] += neu; ldist[-1] += neg
    total += df.height
    print(f"  {inst:8s} {df.height:3d} bars  +1={pos:3d} -1={neg:3d} 0={neu:3d}")

if total > 0:
    print(f"\nTOTAL {total} bars")
    print(f"  +1={ldist[1]} ({100*ldist[1]/total:.1f}%)")
    print(f"  -1={ldist[-1]} ({100*ldist[-1]/total:.1f}%)")
    print(f"   0={ldist[0]} ({100*ldist[0]/total:.1f}%)")

# Merge
cfls = sorted(data_dir.glob('cfl_*.parquet'))
dfs = [pl.read_parquet(f) for f in cfls]
merged = pl.concat(dfs)
out = data_dir / 'cfl_all.parquet'
merged.write_parquet(out)
print(f"Merged: {out} ({merged.height} rows)")
