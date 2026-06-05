#!/usr/bin/env python3
"""
prepare_training_data.py — Adapter: 50d micro features → AlphaCast 178d format

1. Reads CFL-labeled parquet (individual f0...f49 columns + label)
2. Packs features into list column, zero-pads to 178d
3. Outputs parquet compatible with train_alphacast.py

Usage:
  python3 prepare_training_data.py --input cfl_btc.parquet --output train_btc.parquet [--input-dim 50]
"""

import argparse, sys
import numpy as np
from pathlib import Path

try:
    import polars as pl
except ImportError:
    print("ERROR: pip install polars")
    sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, nargs='+', help='CFL parquet(s) to merge')
    p.add_argument('--output', required=True)
    p.add_argument('--input-dim', type=int, default=50, help='Actual feature dim (default 50)')
    p.add_argument('--target-dim', type=int, default=178, help='Target dim for AlphaCast (default 178)')
    args = p.parse_args()

    # Load and merge
    dfs = [pl.read_parquet(f) for f in args.input]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    print(f"Loaded {df.height} rows from {len(args.input)} file(s)")

    # Filter neutral labels
    df_f = df.filter(pl.col("label") != 0)
    vc = df_f["label"].value_counts().sort("label")
    for row in vc.rows():
        print(f"  label={row[0]}: {row[1]} ({100*row[1]/df_f.height:.1f}%)")
    print(f"  after filter: {df_f.height} non-neutral samples")

    # Pack f0...f49 into list column, zero-pad to 178d
    feature_cols = [f"f{i}" for i in range(args.input_dim)]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing[:5]}...")
        sys.exit(1)

    # Convert feature columns to rows of list[float]
    pad_size = args.target_dim - args.input_dim
    
    # Build features list column efficiently
    feature_data = df_f.select(feature_cols).to_numpy()
    features_list = []
    for row in feature_data:
        vec = [float(x) for x in row]
        vec.extend([0.0] * pad_size)  # Zero-pad remaining dims
        features_list.append(vec)

    # Build output DF
    out = df_f.select(["ts_ms", "inst_id"]).with_columns([
        pl.Series("cfl_label", [1 if l > 0 else -1 for l in df_f["label"].to_list()], dtype=pl.Int64),
        pl.Series("cfl_weight", [1.0] * df_f.height, dtype=pl.Float64),
        pl.Series("features_178d", features_list, dtype=pl.List(pl.Float64)),
    ])

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(str(out_path))

    pos_count = sum(1 for l in df_f["label"] if l > 0)
    neg_count = sum(1 for l in df_f["label"] if l < 0)
    print(f"\nOutput: {out_path}")
    print(f"  {out.height} rows, features: {args.input_dim}d→{args.target_dim}d (zero-padded)")
    print(f"  +1={pos_count}({100*pos_count/out.height:.1f}%) -1={neg_count}({100*neg_count/out.height:.1f}%)")
    print(f"  Ready for: python3 scripts/train_alphacast.py --data {out_path}")

if __name__ == '__main__':
    main()
