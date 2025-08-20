from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mean = float(series.mean())
            std = float(series.std())
            zero_ratio = float((series == 0).mean())
        else:
            mean = float('nan')
            std = float('nan')
            zero_ratio = float('nan')
        n_unique = int(series.nunique(dropna=True))
        rows.append({
            'column': col,
            'dtype': str(series.dtype),
            'mean': mean,
            'std': std,
            'zero_ratio': zero_ratio,
            'n_unique': n_unique,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Diagnose feature distributions of preprocessed dataset')
    parser.add_argument('--input', required=True, help='Path to preprocessed dataset (CSV or Parquet)')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save diagnostics (defaults to input directory)')
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == '.csv':
        df = pd.read_csv(in_path)
    else:
        df = pd.read_parquet(in_path)

    stats_df = compute_diagnostics(df)
    csv_path = out_dir / f'{in_path.stem}_diagnostics.csv'
    json_path = out_dir / f'{in_path.stem}_diagnostics.json'
    stats_df.to_csv(csv_path, index=False)
    stats_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(stats_df)


if __name__ == '__main__':
    main()
