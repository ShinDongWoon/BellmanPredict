from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from LGHackerton.preprocess import Preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline and save LGBM training data")
    parser.add_argument(
        "--input",
        type=str,
        default="LGHackerton/data/train.csv",
        help="Path to raw training CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="LGHackerton/artifacts",
        help="Directory to save the preprocessed dataset",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_path)
    pp = Preprocessor()
    df_full = pp.fit_transform_train(df_raw)
    train_df = pp.build_lgbm_train(df_full)

    out_file = out_dir / "lgbm_train.csv"
    train_df.to_csv(out_file, index=False)
    print(f"Saved LGBM train dataset to {out_file}")


if __name__ == "__main__":
    main()
