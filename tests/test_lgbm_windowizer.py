import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (  # noqa: E402
    SampleWindowizer,
    SERIES_COL,
    DATE_COL,
    SALES_COL,
)


def _build_lgbm_train_loop(df: pd.DataFrame, feature_cols, horizon: int) -> pd.DataFrame:
    """Reference loop-based implementation from v1.0 for comparison."""
    d = df.sort_values([SERIES_COL, DATE_COL]).copy()
    if "lag_27" not in d.columns:
        raise ValueError
    rows = []
    for sid, g in d.groupby(SERIES_COL, sort=False):
        g = g.reset_index(drop=True)
        idxs = np.where(g["lag_27"].notna().values)[0]
        for t in idxs:
            for h in range(1, horizon + 1):
                if t + h >= len(g):
                    break
                y = g.loc[t + h, SALES_COL]
                if pd.isna(y):
                    continue
                row = {
                    SERIES_COL: sid,
                    DATE_COL: g.loc[t, DATE_COL],
                    "h": h,
                    "y": float(y),
                }
                row.update(g.loc[t, feature_cols].to_dict())
                rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def test_vectorized_matches_loop():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    records = []
    for sid, offset in [("A", 0), ("B", 100)]:
        for i, d in enumerate(dates):
            records.append(
                {
                    SERIES_COL: sid,
                    DATE_COL: d,
                    SALES_COL: float(i + offset),
                    "lag_27": float(i),
                }
            )
    df = pd.DataFrame(records)

    feature_cols = ["lag_27"]
    win = SampleWindowizer(lookback=28, horizon=3)
    vec = win.build_lgbm_train(df, feature_cols)

    loop = _build_lgbm_train_loop(df, feature_cols, horizon=3)
    loop = loop.astype({"lag_27": "float32"})
    loop = loop[[SERIES_COL, DATE_COL, *feature_cols, "h", "y"]]

    vec = vec.sort_values([SERIES_COL, DATE_COL, "h"]).reset_index(drop=True)
    loop = loop.sort_values([SERIES_COL, DATE_COL, "h"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(vec, loop)

    # Column presence and dtypes
    assert vec.columns.tolist() == [SERIES_COL, DATE_COL, *feature_cols, "h", "y"]
    assert vec.dtypes["lag_27"] == np.float32

