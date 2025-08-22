import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (
    SampleWindowizer,
    SERIES_COL,
    DATE_COL,
    SALES_COL,
    SALES_FILLED_COL,
)


def test_patch_windowizer_dynamic_static():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    records = []
    for i, d in enumerate(dates):
        records.append(
            {
                SERIES_COL: "A",
                DATE_COL: d,
                SALES_COL: float(i + 1),
                SALES_FILLED_COL: float(i + 1),
                "feat_dyn": float(i),
                "feat_static": 42.0,
            }
        )
    df = pd.DataFrame(records)

    feature_cols = ["feat_dyn", "feat_static"]
    static_cols = ["feat_static"]

    win = SampleWindowizer(lookback=3, horizon=2)
    X, Y, sids, dates, dyn_idx, stat_idx = win.build_patch_train(
        df, feature_cols, static_cols
    )

    assert X.shape == (1, 3, 3)
    expected_dyn = df.loc[0:2, [SALES_FILLED_COL, "feat_dyn"]].values
    expected_stat = np.repeat([[42.0]], 3, axis=0)
    expected = np.concatenate([expected_dyn, expected_stat], axis=1)
    np.testing.assert_allclose(X[0], expected)
    np.testing.assert_allclose(Y[0], df.loc[3:4, SALES_COL].values)
    assert win.dynamic_idx[SALES_FILLED_COL] == 0
    assert win.dynamic_idx["feat_dyn"] == 1
    assert win.static_idx["feat_static"] == 2

    X_eval, sids_eval, dates_eval, dyn_idx_e, stat_idx_e = win.build_patch_eval(
        df, feature_cols, static_cols
    )
    assert dyn_idx == dyn_idx_e
    assert stat_idx == stat_idx_e
    assert X_eval.shape == (1, 3, 3)
    expected_eval_dyn = df.loc[2:4, [SALES_FILLED_COL, "feat_dyn"]].values
    expected_eval = np.concatenate([expected_eval_dyn, expected_stat], axis=1)
    np.testing.assert_allclose(X_eval[0], expected_eval)
    assert sids_eval[0] == "A"
    assert dates_eval.shape == (1,)


def test_patch_windowizer_static_multi_series():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    records = []
    for sid, s_val in [("A", 1.0), ("B", 2.0)]:
        for i, d in enumerate(dates):
            records.append(
                {
                    SERIES_COL: sid,
                    DATE_COL: d,
                    SALES_COL: float(i + 1),
                    SALES_FILLED_COL: float(i + 1),
                    "feat_dyn": float(i),
                    "feat_static": s_val,
                }
            )
    df = pd.DataFrame(records)
    feature_cols = ["feat_dyn", "feat_static"]
    static_cols = ["feat_static"]

    win = SampleWindowizer(lookback=2, horizon=1)
    X, Y, sids, dates, dyn_idx, stat_idx = win.build_patch_train(
        df, feature_cols, static_cols
    )
    static_idx = stat_idx["feat_static"]
    for x_window, sid in zip(X, sids):
        expected = 1.0 if sid == "A" else 2.0
        np.testing.assert_allclose(x_window[:, static_idx], expected)
