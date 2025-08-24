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
                "feat_static": 1,
            }
        )
    df = pd.DataFrame(records)

    feature_cols = ["feat_dyn", "feat_static"]
    static_cols = ["feat_static"]

    win = SampleWindowizer(lookback=3, horizon=2)
    X_dyn, S, M, Y, sids, dates, dyn_idx, stat_idx = win.build_patch_train(
        df, feature_cols, static_cols
    )

    assert X_dyn.shape == (1, 3, 2)
    expected_dyn = df.loc[0:2, [SALES_FILLED_COL, "feat_dyn"]].values
    np.testing.assert_allclose(X_dyn[0], expected_dyn)
    np.testing.assert_array_equal(S[0], np.array([2], dtype=np.int64))
    np.testing.assert_allclose(Y[0], df.loc[3:4, SALES_COL].values)
    assert win.dynamic_idx[SALES_FILLED_COL] == 0
    assert win.dynamic_idx["feat_dyn"] == 1
    assert win.static_idx["feat_static"] == 0

    X_eval_dyn, S_eval, M_eval, sids_eval, dates_eval, dyn_idx_e, stat_idx_e = win.build_patch_eval(
        df, feature_cols, static_cols
    )
    assert dyn_idx == dyn_idx_e
    assert stat_idx == stat_idx_e
    assert X_eval_dyn.shape == (1, 3, 2)
    expected_eval_dyn = df.loc[2:4, [SALES_FILLED_COL, "feat_dyn"]].values
    np.testing.assert_allclose(X_eval_dyn[0], expected_eval_dyn)
    np.testing.assert_array_equal(S_eval[0], np.array([2], dtype=np.int64))
    assert sids_eval[0] == "A"
    assert dates_eval.shape == (1,)


def test_patch_windowizer_static_multi_series():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    records = []
    for sid, s_val in [("A", 0), ("B", 1)]:
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
    X_dyn, S, M, Y, sids, dates, dyn_idx, stat_idx = win.build_patch_train(
        df, feature_cols, static_cols
    )
    static_idx = stat_idx["feat_static"]
    for stat_vals, sid in zip(S, sids):
        expected = 1 if sid == "A" else 2
        np.testing.assert_array_equal(stat_vals[static_idx], expected)


def test_static_codes_non_negative_with_unknown():
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
                "feat_static": 0 if i < 3 else -1,
            }
        )
    df = pd.DataFrame(records)
    feature_cols = ["feat_dyn", "feat_static"]
    static_cols = ["feat_static"]
    win = SampleWindowizer(lookback=3, horizon=1)
    _, S, _, _, _, _, _, _ = win.build_patch_train(df, feature_cols, static_cols)
    assert S.shape == (2, 1)
    assert np.array_equal(S[:, 0], np.array([1, 0], dtype=np.int64))
    assert S.min() >= 0
