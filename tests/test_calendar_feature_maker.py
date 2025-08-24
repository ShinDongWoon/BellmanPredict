import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (  # noqa: E402
    CalendarFeatureMaker,
    DATE_COL,
    SHOP_COL,
    Preprocessor,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-08", "2020-02-05", "2020-02-12"]
    )
    return pd.DataFrame({DATE_COL: dates, SHOP_COL: "A"})


def test_cyclical_reduces_columns_and_variance():
    df = _sample_df()
    cyc = CalendarFeatureMaker(dow_mode="cyclical").fit(df).transform(df)
    dum = CalendarFeatureMaker(cyclical=False, dow_mode="cyclical").fit(df).transform(df)

    def _dummy_cols(cols):
        return [
            c
            for c in cols
            if (c.startswith("month_") or c.startswith("woy_"))
            and not (c.endswith("_sin") or c.endswith("_cos"))
        ]

    assert not _dummy_cols(cyc.columns)
    dum_cols = _dummy_cols(dum.columns)
    assert dum_cols

    cyc_cols = [
        c
        for c in cyc.columns
        if (c.endswith("_sin") or c.endswith("_cos")) and not c.startswith("dow_")
    ]
    assert len(cyc_cols) < len(dum_cols)

    cyc_var = cyc[cyc_cols].var().sum()
    dum_var = dum[dum_cols].var().sum()
    assert cyc_var < dum_var


def test_keep_selected_reduces_columns_and_variance():
    df = _sample_df()
    base = CalendarFeatureMaker(cyclical=False, dow_mode="cyclical").fit(df).transform(df)
    kept = CalendarFeatureMaker(cyclical=False, keep_months=[1], keep_woys=[1, 2], dow_mode="cyclical").fit(df).transform(df)

    base_cols = [c for c in base.columns if c.startswith("month_") or c.startswith("woy_")]
    kept_cols = [c for c in kept.columns if c.startswith("month_") or c.startswith("woy_")]

    assert len(kept_cols) < len(base_cols)

    base_var = base[base_cols].var().sum()
    kept_var = kept[kept_cols].var().sum()
    assert kept_var < base_var


@pytest.mark.parametrize("mode", ["cyclical", "integer", "embed"])
def test_dow_modes(mode):
    df = _sample_df()
    out = CalendarFeatureMaker(dow_mode=mode).fit(df).transform(df)
    assert "day" not in out.columns
    if mode == "cyclical":
        assert "dow" in out.columns
        assert {"dow_sin", "dow_cos"}.issubset(out.columns)
    else:
        assert "dow" in out.columns
        assert "dow_sin" not in out.columns and "dow_cos" not in out.columns


def test_dow_cyclical_features_values():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame({DATE_COL: dates, SHOP_COL: "A"})
    out = CalendarFeatureMaker().fit(df).transform(df)
    assert {"dow_sin", "dow_cos"}.issubset(out.columns)
    dow = df[DATE_COL].dt.weekday
    expected_sin = np.sin(2 * np.pi * dow / 7)
    expected_cos = np.cos(2 * np.pi * dow / 7)
    np.testing.assert_allclose(out["dow_sin"].to_numpy(), expected_sin)
    np.testing.assert_allclose(out["dow_cos"].to_numpy(), expected_cos)


def test_holiday_distance_features():
    df = _sample_df()

    class DummyHolidayProvider:
        def compute(self, years):
            return {
                pd.Timestamp("2020-01-01").date(),
                pd.Timestamp("2020-02-05").date(),
            }

    out = (
        CalendarFeatureMaker(holiday_provider=DummyHolidayProvider())
        .fit(df)
        .transform(df)
    )

    assert {"days_since_holiday", "days_to_next_holiday"}.issubset(out.columns)
    assert out["days_since_holiday"].tolist() == [0, 7, 0, 7]
    assert out["days_to_next_holiday"].tolist() == [0, 28, 0, 9999]


def test_preprocessor_cyclical_option_controls_dummies():
    df = _sample_df()
    out_default = Preprocessor().calendar.fit(df).transform(df)
    out_dum = Preprocessor(cyclical=False).calendar.fit(df).transform(df)

    dummy = lambda cols: [
        c
        for c in cols
        if (c.startswith("month_") or c.startswith("woy_"))
        and not (c.endswith("_sin") or c.endswith("_cos"))
    ]

    assert not dummy(out_default.columns)
    assert dummy(out_dum.columns)
