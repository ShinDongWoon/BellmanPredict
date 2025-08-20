import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (  # noqa: E402
    CalendarFeatureMaker,
    DATE_COL,
    SHOP_COL,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-08", "2020-02-05", "2020-02-12"]
    )
    return pd.DataFrame({DATE_COL: dates, SHOP_COL: "A"})


def test_cyclical_reduces_columns_and_variance():
    df = _sample_df()
    base = CalendarFeatureMaker().fit(df).transform(df)
    cyc = CalendarFeatureMaker(cyclical=True).fit(df).transform(df)

    base_cols = [c for c in base.columns if c.startswith("month_") or c.startswith("woy_")]
    cyc_cols = [c for c in cyc.columns if c.endswith("_sin") or c.endswith("_cos")]

    assert len(cyc_cols) < len(base_cols)

    base_var = base[base_cols].var().sum()
    cyc_var = cyc[cyc_cols].var().sum()
    assert cyc_var < base_var


def test_keep_selected_reduces_columns_and_variance():
    df = _sample_df()
    base = CalendarFeatureMaker().fit(df).transform(df)
    kept = CalendarFeatureMaker(keep_months=[1], keep_woys=[1, 2]).fit(df).transform(df)

    base_cols = [c for c in base.columns if c.startswith("month_") or c.startswith("woy_")]
    kept_cols = [c for c in kept.columns if c.startswith("month_") or c.startswith("woy_")]

    assert len(kept_cols) < len(base_cols)

    base_var = base[base_cols].var().sum()
    kept_var = kept[kept_cols].var().sum()
    assert kept_var < base_var
