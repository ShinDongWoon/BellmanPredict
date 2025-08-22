import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure module import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (
    Preprocessor,
    SERIES_COL,
    DATE_COL,
    SALES_COL,
    SALES_FILLED_COL,
)


def test_patchtst_dynamic_channels_drop_lag_roll():
    pp = Preprocessor()
    pp.guard.set_scope("train")

    # Feature columns including lag/roll statistics
    pp.feature_cols = [
        "lag_1",
        "lag_27",
        "roll_mean_7",
        "roll_std_28",
        "dow",
        "shop_code",
    ]
    pp.static_feature_cols = ["shop_code"]
    pp.dynamic_feature_cols = [c for c in pp.feature_cols if c not in pp.static_feature_cols]

    # Compute PatchTST feature lists
    pp._compute_patch_features()

    # Ensure lag/roll columns are removed
    assert all(
        not c.startswith("lag_") and not c.startswith("roll_")
        for c in pp.patch_feature_cols
    )

    # Build minimal training data containing lag/roll columns
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            SERIES_COL: ["S1"] * len(dates),
            DATE_COL: dates,
            SALES_COL: np.arange(len(dates), dtype=float),
        }
    )
    df[SALES_FILLED_COL] = df[SALES_COL]
    df["lag_1"] = df[SALES_FILLED_COL].shift(1)
    df["lag_27"] = df[SALES_FILLED_COL].shift(27)
    df["roll_mean_7"] = df[SALES_FILLED_COL].shift(1).rolling(7, min_periods=1).mean()
    df["roll_std_28"] = df[SALES_FILLED_COL].shift(1).rolling(28, min_periods=1).std()
    df["dow"] = df[DATE_COL].dt.dayofweek
    df["shop_code"] = 1

    pp.build_patch_train(df)

    # Dynamic channel names should omit lag/roll statistics
    assert all(
        not k.startswith("lag_") and not k.startswith("roll_")
        for k in pp.patch_dynamic_idx
    )


def test_patchtst_intermittency_features():
    pp = Preprocessor()
    pp.guard.set_scope("train")

    # Feature columns including intermittency indicators
    pp.feature_cols = [
        "zero_ratio_28",
        "days_since_last_sale",
        "zero_run_len",
        "dow",
    ]
    pp.static_feature_cols = []
    pp.dynamic_feature_cols = [c for c in pp.feature_cols]

    # Compute PatchTST feature lists
    pp._compute_patch_features()

    # Only zero_run_len should remain among intermittency features
    assert "zero_run_len" in pp.patch_feature_cols
    assert "zero_ratio_28" not in pp.patch_feature_cols
    assert "days_since_last_sale" not in pp.patch_feature_cols
