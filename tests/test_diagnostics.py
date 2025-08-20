import numpy as np
import pandas as pd

from LGHackerton.utils.diagnostics import ljung_box_test


def test_ljung_box_test_returns_residuals():
    """ljung_box_test returns validated residuals when requested."""
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    lb, res = ljung_box_test(series, lags=[2], return_residuals=True)
    # Ensure returned DataFrame has expected column and residuals are cleaned
    assert "pvalue" in lb.columns
    assert res.equals(series.dropna())
