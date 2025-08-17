from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _validate_series(residuals: pd.Series, min_len: int) -> pd.Series:
    """Validate residual series for diagnostics functions."""
    if not isinstance(residuals, pd.Series):
        raise TypeError("residuals must be a pandas Series")
    res = residuals.dropna()
    if res.empty:
        raise ValueError("residual series is empty")
    if len(res) <= min_len:
        raise ValueError(f"residual series length must exceed {min_len}")
    return res


def compute_acf(residuals: pd.Series, nlags: int = 40) -> pd.DataFrame:
    """Autocorrelation function with Q-statistics and p-values."""
    res = _validate_series(residuals, nlags)
    try:
        from statsmodels.tsa.stattools import acf
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError("statsmodels is required for compute_acf") from e
    acf_vals, confint, qstat, pvals = acf(res, nlags=nlags, qstat=True, alpha=0.05)
    lags = np.arange(len(acf_vals))
    qstat = np.insert(qstat, 0, np.nan)
    pvals = np.insert(pvals, 0, np.nan)
    return pd.DataFrame(
        {
            "lag": lags,
            "acf": acf_vals,
            "qstat": qstat,
            "pvalue": pvals,
            "confint_lower": confint[:, 0],
            "confint_upper": confint[:, 1],
        }
    )


def compute_pacf(residuals: pd.Series, nlags: int = 40) -> pd.DataFrame:
    """Partial autocorrelation function with approximate p-values."""
    res = _validate_series(residuals, nlags)
    try:
        from statsmodels.tsa.stattools import pacf
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError("statsmodels is required for compute_pacf") from e
    try:
        from scipy import stats
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError("scipy is required for compute_pacf") from e
    pacf_vals, confint = pacf(res, nlags=nlags, alpha=0.05, method="ywmle")
    se = (confint[:, 1] - confint[:, 0]) / (2 * 1.96)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(pacf_vals) / se))
    lags = np.arange(len(pacf_vals))
    return pd.DataFrame(
        {
            "lag": lags,
            "pacf": pacf_vals,
            "pvalue": pvals,
            "confint_lower": confint[:, 0],
            "confint_upper": confint[:, 1],
        }
    )


def ljung_box_test(residuals: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation."""
    if not lags:
        raise ValueError("lags must be a non-empty list")
    max_lag = max(lags)
    res = _validate_series(residuals, max_lag)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError("statsmodels is required for ljung_box_test") from e
    lb = acorr_ljungbox(res, lags=lags, return_df=True)
    lb = lb.rename(columns={"lb_stat": "stat", "lb_pvalue": "pvalue"})
    lb.insert(0, "lag", lags)
    return lb


def white_test(residuals: pd.Series, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """White's test for heteroskedasticity."""
    res = _validate_series(residuals, 1)
    try:
        from statsmodels.stats.diagnostic import het_white
        import statsmodels.api as sm
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError("statsmodels is required for white_test") from e
    if exog is None:
        exog = pd.DataFrame(
            {"const": np.ones(len(res)), "trend": np.arange(len(res))},
            index=res.index,
        )
    else:
        if not isinstance(exog, pd.DataFrame):
            raise TypeError("exog must be a pandas DataFrame or None")
        if len(exog) != len(res):
            raise ValueError("exog must have same length as residuals")
        exog = exog.reindex(res.index)
        exog = sm.add_constant(exog, has_constant="add")
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(res, exog)
    return pd.DataFrame(
        {
            "lm_stat": [lm_stat],
            "lm_pvalue": [lm_pvalue],
            "f_stat": [f_stat],
            "f_pvalue": [f_pvalue],
        }
    )


def plot_residuals(residuals: pd.Series, out_dir: Path) -> None:
    """Plot residual diagnostics and save to directory."""
    res = _validate_series(residuals, 1)
    try:
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError(
            "matplotlib and statsmodels are required for plot_residuals"
        ) from e
    os.makedirs(out_dir, exist_ok=True)
    # time series plot
    fig, ax = plt.subplots()
    res.plot(ax=ax)
    ax.set_title("Residuals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    fig.savefig(Path(out_dir) / "residuals.png", bbox_inches="tight")
    plt.close(fig)
    # histogram
    fig, ax = plt.subplots()
    ax.hist(res, bins=20, edgecolor="black")
    ax.set_title("Residual Histogram")
    fig.savefig(Path(out_dir) / "histogram.png", bbox_inches="tight")
    plt.close(fig)
    # ACF plot
    nlags = min(40, len(res) - 1)
    fig = plot_acf(res, lags=nlags)
    fig.savefig(Path(out_dir) / "acf.png", bbox_inches="tight")
    plt.close(fig)
    # PACF plot
    fig = plot_pacf(res, lags=nlags)
    fig.savefig(Path(out_dir) / "pacf.png", bbox_inches="tight")
    plt.close(fig)

