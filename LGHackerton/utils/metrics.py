
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Optional

PRIORITY_OUTLETS = {"담하", "미라시아"}

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    if eps > 0:
        denom = denom + eps
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

def weighted_smape_np(y_true: np.ndarray, y_pred: np.ndarray,
                      outlet_names: Optional[Iterable[str]] = None,
                      priority_weight: float = 3.0,
                      eps: float = 0.0) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    if eps > 0:
        denom = denom + eps
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    sm = np.zeros_like(y_true, dtype=float)
    sm[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    if outlet_names is None:
        return float(np.mean(sm[mask]))
    outlets = np.asarray(list(outlet_names))
    w = np.where(np.isin(outlets, list(PRIORITY_OUTLETS)), priority_weight, 1.0).astype(float)
    w = np.where(mask, w, 0.0)
    if w.sum() <= 0:
        return 0.0
    return float(np.sum(sm * w) / np.sum(w))

def lgbm_weighted_smape(preds, dataset, use_asinh_target: bool = False):
    import numpy as np
    y = dataset.get_label()
    if use_asinh_target:
        y = np.sinh(y)
        preds = np.sinh(preds)
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y, dtype=float)
    denom = (np.abs(y) + np.abs(preds)) / 2.0
    mask = denom > 0
    sm = np.zeros_like(y, dtype=float)
    sm[mask] = np.abs(y[mask] - preds[mask]) / denom[mask]
    w = np.where(mask, w, 0.0)
    val = float(np.sum(sm * w) / np.sum(w)) if np.sum(w) > 0 else 0.0
    return ("wSMAPE", val, False)


# ---------------------------------------------------------------------------
# Baseline forecasting utilities
# ---------------------------------------------------------------------------

def naive_forecast(series: Iterable[float] | pd.Series,
                   horizon: int,
                   frequency: int | str | None = None) -> np.ndarray:
    """Forecast by repeating the last observed value."""
    arr = pd.Series(series).astype(float).dropna().values
    if len(arr) == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(arr[-1], horizon)


def seasonal_naive_forecast(series: Iterable[float] | pd.Series,
                            horizon: int,
                            frequency: int = 7) -> np.ndarray:
    """Repeat the observations from one seasonal cycle ago."""
    arr = pd.Series(series).astype(float).dropna().values
    if len(arr) == 0:
        return np.zeros(horizon, dtype=float)
    if len(arr) < frequency or frequency <= 0:
        return naive_forecast(arr, horizon)
    last_cycle = arr[-frequency:]
    reps = int(np.ceil(horizon / frequency))
    return np.tile(last_cycle, reps)[:horizon]


def ets_forecast(series: Iterable[float] | pd.Series,
                 horizon: int,
                 frequency: int | None = None) -> np.ndarray:
    """Simple ETS (Exponential Smoothing) forecast using statsmodels."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    arr = pd.Series(series).astype(float).dropna().values
    if len(arr) == 0:
        return np.zeros(horizon, dtype=float)
    seasonal = None
    seasonal_periods = None
    if frequency and frequency > 1:
        seasonal = "add"
        seasonal_periods = int(frequency)
    model = ExponentialSmoothing(arr,
                                 trend=None,
                                 seasonal=seasonal,
                                 seasonal_periods=seasonal_periods,
                                 initialization_method="estimated")
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)


def prophet_forecast(series: pd.Series,
                     horizon: int,
                     frequency: str = "D") -> np.ndarray:
    """Forecast using Prophet with holidays disabled."""
    from prophet import Prophet

    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    ds = series.index
    if not isinstance(ds, pd.DatetimeIndex):
        # create a dummy daily index starting today
        ds = pd.date_range(pd.Timestamp.today(), periods=len(series), freq=frequency)
    df = pd.DataFrame({"ds": ds, "y": series.astype(float).values})
    m = Prophet(holidays=None,
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False)
    m.fit(df, iter=1000)
    future = m.make_future_dataframe(periods=horizon, freq=frequency, include_history=False)
    fcst = m.predict(future)
    return fcst["yhat"].values
