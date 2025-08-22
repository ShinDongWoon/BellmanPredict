from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from LGHackerton.config.default import SAMPLE_SUB_PATH


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def _missing_checks(df: pd.DataFrame) -> None:
    """Warn about missing dates or series compared to sample submission."""
    sample_df = _read_table(SAMPLE_SUB_PATH)
    missing_dates = set(sample_df.iloc[:, 0]) - set(df["date"])
    missing_cols = set(sample_df.columns[1:]) - set(df["series_id"].unique())
    if missing_dates:
        logging.warning("Missing dates in predictions: %s", sorted(missing_dates))
    if missing_cols:
        logging.warning("Missing columns in predictions: %s", sorted(missing_cols))


def aggregate_predictions(
    pred_dfs: List[pd.DataFrame],
    weights: Optional[List[float]] = None,
    how: str = "mean",
) -> pd.DataFrame:
    """Aggregate model predictions into a single dataframe.

    Parameters
    ----------
    pred_dfs : List[pd.DataFrame]
        List of prediction dataframes. Each must contain at least
        ``series_id`` and either ``date`` or both ``test_id`` and ``h``
        columns, plus a single ``yhat_<model>`` column.
    weights : Optional[List[float]], default None
        Weights for weighted mean. When ``None`` simple averages are used.
    how : str, default "mean"
        Currently only ``"mean"`` is supported.
    """

    if not pred_dfs:
        raise ValueError("pred_dfs must not be empty")

    if weights is not None and len(weights) != len(pred_dfs):
        raise ValueError("weights length must match number of dataframes")

    if weights is None:
        weights = [1.0] * len(pred_dfs)
    weights_arr = np.asarray(weights, dtype=float)
    if how != "mean":
        raise ValueError("Currently only 'mean' aggregation is supported")

    merged: Optional[pd.DataFrame] = None
    model_cols: List[str] = []
    key_cols = ["series_id", "date"]

    for idx, (df, w) in enumerate(zip(pred_dfs, weights_arr)):
        df = df.copy()
        yhat_cols = [c for c in df.columns if c.startswith("yhat_")]
        if len(yhat_cols) != 1:
            raise ValueError("Each dataframe must have exactly one 'yhat_' column")
        ycol = yhat_cols[0]

        if "date" not in df.columns:
            if {"test_id", "h"}.issubset(df.columns):
                df["date"] = df["test_id"].astype(str) + "+" + df["h"].astype(str) + "일"
            else:
                raise ValueError("Prediction dataframe must contain 'date' or ('test_id' and 'h') columns")

        df = df["series_id"].to_frame().join(df["date"]).join(df[ycol])
        new_col = f"yhat_model_{idx}"
        df.rename(columns={ycol: new_col}, inplace=True)

        dup = df.duplicated(subset=key_cols)
        if dup.any():
            logging.warning("Duplicate predictions found for some series/date pairs")
            df = df[~dup]

        merged = df if merged is None else pd.merge(merged, df, on=key_cols, how="outer")
        model_cols.append(new_col)

    values = merged[model_cols].to_numpy(dtype=float)
    mask = np.isnan(values)

    if weights_arr.sum() != 0:
        norm_w = weights_arr / weights_arr.sum()
    else:
        norm_w = weights_arr

    weighted = values * norm_w
    denom = (~mask * norm_w).sum(axis=1)
    yhat = np.nansum(weighted, axis=1) / denom

    merged["yhat_ens"] = yhat
    merged = merged[key_cols + ["yhat_ens"]]

    if np.isnan(merged["yhat_ens"]).any():
        logging.warning("Some predictions are missing after aggregation")

    _missing_checks(merged)

    return merged


def convert_to_submission(
    preds: Union[pd.DataFrame, List[pd.DataFrame]],
    weights: Optional[List[float]] = None,
    how: str = "mean",
) -> pd.DataFrame:
    """Convert predictions to the official submission format.

    Parameters
    ----------
    preds : Union[pd.DataFrame, List[pd.DataFrame]]
        Either a single aggregated dataframe containing ``yhat_ens`` or a
        list of model prediction dataframes. When a list is provided,
        :func:`aggregate_predictions` is called internally.
    weights : Optional[List[float]], default None
        Weights used when ``preds`` is a list of dataframes.
    how : str, default "mean"
        Aggregation method passed to :func:`aggregate_predictions`.
    """

    if isinstance(preds, list):
        pred_df = aggregate_predictions(preds, weights=weights, how=how)
    else:
        pred_df = preds.copy()

    sample_df = _read_table(SAMPLE_SUB_PATH)

    pred_df["series_id"] = pred_df["series_id"].str.replace("::", "_", n=1)

    wide = pred_df.pivot(index="date", columns="series_id", values="yhat_ens")
    wide = wide.reindex(sample_df.iloc[:, 0]).reindex(
        columns=sample_df.columns[1:], fill_value=0.0
    )
    wide = wide.astype(float)

    _missing_checks(pred_df)

    out_df = sample_df.copy()
    out_df = out_df.astype({col: float for col in out_df.columns[1:]})
    out_df.iloc[:, 1:] = wide.to_numpy()
    assert list(out_df.columns) == list(sample_df.columns)
    return out_df
