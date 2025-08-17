from __future__ import annotations
import argparse
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import yaml

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess import Preprocessor, DATE_COL, SERIES_COL, SALES_COL
from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SHOP_COL
from LGHackerton.config.default import TRAIN_PATH, ARTIFACTS_DIR
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.utils.metrics import (
    naive_forecast,
    seasonal_naive_forecast,
    ets_forecast,
    prophet_forecast,
    smape,
    weighted_smape_np,
)
from LGHackerton.utils.seed import set_seed

BASELINES: Dict[str, callable] = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "ets": ets_forecast,
    "prophet": prophet_forecast,
}


def file_md5(path: str) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            m.update(chunk)
    return m.hexdigest()


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    outlets: np.ndarray,
                    series_ids: np.ndarray,
                    train_series: Dict[str, np.ndarray]) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    smape_val = float(smape(y_true, y_pred))
    wsmape_val = float(weighted_smape_np(y_true, y_pred, outlets))
    mase_list = []
    rmsse_list = []
    # compute per-series MASE/RMSSE
    for sid, tr in train_series.items():
        diff = np.diff(tr)
        scale = np.mean(np.abs(diff)) if len(diff) > 0 else np.nan
        scale2 = np.mean(diff ** 2) if len(diff) > 0 else np.nan
        mask = series_ids == sid
        if np.isnan(scale) or scale == 0 or np.isnan(scale2) or scale2 == 0:
            continue
        err = y_true[mask] - y_pred[mask]
        mase_list.append(np.mean(np.abs(err)) / scale)
        rmsse_list.append(np.sqrt(np.mean(err ** 2)) / np.sqrt(scale2))
    mase = float(np.mean(mase_list)) if mase_list else float("nan")
    rmsse = float(np.mean(rmsse_list)) if rmsse_list else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_val,
        "wSMAPE": wsmape_val,
        "MASE": mase,
        "RMSSE": rmsse,
    }


def make_folds(dates: np.ndarray, cfg: TrainConfig):
    """Generate rolling-origin folds (start, end) tuples."""
    purge_days = cfg.purge_days if cfg.purge_days > 0 else 28
    folds = []
    used = np.zeros(len(dates), dtype=bool)
    for i in range(cfg.rocv_n_folds):
        end = dates[-1] - np.timedelta64(i * cfg.rocv_stride_days, "D")
        start = end - np.timedelta64(cfg.rocv_val_span_days - 1, "D")
        va_mask = (dates >= start) & (dates <= end) & (~used)
        if not va_mask.any():
            continue
        used |= va_mask
        train_end = start - np.timedelta64(purge_days, "D")
        folds.append((train_end, start, end))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model", type=str, choices=list(BASELINES.keys()), default="naive")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    set_seed(cfg.seed)
    logging.info("seed=%s", cfg.seed)
    logging.info("data_hash=%s", file_md5(TRAIN_PATH))
    logging.info("numpy=%s pandas=%s", np.__version__, pd.__version__)
    try:
        import statsmodels
        logging.info("statsmodels=%s", statsmodels.__version__)
    except Exception as e:  # pragma: no cover - optional
        logging.info("statsmodels not available: %s", e)
    try:
        import prophet
        logging.info("prophet=%s", prophet.__version__)
    except Exception as e:  # pragma: no cover - optional
        logging.info("prophet not available: %s", e)

    df_raw = pd.read_csv(TRAIN_PATH)
    pp = Preprocessor(show_progress=False)
    df_full = pp.fit_transform_train(df_raw)

    dates = np.sort(df_full[DATE_COL].unique())
    folds = make_folds(dates, cfg)
    model_fn = BASELINES[args.model]
    all_fold_metrics = []

    for idx, (train_end, start, end) in enumerate(folds):
        tr_mask = df_full[DATE_COL] < train_end
        va_mask = (df_full[DATE_COL] >= start) & (df_full[DATE_COL] <= end)
        df_tr = df_full[tr_mask]
        df_va = df_full[va_mask]
        y_true_all = []
        y_pred_all = []
        outlet_all = []
        sid_all = []
        train_series = {}
        for sid, g_tr in df_tr.groupby(SERIES_COL, sort=False):
            g_va = df_va[df_va[SERIES_COL] == sid].sort_values(DATE_COL)
            if g_va.empty:
                continue
            series = g_tr.sort_values(DATE_COL)[SALES_COL].astype(float)
            train_series[sid] = series.values
            horizon = len(g_va)
            if args.model == "prophet":
                series = series
            preds = model_fn(series, horizon, 7)
            y_true = g_va[SALES_COL].astype(float).values
            y_true_all.append(y_true)
            y_pred_all.append(preds)
            outlet_all.extend(g_va[SHOP_COL].tolist())
            sid_all.extend([sid] * len(g_va))
        if not y_true_all:
            continue
        y_true_arr = np.concatenate(y_true_all)
        y_pred_arr = np.concatenate(y_pred_all)
        outlets_arr = np.array(outlet_all, dtype=object)
        sid_arr = np.array(sid_all, dtype=object)
        met = compute_metrics(y_true_arr, y_pred_arr, outlets_arr, sid_arr, train_series)
        met.update({"model": args.model, "fold": idx,
                    "timestamp": datetime.now().isoformat()})
        all_fold_metrics.append(met)
        logging.info("fold %s metrics: %s", idx, met)

    if all_fold_metrics:
        overall = {k: float(np.nanmean([m[k] for m in all_fold_metrics]))
                   for k in ["MAE", "RMSE", "sMAPE", "wSMAPE", "MASE", "RMSSE"]}
        overall.update({"model": args.model, "fold": "overall",
                        "timestamp": datetime.now().isoformat()})
        all_fold_metrics.append(overall)

    out_df = pd.DataFrame(all_fold_metrics)
    out_path = Path(ARTIFACTS_DIR) / "baseline_metrics.csv"
    out_df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    logging.info("results appended to %s", out_path)


if __name__ == "__main__":
    main()

