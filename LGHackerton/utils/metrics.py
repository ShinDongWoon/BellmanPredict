
from __future__ import annotations
import numpy as np
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
    sm = np.zeros_like(y_true, dtype=float)
    sm[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    if outlet_names is None:
        return float(np.mean(sm[mask])) if np.any(mask) else 0.0
    import numpy as np
    outlets = np.array(list(outlet_names))
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
