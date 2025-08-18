"""Training utilities for LightGBM classification models.

This module provides helpers to train a binary classifier that distinguishes
between zero and non-zero demand. The classifier's out-of-fold predictions can
be combined with regression model outputs (e.g. PatchTST) to form a hurdle
model. Final predictions are zero when the classifier predicts ``non-zero``
probability below a threshold and otherwise fall back to the regression
prediction.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def train_zero_classifier(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[List[lgb.Booster], np.ndarray]:
    """Train LightGBM classifiers for zero/non-zero demand.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Target demand values. ``0`` indicates no sales.
    params:
        Optional LightGBM parameter dictionary. Reasonable defaults are used
        when ``None``.
    n_folds:
        Number of stratified CV folds.
    random_state:
        Seed for the cross-validation splitter.

    Returns
    -------
    models, oof
        List of trained boosters and the out-of-fold probability predictions
        for the non-zero class.
    """

    z = (y > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    models: List[lgb.Booster] = []
    oof = np.zeros(len(y), dtype=float)

    lgb_params: Dict[str, float] = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "metric": "binary_logloss",
    }
    if params:
        lgb_params.update(params)

    for tr_idx, va_idx in skf.split(X, z):
        dtrain = lgb.Dataset(X[tr_idx], label=z[tr_idx])
        dvalid = lgb.Dataset(X[va_idx], label=z[va_idx])
        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            verbose_eval=False,
        )
        models.append(booster)
        oof[va_idx] = booster.predict(X[va_idx], num_iteration=booster.best_iteration)

    return models, oof


def predict_zero_probability(models: Iterable[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """Average predictions from a list of LightGBM classifiers."""

    preds = [m.predict(X, num_iteration=m.best_iteration) for m in models]
    return np.mean(preds, axis=0)


def combine_with_regression(
    clf_prob: np.ndarray,
    reg_pred: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Combine classifier probabilities and regression predictions.

    Parameters
    ----------
    clf_prob:
        Probability estimates of non-zero demand.
    reg_pred:
        Regression model predictions for the same samples.
    threshold:
        Decision threshold for the classifier. Predictions below the threshold
        are set to ``0``.
    """

    return np.where(clf_prob >= threshold, reg_pred, 0.0)


__all__ = [
    "train_zero_classifier",
    "predict_zero_probability",
    "combine_with_regression",
]

