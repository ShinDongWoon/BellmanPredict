r"""Helpers for hurdle models."""

from __future__ import annotations

import numpy as np


def combine_with_regression(clf_prob: np.ndarray, reg_pred: np.ndarray) -> np.ndarray:
    r"""Combine classifier probabilities and regression predictions.

    Parameters
    ----------
    clf_prob : np.ndarray
        Probability estimates of non-zero demand.
    reg_pred : np.ndarray
        Regression model predictions for the same samples.

    Returns
    -------
    np.ndarray
        Final demand forecasts ``p * y_hat`` clipped at zero.
    """

    return np.clip(clf_prob * reg_pred, 0.0, None)
