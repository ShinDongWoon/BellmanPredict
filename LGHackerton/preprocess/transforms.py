"""Preprocessing utilities for handling negative values.

Provides symmetric transformations to ensure that negative values are
properly preserved through the modeling pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def symmetric_transform(series: pd.Series) -> pd.Series:
    """Apply an ``arcsinh`` transform to ``series``.

    The ``arcsinh`` transform behaves like ``log1p`` for large positive values
    while remaining defined for negatives. This makes it suitable for data that
    may contain returns or refunds.
    """
    return np.arcsinh(series)


def inverse_symmetric_transform(values: pd.Series | np.ndarray) -> np.ndarray:
    """Inverse of :func:`symmetric_transform` using ``sinh``.

    Parameters
    ----------
    values:
        Transformed values to convert back to the original scale.

    Returns
    -------
    np.ndarray
        Values on the original scale where negative predictions are allowed.
    """
    return np.sinh(values)


__all__ = ["symmetric_transform", "inverse_symmetric_transform"]
