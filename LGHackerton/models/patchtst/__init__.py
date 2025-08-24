"""PatchTST training utilities.

This package exposes loss functions and helpers used for training the
PatchTST model. It is intentionally lightweight so that other modules can
import the losses without pulling in heavy training dependencies.
"""

from .train import (
    WeightedSMAPELoss,
    build_loss,
    trunc_nb_nll,
    focal_loss,
    combine_predictions,
    combine_predictions_thresholded,
    weighted_smape_oof,
    weighted_smape_oof_thresholded,
)

__all__ = [
    "WeightedSMAPELoss",
    "build_loss",
    "trunc_nb_nll",
    "focal_loss",
    "combine_predictions",
    "combine_predictions_thresholded",
    "weighted_smape_oof",
    "weighted_smape_oof_thresholded",
]
