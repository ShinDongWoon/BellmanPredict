import pathlib
import sys

import torch

# Add project root to sys.path for module imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from LGHackerton.models.patchtst import (
    combine_predictions,
    focal_loss,
    trunc_nb_nll,
    weighted_smape_oof,
)


def test_weighted_smape_oof_accepts_tensor_kappa():
    """weighted_smape_oof should handle per-sample kappa tensors."""
    y_true = torch.tensor([0.0, 1.0])
    clf_prob = torch.tensor([0.2, 0.8])
    reg_pred = torch.tensor([0.3, 0.5])
    kappa = torch.tensor([1.0, 2.0])
    loss = weighted_smape_oof(y_true, clf_prob, reg_pred, kappa, 0.1)
    assert loss.ndim == 0


def test_patchtst_exports_loss_helpers():
    """The patchtst package should expose utility functions for external use."""
    for fn in (trunc_nb_nll, focal_loss, combine_predictions, weighted_smape_oof):
        assert callable(fn)
