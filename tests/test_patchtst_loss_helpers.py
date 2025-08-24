import pathlib
import sys

import torch

# Add project root to sys.path for module imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from LGHackerton.models.patchtst import (
    combine_predictions,
    combine_predictions_thresholded,
    hurdle_nll,
    trunc_nb_nll,
    weighted_smape_oof,
    weighted_smape_oof_thresholded,
)


def test_weighted_smape_oof_accepts_tensor_kappa():
    """weighted_smape_oof should handle per-sample kappa tensors."""
    y_true = torch.tensor([0.0, 1.0])
    clf_prob = torch.tensor([0.2, 0.8])
    mu = torch.tensor([0.3, 0.5])
    kappa = torch.tensor([1.0, 2.0])
    loss = weighted_smape_oof(y_true, clf_prob, mu, kappa, 0.1)
    assert loss.ndim == 0


def test_weighted_smape_oof_thresholded_accepts_tensor_kappa():
    """weighted_smape_oof_thresholded should handle per-sample kappa tensors."""
    y_true = torch.tensor([0.0, 1.0])
    clf_prob = torch.tensor([0.2, 0.8])
    mu = torch.tensor([0.3, 0.5])
    kappa = torch.tensor([1.0, 2.0])
    loss = weighted_smape_oof_thresholded(y_true, p=clf_prob, mu=mu, kappa=kappa, tau=0.5)
    assert loss.ndim == 0


def test_combine_predictions_thresholded_gate_behaviour():
    """Hard and soft gates should behave as expected."""
    p = torch.tensor([0.3, 0.7])
    logits = torch.logit(p)
    mu = torch.tensor([1.0, 2.0])
    kappa = torch.tensor([1.0, 1.0])
    hard = combine_predictions_thresholded(p=p, mu=mu, kappa=kappa, tau=0.5)
    soft = combine_predictions_thresholded(logits=logits, mu=mu, kappa=kappa, tau=0.5, temperature=1.0)
    expected_hard = torch.tensor([0.0, 3.0])
    expected_soft = torch.tensor([0.6, 2.1])
    assert torch.allclose(hard, expected_hard)
    assert torch.allclose(soft, expected_soft)


def test_hurdle_nll_matches_bce_with_logits():
    """hurdle_nll should match BCE with logits for identical inputs."""
    logits = torch.tensor([0.2, -0.3])
    targets = torch.tensor([1.0, 0.0])
    w = torch.tensor([1.0, 2.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=w
    )
    actual = hurdle_nll(logits, targets, w)
    assert torch.allclose(actual, expected)


def test_patchtst_exports_loss_helpers():
    """The patchtst package should expose utility functions for external use."""
    for fn in (
        trunc_nb_nll,
        hurdle_nll,
        combine_predictions,
        combine_predictions_thresholded,
        weighted_smape_oof,
        weighted_smape_oof_thresholded,
    ):
        assert callable(fn)
