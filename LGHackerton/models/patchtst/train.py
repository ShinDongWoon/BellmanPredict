"""Training helpers for PatchTST.

The module provides the :class:`WeightedSMAPELoss` which implements a
weighted symmetric mean absolute percentage error (SMAPE) loss.  It also
exposes a simple ``build_loss`` helper and a small CLI so that the loss can
be selected when running the training script directly::

    python -m LGHackerton.models.patchtst.train --loss smape

By default the ``WeightedSMAPELoss`` is used.  ``--loss l1`` selects a
standard L1 loss and ``--loss hybrid`` combines L1 and SMAPE losses using a
mixing parameter ``alpha``.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class WeightedSMAPELoss(nn.Module):
    """Symmetric MAPE with optional sample weights.

    Parameters
    ----------
    eps:
        Small value added to the denominator for numerical stability.
    reduction:
        Specifies the reduction to apply to the output: ``'mean'``, ``'sum'``
        or ``'none'``.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor, w: Optional[Tensor] = None) -> Tensor:  # type: ignore[override]
        """Compute weighted SMAPE.

        The loss is calculated as::

            2 * |y_true - y_pred| / clamp(|y_true| + |y_pred|, min=eps)

        If ``w`` is provided it is multiplied element-wise with the loss before
        applying reduction.
        """

        denom = torch.clamp(torch.abs(y_true) + torch.abs(y_pred), min=self.eps)
        loss = 2.0 * torch.abs(y_true - y_pred) / denom
        if w is not None:
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HybridLoss(nn.Module):
    """Blend of L1 and SMAPE losses.

    The loss is computed as ``alpha * L1 + (1 - alpha) * SMAPE``.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.l1 = nn.L1Loss(reduction="none")
        self.smape = WeightedSMAPELoss(eps=eps, reduction="none")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor, w: Optional[Tensor] = None) -> Tensor:  # type: ignore[override]
        l1 = self.l1(y_pred, y_true)
        smape = self.smape(y_pred, y_true)
        if w is not None:
            if w.ndim == 1:
                w = w.unsqueeze(1)
            l1 = l1 * w
            smape = smape * w
        loss = self.alpha * l1 + (1.0 - self.alpha) * smape
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, alpha: float = 0.5, eps: float = 1e-8, reduction: str = "mean") -> nn.Module:
    """Return a loss function according to ``name``.

    Parameters
    ----------
    name:
        One of ``"smape"``, ``"l1"`` or ``"hybrid"``.
    alpha:
        Mixing factor for the hybrid loss.
    eps:
        Epsilon passed to :class:`WeightedSMAPELoss`.
    reduction:
        Reduction method for the loss.
    """

    name = name.lower()
    if name == "smape":
        return WeightedSMAPELoss(eps=eps, reduction=reduction)
    if name == "l1":
        return nn.L1Loss(reduction=reduction)
    if name == "hybrid":
        return HybridLoss(alpha=alpha, eps=eps, reduction=reduction)
    raise ValueError(f"Unknown loss '{name}'")


def combine_predictions(
    clf_prob: Tensor, reg_pred: Tensor, kappa: float, epsilon_leaky: float
) -> Tensor:
    r"""Combine classifier probability with conditional mean adjustment.

    The regression prediction ``reg_pred`` is interpreted as the unconditional
    mean :math:`\mu_u`.  The conditional mean under zero truncation is
    calculated using ``kappa`` and then scaled by the classifier probability
    ``clf_prob`` with a small ``epsilon_leaky`` to avoid collapse::

        P0 = (kappa / (kappa + mu_u)) ** kappa
        cond_mean = mu_u / max(1 - P0, 1e-6)
        y_hat = ((1 - eps) * p + eps) * cond_mean

    Parameters
    ----------
    clf_prob:
        Probability output of the zero/non-zero classifier where higher values
        indicate non-zero demand.
    reg_pred:
        Regression model predictions corresponding to the unconditional mean
        ``mu_u``.
    kappa:
        Shape parameter controlling the zero probability ``P0``.
    epsilon_leaky:
        Small constant added to the classifier probability.

    Returns
    -------
    Tensor
        Final demand prediction after conditional mean adjustment.
    """
    k = torch.as_tensor(kappa, dtype=reg_pred.dtype, device=reg_pred.device)
    eps = torch.as_tensor(epsilon_leaky, dtype=reg_pred.dtype, device=reg_pred.device)
    p0 = torch.pow(k / (k + reg_pred), k)
    cond_mean = reg_pred / torch.clamp(1.0 - p0, min=1e-6)
    return ((1 - eps) * clf_prob + eps) * cond_mean


def weighted_smape_oof(
    y_true: Tensor,
    clf_prob: Tensor,
    reg_pred: Tensor,
    kappa: float,
    epsilon_leaky: float,
    w: Optional[Tensor] = None,
) -> Tensor:
    """Compute weighted sMAPE for combined OOF predictions.

    Parameters
    ----------
    y_true:
        Ground truth demand.
    clf_prob:
        Probability estimates from the classifier.
    reg_pred:
        Regression predictions representing ``mu_u``.
    kappa:
        Shape parameter for the zero probability computation.
    epsilon_leaky:
        Small constant added to the classifier probability.
    w:
        Optional sample weights.
    """
    final_pred = combine_predictions(clf_prob, reg_pred, kappa, epsilon_leaky)
    loss_fn = WeightedSMAPELoss(reduction="mean")
    return loss_fn(final_pred, y_true, w=w)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST training helper")
    parser.add_argument("--loss", choices=["smape", "l1", "hybrid"], default="smape",
                        help="Loss function to use (default: smape)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="L1 weight for hybrid loss")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Denominator epsilon for SMAPE")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    loss = build_loss(args.loss, alpha=args.alpha, eps=args.eps)
    # For demonstration purposes we simply print the selected loss.
    # Real training pipelines would pass ``loss`` to the optimisation loop.
    print(f"Using loss: {loss.__class__.__name__}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
