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


def trunc_nb_nll(y: Tensor, mu: Tensor, kappa: Tensor) -> Tensor:
    r"""Zero-truncated negative binomial negative log-likelihood.

    The probability mass function of the negative binomial distribution is

    .. math::

        \Pr(Y=y) = \frac{\Gamma(y+\kappa)}{\Gamma(\kappa)\,\Gamma(y+1)}
        \Big(\frac{\kappa}{\kappa+\mu}\Big)^{\kappa}
        \Big(\frac{\mu}{\kappa+\mu}\Big)^y

    Conditioning on :math:`Y>0` yields the zero-truncated version used here.

    Parameters
    ----------
    y, mu, kappa:
        Observed counts, unconditional mean and shape parameters.  All tensors
        must be broadcastable to a common shape.
    """

    y = y.to(mu.dtype)
    if (y <= 0).any():
        raise ValueError("trunc_nb_nll requires y > 0")
    kappa = torch.clamp(
        torch.as_tensor(kappa, dtype=mu.dtype, device=mu.device), min=1e-8
    )
    mu = torch.clamp(mu, min=1e-8)

    log_kappa = torch.log(kappa)
    log_mu = torch.log(mu)
    log_kappa_plus_mu = log_kappa + torch.log1p(mu / kappa)

    log_pmf = (
        torch.lgamma(y + kappa)
        - torch.lgamma(kappa)
        - torch.lgamma(y + 1.0)
        + kappa * (log_kappa - log_kappa_plus_mu)
        + y * (log_mu - log_kappa_plus_mu)
    )

    log_p0 = kappa * (log_kappa - log_kappa_plus_mu)
    p0 = torch.exp(log_p0)
    p0 = torch.clamp(p0, max=1.0 - 1e-6)
    log1m_p0 = torch.log1p(-p0)

    return -(log_pmf - log1m_p0)


def hurdle_nll(logits: Tensor, z: Tensor, w: Optional[Tensor] = None) -> Tensor:
    r"""Binary hurdle negative log-likelihood.

    Parameters
    ----------
    logits:
        Logits for the probability of observing a non-zero target.
    z:
        Ground truth indicators where ``1`` denotes a non-zero target.
    w:
        Optional sample weights applied before taking the mean.
    """

    p0 = torch.sigmoid(-logits)
    p0 = torch.clamp(p0, min=1e-8, max=1 - 1e-8)
    loss = z * -torch.log1p(-p0) + (1 - z) * -torch.log(p0)
    if w is not None:
        loss = loss * w
    return loss.mean()


def combine_predictions(
    p: Tensor, mu: Tensor, kappa: Tensor, epsilon: Tensor
) -> Tensor:
    r"""Combine classifier probability with conditional mean adjustment.

    The regression prediction ``mu`` is interpreted as the unconditional mean
    :math:`\mu_u`. The conditional mean under zero truncation is calculated
    using the per-sample ``kappa`` and then scaled by the classifier
    probability ``p`` with a small ``epsilon`` to avoid collapse::

        P0 = (kappa / (kappa + mu)) ** kappa
        cond_mean = mu / max(1 - P0, 1e-6)
        y_hat = ((1 - eps) * p + eps) * cond_mean

    Parameters
    ----------
    p:
        Probability output of the zero/non-zero classifier where higher values
        indicate non-zero demand.
    mu:
        Regression model predictions corresponding to the unconditional mean
        ``mu_u``.
    kappa:
        Shape parameter controlling the zero probability ``P0``.
    epsilon:
        Small constant added to the classifier probability. May be per-sample.

    Returns
    -------
    Tensor
        Final demand prediction after conditional mean adjustment.
    """
    mu = mu.to(p.dtype)
    kappa = torch.clamp(
        torch.as_tensor(kappa, dtype=mu.dtype, device=mu.device), min=1e-8
    )
    epsilon = torch.as_tensor(epsilon, dtype=mu.dtype, device=mu.device)

    p0 = torch.pow(kappa / (kappa + mu), kappa)
    cond_mean = mu / torch.clamp(1.0 - p0, min=1e-6)
    return ((1 - epsilon) * p + epsilon) * cond_mean


def combine_predictions_thresholded(
    p: Optional[Tensor] = None,
    logits: Optional[Tensor] = None,
    mu: Tensor | None = None,
    kappa: Tensor | None = None,
    tau: float = 0.5,
    temperature: Optional[float] = None,
) -> Tensor:
    """Combine predictions using a thresholded gate.

    The conditional mean is computed in the same way as in
    :func:`combine_predictions`.  The classifier output is converted into a
    gating value according to ``tau`` and ``temperature``:

    - If ``temperature`` is ``None`` a hard gate is applied using the
      probabilities ``p`` as ``gate = (p >= tau)``.
    - Otherwise the provided ``logits`` are transformed using
      ``sigmoid((logits - b) / temperature)`` where ``b`` is the logit of
      ``tau``.

    Parameters
    ----------
    p, logits:
        Probability or logit outputs from the classifier.  Exactly one of
        these must be provided depending on ``temperature``.
    mu:
        Regression model predictions corresponding to the unconditional mean
        ``mu_u``.
    kappa:
        Shape parameter controlling the zero probability ``P0``.
    tau:
        Threshold applied to the classifier output.
    temperature:
        Temperature for the soft gate.  If ``None`` a hard gate is used.
    """

    if mu is None or kappa is None:
        raise ValueError("mu and kappa must be provided")

    dtype = mu.dtype
    if p is not None:
        dtype = p.dtype
    elif logits is not None:
        dtype = logits.dtype
    mu = mu.to(dtype)
    device = mu.device
    kappa_t = torch.clamp(torch.as_tensor(kappa, dtype=dtype, device=device), min=1e-8)

    # Compute conditional mean with numerical safeguards
    p0 = torch.pow(kappa_t / (kappa_t + mu), kappa_t)
    cond_mean = mu / torch.clamp(1.0 - p0, min=1e-6)

    if temperature is None:
        if p is None:
            raise ValueError("p must be provided when temperature is None")
        gate = (p >= tau).to(mu.dtype)
    else:
        if logits is None:
            raise ValueError("logits must be provided when temperature is not None")
        logits = logits.to(mu.dtype)
        b = torch.logit(torch.tensor(tau, dtype=mu.dtype, device=device))
        gate = torch.sigmoid((logits - b) / temperature)

    return gate * cond_mean


def weighted_smape_oof(
    y_true: Tensor,
    clf_prob: Tensor,
    mu: Tensor,
    kappa: Tensor,
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
    mu:
        Regression predictions representing ``mu_u``.
    kappa:
        Per-sample shape parameters controlling the zero probability.
    epsilon_leaky:
        Small constant added to the classifier probability.
    w:
        Optional sample weights.
    """
    final_pred = combine_predictions(clf_prob, mu, kappa, epsilon_leaky)
    loss_fn = WeightedSMAPELoss(reduction="mean")
    return loss_fn(final_pred, y_true, w=w)


def weighted_smape_oof_thresholded(
    y_true: Tensor,
    p: Optional[Tensor] = None,
    logits: Optional[Tensor] = None,
    mu: Tensor | None = None,
    kappa: Tensor | None = None,
    tau: float = 0.5,
    temperature: Optional[float] = None,
    w: Optional[Tensor] = None,
) -> Tensor:
    """Compute weighted sMAPE for thresholded combined predictions."""

    final_pred = combine_predictions_thresholded(
        p=p,
        logits=logits,
        mu=mu,
        kappa=kappa,
        tau=tau,
        temperature=temperature,
    )
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
