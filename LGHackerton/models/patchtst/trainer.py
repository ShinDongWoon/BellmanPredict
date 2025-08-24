from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass, asdict
from typing import Tuple, Iterable, Optional, List, Any, Dict, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception as _e:
    TORCH_OK = False
    _TORCH_ERR = _e

from LGHackerton.models.base_trainer import BaseModel, TrainConfig
from LGHackerton.utils.metrics import smape, weighted_smape_np, PRIORITY_OUTLETS
from LGHackerton.preprocess import Preprocessor, L as DEFAULT_L, H as DEFAULT_H
from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SALES_FILLED_COL
from .train import (
    trunc_nb_nll,
    hurdle_nll,
    WeightedSMAPELoss,
    combine_predictions_thresholded,
)

SINH_MAX = 15.0


def temperature_schedule(
    epoch: int,
    max_epochs: int,
    start: float,
    end: float,
    anneal_epochs: int,
) -> float:
    """Linearly anneal a value from ``start`` to ``end``.

    The value is decayed from ``start`` to ``end`` over
    ``max_epochs - anneal_epochs`` steps and then held constant
    for the remaining ``anneal_epochs`` epochs.
    """

    span = max(1, max_epochs - anneal_epochs)
    if epoch >= span:
        return end
    frac = float(epoch) / float(span)
    return start + (end - start) * frac

@dataclass
class PatchTSTParams:
    """Hyperparameters controlling PatchTST training.

    Attributes
    ----------
    d_model : int
        Dimension of the model embeddings.
    n_heads : int
        Number of attention heads.
    depth : int
        Number of Transformer blocks.
    patch_len : int
        Length of each temporal patch.
    stride : int
        Stride between successive patches.
    dropout : float
        Dropout applied inside Transformer blocks.
    id_embed_dim : int
        Dimensionality of optional series ID embeddings. Set to ``0`` to disable.
    static_embed_dim : int
        Dimensionality of embeddings used for static categorical features.
    static_cardinalities : Optional[List[int]]
        Number of classes for each static categorical feature. Computed from
        the data and used to size embedding layers.
    enable_covariates : bool
        Whether input windows contain additional covariate channels beyond the
        primary target series.
    input_dim : int
        Total number of channels in the input window. This is inferred from the
        data when training.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    grad_clip : float
        Maximum norm for gradient clipping.
    batch_size : int
        Mini-batch size used during training and inference.
    max_epochs : int
        Maximum number of training epochs per fold.
    patience : int
        Early stopping patience measured in epochs.
    lambda_nb : float
        Weight for the negative binomial regression loss component.
    lambda_s : float
        Scaling applied to the sparsity regularisation term.
    lambda_smooth : float
        Weight for temporal smoothness.
    tau : float
        Threshold applied to the classifier probability for gating.
    temp_start : float
        Initial temperature for the gating schedule.
    temp_end : float
        Final temperature after annealing.
    temp_anneal_epochs : int
        Number of final epochs to keep ``temp_end`` constant.
    scaler : str
        Scaling strategy ("per_series" or "revin").
    channel_fusion : str
        Strategy to combine channel representations ("attention", "linear" or "mean").
        val_policy : str
            Validation split policy.
    val_ratio : float
        Ratio for the validation split when ``val_policy`` is ``"ratio"``.
    val_span_days : int
        Number of days used for validation when ``val_policy`` is ``"span"``.
    rocv_n_folds : int
        Rolling-origin cross-validation folds.
    rocv_stride_days : int
        Stride between ROCV folds in days.
    rocv_val_span_days : int
        Validation span in days for ROCV.
    purge_days : int
        Purge window between training and validation sets.
    min_val_samples : int
        Minimum number of samples in the validation set.
    num_workers : int
        Number of worker processes used by :class:`torch.utils.data.DataLoader`.
    """

    d_model: int = 512
    n_heads: int = 8
    depth: int = 4
    patch_len: int = 4
    stride: int = 1
    dropout: float = 0.1
    id_embed_dim: int = 16
    static_embed_dim: int = 16
    static_cardinalities: Optional[List[int]] = None
    enable_covariates: bool = True
    input_dim: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    lambda_nb: float = 1.0
    lambda_s: float = 0.05
    lambda_smooth: float = 0.0
    tau: float = 0.5
    temp_start: float = 1.0
    temp_end: float = 0.05
    temp_anneal_epochs: int = 5
    scaler: str = "per_series"
    channel_fusion: str = "attention"
    num_workers: int = 0
    # validation settings (mirrors TrainConfig)
    val_policy: str = "ratio"
    val_ratio: float = 0.2
    val_span_days: int = 28
    rocv_n_folds: int = 3
    rocv_stride_days: int = 7
    rocv_val_span_days: int = 7
    purge_days: int = 0
    min_val_samples: int = 28

class _SeriesDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        S: Optional[np.ndarray] = None,
        series_ids: Optional[Iterable[int]] = None,
        scaler: str = "per_series",
        dyn_idx: Optional[Iterable[int]] = None,
        static_idx: Optional[Iterable[int]] = None,
    ):
        X = X.astype(np.float32)
        if X.ndim == 2:  # (N,L) -> (N,L,1)
            X = X[..., None]
        self.X = X
        self.y = y.astype(np.float32)
        if S is None:
            S = np.zeros((len(X), 0), dtype=np.int64)
        self.S = S.astype(np.int64)
        if series_ids is None:
            series_ids = [0 for _ in range(len(X))]
        self.sids = np.array(list(series_ids), dtype=np.int64)
        self.scaler = scaler
        # handle dynamic/static channel indices
        if isinstance(dyn_idx, dict):
            dyn_idx = dyn_idx.values()
        if isinstance(static_idx, dict):
            static_idx = static_idx.values()
        if dyn_idx is None or len(dyn_idx) == 0:
            raise ValueError("dynamic channel indices required")
        self.dyn_idx = sorted(dyn_idx)
        self.static_idx = sorted(static_idx) if static_idx is not None else []

        # Pre-compute mean and std for each dynamic channel
        dyn = self.X[:, :, self.dyn_idx]
        mu = dyn.mean(axis=1)
        std = dyn.std(axis=1)
        std[std == 0] = 1.0
        self.mu = mu.astype(np.float32)
        self.std = std.astype(np.float32)
        # base channel statistics (first dynamic channel)
        self.mu_base = self.mu[:, 0]
        self.std_base = self.std[:, 0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        mu = np.float32(self.mu[idx])
        std = np.float32(self.std[idx])
        # normalise all dynamic channels using their own statistics
        x[:, self.dyn_idx] = (x[:, self.dyn_idx] - mu) / std
        y = self.y[idx]
        if self.scaler == "revin":
            y = (y - self.mu_base[idx]) / self.std_base[idx]
        static_codes = self.S[idx].copy()
        return (
            x,
            y,
            int(self.sids[idx]),
            np.float32(self.mu_base[idx]),
            np.float32(self.std_base[idx]),
            static_codes,
        )


def _make_rocv_slices(
    label_dates: np.ndarray,
    n_folds: int,
    stride: int,
    span: int,
    purge: np.timedelta64,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate ROCV train/validation boolean masks.

    Parameters
    ----------
    label_dates : np.ndarray
        Array of label dates sorted in chronological order.
    n_folds : int
        Number of ROCV folds to generate.
    stride : int
        Distance between consecutive validation windows in days.
    span : int
        Validation window length in days.
    purge : np.timedelta64
        Purge gap applied between training and validation windows.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_mask, val_mask) boolean arrays.
    """

    n = len(label_dates)
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    used = np.zeros(n, dtype=bool)
    dates = np.sort(np.unique(label_dates))

    for i in range(n_folds):
        end = dates[-1] - np.timedelta64(i * stride, "D")
        start = end - np.timedelta64(span - 1, "D")
        va_mask = (label_dates >= start) & (label_dates <= end) & (~used)
        tr_mask = label_dates < (start - purge)
        if va_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        assert label_dates[tr_mask].max() < label_dates[va_mask].min() - purge
        slices.append((tr_mask, va_mask))
        used |= va_mask

    return slices

if TORCH_OK:
    class PatchTSTBlock(nn.Module):
        def __init__(self, d_model, n_heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
            )
            self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        def forward(self, x):
            a,_ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
            x = x + a
            x = x + self.ff(self.n2(x))
            return x

    class PatchTSTNet(nn.Module):
        def __init__(
            self,
            L: int,
            H: int,
            d_model: int,
            n_heads: int,
            depth: int,
            patch_len: int,
            stride: int,
            dropout: float,
            id_embed_dim: int = 0,
            num_series: int = 0,
            input_dim: int = 1,
            static_cardinalities: Optional[Iterable[int]] = None,
            static_emb_dim: int = 16,
            channel_fusion: str = "attention",
        ):
            super().__init__()
            self.L, self.H = L, H
            self.patch_len = patch_len
            self.stride = stride
            n_patches = 1 + (L - patch_len) // stride
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, 1, d_model))
            # project each channel's patch separately
            self.proj = nn.Linear(patch_len, d_model)
            self.blocks = nn.ModuleList([PatchTSTBlock(d_model, n_heads, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(d_model)
            self.channel_fusion = channel_fusion
            if channel_fusion == "attention":
                self.channel_query = nn.Parameter(torch.randn(1, 1, d_model))
                self.channel_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            elif channel_fusion == "linear":
                self.channel_lin = nn.Linear(d_model, 1)
            else:
                self.channel_attn = None
                self.channel_lin = None
            # separate heads for classification and regression
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * H)
            )
            self.clf_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, H)
            )
            if id_embed_dim > 0 and num_series > 0:
                self.id_embed = nn.Embedding(num_series, id_embed_dim)
                self.id_proj = nn.Linear(id_embed_dim, d_model) if id_embed_dim != d_model else None
            else:
                self.id_embed = None
                self.id_proj = None

            self.static_embeds = nn.ModuleList()
            self.static_mlp = None
            if static_cardinalities:
                self.static_embeds = nn.ModuleList(
                    [nn.Embedding(nc + 1, static_emb_dim, padding_idx=0) for nc in static_cardinalities]
                )
                in_dim = len(static_cardinalities) * static_emb_dim
                self.static_mlp = nn.Sequential(
                    nn.Linear(in_dim, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model * 2),
                )

        def forward(self, x, sid_idx=None, static_codes=None):
            B, L, C = x.shape
            p = x.unfold(1, self.patch_len, self.stride).contiguous()  # (B, n_patches, C, patch_len)
            z = self.proj(p) + self.pos_embed  # (B, n_patches, C, d_model)
            if self.id_embed is not None and sid_idx is not None:
                e = self.id_embed(sid_idx)
                if self.id_proj is not None:
                    e = self.id_proj(e)
                z = z + e.unsqueeze(1).unsqueeze(2)
            if self.static_mlp is not None and static_codes is not None:
                embs = []
                for i, emb in enumerate(self.static_embeds):
                    embs.append(emb(static_codes[:, i].long()))
                stat = torch.cat(embs, dim=-1)
                gamma_beta = self.static_mlp(stat)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                z = gamma.unsqueeze(1).unsqueeze(1) * z + beta.unsqueeze(1).unsqueeze(1)
            B, n_patches, C, _ = z.shape
            z = z.view(B * C, n_patches, -1)
            for blk in self.blocks:
                z = blk(z)
            z = z.view(B, C, n_patches, -1)
            z = self.norm(z).mean(2)  # (B, C, d_model)
            if self.channel_fusion == "attention" and self.channel_attn is not None:
                q = self.channel_query.expand(B, -1, -1)
                z, _ = self.channel_attn(q, z, z)
                z = z.squeeze(1)
            elif self.channel_fusion == "linear" and self.channel_lin is not None:
                w = torch.softmax(self.channel_lin(z).squeeze(-1), dim=1).unsqueeze(-1)
                z = (w * z).sum(dim=1)
            else:  # mean
                z = z.mean(dim=1)
            params = self.reg_head(z)
            mu_raw, kappa_raw = params.chunk(2, dim=-1)
            logits = self.clf_head(z)
            return logits, mu_raw, kappa_raw

class PatchTSTTrainer(BaseModel):
    """Trainer for PatchTST models.

    The class exposes a small callback API that fires whenever rolling-origin
    cross-validation (ROCV) folds are generated. External consumers may
    register callbacks via :meth:`register_rocv_callback` to receive the seed,
    fold index and corresponding train/validation masks. Errors raised inside
    callbacks are isolated from training by being converted to warnings.
    """
    #: column suffix used for prediction outputs
    prediction_column_name = "patch"

    #: Callbacks executed for each ROCV fold immediately after slices are
    #: computed and before training begins. Each callback must accept
    #: ``(seed, fold_idx, train_mask, val_mask, cfg)``. Exceptions are caught
    #: and downgraded to warnings so that failures in user code do not halt
    #: model training.
    _rocv_callbacks: List[Callable[[int, int, np.ndarray, np.ndarray, TrainConfig], None]] = []

    @classmethod
    def register_rocv_callback(
        cls,
        cb: Callable[[int, int, np.ndarray, np.ndarray, TrainConfig], None],
    ) -> None:
        """Register a callback invoked for every ROCV fold.

        Parameters
        ----------
        cb : Callable[[int, int, np.ndarray, np.ndarray, TrainConfig], None]
            Function accepting ``(seed, fold_idx, train_mask, val_mask, cfg)``.

        Notes
        -----
        Exceptions raised by callbacks are caught and reported as warnings so
        that training can proceed uninterrupted.
        """

        cls._rocv_callbacks.append(cb)

    @classmethod
    def _notify_rocv_callbacks(
        cls,
        seed: int,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        cfg: TrainConfig,
    ) -> None:
        """Execute registered ROCV callbacks.

        Each callback is called with ``(seed, fold_idx, train_mask, val_mask, cfg)``.
        Errors are swallowed and surfaced as warnings to avoid aborting
        training when a callback fails.
        """

        for i, (tr_mask, va_mask) in enumerate(folds):
            for cb in cls._rocv_callbacks:
                try:
                    cb(seed, i, tr_mask, va_mask, cfg)
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.warn(
                        f"ROCV callback {getattr(cb, '__name__', repr(cb))} failed: {exc}"
                    )

    @staticmethod
    def build_dataset(pp: Preprocessor, df_full: pd.DataFrame, input_len: int | None = None):
        if input_len is not None:
            horizon = getattr(pp.windowizer, "H", DEFAULT_H)
            pp.windowizer.L = input_len
            pp.windowizer.H = horizon
        X_dyn, S_stat, Y, sids, dates = pp.build_patch_train(df_full)
        # Guarantee channel index maps exist on the preprocessor
        if not hasattr(pp, "patch_dynamic_idx"):
            pp.patch_dynamic_idx = {}
        if not hasattr(pp, "patch_static_idx"):
            pp.patch_static_idx = {}
        return X_dyn, S_stat, Y, sids, dates

    @staticmethod
    def build_eval_dataset(pp: Preprocessor, df_full: pd.DataFrame):
        """Build evaluation dataset for prediction."""
        X_dyn, S_stat, sids, dates = pp.build_patch_eval(df_full)
        if not hasattr(pp, "patch_dynamic_idx"):
            pp.patch_dynamic_idx = {}
        if not hasattr(pp, "patch_static_idx"):
            pp.patch_static_idx = {}
        return X_dyn, S_stat, sids, dates

    def __init__(
        self,
        params: PatchTSTParams,
        model_dir: str,
        device: str,
        L: int = DEFAULT_L,
        H: int = DEFAULT_H,
    ):
        super().__init__(model_params=asdict(params), model_dir=model_dir)
        self.params = params
        self.L = L
        self.H = H
        self.models: List[Any] = []
        self.device = device  # 'cpu', 'cuda', or 'mps'
        self.id2idx = {}
        self.idx2id = []
        self.oof_records: List[Dict[str, Any]] = []
        # Channel index maps for dynamic/static features
        self.dynamic_idx_map: Dict[str, int] = {}
        self.static_idx_map: Dict[str, int] = {}
        # calibration storage and calibrated tau
        self.calib_records: List[Tuple[float, float, float, float, str, int]] = []
        self.tau: float | torch.Tensor = self.params.tau

    def _ensure_torch(self):
        if not TORCH_OK: raise RuntimeError(f"PyTorch not available: {_TORCH_ERR}")
    def train(
        self,
        X_train: np.ndarray,
        S_train: np.ndarray,
        y_train: np.ndarray,
        series_ids: np.ndarray,
        label_dates: np.ndarray,
        cfg: TrainConfig,
        preprocessors: Optional[List[Any]] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> None:
        """Train PatchTST models under various validation policies.

        Parameters
        ----------
        preprocessors : Optional[List[Any]]
            Optional list of fold-specific preprocessors or artifacts.
        trial : Optional[optuna.Trial]
            Optuna trial for reporting intermediate values and pruning.
        """
        self.preprocessors = preprocessors
        if preprocessors:
            pp0 = preprocessors[0]
            self.dynamic_idx_map = getattr(pp0, "patch_dynamic_idx", {})
            self.static_idx_map = getattr(pp0, "patch_static_idx", {})
        self._ensure_torch(); import torch
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Using gradient clipping max_norm={self.params.grad_clip}")
        self.oof_records = []
        self.calib_records = []
        self.calib_priority_weight = cfg.priority_weight
        order = np.argsort(label_dates)
        X_train = X_train[order]
        S_train = S_train[order]
        y_train = y_train[order]
        series_ids = np.array(series_ids)[order]
        label_dates = label_dates[order]
        if X_train.ndim == 3:
            self.params.input_dim = X_train.shape[2]
        else:
            self.params.input_dim = 1
        self.params.enable_covariates = self.params.input_dim > 1
        self.model_params["input_dim"] = self.params.input_dim
        self.model_params["enable_covariates"] = self.params.enable_covariates
        unique_sids = sorted(set(series_ids))
        self.id2idx = {sid:i for i,sid in enumerate(unique_sids)}
        self.idx2id = unique_sids
        series_idx = np.array([self.id2idx[s] for s in series_ids], dtype=np.int64)
        if S_train.size > 0:
            static_cardinalities = [int(S_train[:, j].max()) for j in range(S_train.shape[1])]
        else:
            static_cardinalities = []
        self.params.static_cardinalities = static_cardinalities
        self.model_params["static_cardinalities"] = static_cardinalities
        # compute per-series weights based on zero ratio
        self.series_weight_map = {}
        for sid, idx in self.id2idx.items():
            y_series = y_train[series_idx == idx].ravel()
            zero_ratio = float(np.mean(y_series == 0)) if y_series.size > 0 else 0.0
            self.series_weight_map[sid] = cfg.non_zero_weight * (1.0 + zero_ratio)
        self.series_weight_tensor = torch.tensor(
            [self.series_weight_map[sid] for sid in self.idx2id],
            device=self.device,
            dtype=torch.float32,
        )
        n = len(label_dates)
        min_samples = max(cfg.min_val_samples, 5 * self.params.batch_size)
        purge_days = cfg.purge_days if cfg.purge_days > 0 else (
            self.L + self.H if cfg.purge_mode == "L+H" else self.L
        )
        purge = np.timedelta64(purge_days, "D")
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        if cfg.val_policy == "ratio":
            n_val = max(min_samples, int(cfg.val_ratio * n))
            n_val = min(n_val, n - 1)
            cutoff_date = label_dates[-n_val]
            va_mask = label_dates >= cutoff_date
            tr_mask = label_dates < (cutoff_date - purge)
            if tr_mask.sum() > 0 and va_mask.sum() >= min_samples:
                folds = [(tr_mask, va_mask)]
            else:
                purge_win = cfg.purge_days if cfg.purge_days > 0 else (
                    self.L + self.H if cfg.purge_mode == "L+H" else self.L
                )
                purge_win = max(1, purge_win)
                n_val = min(n_val, n - purge_win - 1)
                if n_val < min_samples:
                    if n - purge_win - 1 >= min_samples:
                        n_val = min_samples
                    else:
                        warnings.warn(
                            f"Validation set only {n_val} samples; minimum is {min_samples}"
                        )
                n_tr = max(1, n - n_val - purge_win)
                idx = np.arange(n)
                tr_mask = idx < n_tr
                va_mask = idx >= n_tr + purge_win
                if tr_mask.sum() > 0 and va_mask.sum() > 0:
                    folds = [(tr_mask, va_mask)]
        elif cfg.val_policy == "span":
            end = label_dates[-1]
            start = end - np.timedelta64(cfg.val_span_days - 1, "D")
            va_mask = label_dates >= start
            if va_mask.sum() < min_samples:
                start = label_dates[-min_samples]
                va_mask = label_dates >= start
            tr_mask = label_dates < (start - purge)
            if tr_mask.sum() > 0 and va_mask.sum() >= min_samples:
                folds = [(tr_mask, va_mask)]
            else:
                purge_win = cfg.purge_days if cfg.purge_days > 0 else (
                    self.L + self.H if cfg.purge_mode == "L+H" else self.L
                )
                purge_win = max(1, purge_win)
                n_val = min(min_samples, n - purge_win - 1)
                if n_val < min_samples and n - purge_win - 1 < min_samples:
                    warnings.warn(
                        f"Validation set only {n_val} samples; minimum is {min_samples}"
                    )
                n_tr = max(1, n - n_val - purge_win)
                idx = np.arange(n)
                tr_mask = idx < n_tr
                va_mask = idx >= n_tr + purge_win
                if tr_mask.sum() > 0 and va_mask.sum() > 0:
                    folds = [(tr_mask, va_mask)]
        elif cfg.val_policy == "rocv":
            span = cfg.rocv_val_span_days
            n_folds = cfg.rocv_n_folds
            while True:
                folds = _make_rocv_slices(label_dates, n_folds, cfg.rocv_stride_days, span, purge)
                if folds and min(f.sum() for _, f in folds) >= min_samples:
                    break
                if span > 1:
                    span = max(1, span // 2)
                    continue
                if n_folds > 1:
                    n_folds -= 1
                    continue
                folds = _make_rocv_slices(label_dates, 1, cfg.rocv_stride_days, span, purge)
                break
            # Inform any registered listeners about the ROCV folds. Callbacks
            # run once per fold with ``(seed, fold_idx, train_mask, val_mask, cfg)``
            # and failures inside callbacks only generate warnings.
            self._notify_rocv_callbacks(cfg.seed, folds, cfg)
        else:
            raise ValueError(f"Unknown val_policy: {cfg.val_policy}")
        assert folds, "No valid folds generated"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for i, (tr_mask, va_mask) in enumerate(folds):
            pp = preprocessors[i] if preprocessors and i < len(preprocessors) else None
            dyn_idx = getattr(pp, "patch_dynamic_idx", self.dynamic_idx_map)
            stat_idx = getattr(pp, "patch_static_idx", self.static_idx_map)
            tr_ds = _SeriesDataset(
                X_train[tr_mask],
                y_train[tr_mask],
                S_train[tr_mask],
                series_idx[tr_mask],
                scaler=self.params.scaler,
                dyn_idx=dyn_idx,
                static_idx=stat_idx,
            )
            va_ds = _SeriesDataset(
                X_train[va_mask],
                y_train[va_mask],
                S_train[va_mask],
                series_idx[va_mask],
                scaler=self.params.scaler,
                dyn_idx=dyn_idx,
                static_idx=stat_idx,
            )
            pin = self.device != "cpu"
            tr_loader = DataLoader(
                tr_ds,
                batch_size=self.params.batch_size,
                shuffle=True,
                pin_memory=pin,
                num_workers=self.params.num_workers,
            )
            va_loader = DataLoader(
                va_ds,
                batch_size=self.params.batch_size,
                shuffle=False,
                pin_memory=pin,
                num_workers=self.params.num_workers,
            )
            net = PatchTSTNet(
                self.L,
                self.H,
                self.params.d_model,
                self.params.n_heads,
                self.params.depth,
                self.params.patch_len,
                self.params.stride,
                self.params.dropout,
                self.params.id_embed_dim,
                len(self.id2idx),
                self.params.input_dim,
                self.params.static_cardinalities,
                self.params.static_embed_dim,
                self.params.channel_fusion,
            ).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
            smape_loss = WeightedSMAPELoss(reduction="mean")
            best=float("inf"); best_state=None; bad=0
            stop_training = False
            for ep in range(self.params.max_epochs):
                curr_T = temperature_schedule(
                    ep,
                    self.params.max_epochs,
                    self.params.temp_start,
                    self.params.temp_end,
                    self.params.temp_anneal_epochs,
                )
                net.train()
                nb_sum = clf_sum = s_sum = sm_sum = 0.0
                batch_count = 0
                for xb, yb, sb, mu_s, std_s, static_codes in tr_loader:
                    xb = xb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    sb = sb.to(self.device, non_blocking=pin)
                    mu_s = mu_s.to(self.device, non_blocking=pin)
                    std_s = std_s.to(self.device, non_blocking=pin)
                    static_codes = static_codes.to(self.device, non_blocking=pin)
                    opt.zero_grad()
                    logits, mu_raw, kappa_raw = net(xb, sb, static_codes)
                    mu = F.softplus(mu_raw) + 1e-6
                    kappa = F.softplus(kappa_raw) + 1e-6
                    y_raw = yb
                    mu_unscaled = mu
                    if self.params.scaler == "revin":
                        y_raw = y_raw * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                    y_raw = torch.clamp(y_raw, -SINH_MAX, SINH_MAX)
                    mu_unscaled = torch.clamp(mu_unscaled, -SINH_MAX, SINH_MAX)
                    y_count = torch.sinh(y_raw)
                    mu_count = torch.sinh(mu_unscaled)
                    M = y_count > 0  # existing mask of positive counts
                    z = M.float()
                    series_w = self.series_weight_tensor[sb]
                    w = torch.where(M, series_w.unsqueeze(1), torch.ones_like(y_count))
                    if cfg.use_weighted_loss:
                        outlets = [self.idx2id[int(i)].split("::")[0] for i in sb.cpu().tolist()]
                        priority_w = torch.tensor(
                            [cfg.priority_weight if o in PRIORITY_OUTLETS else 1.0 for o in outlets],
                            device=self.device,
                        ).view(-1, 1)
                        w = w * priority_w
                    nb_loss = torch.zeros_like(y_count)
                    if M.any():
                        nb_loss[M] = trunc_nb_nll(
                            y_count[M],
                            mu_count[M],
                            kappa[M],
                        )
                    L_nb = (nb_loss * w * z).sum() / torch.clamp(z.sum(), min=1.0)
                    L_clf = hurdle_nll(logits, z, w)
                    y_hat = combine_predictions_thresholded(
                        logits=logits,
                        mu=mu_count,
                        kappa=kappa,
                        tau=self.params.tau,
                        temperature=curr_T,
                    )
                    L_s = smape_loss(y_hat, y_count, w)
                    p = torch.sigmoid(logits)
                    L_sm = torch.mean(torch.abs(p[:, 1:] - p[:, :-1]))
                    loss = (
                        self.params.lambda_nb * L_nb
                        + L_clf
                        + self.params.lambda_s * L_s
                        + self.params.lambda_smooth * L_sm
                    )
                    if not torch.isfinite(loss):
                        warnings.warn("Non-finite loss; stopping training")
                        stop_training = True
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.params.grad_clip)
                    opt.step()
                    nb_sum += float(L_nb.item())
                    clf_sum += float(L_clf.item())
                    s_sum += float(L_s.item())
                    sm_sum += float(L_sm.item())
                    batch_count += 1
                print(
                    f"Fold {i} Epoch {ep} Train: L_nb={nb_sum/batch_count:.4f} "
                    f"L_clf={clf_sum/batch_count:.4f} L_s={s_sum/batch_count:.4f} "
                    f"L_sm={sm_sum/batch_count:.4f}"
                )
                if stop_training:
                    break
                net.eval()
                P = []
                T = []
                S = []
                with torch.no_grad():
                    for xb, yb, sb, mu_s, std_s, static_codes in va_loader:
                        xb = xb.to(self.device, non_blocking=pin)
                        yb = yb.to(self.device, non_blocking=pin)
                        sb = sb.to(self.device, non_blocking=pin)
                        mu_s = mu_s.to(self.device, non_blocking=pin)
                        std_s = std_s.to(self.device, non_blocking=pin)
                        static_codes = static_codes.to(self.device, non_blocking=pin)
                        logits, mu_raw, kappa_raw = net(xb, sb, static_codes)
                        p = torch.sigmoid(logits)
                        mu = F.softplus(mu_raw) + 1e-6
                        kappa = F.softplus(kappa_raw) + 1e-6
                        mu_unscaled = mu
                        if self.params.scaler == "revin":
                            mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                            yb = yb * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        yb = torch.clamp(yb, -SINH_MAX, SINH_MAX)
                        mu_unscaled = torch.clamp(mu_unscaled, -SINH_MAX, SINH_MAX)
                        yb_count = torch.sinh(yb)
                        mu_count = torch.sinh(mu_unscaled)
                        _ = combine_predictions_thresholded(
                            logits=logits,
                            mu=mu_count,
                            kappa=kappa,
                            tau=self.params.tau,
                            temperature=curr_T,
                        )
                        final = combine_predictions_thresholded(
                            p=p,
                            mu=mu_count,
                            kappa=kappa,
                            tau=self.params.tau,
                            temperature=None,
                        )
                        P.append(final.cpu().numpy())
                        T.append(yb_count.cpu().numpy())
                        S.extend(sb.cpu().tolist())
                y_pred = np.clip(np.concatenate(P, 0), 0, None)
                y_true = np.concatenate(T, 0)
                series_ids_val = np.array([self.idx2id[int(i)] for i in S])
                outlets = [sid.split("::")[0] for sid in series_ids_val]
                eps = 1e-8
                w_val = weighted_smape_np(
                    y_true.ravel(),
                    y_pred.ravel(),
                    np.repeat(outlets, self.H),
                    cfg.priority_weight,
                    eps,
                    series_weight_map=self.series_weight_map,
                    series_ids=np.repeat(series_ids_val, self.H),
                )
                s_val = smape(y_true.ravel(), y_pred.ravel(), eps)
                mae_val = float(np.mean(np.abs(y_true - y_pred)))
                if not np.isfinite(w_val):
                    warnings.warn("Non-finite validation metric; treating as failure")
                    w_val = float("inf")
                if w_val + 1e-8 < best:
                    best = w_val; best_state = net.state_dict(); bad = 0
                else:
                    bad += 1
                print(
                    f"Fold {i} Epoch {ep}: wSMAPE={w_val:.4f} SMAPE={s_val:.4f} MAE={mae_val:.4f}"
                )
                if trial is not None:
                    trial.report(w_val, step=ep)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if bad >= self.params.patience:
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            # compute OOF predictions for this fold
            net.eval()
            P = []
            Y = []
            S = []
            with torch.no_grad():
                for xb, yb, sb, mu_s, std_s, static_codes in va_loader:
                    xb = xb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    sb = sb.to(self.device, non_blocking=pin)
                    mu_s = mu_s.to(self.device, non_blocking=pin)
                    std_s = std_s.to(self.device, non_blocking=pin)
                    static_codes = static_codes.to(self.device, non_blocking=pin)
                    logits, mu_raw, kappa_raw = net(xb, sb, static_codes)
                    p = torch.sigmoid(logits)
                    mu = F.softplus(mu_raw) + 1e-6
                    kappa = F.softplus(kappa_raw) + 1e-6
                    mu_unscaled = mu
                    if self.params.scaler == "revin":
                        mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        yb = yb * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                    yb = torch.clamp(yb, -SINH_MAX, SINH_MAX)
                    mu_unscaled = torch.clamp(mu_unscaled, -SINH_MAX, SINH_MAX)
                    yb_count = torch.sinh(yb)
                    mu_count = torch.sinh(mu_unscaled)
                    _ = combine_predictions_thresholded(
                        logits=logits,
                        mu=mu_count,
                        kappa=kappa,
                        tau=self.params.tau,
                        temperature=curr_T,
                    )
                    final = combine_predictions_thresholded(
                        p=p,
                        mu=mu_count,
                        kappa=kappa,
                        tau=self.params.tau,
                        temperature=None,
                    )
                    # record calibration tuples
                    logits_np = logits.cpu().numpy()
                    mu_np = mu_count.cpu().numpy()
                    kappa_np = kappa.cpu().numpy()
                    y_np = yb_count.cpu().numpy()
                    sid_list = [self.idx2id[int(i)] for i in sb.cpu().tolist()]
                    for idx_rec, sid in enumerate(sid_list):
                        for h in range(self.H):
                            self.calib_records.append(
                                (
                                    float(logits_np[idx_rec, h]),
                                    float(mu_np[idx_rec, h]),
                                    float(kappa_np[idx_rec, h]),
                                    float(y_np[idx_rec, h]),
                                    sid,
                                    h + 1,
                                )
                            )
                    final = final.cpu()
                    P.append(final.numpy())
                    Y.append(y_np)
                    S.extend(sb.cpu().tolist())
            y_pred = np.clip(np.concatenate(P, 0), 0, None)
            y_true = np.concatenate(Y, 0)
            series_ids_val = np.array([self.idx2id[int(i)] for i in S])
            oof_df = pd.DataFrame(
                {
                    "series_id": np.repeat(series_ids_val, self.H),
                    "h": np.tile(np.arange(1, self.H + 1), len(series_ids_val)),
                    "y": y_true.reshape(-1),
                    "yhat": y_pred.reshape(-1),
                }
            )
            self.oof_records.extend(oof_df.to_dict("records"))
            self.models.append(net)
        # calibrate gating threshold using collected records
        self.calibrate_tau()
        # produce reliability plot from calibration records
        self.plot_reliability()
        self.save(os.path.join(self.model_dir,"patchtst.pt"))
    def predict(
        self,
        X_eval: np.ndarray,
        S_eval: np.ndarray,
        series_idx: Optional[Iterable[int]] = None,
        dyn_idx: Optional[Iterable[int]] = None,
        static_idx: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        self._ensure_torch(); import torch
        if not self.models:
            raise RuntimeError("Model not loaded.")
        if series_idx is None:
            series_idx = np.zeros(len(X_eval), dtype=np.int64)
        ds = _SeriesDataset(
            X_eval,
            np.zeros((X_eval.shape[0], self.H), dtype=np.float32),
            S_eval,
            series_idx,
            scaler=self.params.scaler,
            dyn_idx=dyn_idx or self.dynamic_idx_map,
            static_idx=static_idx or self.static_idx_map,
        )
        pin = self.device != "cpu"
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.params.batch_size,
            shuffle=False,
            pin_memory=pin,
            num_workers=self.params.num_workers,
        )
        outs = []
        with torch.no_grad():
            for xb, _, sb, mu_s, std_s, static_codes in loader:
                xb = xb.to(self.device, non_blocking=pin)
                sb = sb.to(self.device, non_blocking=pin)
                mu_s = mu_s.to(self.device, non_blocking=pin)
                std_s = std_s.to(self.device, non_blocking=pin)
                static_codes = static_codes.to(self.device, non_blocking=pin)
                preds = [m(xb, sb, static_codes) for m in self.models]
                logits, mu_raw, kappa_raw = zip(*preds)
                logits = torch.stack(logits)
                p = torch.sigmoid(logits)
                mu = torch.stack([F.softplus(m) + 1e-6 for m in mu_raw])
                kappa = torch.stack([F.softplus(k) + 1e-6 for k in kappa_raw])
                if self.params.scaler == "revin":
                    mu = mu * std_s.view(1, -1, 1) + mu_s.view(1, -1, 1)
                mu = torch.clamp(mu, -SINH_MAX, SINH_MAX)
                mu = torch.sinh(mu)
                out = combine_predictions_thresholded(
                    p=p,
                    mu=mu,
                    kappa=kappa,
                    tau=self.tau,
                    temperature=None,
                ).mean(0)
                outs.append(out.cpu().numpy())
        yhat = np.clip(np.concatenate(outs, 0), 0, None)
        return yhat

    def predict_df(self, eval_df):
        """Return conditional-mean prediction dataframe for PatchTST."""
        X_eval, S_eval, sids, _ = eval_df
        sid_idx = np.array([self.id2idx.get(sid, 0) for sid in sids])
        y_pred = self.predict(
            X_eval,
            S_eval,
            sid_idx,
            dyn_idx=self.dynamic_idx_map,
            static_idx=self.static_idx_map,
        )
        reps = np.repeat(sids, self.H)
        hs = np.tile(np.arange(1, self.H + 1), len(sids))
        out = pd.DataFrame({"series_id": reps, "h": hs, "yhat_patch": y_pred.reshape(-1)})
        return out

    def calibrate_tau(
        self,
        tau_grid: Iterable[float] | None = None,
        per_horizon: bool = False,
    ) -> float | torch.Tensor:
        """Calibrate gating threshold ``tau`` using stored validation outputs."""
        self._ensure_torch(); import torch
        if not self.calib_records:
            self.tau = self.params.tau
            return self.tau
        if tau_grid is None:
            tau_grid = np.linspace(0.05, 0.95, 19)
        records = self.calib_records
        logits = torch.tensor([r[0] for r in records], dtype=torch.float32)
        mu = torch.tensor([r[1] for r in records], dtype=torch.float32)
        kappa = torch.tensor([r[2] for r in records], dtype=torch.float32)
        y_true = torch.tensor([r[3] for r in records], dtype=torch.float32)
        series_ids = np.array([r[4] for r in records])
        hs = np.array([r[5] for r in records], dtype=int)
        outlets = np.array([sid.split("::")[0] for sid in series_ids])
        priority = getattr(self, "calib_priority_weight", 1.0)

        def _search(mask: np.ndarray) -> float:
            best_tau = float(tau_grid[0])
            best_score = float("inf")
            lg = logits[mask]
            mu_u = mu[mask]
            kp = kappa[mask]
            yt = y_true[mask]
            outs = outlets[mask]
            sids = series_ids[mask]
            p = torch.sigmoid(lg)
            for t in tau_grid:
                pred = combine_predictions_thresholded(
                    p=p,
                    mu=mu_u,
                    kappa=kp,
                    tau=float(t),
                    temperature=None,
                ).cpu().numpy()
                score = weighted_smape_np(
                    yt.cpu().numpy(),
                    pred,
                    outs,
                    priority,
                    1e-8,
                    series_weight_map=self.series_weight_map,
                    series_ids=sids,
                )
                if score < best_score:
                    best_score = score
                    best_tau = float(t)
            return best_tau

        if per_horizon:
            taus = []
            for h in range(1, self.H + 1):
                mask = hs == h
                if mask.any():
                    taus.append(_search(mask))
                else:
                    taus.append(self.params.tau)
            self.tau = torch.tensor(taus, dtype=torch.float32)
        else:
            self.tau = _search(np.ones_like(hs, dtype=bool))
        return self.tau

    def plot_reliability(
        self,
        records: Optional[List[Tuple[float, float, float, float, str, int]]] = None,
        bins: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a reliability diagram for gating probabilities.

        Parameters
        ----------
        records : Optional list of calibration tuples
            Each record is expected to contain ``(logit, mu, kappa, y, sid, h)``.
            If ``None``, ``self.calib_records`` is used.
        bins : int, default 10
            Number of probability bins.
        save_path : Optional[str]
            Destination path for the generated figure. Defaults to
            ``model_dir/reliability.png``.
        """

        if records is None:
            records = self.calib_records
        if not records:
            warnings.warn("No calibration records to plot reliability.")
            return
        import matplotlib.pyplot as plt

        logits = np.array([r[0] for r in records])
        y_true = np.array([r[3] for r in records])
        probs = 1.0 / (1.0 + np.exp(-logits))
        z = (y_true > 0).astype(float)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_idx = np.digitize(probs, bin_edges, right=True)
        exp, act = [], []
        for i in range(1, bins + 1):
            mask = bin_idx == i
            if mask.any():
                exp.append(probs[mask].mean())
                act.append(z[mask].mean())
        plt.figure()
        plt.plot(exp, act, marker="o")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical frequency")
        plt.title("Reliability Diagram")
        save_path = save_path or os.path.join(self.model_dir, "reliability.png")
        plt.savefig(save_path)
        plt.close()

    def get_oof(self) -> pd.DataFrame:
        return pd.DataFrame(self.oof_records)

    def save(self, path:str)->None:
        if not TORCH_OK: return
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index = []
        for i, m in enumerate(self.models):
            fpath = os.path.join(self.model_dir, f"patchtst_fold{i}.pt")
            torch.save(m.state_dict(), fpath)
            index.append(os.path.basename(fpath))
        meta = {
            "params": self.model_params,
            "L": self.L,
            "H": self.H,
            "index": index,
            "id2idx": self.id2idx,
            "patch_dynamic_idx": self.dynamic_idx_map,
            "patch_static_idx": self.static_idx_map,
            "static_cardinalities": self.params.static_cardinalities,
            "tau": self.tau.tolist() if isinstance(self.tau, torch.Tensor) else self.tau,
        }
        with open(path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        torch.save(meta, path)
    def load(self, path:str)->None:
        self._ensure_torch(); import torch, json, os
        meta=path.replace(".pt",".json")
        if os.path.exists(meta):
            with open(meta,"r",encoding="utf-8") as f:
                m=json.load(f)
                self.model_params=m.get("params",self.model_params)
                # ensure embedding cardinalities are available when rebuilding
                self.model_params["static_cardinalities"] = m.get(
                    "static_cardinalities",
                    self.model_params.get("static_cardinalities", []),
                )
                self.L=int(m.get("L",self.L))
                self.H=int(m.get("H",self.H))
                index=m.get("index",[])
                self.id2idx=m.get("id2idx",{})
                self.dynamic_idx_map=m.get("patch_dynamic_idx",{})
                self.static_idx_map=m.get("patch_static_idx",{})
                tau_meta = m.get("tau", self.params.tau)
                if isinstance(tau_meta, list):
                    self.tau = torch.tensor(tau_meta, dtype=torch.float32)
                else:
                    self.tau = float(tau_meta)
        else:
            index=[]
            self.id2idx={}
            self.dynamic_idx_map={}
            self.static_idx_map={}
            self.tau = self.params.tau
        self.idx2id=[None]*len(self.id2idx)
        for sid,idx in self.id2idx.items():
            self.idx2id[int(idx)] = sid
        self.params = PatchTSTParams(**self.model_params)
        self.model_params = asdict(self.params)
        self.models=[]
        for fname in index:
            net = PatchTSTNet(
                self.L,
                self.H,
                self.params.d_model,
                self.params.n_heads,
                self.params.depth,
                self.params.patch_len,
                self.params.stride,
                self.params.dropout,
                self.params.id_embed_dim,
                len(self.id2idx),
                self.params.input_dim,
                self.params.static_cardinalities,
                self.params.static_embed_dim,
                self.params.channel_fusion,
            )
            net.load_state_dict(
                torch.load(os.path.join(self.model_dir, fname), map_location=torch.device(self.device)),
                strict=False,
            )
            net.to(self.device)
            self.models.append(net)
