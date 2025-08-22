from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
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
from .train import trunc_nb_nll, focal_loss, WeightedSMAPELoss, combine_predictions

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
    batch_size : int
        Mini-batch size used during training and inference.
    max_epochs : int
        Maximum number of training epochs per fold.
    patience : int
        Early stopping patience measured in epochs.
    lambda_nb : float
        Weight for the negative binomial regression loss component.
    lambda_clf : float
        Weight for the classifier loss component.
    lambda_s : float
        Scaling applied to the sparsity regularisation term.
    gamma : float
        Gamma parameter for focal loss.
    alpha : float
        Alpha parameter for focal loss.
    epsilon_leaky : float
        Small constant added for numerical stability in leaky operations.
    scaler : str
        Scaling strategy ("per_series" or "revin").
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

    d_model: int = 128
    n_heads: int = 8
    depth: int = 4
    patch_len: int = 4
    stride: int = 1
    dropout: float = 0.1
    id_embed_dim: int = 16
    enable_covariates: bool = False
    input_dim: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    lambda_nb: float = 1.0
    lambda_clf: float = 1.0
    lambda_s: float = 0.05
    gamma: float = 2.0
    alpha: float = 0.25
    epsilon_leaky: float = 0.0
    scaler: str = "per_series"
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
        if series_ids is None:
            series_ids = [0 for _ in range(len(X))]
        self.sids = np.array(list(series_ids), dtype=np.int64)
        self.scaler = scaler
        # handle dynamic/static channel indices
        if isinstance(dyn_idx, dict):
            dyn_idx = dyn_idx.values()
        if isinstance(static_idx, dict):
            static_idx = static_idx.values()
        self.dyn_idx = sorted(dyn_idx) if dyn_idx is not None else [0]
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
        # normalise dynamic channels only
        x[:, self.dyn_idx] = (x[:, self.dyn_idx] - mu) / std
        y = self.y[idx]
        if self.scaler == "revin":
            y = (y - self.mu_base[idx]) / self.std_base[idx]
        return x, y, int(self.sids[idx]), np.float32(self.mu_base[idx]), np.float32(self.std_base[idx])


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
        ):
            super().__init__()
            self.L, self.H = L, H
            self.patch_len = patch_len
            self.stride = stride
            self.proj = nn.Linear(patch_len * input_dim, d_model)
            self.blocks = nn.ModuleList([PatchTSTBlock(d_model, n_heads, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(d_model)
            # separate heads for classification and regression
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * H)
            )
            self.clf_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, H), nn.Sigmoid()
            )
            if id_embed_dim > 0 and num_series > 0:
                self.id_embed = nn.Embedding(num_series, id_embed_dim)
                self.id_proj = nn.Linear(id_embed_dim, d_model) if id_embed_dim != d_model else None
            else:
                self.id_embed = None
                self.id_proj = None

        def forward(self, x, sid_idx=None):
            B, L, C = x.shape
            p = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len, C)
            p = p.contiguous().view(B, -1, self.patch_len * C)
            z = self.proj(p)
            if self.id_embed is not None and sid_idx is not None:
                e = self.id_embed(sid_idx)
                if self.id_proj is not None:
                    e = self.id_proj(e)
                z = z + e.unsqueeze(1)
            for blk in self.blocks:
                z = blk(z)
            z = self.norm(z).mean(1)
            params = self.reg_head(z)
            mu_raw, kappa_raw = params.chunk(2, dim=-1)
            p_clf = self.clf_head(z)
            return p_clf, mu_raw, kappa_raw

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
        return pp.build_patch_train(df_full)

    @staticmethod
    def build_eval_dataset(pp: Preprocessor, df_full: pd.DataFrame):
        """Build evaluation dataset for prediction."""
        return pp.build_patch_eval(df_full)

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

    def _ensure_torch(self):
        if not TORCH_OK: raise RuntimeError(f"PyTorch not available: {_TORCH_ERR}")
    def train(self, X_train: np.ndarray, y_train: np.ndarray, series_ids: np.ndarray, label_dates: np.ndarray, cfg: TrainConfig,
              preprocessors: Optional[List[Any]] = None) -> None:
        """Train PatchTST models under various validation policies.

        Parameters
        ----------
        preprocessors : Optional[List[Any]]
            Optional list of fold-specific preprocessors or artifacts.
        """
        self.preprocessors = preprocessors
        if preprocessors:
            pp0 = preprocessors[0]
            self.dynamic_idx_map = getattr(pp0, "patch_dynamic_idx", {})
            self.static_idx_map = getattr(pp0, "patch_static_idx", {})
        self._ensure_torch(); import torch
        os.makedirs(self.model_dir, exist_ok=True)
        self.oof_records = []
        order = np.argsort(label_dates)
        X_train = X_train[order]; y_train = y_train[order]
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
            tr_ds = _SeriesDataset(
                X_train[tr_mask], y_train[tr_mask], series_idx[tr_mask],
                scaler=self.params.scaler,
                dyn_idx=self.dynamic_idx_map,
                static_idx=self.static_idx_map,
            )
            va_ds = _SeriesDataset(
                X_train[va_mask], y_train[va_mask], series_idx[va_mask],
                scaler=self.params.scaler,
                dyn_idx=self.dynamic_idx_map,
                static_idx=self.static_idx_map,
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
            ).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
            smape_loss = WeightedSMAPELoss(reduction="mean")
            best=float("inf"); best_state=None; bad=0
            for ep in range(self.params.max_epochs):
                net.train()
                nb_sum = clf_sum = s_sum = 0.0
                batch_count = 0
                for xb, yb, sb, mu_s, std_s in tr_loader:
                    xb = xb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    sb = sb.to(self.device, non_blocking=pin)
                    mu_s = mu_s.to(self.device, non_blocking=pin)
                    std_s = std_s.to(self.device, non_blocking=pin)
                    opt.zero_grad()
                    p, mu_raw, kappa_raw = net(xb, sb)
                    mu = F.softplus(mu_raw) + 1e-6
                    kappa = F.softplus(kappa_raw) + 1e-6
                    y_raw = yb
                    mu_unscaled = mu
                    if self.params.scaler == "revin":
                        y_raw = y_raw * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                    M = (y_raw > 0)
                    z = M.float()
                    series_w = self.series_weight_tensor[sb]
                    w = torch.where(M, series_w.unsqueeze(1), torch.ones_like(y_raw))
                    if cfg.use_weighted_loss:
                        outlets = [self.idx2id[int(i)].split("::")[0] for i in sb.cpu().tolist()]
                        priority_w = torch.tensor(
                            [cfg.priority_weight if o in PRIORITY_OUTLETS else 1.0 for o in outlets],
                            device=self.device,
                        ).view(-1, 1)
                        w = w * priority_w
                    nb_loss = trunc_nb_nll(y_raw, mu_unscaled, kappa)
                    L_nb = (nb_loss * w * z).sum() / torch.clamp(z.sum(), min=1.0)
                    L_clf = focal_loss(p, z, self.params.gamma, self.params.alpha, w)
                    P0 = torch.pow(kappa / (kappa + mu_unscaled), kappa)
                    cond_mean = mu_unscaled / torch.clamp(1.0 - P0, min=1e-6)
                    y_hat = ((1 - self.params.epsilon_leaky) * p + self.params.epsilon_leaky) * cond_mean
                    L_s = smape_loss(y_hat, y_raw, w)
                    loss = (
                        self.params.lambda_nb * L_nb
                        + self.params.lambda_clf * L_clf
                        + self.params.lambda_s * L_s
                    )
                    loss.backward()
                    opt.step()
                    nb_sum += float(L_nb.item())
                    clf_sum += float(L_clf.item())
                    s_sum += float(L_s.item())
                    batch_count += 1
                print(
                    f"Fold {i} Epoch {ep} Train: L_nb={nb_sum/batch_count:.4f} "
                    f"L_clf={clf_sum/batch_count:.4f} L_s={s_sum/batch_count:.4f}"
                )
                net.eval()
                P = []
                T = []
                S = []
                with torch.no_grad():
                    for xb, yb, sb, mu_s, std_s in va_loader:
                        xb = xb.to(self.device, non_blocking=pin)
                        yb = yb.to(self.device, non_blocking=pin)
                        sb = sb.to(self.device, non_blocking=pin)
                        mu_s = mu_s.to(self.device, non_blocking=pin)
                        std_s = std_s.to(self.device, non_blocking=pin)
                        p, mu_raw, kappa_raw = net(xb, sb)
                        mu = F.softplus(mu_raw) + 1e-6
                        kappa = F.softplus(kappa_raw) + 1e-6
                        mu_unscaled = mu
                        if self.params.scaler == "revin":
                            mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                            yb = yb * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        final = combine_predictions(
                            p, mu_unscaled, kappa, self.params.epsilon_leaky
                        )
                        P.append(final.cpu().numpy())
                        T.append(yb.cpu().numpy())
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
                if w_val + 1e-8 < best:
                    best = w_val; best_state = net.state_dict(); bad = 0
                else:
                    bad += 1
                print(
                    f"Fold {i} Epoch {ep}: wSMAPE={w_val:.4f} SMAPE={s_val:.4f} MAE={mae_val:.4f}"
                )
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
                for xb, yb, sb, mu_s, std_s in va_loader:
                    xb = xb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    sb = sb.to(self.device, non_blocking=pin)
                    mu_s = mu_s.to(self.device, non_blocking=pin)
                    std_s = std_s.to(self.device, non_blocking=pin)
                    p, mu_raw, kappa_raw = net(xb, sb)
                    mu = F.softplus(mu_raw) + 1e-6
                    kappa = F.softplus(kappa_raw) + 1e-6
                    mu_unscaled = mu
                    if self.params.scaler == "revin":
                        mu_unscaled = mu * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                        yb = yb * std_s.unsqueeze(1) + mu_s.unsqueeze(1)
                    final = combine_predictions(
                        p, mu_unscaled, kappa, self.params.epsilon_leaky
                    )
                    final = final.cpu()
                    P.append(final.numpy())
                    Y.append(yb.cpu().numpy())
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
        self.save(os.path.join(self.model_dir,"patchtst.pt"))
    def predict(self, X_eval: np.ndarray, series_idx: Optional[Iterable[int]] = None) -> np.ndarray:
        self._ensure_torch(); import torch
        if not self.models:
            raise RuntimeError("Model not loaded.")
        if series_idx is None:
            series_idx = np.zeros(len(X_eval), dtype=np.int64)
        ds = _SeriesDataset(
            X_eval,
            np.zeros((X_eval.shape[0], self.H), dtype=np.float32),
            series_idx,
            scaler=self.params.scaler,
            dyn_idx=self.dynamic_idx_map,
            static_idx=self.static_idx_map,
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
            for xb, _, sb, mu_s, std_s in loader:
                xb = xb.to(self.device, non_blocking=pin)
                sb = sb.to(self.device, non_blocking=pin)
                mu_s = mu_s.to(self.device, non_blocking=pin)
                std_s = std_s.to(self.device, non_blocking=pin)
                preds = [m(xb, sb) for m in self.models]
                p, mu_raw, kappa_raw = zip(*preds)
                p = torch.stack(p)
                mu = torch.stack([F.softplus(m) + 1e-6 for m in mu_raw])
                kappa = torch.stack([F.softplus(k) + 1e-6 for k in kappa_raw])
                if self.params.scaler == "revin":
                    mu = mu * std_s.view(1, -1, 1) + mu_s.view(1, -1, 1)
                out = combine_predictions(
                    p, mu, kappa, self.params.epsilon_leaky
                ).mean(0)
                outs.append(out.cpu().numpy())
        yhat = np.clip(np.concatenate(outs, 0), 0, None)
        return yhat

    def predict_df(self, eval_df):
        """Return conditional-mean prediction dataframe for PatchTST."""
        X_eval, sids, _ = eval_df
        sid_idx = np.array([self.id2idx.get(sid, 0) for sid in sids])
        y_pred = self.predict(X_eval, sid_idx)
        reps = np.repeat(sids, self.H)
        hs = np.tile(np.arange(1, self.H + 1), len(sids))
        out = pd.DataFrame({"series_id": reps, "h": hs, "yhat_patch": y_pred.reshape(-1)})
        return out

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
                self.L=int(m.get("L",self.L))
                self.H=int(m.get("H",self.H))
                index=m.get("index",[])
                self.id2idx=m.get("id2idx",{})
                self.dynamic_idx_map=m.get("patch_dynamic_idx",{})
                self.static_idx_map=m.get("patch_static_idx",{})
        else:
            index=[]
            self.id2idx={}
            self.dynamic_idx_map={}
            self.static_idx_map={}
        self.idx2id=[None]*len(self.id2idx)
        for sid,idx in self.id2idx.items():
            self.idx2id[int(idx)] = sid
        self.params = PatchTSTParams(**self.model_params)
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
            )
            net.load_state_dict(torch.load(os.path.join(self.model_dir,fname), map_location=torch.device(self.device)))
            net.to(self.device)
            self.models.append(net)
