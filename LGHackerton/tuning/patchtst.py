from __future__ import annotations

import gc
import json
import warnings
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import optuna

from LGHackerton.config.default import ARTIFACTS_DIR, PATCH_PARAMS
from LGHackerton.models.patchtst_trainer import (
    PatchTSTParams,
    PatchTSTTrainer,
    TORCH_OK,
)
from LGHackerton.preprocess import H
from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SampleWindowizer
from LGHackerton.tuning.base import HyperparameterTuner
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.utils.seed import set_seed

try:  # torch is optional for CPU-only environments
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@dataclass
class PatchTSTSearchSpace:
    """Search space definitions for PatchTST hyperparameters.

    Attributes
    ----------
    d_model : tuple[int, ...]
        Candidate model embedding dimensions.
    n_heads : tuple[int, ...]
        Possible numbers of attention heads.
    depth : tuple[int, int]
        Minimum and maximum number of Transformer layers (inclusive).
    patch_len : tuple[int, ...]
        Allowed patch lengths; ``stride`` will mirror this value.
    stride : tuple[int, ...]
        Candidate strides between patches. Typically equals ``patch_len``.
    dropout : tuple[float, float]
        Range of dropout probabilities.
    lr : tuple[float, float]
        Log-uniform range for learning rate.
    weight_decay : tuple[float, float]
        Log-uniform range for AdamW weight decay.
    id_embed_dim : tuple[int, ...]
        Optional dimensionality for series ID embeddings.
    batch_size : tuple[int, ...]
        Candidate mini-batch sizes.
    max_epochs : tuple[int, int]
        Inclusive range for maximum training epochs.
    patience : tuple[int, int]
        Inclusive range for early-stopping patience.
    """

    d_model: Tuple[int, ...] = (64, 128, 256)
    n_heads: Tuple[int, ...] = (4, 8)
    depth: Tuple[int, int] = (2, 6)
    patch_len: Tuple[int, ...] = (8, 12, 14, 16, 24)
    stride: Tuple[int, ...] = (8, 12, 14, 16, 24)
    dropout: Tuple[float, float] = (0.0, 0.5)
    lr: Tuple[float, float] = (1e-4, 1e-2)
    weight_decay: Tuple[float, float] = (1e-6, 1e-3)
    id_embed_dim: Tuple[int, ...] = (0, 16)
    batch_size: Tuple[int, ...] = (64, 128, 256)
    max_epochs: Tuple[int, int] = (50, 200)
    patience: Tuple[int, int] = (5, 30)


def _log_fold_start(
    prefix: str,
    seed: int,
    fold_name: str,
    tr_mask: np.ndarray,
    va_mask: np.ndarray,
    cfg: PatchTSTParams | object,
) -> None:
    """Persist fold information for a trial to the artifacts directory."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "seed": seed,
        "fold": fold_name,
        "train_indices": np.where(tr_mask)[0].tolist(),
        "val_indices": np.where(va_mask)[0].tolist(),
        "config": asdict(cfg) if hasattr(cfg, "__dict__") else {},
    }
    out = ARTIFACTS_DIR / f"{prefix}_{fold_name}_{timestamp}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class PatchTSTTuner(HyperparameterTuner):
    """Optuna-based hyperparameter tuner for PatchTST."""

    search_space: PatchTSTSearchSpace = PatchTSTSearchSpace()

    def __init__(self, pp, df, cfg) -> None:  # type: ignore[override]
        super().__init__(pp, df, cfg)
        # Ensure artifacts are stored under the "patchtst" directory for
        # backward compatibility with existing utilities.
        self.model_name = "patchtst"
        self.artifact_dir = ARTIFACTS_DIR / self.model_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_cache: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.best_input_len: int | None = None

    def validate_params(self, params: dict) -> None:  # type: ignore[override]
        required = {f.name for f in fields(PatchTSTParams)}
        missing = required - params.keys()
        if missing:
            raise ValueError(f"Missing hyperparameters: {sorted(missing)}")
        if params.get("patch_len") != params.get("stride"):
            raise ValueError("stride must equal patch_len")

    def run(self, n_trials: int, force: bool = False) -> dict:  # type: ignore[override]
        """Execute Optuna search over :class:`PatchTSTSearchSpace`.

        Parameters
        ----------
        n_trials : int
            Number of trials to evaluate.
        force : bool
            Ignored. Included for interface compatibility.
        """

        if not TORCH_OK:
            raise RuntimeError("PyTorch not available for PatchTST")

        study = optuna.create_study(direction="minimize")
        input_lens = getattr(self.cfg, "input_lens", None) or [96, 168, 336]
        if not isinstance(input_lens, (list, tuple)):
            input_lens = [input_lens]

        search = self.search_space

        def objective(trial: optuna.Trial) -> float:
            trainer = None
            import LGHackerton.models.patchtst_trainer as pt
            callback_registered = False
            original_rocv = None

            def _cb(seed, fold_idx, tr_mask, va_mask, cfg_inner):
                _log_fold_start(
                    "tune_patchtst",
                    seed,
                    f"trial{trial.number}_fold{fold_idx}",
                    tr_mask,
                    va_mask,
                    cfg_inner,
                )

            try:
                if hasattr(PatchTSTTrainer, "register_rocv_callback"):
                    PatchTSTTrainer.register_rocv_callback(_cb)
                    callback_registered = True
                elif hasattr(pt, "_make_rocv_slices"):
                    warnings.warn(
                        "PatchTSTTrainer.register_rocv_callback not found; wrapping _make_rocv_slices for fold logging",
                        stacklevel=2,
                    )
                    original_rocv = pt._make_rocv_slices

                    def _logged_rocv(label_dates, n_folds, stride, span, purge):
                        slices = original_rocv(label_dates, n_folds, stride, span, purge)
                        for i, (tr_mask, va_mask) in enumerate(slices):
                            _log_fold_start(
                                "tune_patchtst",
                                self.cfg.seed,
                                f"trial{trial.number}_fold{i}",
                                tr_mask,
                                va_mask,
                                self.cfg,
                            )
                        return slices

                    pt._make_rocv_slices = _logged_rocv
                else:  # pragma: no cover - defensive fallback
                    warnings.warn(
                        "No PatchTST fold logging hooks found; fold information will not be logged",
                        stacklevel=2,
                    )

                set_seed(self.cfg.seed)
                input_len = trial.suggest_categorical("input_len", input_lens)
                if input_len not in self._dataset_cache:
                    self.pp.windowizer = SampleWindowizer(lookback=input_len, horizon=H)
                    self._dataset_cache[input_len] = self.pp.build_patch_train(self.df)
                X, y, series_ids, label_dates = self._dataset_cache[input_len]

                params = {
                    "d_model": trial.suggest_categorical("d_model", search.d_model),
                    "n_heads": trial.suggest_categorical("n_heads", search.n_heads),
                    "depth": trial.suggest_int("depth", *search.depth),
                    "patch_len": trial.suggest_categorical("patch_len", search.patch_len),
                    "dropout": trial.suggest_float("dropout", *search.dropout),
                    "id_embed_dim": trial.suggest_categorical(
                        "id_embed_dim", search.id_embed_dim
                    ),
                    "lr": trial.suggest_float("lr", *search.lr, log=True),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", *search.weight_decay, log=True
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", search.batch_size),
                    "max_epochs": trial.suggest_int("max_epochs", *search.max_epochs),
                    "patience": trial.suggest_int("patience", *search.patience),
                }
                patch_len = params["patch_len"]
                params["stride"] = patch_len
                params["num_workers"] = PATCH_PARAMS.get("num_workers", 0)
                if input_len % patch_len != 0:
                    raise optuna.TrialPruned()

                trainer_params = PatchTSTParams(**params)
                device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                trainer = PatchTSTTrainer(
                    params=trainer_params,
                    L=input_len,
                    H=H,
                    model_dir=getattr(self.cfg, "model_dir", "."),
                    device=device,
                )

                trainer.train(X, y, series_ids, label_dates, self.cfg)
                oof = trainer.get_oof()
                outlets = oof["series_id"].str.split("::").str[0].values
                score = weighted_smape_np(
                    oof["y"].values,
                    oof["yhat"].values,
                    outlets,
                    priority_weight=getattr(self.cfg, "priority_weight", 1.0),
                )
                return float(score)
            except Exception as e:  # pragma: no cover - robustness
                trial.set_user_attr("status", "failed")
                raise optuna.TrialPruned() from e
            finally:
                if callback_registered:
                    try:
                        PatchTSTTrainer._rocv_callbacks.remove(_cb)  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensive
                        pass
                if original_rocv is not None:
                    pt._make_rocv_slices = original_rocv  # type: ignore[assignment]
                if trainer is not None:
                    del trainer
                gc.collect()
                if torch and torch.cuda.is_available():  # pragma: no cover - GPU only
                    torch.cuda.empty_cache()

        timeout = getattr(self.cfg, "timeout", None)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        if study.best_trial is None:
            raise RuntimeError("Optuna study finished without any completed trials")

        best = dict(study.best_trial.params)
        self.best_input_len = best.pop("input_len", None)
        best["stride"] = best.get("patch_len")
        best["num_workers"] = PATCH_PARAMS.get("num_workers", 0)
        params = PatchTSTParams(**best)
        self._best_params = asdict(params)
        self.validate_params(self._best_params)
        return self._best_params
