from __future__ import annotations

import gc
import json
import logging
import warnings
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import TPESampler

from LGHackerton.config.default import ARTIFACTS_DIR, PATCH_PARAMS
from LGHackerton.models.patchtst.trainer import (
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

    This search space mirrors the recommended tuning ranges:

    ``d_model``={128, 256, 384, 512}, ``n_heads``={8, 16}, ``depth`` in [2, 6],
    ``patch_len``/``stride``={7,14, 21, 28}, ``dropout`` in [0.10, 0.30],
    ``lr`` in [1e-4, 3e-3], ``weight_decay`` in [1e-5, 1e-3],
    ``id_embed_dim``={16, 32, 64}, ``batch_size``={64, 128},
    ``max_epochs`` in [80, 150], ``patience`` in [15, 30].

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

    d_model: Tuple[int, ...] = (256, 384)
    n_heads: Tuple[int, ...] = (8, 16)
    depth: Tuple[int, int] = (2, 4)
    patch_len: Tuple[int, ...] = (7, 14)
    stride: Tuple[int, ...] = (7,)
    dropout: Tuple[float, float] = (0.10, 0.30)
    lr: Tuple[float, float] = (1e-4, 5e-4)
    weight_decay: Tuple[float, float] = (1e-5, 1e-3)
    id_embed_dim: Tuple[int, ...] = (32, 64)
    batch_size: Tuple[int, ...] = (64, 128)
    max_epochs: Tuple[int, int] = (80, 150)
    patience: Tuple[int, int] = (15, 30)


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
        self._dataset_cache: Dict[
            int,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Dict[str, int],
                Dict[str, int],
            ],
        ] = {}
        self.best_input_len: int | None = None

    def validate_params(self, params: dict) -> None:  # type: ignore[override]
        required = {f.name for f in fields(PatchTSTParams)}
        for field in required:
            if field not in params:
                raise TypeError(f"Missing field: {field}")
        s = self.search_space
        choice_fields = {
            "d_model": s.d_model,
            "n_heads": s.n_heads,
            "patch_len": s.patch_len,
            "stride": s.stride,
            "id_embed_dim": s.id_embed_dim,
            "batch_size": s.batch_size,
        }
        for name, choices in choice_fields.items():
            val = params.get(name)
            if not isinstance(val, int) or val not in choices:
                raise ValueError(f"{name} out of range: {val}")
        range_fields = {
            "depth": s.depth,
            "max_epochs": s.max_epochs,
            "patience": s.patience,
        }
        for name, (lo, hi) in range_fields.items():
            val = params.get(name)
            if not isinstance(val, int) or not (lo <= val <= hi):
                raise ValueError(f"{name} out of range: {val}")
        float_fields = {
            "dropout": s.dropout,
            "lr": s.lr,
            "weight_decay": s.weight_decay,
        }
        for name, (lo, hi) in float_fields.items():
            val = params.get(name)
            if not isinstance(val, (float, int)) or not (lo <= float(val) <= hi):
                raise ValueError(f"{name} out of range: {val}")
        patch_len = params.get("patch_len")
        stride = params.get("stride")
        if patch_len is None or stride is None:
            raise ValueError("patch_len and stride must be provided")
        n_patches_eval = 1 + (28 - patch_len) // stride
        if n_patches_eval < 8:
            raise ValueError("n_patches_eval < 8")

    def run(self, n_trials: int, force: bool) -> dict:  # type: ignore[override]
        """Execute Optuna search over :class:`PatchTSTSearchSpace`.

        If ``best_params.json`` exists and ``force`` is ``False``, the cached
        parameters are returned without running the optimisation.

        Parameters
        ----------
        n_trials : int
            Number of trials to evaluate.
        force : bool
            If ``True``, ignore any cached results and rerun the search.
        """

        if not force:
            cache = self.artifact_dir / "best_params.json"
            if cache.exists():
                with cache.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                try:
                    self.validate_params(cached)
                except ValueError as exc:
                    logging.info(
                        "Invalid cached parameters; ignoring and starting a new search: %s",
                        exc,
                    )
                else:
                    self._best_params = cached
                    self.best_input_len = cached.get("input_len")
                    return cached

        if not TORCH_OK:
            raise RuntimeError("PyTorch not available for PatchTST")

        input_lens = getattr(self.cfg, "input_lens", None) or [336, 448]
        if not isinstance(input_lens, (list, tuple)):
            input_lens = [input_lens]

        param_count = 11
        sampler = TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=max(20, 2 * param_count),
            n_ei_candidates=64,
            seed=getattr(self.cfg, "seed", 42),
            warn_independent_sampling=True,
        )

        def _constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
            d_model = trial.params.get("d_model")
            n_heads = trial.params.get("n_heads")
            patch_len = trial.params.get("patch_len")
            input_len = trial.params.get("input_len")
            return (
                (d_model % n_heads) if d_model is not None and n_heads is not None else 0.0,
                (patch_len - input_len)
                if patch_len is not None and input_len is not None
                else 0.0,
            )

        pruner = PatientPruner(MedianPruner(n_warmup_steps=12), patience=4)
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )
        study.sampler.constraints_func = _constraints_func

        search = self.search_space

        def objective(trial: optuna.Trial) -> float:
            trainer = None
            import LGHackerton.models.patchtst.trainer as pt
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
                    X, S, M, y, series_ids, label_dates = self.pp.build_patch_train(self.df)
                    dyn_idx = getattr(self.pp, "patch_dynamic_idx", {}).copy()
                    stat_idx = getattr(self.pp, "patch_static_idx", {}).copy()
                    self._dataset_cache[input_len] = (
                        X,
                        S,
                        M,
                        y,
                        series_ids,
                        label_dates,
                        dyn_idx,
                        stat_idx,
                    )
                X, S, M, y, series_ids, label_dates, dyn_idx, stat_idx = self._dataset_cache[input_len]
                # Reassign channel index maps to ensure they reflect the cached
                # dataset's indices even if downstream steps mutate them.
                self.pp.patch_dynamic_idx = dyn_idx.copy()
                self.pp.patch_static_idx = stat_idx.copy()
                reg_mode = trial.suggest_categorical("reg_mode", ["light", "strong"])
                dropout = trial.suggest_float("dropout", *search.dropout)
                if reg_mode == "light":
                    weight_decay = trial.suggest_float(
                        "weight_decay", 1e-4, 3e-4, log=True
                    )
                else:
                    weight_decay = trial.suggest_float(
                        "weight_decay", 3e-4, 5e-4, log=True
                    )

                params = {
                    "d_model": trial.suggest_categorical("d_model", search.d_model),
                    "n_heads": trial.suggest_categorical("n_heads", search.n_heads),
                    "depth": trial.suggest_int("depth", *search.depth),
                    "patch_len": trial.suggest_categorical("patch_len", search.patch_len),
                    "stride": trial.suggest_categorical("stride", search.stride),
                    "dropout": dropout,
                    "id_embed_dim": trial.suggest_categorical(
                        "id_embed_dim", search.id_embed_dim
                    ),
                    "lr": trial.suggest_float("lr", *search.lr, log=True),
                    "weight_decay": weight_decay,
                    "batch_size": trial.suggest_categorical("batch_size", search.batch_size),
                    "max_epochs": trial.suggest_int("max_epochs", *search.max_epochs),
                    "patience": trial.suggest_int("patience", *search.patience),
                }
                patch_len = params["patch_len"]
                stride = params["stride"]
                n_patches_eval = 1 + (input_len - patch_len) // stride
                if n_patches_eval < 8:
                    # ensure at least 8 training patches
                    raise optuna.TrialPruned()
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

                trainer.train(
                    X,
                    S,
                    M,
                    y,
                    series_ids,
                    label_dates,
                    self.cfg,
                    [self.pp],
                    trial=trial,
                )
                oof = trainer.get_oof()
                outlets = oof["series_id"].str.split("::").str[0].values
                score = weighted_smape_np(
                    oof["y"].values,
                    oof["yhat"].values,
                    outlets,
                    priority_weight=getattr(self.cfg, "priority_weight", 1.0),
                )
                return float(score)
            except optuna.TrialPruned:
                raise
            except Exception as e:  # pragma: no cover - robustness
                logging.exception("PatchTST trial %s failed", trial.number)
                trial.set_user_attr("status", "failed")
                trial.set_user_attr("fail_reason", repr(e))
                raise
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
            logging.warning("Hyperparameter search stopped early; no trials completed")
            self._best_params = {}
            return self._best_params

        best = dict(study.best_trial.params)
        best.pop("reg_mode", None)
        self.best_input_len = best.pop("input_len", None)
        best["stride"] = best.get("patch_len")
        best["num_workers"] = PATCH_PARAMS.get("num_workers", 0)
        params = PatchTSTParams(**best)
        self._best_params = asdict(params)
        if self.best_input_len is not None:
            self._best_params["input_len"] = self.best_input_len
        self.validate_params(self._best_params)
        return self.best_params()
