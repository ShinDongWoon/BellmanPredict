"""Hyperparameter tuning utilities using Optuna.

This module focuses on searches for the PatchTST model while retaining a
minimal Optuna demo objective for backward compatibility.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Any, List
import sys

import itertools

import numpy as np
import optuna
import pandas as pd
import yaml
from dataclasses import asdict
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:  # torch is optional; used only for GPU cache clearing
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from LGHackerton.config.default import OPTUNA_DIR, TRAIN_PATH, TRAIN_CFG, ARTIFACTS_DIR
from LGHackerton.preprocess import Preprocessor
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# simple demo objective (kept for backward compatibility)
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Simple objective function for demonstration purposes."""

    x = trial.suggest_float("x", -10.0, 10.0)
    return x * x


def demo_study(n_trials: int = 20) -> None:
    """Run a sample Optuna study and persist the results."""

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    out_path = OPTUNA_DIR / "study.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"value": t.value, "params": t.params} for t in study.trials],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Study results saved to {out_path}")


def _log_fold_start(
    prefix: str,
    seed: int,
    fold_name: str,
    tr_mask: np.ndarray,
    va_mask: np.ndarray,
    cfg: TrainConfig,
) -> None:
    """Persist fold information for a trial to artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "seed": seed,
        "fold": fold_name,
        "train_indices": np.where(tr_mask)[0].tolist(),
        "val_indices": np.where(va_mask)[0].tolist(),
        "config": asdict(cfg),
    }
    out = ARTIFACTS_DIR / f"{prefix}_{fold_name}_{timestamp}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tune_patchtst(pp, df_full, cfg):
    """Tune PatchTST hyperparameters using Optuna."""

    from LGHackerton.models.patchtst_trainer import (
        PatchTSTParams,
        PatchTSTTrainer,
        TORCH_OK,
    )
    from LGHackerton.preprocess import H
    from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SampleWindowizer

    if not TORCH_OK:
        raise RuntimeError("PyTorch not available for PatchTST")

    study = optuna.create_study(direction="minimize")

    dataset_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    input_lens = getattr(cfg, "input_lens", None) or [96, 168, 336]
    if not isinstance(input_lens, (list, tuple)):
        input_lens = [input_lens]

    def objective(trial: optuna.Trial) -> float:
        """Train a PatchTST model for a single Optuna trial.

        Prior to training we attempt to hook into the generation of ROCV
        folds so that each fold can be logged via :func:`_log_fold_start`.
        Newer versions expose ``PatchTSTTrainer.register_rocv_callback``;
        older versions require temporarily wrapping the module-level
        ``_make_rocv_slices`` helper.  Hooks are removed in the ``finally``
        block to avoid leaking state across trials.
        """

        trainer = None
        import LGHackerton.models.patchtst_trainer as pt
        from LGHackerton.models.patchtst_trainer import PatchTSTTrainer

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
                            cfg.seed,
                            f"trial{trial.number}_fold{i}",
                            tr_mask,
                            va_mask,
                            cfg,
                        )
                    return slices

                pt._make_rocv_slices = _logged_rocv
            else:  # pragma: no cover - defensive fallback
                warnings.warn(
                    "No PatchTST fold logging hooks found; fold information will not be logged",
                    stacklevel=2,
                )

            set_seed(cfg.seed)
            input_len = trial.suggest_categorical("input_len", input_lens)
            if input_len not in dataset_cache:
                pp.windowizer = SampleWindowizer(lookback=input_len, horizon=H)
                dataset_cache[input_len] = pp.build_patch_train(df_full)
            X, y, series_ids, label_dates = dataset_cache[input_len]

            sampled_params = {
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "depth": trial.suggest_int("depth", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "id_embed_dim": trial.suggest_categorical("id_embed_dim", [0, 16]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "max_epochs": trial.suggest_int("max_epochs", 50, 200),
                "patience": trial.suggest_int("patience", 5, 30),
            }
            patch_len = trial.suggest_categorical(
                "patch_len",
                [8, 12, 14, 16, 24],
            )
            sampled_params["patch_len"] = patch_len
            sampled_params["stride"] = patch_len
            if input_len % patch_len != 0:
                raise optuna.TrialPruned()

            params = PatchTSTParams(**sampled_params)
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            trainer = PatchTSTTrainer(
                params=params,
                L=input_len,
                H=H,
                model_dir=getattr(cfg, "model_dir", "."),
                device=device,
            )

            trainer.train(X, y, series_ids, label_dates, cfg)
            oof = trainer.get_oof()
            outlets = oof["series_id"].str.split("::").str[0].values
            score = weighted_smape_np(
                oof["y"].values,
                oof["yhat"].values,
                outlets,
                priority_weight=getattr(cfg, "priority_weight", 1.0),
            )

            return float(score)
        except Exception as e:
            trial.set_user_attr("status", "failed")
            raise optuna.TrialPruned() from e
        finally:
            if callback_registered:
                try:
                    PatchTSTTrainer._rocv_callbacks.remove(_cb)
                except Exception:  # pragma: no cover - defensive
                    pass
            if original_rocv is not None:
                pt._make_rocv_slices = original_rocv
            if trainer is not None:
                del trainer
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_trials = getattr(cfg, "n_trials", 20)
    timeout = getattr(cfg, "timeout", None)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    if study.best_trial is None:
        logger.error("Optuna study finished without any completed trials")
        raise RuntimeError("No completed trials; cannot retrieve best parameters")

    best_path = OPTUNA_DIR / "patchtst_best.json"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as f:
        best = {**study.best_params, "stride": study.best_params["patch_len"]}
        json.dump(best, f, ensure_ascii=False, indent=2)

    return study


def run_patchtst_grid_search(cfg_path: str | Path) -> None:
    """Run a simple grid search over PatchTST hyperparameters."""

    from LGHackerton.models.patchtst_trainer import (
        PatchTSTParams,
        PatchTSTTrainer,
        TORCH_OK,
    )
    from LGHackerton.preprocess import H
    from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SampleWindowizer

    if not TORCH_OK:
        raise RuntimeError("PyTorch not available for PatchTST")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)

    df_raw = pd.read_csv(TRAIN_PATH)
    pp = Preprocessor(show_progress=False)
    df_full = pp.fit_transform_train(df_raw)

    input_lens = [96, 168, 336]
    patch_lens = [16, 24, 32]
    lrs = [1e-4, 5e-4, 1e-3]
    scalers = ["per_series", "revin"]

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    results: List[dict[str, Any]] = []
    dataset_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # Prebuild datasets for each unique input length
    for inp in input_lens:
        set_seed(42)
        pp.windowizer = SampleWindowizer(lookback=inp, horizon=H)
        dataset_cache[inp] = pp.build_patch_train(df_full)

    # Iterate over grid while reusing cached datasets
    for inp in input_lens:
        X, y, series_ids, label_dates = dataset_cache[inp]
        for patch, lr, scaler in itertools.product(patch_lens, lrs, scalers):
            if inp % patch != 0:
                continue
            try:
                set_seed(42)
                params = PatchTSTParams(
                    patch_len=patch, stride=patch, lr=lr, scaler=scaler
                )
                trainer = PatchTSTTrainer(
                    params=params, L=inp, H=H, model_dir=cfg.model_dir, device=device
                )
                trainer.train(X, y, series_ids, label_dates, cfg)
                oof = trainer.get_oof()
                outlets = oof["series_id"].str.split("::").str[0].values
                val_w = weighted_smape_np(
                    oof["y"].values,
                    oof["yhat"].values,
                    outlets,
                    priority_weight=getattr(cfg, "priority_weight", 1.0),
                )
                val_mae = float(np.mean(np.abs(oof["y"].values - oof["yhat"].values)))
                results.append(
                    {
                        "input_len": inp,
                        "patch_len": patch,
                        "lr": lr,
                        "scaler": scaler,
                        "val_wsmape": float(val_w),
                        "val_mae": val_mae,
                    }
                )
                logger.info(
                    "inp=%s patch=%s lr=%s scaler=%s wSMAPE=%.4f MAE=%.4f",
                    inp,
                    patch,
                    lr,
                    scaler,
                    val_w,
                    val_mae,
                )
            except Exception as e:  # pragma: no cover - robustness
                logger.exception(
                    "Grid combo failed for input_len=%s patch_len=%s lr=%s scaler=%s",
                    inp,
                    patch,
                    lr,
                    scaler,
                )
                results.append(
                    {
                        "input_len": inp,
                        "patch_len": patch,
                        "lr": lr,
                        "scaler": scaler,
                        "error": str(e),
                    }
                )
                continue

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(ARTIFACTS_DIR / "patchtst_search.csv", index=False)


def main() -> None:  # pragma: no cover - CLI entry point
    """Entry point for command-line usage."""

    parser = argparse.ArgumentParser(description="Hyperparameter tuning utilities")
    parser.add_argument("--task", type=str, default=None, help="special task to run")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml", help="config path"
    )
    parser.add_argument(
        "--patch", action="store_true", help="tune PatchTST hyperparameters"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="time limit for tuning in seconds"
    )
    args = parser.parse_args()

    if args.task == "patchtst_grid":
        run_patchtst_grid_search(args.config)
        return
    if args.patch:
        df_raw = pd.read_csv(TRAIN_PATH)
        pp = Preprocessor(show_progress=False)
        df_full = pp.fit_transform_train(df_raw)
        cfg = TrainConfig(**TRAIN_CFG)
        cfg.n_trials = args.n_trials
        cfg.timeout = args.timeout
        if getattr(cfg, "input_lens", None) is None:
            cfg.input_lens = [96, 168, 336]
        tune_patchtst(pp, df_full, cfg)

    if not args.patch:
        demo_study(args.n_trials)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
