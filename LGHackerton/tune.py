"""Hyperparameter tuning utilities using Optuna.

This module originally provided only a very small example of running an
Optuna study.  It now also contains helper functions for tuning a LightGBM
model on the project's training data using the weighted sMAPE metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import warnings
import os
import tempfile
from pathlib import Path
from typing import Any, List, Tuple
import sys

import itertools

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
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
from LGHackerton.models.lgbm_trainer import LGBMParams, LGBMTrainer
from LGHackerton.utils.hurdle import combine_with_regression
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


# ---------------------------------------------------------------------------
# LightGBM tuning utilities
# ---------------------------------------------------------------------------

_LGBM_TRAIN: pd.DataFrame | None = None
_FEATURE_COLS: List[str] | None = None
_TRIAL_LOG: List[dict[str, Any]] = []
_FEATURE_IMPORTANCE: List[dict[str, Any]] = []

TRIAL_LOG_PATH = OPTUNA_DIR / "lgbm_trials.json"


def _log_fold_start(prefix: str, seed: int, fold_name: str, tr_mask: np.ndarray, va_mask: np.ndarray, cfg: TrainConfig) -> None:
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


def _prepare_lgbm_train(
    df: pd.DataFrame | None = None, feature_cols: List[str] | None = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Register LightGBM training frame and feature columns.

    The training dataframe and list of feature columns may be provided by the
    caller (e.g. :func:`tune_lgbm`).  They are cached in module-level globals so
    subsequent calls reuse the same objects without re-loading data.
    """

    global _LGBM_TRAIN, _FEATURE_COLS

    if df is not None and feature_cols is not None:
        _LGBM_TRAIN = df
        _FEATURE_COLS = feature_cols

    if _LGBM_TRAIN is None or _FEATURE_COLS is None:
        raise RuntimeError("Training data has not been provided")

    return _LGBM_TRAIN, _FEATURE_COLS


def _log_trial(record: dict[str, Any]) -> None:
    """Append a trial record to the JSON log under :data:`OPTUNA_DIR`."""

    _TRIAL_LOG.append(record)
    TRIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIAL_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(_TRIAL_LOG, f, ensure_ascii=False, indent=2)


def objective_lgbm(trial: optuna.Trial) -> float:
    """Train LightGBM with trial params and evaluate wSMAPE."""

    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)
    df, feat_cols = _prepare_lgbm_train()

    # Maximum number of boosting rounds. Training may stop earlier
    # due to the early_stopping callback configured below.
    num_boost_round = trial.suggest_int("num_boost_round", 200, 2000)

    params = {
        "objective": "tweedie",
        "metric": "None",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "early_stopping_rounds": 50,
    }
    params["device_type"] = "gpu" if torch and torch.cuda.is_available() else "cpu"

    scores: List[float] = []
    priority_w = TRAIN_CFG.get("priority_weight", 1.0)

    for h in range(1, 8):
        try:
            dfh = df[df["h"] == h]
            if dfh.empty:
                continue
            dfh = dfh.sort_values("date")

            # compute feature variances once to avoid repeated heavy computation
            feat_var = dfh[feat_cols].var()
            if (feat_var == 0).all():
                logger.warning("h%s: all feature variances are zero; skipping", h)
                continue

            dates = dfh["date"].sort_values().unique()
            if len(dates) < 2:
                continue
            split = int(len(dates) * 0.8)
            tr_dates, va_dates = dates[:split], dates[split:]
            tr_mask = dfh["date"].isin(tr_dates)
            va_mask = dfh["date"].isin(va_dates)
            _log_fold_start("tune_lgbm", cfg.seed, f"trial{trial.number}_h{h}", tr_mask.values, va_mask.values, cfg)

            X_tr = dfh.loc[tr_mask, feat_cols].values.astype("float32")
            y_tr = dfh.loc[tr_mask, "y"].values.astype("float32")
            if not np.any(y_tr > 0):
                logger.warning("h%s: no positive samples; skipping", h)
                continue

            X_va = dfh.loc[va_mask, feat_cols].values.astype("float32")
            y_va = dfh.loc[va_mask, "y"].values.astype("float32")
            outlets = dfh.loc[va_mask, "series_id"].str.split("::").str[0].values

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_va, label=y_va)

            fold_params = params.copy()
            callbacks = [lgb.log_evaluation(period=0)]
            if "early_stopping_rounds" in fold_params:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=fold_params["early_stopping_rounds"],
                        verbose=False,
                    )
                )
                fold_params.pop("early_stopping_rounds")

            def wsmape_eval(preds: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
                return (
                    "wSMAPE",
                    float(
                        weighted_smape_np(
                            dataset.get_label(), preds, outlets, priority_weight=priority_w
                        )
                    ),
                    False,
                )

            booster = lgb.train(
                params=fold_params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
                feval=wsmape_eval,
                callbacks=callbacks,
            )

            preds = booster.predict(X_va, num_iteration=booster.best_iteration)
            score = weighted_smape_np(y_va, preds, outlets, priority_weight=priority_w)
            scores.append(score)

            # record feature importance per trial and fold
            fi = booster.feature_importance(importance_type="gain")
            for feat, imp in zip(feat_cols, fi):
                _FEATURE_IMPORTANCE.append(
                    {
                        "feature": feat,
                        "fold": h,
                        "importance": float(imp),
                        "trial": trial.number,
                    }
                )

            del booster, dtrain, dvalid
        except Exception as e:  # pragma: no cover - robustness
            logger.exception("Trial %s fold %s failed: %s", trial.number, h, e)
            continue

    final_score = float(np.mean(scores)) if scores else float("inf")
    _log_trial({"number": trial.number, "value": final_score, "params": trial.params})
    logger.info("Trial %d completed score=%.5f params=%s", trial.number, final_score, trial.params)

    gc.collect()
    if torch and torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.empty_cache()

    return final_score


def objective_lgbm_hurdle(trial: optuna.Trial) -> float:
    """Train :class:`LGBMTrainer` with hurdle option and evaluate wSMAPE."""

    df, feat_cols = _prepare_lgbm_train()

    params = LGBMParams(
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", -1, 16),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 200),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        n_estimators=trial.suggest_int("n_estimators", 500, 3000),
        early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 50, 300),
    )

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    cfg_dict = dict(TRAIN_CFG)
    cfg_dict["use_hurdle"] = True
    cfg = TrainConfig(**cfg_dict)
    cfg.n_folds = min(getattr(cfg, "n_folds", 3), 2)
    cfg.cv_stride = min(getattr(cfg, "cv_stride", 7), 7)
    set_seed(cfg.seed)

    with tempfile.TemporaryDirectory() as model_dir:
        trainer = LGBMTrainer(params, feat_cols, model_dir, device)

        original_make_cv = LGBMTrainer._make_cv_slices

        def _logged_cv(self, df_h, cfg_inner, date_col):
            slices = original_make_cv(self, df_h, cfg_inner, date_col)
            h_val = int(df_h["h"].iloc[0]) if "h" in df_h.columns and not df_h.empty else -1
            for i, (tr_mask, va_mask) in enumerate(slices):
                _log_fold_start(
                    "tune_lgbm",
                    cfg.seed,
                    f"trial{trial.number}_h{h_val}_fold{i}",
                    tr_mask,
                    va_mask,
                    cfg_inner,
                )
            return slices

        LGBMTrainer._make_cv_slices = _logged_cv
        try:
            trainer.train(df, cfg)
        finally:
            LGBMTrainer._make_cv_slices = original_make_cv

        oof = trainer.get_oof()
        outlets = oof["series_id"].str.split("::").str[0].values
        if {"prob", "reg_pred"}.issubset(oof.columns):
            # ensure the same probability-multiplication logic is used during tuning
            oof["yhat"] = combine_with_regression(oof["prob"].values, oof["reg_pred"].values)
        score = weighted_smape_np(
            oof["y"].values,
            oof["yhat"].values,
            outlets,
            priority_weight=cfg.priority_weight,
        )

        for h in trainer.models:
            for booster in trainer.models[h]["reg"]:
                fi = booster.feature_importance(importance_type="gain")
                for feat, imp in zip(feat_cols, fi):
                    _FEATURE_IMPORTANCE.append(
                        {
                            "feature": feat,
                            "fold": h,
                            "importance": float(imp),
                            "trial": trial.number,
                        }
                    )

    final_score = float(score)
    _log_trial({"number": trial.number, "value": final_score, "params": trial.params})
    logger.info(
        "Trial %d completed score=%.5f params=%s", trial.number, final_score, trial.params
    )

    gc.collect()
    if torch and torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.empty_cache()

    return final_score


def tune_lgbm(
    n_trials: int,
    timeout: int | None = None,
    search: str = "bayes",
    train_df: pd.DataFrame | None = None,
    feature_cols: List[str] | None = None,
    use_hurdle: bool = True,
) -> optuna.Study:
    """Run an Optuna study to tune LightGBM hyperparameters.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to run.
    timeout : int | None, optional
        Time limit for the study in seconds, by default ``None``.
    search : str, optional
        Search strategy ("bayes" or "random"), by default ``"bayes"``.
    train_df : pd.DataFrame | None, optional
        Preconstructed LightGBM training dataframe.
    feature_cols : List[str] | None, optional
        List of feature column names corresponding to ``train_df``.
    """

    _FEATURE_IMPORTANCE.clear()

    if TRIAL_LOG_PATH.exists():
        try:
            _TRIAL_LOG.extend(json.load(TRIAL_LOG_PATH.open()))
        except Exception:
            pass

    _prepare_lgbm_train(train_df, feature_cols)  # ensure data prepared once

    seed = TRAIN_CFG.get("seed", 42)
    if search == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective_fn = objective_lgbm_hurdle if use_hurdle else objective_lgbm
    try:
        study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)
    except Exception as e:  # pragma: no cover - robustness
        logger.exception("Study failed: %s", e)

    out_path = OPTUNA_DIR / "lgbm_study.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"value": t.value, "params": t.params} for t in study.trials],
            f,
            ensure_ascii=False,
            indent=2,
        )

    if _FEATURE_IMPORTANCE:
        fi_df = pd.DataFrame(_FEATURE_IMPORTANCE)
        best_trial_num = study.best_trial.number
        fi_df = fi_df[fi_df["trial"] == best_trial_num]
        if not fi_df.empty:
            fi_agg = fi_df.groupby(["feature", "fold"], as_index=False)["importance"].mean()
            os.makedirs("artifacts", exist_ok=True)
            fi_agg.to_csv(Path("artifacts") / "lgbm_feature_importance.csv", index=False)
    _FEATURE_IMPORTANCE.clear()

    return study


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
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
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
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    return study


def run_patchtst_grid_search(cfg_path: str | Path) -> None:
    """Run a simple grid search over PatchTST hyperparameters."""

    from LGHackerton.models.patchtst_trainer import PatchTSTParams, PatchTSTTrainer, TORCH_OK
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
                params = PatchTSTParams(patch_len=patch, stride=patch, lr=lr, scaler=scaler)
                trainer = PatchTSTTrainer(params=params, L=inp, H=H, model_dir=cfg.model_dir, device=device)
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
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="config path")
    parser.add_argument("--lgbm", action="store_true", help="tune LightGBM hyperparameters")
    parser.add_argument("--patch", action="store_true", help="tune PatchTST hyperparameters")
    parser.add_argument("--n-trials", type=int, default=30, help="number of Optuna trials")
    parser.add_argument(
        "--search", choices=["random", "bayes"], default="bayes", help="sampler type"
    )
    parser.add_argument("--timeout", type=int, default=None, help="time limit for tuning in seconds")
    args = parser.parse_args()

    if args.task == "patchtst_grid":
        run_patchtst_grid_search(args.config)
        return

    if args.lgbm:
        tune_lgbm(args.n_trials, args.timeout, args.search)

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

    if not args.lgbm and not args.patch:
        demo_study(args.n_trials)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

