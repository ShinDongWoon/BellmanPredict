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
import os
from pathlib import Path
from typing import Any, List, Tuple
import sys

import itertools

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:  # torch is optional; used only for GPU cache clearing
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from LGHackerton.config.default import OPTUNA_DIR, TRAIN_PATH, TRAIN_CFG
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


# ---------------------------------------------------------------------------
# LightGBM tuning utilities
# ---------------------------------------------------------------------------

_LGBM_TRAIN: pd.DataFrame | None = None
_FEATURE_COLS: List[str] | None = None
_TRIAL_LOG: List[dict[str, Any]] = []
_FEATURE_IMPORTANCE: List[dict[str, Any]] = []

TRIAL_LOG_PATH = OPTUNA_DIR / "lgbm_trials.json"


def _prepare_lgbm_train() -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess training data for LightGBM once per process."""

    global _LGBM_TRAIN, _FEATURE_COLS
    if _LGBM_TRAIN is None or _FEATURE_COLS is None:
        df_raw = pd.read_csv(TRAIN_PATH)
        pp = Preprocessor(show_progress=False)
        df_full = pp.fit_transform_train(df_raw)
        _LGBM_TRAIN = pp.build_lgbm_train(df_full)
        _FEATURE_COLS = pp.feature_cols
    return _LGBM_TRAIN, _FEATURE_COLS


def _log_trial(record: dict[str, Any]) -> None:
    """Append a trial record to the JSON log under :data:`OPTUNA_DIR`."""

    _TRIAL_LOG.append(record)
    TRIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIAL_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(_TRIAL_LOG, f, ensure_ascii=False, indent=2)


def objective_lgbm(trial: optuna.Trial) -> float:
    """Train LightGBM with trial params and evaluate wSMAPE."""

    set_seed(TRAIN_CFG.get("seed", 42))
    df, feat_cols = _prepare_lgbm_train()

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
        "n_estimators": 200,
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
            dates = dfh["date"].sort_values().unique()
            if len(dates) < 2:
                continue
            split = int(len(dates) * 0.8)
            tr_dates, va_dates = dates[:split], dates[split:]
            tr_mask = dfh["date"].isin(tr_dates)
            va_mask = dfh["date"].isin(va_dates)

            X_tr = dfh.loc[tr_mask, feat_cols].values.astype("float32")
            y_tr = dfh.loc[tr_mask, "y"].values.astype("float32")
            X_va = dfh.loc[va_mask, feat_cols].values.astype("float32")
            y_va = dfh.loc[va_mask, "y"].values.astype("float32")

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_va, label=y_va)
            callbacks = [lgb.log_evaluation(period=0)]
            if "early_stopping_rounds" in params:
                if str(params.get("metric", "")).lower() != "none":
                    callbacks.append(
                        lgb.early_stopping(
                            stopping_rounds=params["early_stopping_rounds"],
                            verbose=False,
                        )
                    )
                params.pop("early_stopping_rounds")
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                callbacks=callbacks,
            )

            preds = booster.predict(X_va, num_iteration=booster.best_iteration)
            outlets = dfh.loc[va_mask, "series_id"].str.split("::").str[0].values
            score = weighted_smape_np(y_va, preds, outlets, priority_weight=priority_w)
            scores.append(score)

            # record feature importance
            fi = booster.feature_importance(importance_type="gain")
            for feat, imp in zip(feat_cols, fi):
                _FEATURE_IMPORTANCE.append({"feature": feat, "fold": h, "importance": float(imp)})

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


def tune_lgbm(
    n_trials: int,
    timeout: int | None = None,
    search: str = "bayes",
) -> optuna.Study:
    """Run an Optuna study to tune LightGBM hyperparameters."""

    if TRIAL_LOG_PATH.exists():
        try:
            _TRIAL_LOG.extend(json.load(TRIAL_LOG_PATH.open()))
        except Exception:
            pass

    _prepare_lgbm_train()  # ensure data prepared once

    seed = TRAIN_CFG.get("seed", 42)
    if search == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    try:
        study.optimize(objective_lgbm, n_trials=n_trials, timeout=timeout)
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
        fi_agg = fi_df.groupby(["feature", "fold"], as_index=False)["importance"].mean()
        os.makedirs("artifacts", exist_ok=True)
        fi_agg.to_csv(Path("artifacts") / "lgbm_feature_importance.csv", index=False)

    return study


def tune_patchtst(X, y, series_ids, label_dates, cfg):
    """Tune PatchTST hyperparameters using Optuna."""

    from LGHackerton.models.patchtst_trainer import (
        PatchTSTParams,
        PatchTSTTrainer,
        TORCH_OK,
    )
    from LGHackerton.preprocess import L, H

    if not TORCH_OK:
        raise RuntimeError("PyTorch not available for PatchTST")

    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        trainer = None
        try:
            set_seed(cfg.seed)
            sampled_params = {
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "depth": trial.suggest_int("depth", 2, 6),
                "patch_len": trial.suggest_categorical("patch_len", [4, 8]),
                "stride": trial.suggest_categorical("stride", [1, 2]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "id_embed_dim": trial.suggest_categorical("id_embed_dim", [0, 16]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "max_epochs": trial.suggest_int("max_epochs", 50, 200),
                "patience": trial.suggest_int("patience", 5, 30),
            }

            params = PatchTSTParams(**sampled_params)
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            trainer = PatchTSTTrainer(
                params=params,
                L=L,
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
            if trainer is not None:
                del trainer
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_trials = getattr(cfg, "n_trials", 20)
    timeout = getattr(cfg, "timeout", None)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

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

    for inp, patch, lr, scaler in itertools.product(input_lens, patch_lens, lrs, scalers):
        if inp % patch != 0:
            continue
        try:
            set_seed(42)
            pp.windowizer = SampleWindowizer(lookback=inp, horizon=H)
            X, y, series_ids, label_dates = pp.build_patch_train(df_full)
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

    os.makedirs("artifacts", exist_ok=True)
    pd.DataFrame(results).to_csv(Path("artifacts") / "patchtst_search.csv", index=False)


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
        X, y, series_ids, label_dates = pp.build_patch_train(df_full)
        cfg = TrainConfig(**TRAIN_CFG)
        cfg.n_trials = args.n_trials
        cfg.timeout = args.timeout
        tune_patchtst(X, y, series_ids, label_dates, cfg)

    if not args.lgbm and not args.patch:
        demo_study(args.n_trials)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

