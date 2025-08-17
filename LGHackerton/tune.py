from __future__ import annotations
import json
from pathlib import Path
from typing import List

import optuna
import pandas as pd

from LGHackerton.models.lgbm_trainer import LGBMParams, LGBMTrainer
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.config.default import ARTIFACTS_DIR


def tune_lgbm(df_train: pd.DataFrame, features: List[str], cfg: TrainConfig):
    """Hyper-parameter tuning for LightGBM using Optuna.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Preprocessed training dataframe ready for LGBMTrainer.
    features : list[str]
        List of feature column names to use for training.
    cfg : TrainConfig
        Training configuration. Should contain attributes for
        "model_dir" and optional "n_trials" and "timeout" for optuna.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        sampled_params = {
            "objective": "tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.6),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
            "early_stopping_rounds": 200,
        }

        params = LGBMParams(**sampled_params)
        trial_dir = Path(getattr(cfg, "model_dir", ".")) / f"optuna_trial_{trial.number}"
        trainer = LGBMTrainer(
            params=params,
            features=features,
            model_dir=str(trial_dir),
            device=getattr(cfg, "device", "cpu"),
        )
        trainer.train(df_train, cfg)
        oof = trainer.get_oof()
        outlets = oof["series_id"].str.split("::").str[0]
        loss = weighted_smape_np(
            oof["y"].values,
            oof["yhat"].values,
            outlet_names=outlets,
            priority_weight=getattr(cfg, "priority_weight", 3.0),
        )
        return loss

    study = optuna.create_study(direction="minimize")
    n_trials = int(getattr(cfg, "n_trials", 20))
    timeout = getattr(cfg, "timeout", None)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    optuna_dir = ARTIFACTS_DIR / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    out_path = optuna_dir / "lgbm_best.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"params": study.best_params, "value": study.best_value}, f, ensure_ascii=False, indent=2)

    return study
