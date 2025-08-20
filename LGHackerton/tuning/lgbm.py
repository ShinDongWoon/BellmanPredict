from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, fields
from typing import List, Tuple

import json
import logging
import optuna
import pandas as pd

from LGHackerton.config.default import ARTIFACTS_DIR
from LGHackerton.models.lgbm_trainer import LGBMParams, LGBMTrainer
from LGHackerton.tuning.base import HyperparameterTuner
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.utils.seed import set_seed


@dataclass
class LGBMSearchSpace:
    """Search space for LightGBM hyperparameters."""

    num_leaves: Tuple[int, int] = (31, 255)
    max_depth: Tuple[int, int] = (3, 16)
    learning_rate: Tuple[float, float] = (1e-3, 0.3)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    min_data_in_leaf: Tuple[int, int] = (10, 200)
    reg_alpha: Tuple[float, float] = (1e-8, 1.0)
    reg_lambda: Tuple[float, float] = (1e-8, 1.0)
    n_estimators: Tuple[int, int] = (500, 4000)


class LGBMTuner(HyperparameterTuner):
    """Optuna tuner for the LightGBM model."""

    search_space: LGBMSearchSpace = LGBMSearchSpace()

    def __init__(self, pp, df, cfg) -> None:  # type: ignore[override]
        super().__init__(pp, df, cfg)
        self._dataset: tuple[pd.DataFrame, List[str]] | None = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def prepare_dataset(self) -> tuple[pd.DataFrame, List[str]]:
        """Build the training dataframe and feature list."""

        if self._dataset is None:
            df_train = self.pp.build_lgbm_train(self.df)
            feats = list(self.pp.feature_cols)
            self._dataset = (df_train, feats)
        return self._dataset

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> None:  # type: ignore[override]
        required = {f.name for f in fields(LGBMParams)}
        for field in required:
            if field not in params:
                raise TypeError(f"Missing field: {field}")
        s = self.search_space
        int_fields = {
            "num_leaves": s.num_leaves,
            "max_depth": s.max_depth,
            "min_data_in_leaf": s.min_data_in_leaf,
            "n_estimators": s.n_estimators,
        }
        for name, (lo, hi) in int_fields.items():
            val = params.get(name)
            if not isinstance(val, int) or not (lo <= val <= hi):
                raise ValueError(f"{name} out of range: {val}")
        float_fields = {
            "learning_rate": s.learning_rate,
            "subsample": s.subsample,
            "colsample_bytree": s.colsample_bytree,
            "reg_alpha": s.reg_alpha,
            "reg_lambda": s.reg_lambda,
        }
        for name, (lo, hi) in float_fields.items():
            val = params.get(name)
            if not isinstance(val, (float, int)) or not (lo <= float(val) <= hi):
                raise ValueError(f"{name} out of range: {val}")

    def run(self, n_trials: int, force: bool) -> dict:  # type: ignore[override]
        df_train, feat_cols = self.prepare_dataset()
        if not force:
            cache = self.artifact_dir / "best_params.json"
            if cache.exists():
                with cache.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                self.validate_params(cached)
                self._best_params = cached
                return cached
        study = optuna.create_study(direction="minimize")
        search = self.search_space

        def objective(trial: optuna.Trial) -> float:
            set_seed(self.cfg.seed)
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *search.num_leaves),
                "max_depth": trial.suggest_int("max_depth", *search.max_depth),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *search.learning_rate, log=True
                ),
                "subsample": trial.suggest_float("subsample", *search.subsample),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *search.colsample_bytree
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", *search.min_data_in_leaf
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *search.reg_alpha, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *search.reg_lambda, log=True
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators", *search.n_estimators
                ),
            }
            trainer_params = LGBMParams(**params)
            trainer = LGBMTrainer(
                trainer_params,
                feat_cols,
                getattr(self.cfg, "model_dir", "."),
                device="cpu",
            )
            trainer.train(df_train, self.cfg, preprocessors=[self.pp])
            oof = trainer.get_oof()
            if oof.empty:
                raise optuna.TrialPruned()
            outlets = oof["series_id"].str.split("::").str[0].values
            score = weighted_smape_np(
                oof["y"].to_numpy(),
                oof["yhat"].to_numpy(),
                outlets,
                priority_weight=getattr(self.cfg, "priority_weight", 1.0),
            )
            gc.collect()
            return float(score)

        study.optimize(objective, n_trials=n_trials)
        try:
            best = dict(study.best_trial.params)
        except Exception:
            logging.warning("Hyperparameter search stopped early; no trials completed")
            self._best_params = {}
            return self._best_params
        params = LGBMParams(**best)
        self._best_params = asdict(params)
        self.validate_params(self._best_params)
        return self.best_params()
