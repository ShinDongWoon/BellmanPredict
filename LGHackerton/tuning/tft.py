from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Tuple

import json
import logging
import optuna
import numpy as np

from LGHackerton.config.default import ARTIFACTS_DIR
from LGHackerton.tuning.base import HyperparameterTuner
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.utils.seed import set_seed

try:  # optional, trainer may not be present
    from LGHackerton.models.tft_trainer import TFTParams, TFTTrainer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TFTParams = None  # type: ignore
    TFTTrainer = None  # type: ignore

try:  # torch is optional for CPU-only environments
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@dataclass
class TFTSearchSpace:
    """Search space for TFT hyperparameters."""

    hidden_size: Tuple[int, int] = (64, 512)
    lstm_layers: Tuple[int, int] = (1, 4)
    dropout: Tuple[float, float] = (0.0, 0.5)
    attention_heads: Tuple[int, int] = (1, 8)
    learning_rate: Tuple[float, float] = (1e-4, 1e-2)
    batch_size: Tuple[int, int] = (32, 256)
    max_epochs: Tuple[int, int] = (50, 200)
    patience: Tuple[int, int] = (5, 30)


class TFTTuner(HyperparameterTuner):
    """Optuna tuner for Temporal Fusion Transformer models."""

    search_space: TFTSearchSpace = TFTSearchSpace()

    def __init__(self, pp, df, cfg) -> None:  # type: ignore[override]
        super().__init__(pp, df, cfg)
        self._dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def prepare_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct training tensors for TFT."""

        if self._dataset is None:
            X, S, M, y, series_ids, label_dates = self.pp.build_patch_train(self.df)
            # TFT expects dynamic inputs X, targets y and identifiers
            self._dataset = (X, y, series_ids, label_dates)
        return self._dataset

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> None:  # type: ignore[override]
        required = {
            "hidden_size",
            "lstm_layers",
            "dropout",
            "attention_heads",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "patience",
        }
        for field in required:
            if field not in params:
                raise TypeError(f"Missing field: {field}")
        s = self.search_space
        def _chk_int(name: str, bounds: Tuple[int, int]) -> None:
            val = params.get(name)
            if not isinstance(val, int) or not (bounds[0] <= val <= bounds[1]):
                raise ValueError(f"{name} out of range: {val}")
        def _chk_float(name: str, bounds: Tuple[float, float]) -> None:
            val = params.get(name)
            if not isinstance(val, (float, int)) or not (bounds[0] <= float(val) <= bounds[1]):
                raise ValueError(f"{name} out of range: {val}")
        _chk_int("hidden_size", s.hidden_size)
        _chk_int("lstm_layers", s.lstm_layers)
        _chk_float("dropout", s.dropout)
        _chk_int("attention_heads", s.attention_heads)
        _chk_float("learning_rate", s.learning_rate)
        _chk_int("batch_size", s.batch_size)
        _chk_int("max_epochs", s.max_epochs)
        _chk_int("patience", s.patience)

    def run(self, n_trials: int, force: bool) -> dict:  # type: ignore[override]
        if TFTTrainer is None or TFTParams is None:
            raise RuntimeError("TFTTrainer not available")
        X, y, series_ids, label_dates = self.prepare_dataset()
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
                "hidden_size": trial.suggest_int("hidden_size", *search.hidden_size),
                "lstm_layers": trial.suggest_int("lstm_layers", *search.lstm_layers),
                "dropout": trial.suggest_float("dropout", *search.dropout),
                "attention_heads": trial.suggest_int("attention_heads", *search.attention_heads),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *search.learning_rate, log=True
                ),
                "batch_size": trial.suggest_int("batch_size", *search.batch_size),
                "max_epochs": trial.suggest_int("max_epochs", *search.max_epochs),
                "patience": trial.suggest_int("patience", *search.patience),
            }
            trainer_params = TFTParams(**params)
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            trainer = TFTTrainer(
                params=trainer_params,
                L=X.shape[1],
                H=y.shape[1],
                model_dir=getattr(self.cfg, "model_dir", "."),
                device=device,
            )
            trainer.train(X, y, series_ids, label_dates, self.cfg)
            oof = trainer.get_oof()
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
        params = TFTParams(**best)
        self._best_params = asdict(params)
        self.validate_params(self._best_params)
        return self.best_params()
