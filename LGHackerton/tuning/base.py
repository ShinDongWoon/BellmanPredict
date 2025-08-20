from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from LGHackerton.config.default import ARTIFACTS_DIR
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.preprocess import Preprocessor


class HyperparameterTuner(ABC):
    """Abstract base class for hyperparameter search utilities.

    Parameters
    ----------
    pp : Preprocessor
        Preprocessor instance used to generate features.
    df : pd.DataFrame
        Raw dataframe containing the training data.
    cfg : TrainConfig
        Training configuration for the underlying model.
    """

    def __init__(self, pp: Preprocessor, df: pd.DataFrame, cfg: TrainConfig) -> None:
        self.pp = pp
        self.df = df
        self.cfg = cfg
        self.model_name = getattr(cfg, "model_name", self.__class__.__name__)
        self.artifact_dir: Path = ARTIFACTS_DIR / self.model_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._best_params: dict | None = None

    @abstractmethod
    def run(self, n_trials: int, force: bool) -> dict:
        """Execute a hyperparameter search strategy.

        Implementations should perform the optimisation (e.g., via Optuna) and
        populate ``self._best_params`` with the best hyperparameter dictionary.
        ``validate_params`` **must** be invoked on the final selection before
        returning.

        Parameters
        ----------
        n_trials : int
            Number of trials to evaluate.
        force : bool
            If ``True``, ignore any cached studies and rerun the search from
            scratch.

        Returns
        -------
        dict
            Best hyperparameter set discovered during the search.
        """

    def best_params(self) -> dict:
        """Return and persist the best hyperparameter dictionary.

        The parameters are written to ``best_params.json`` inside
        ``ARTIFACTS_DIR / model_name`` for reproducibility.

        Returns
        -------
        dict
            The best hyperparameters from the last :meth:`run` execution.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been executed yet.
        """

        if self._best_params is None:
            raise RuntimeError("run() must be executed before accessing best_params")
        out_path = self.artifact_dir / "best_params.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self._best_params, f, ensure_ascii=False, indent=2)
        return self._best_params

    @abstractmethod
    def validate_params(self, params: dict) -> None:
        """Validate a candidate hyperparameter configuration.

        Parameters
        ----------
        params : dict
            Hyperparameter dictionary returned by :meth:`run`.

        Raises
        ------
        ValueError
            If the provided parameters are invalid or incomplete.
        """
