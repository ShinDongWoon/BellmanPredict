"""Hyperparameter tuning utilities using Optuna.

This module provides a minimal example of running an Optuna study and
storing its results under :data:`ARTIFACTS_DIR/"optuna"`.
"""

from __future__ import annotations

import json
import optuna

from LGHackerton.config.default import OPTUNA_DIR


def objective(trial: optuna.Trial) -> float:
    """Simple objective function for demonstration purposes.

    The function is intentionally lightweight so that it can be executed in
    restricted environments. Replace this logic with model training and
    evaluation to perform real hyperparameter optimisation.
    """

    x = trial.suggest_float("x", -10.0, 10.0)
    return x * x


def main(n_trials: int = 20) -> None:
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


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

