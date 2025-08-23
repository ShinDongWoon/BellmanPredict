"""Hyperparameter tuning utilities using Optuna.

This module focuses on searches for the PatchTST model while retaining a
minimal Optuna demo objective for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List
import sys

import itertools

import numpy as np
import optuna
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:  # torch is optional; used only for GPU cache clearing
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from LGHackerton.config.default import (
    OPTUNA_DIR,
    TRAIN_PATH,
    TRAIN_CFG,
    ARTIFACTS_DIR,
    PATCH_PARAMS,
)
from LGHackerton.preprocess import Preprocessor
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.utils.metrics import weighted_smape_np
from LGHackerton.utils.seed import set_seed
from LGHackerton.tuning.patchtst import PatchTSTTuner

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

def run_patchtst_grid_search(cfg_path: str | Path) -> None:
    """Run a simple grid search over PatchTST hyperparameters."""

    from LGHackerton.models.patchtst.trainer import (
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
    dataset_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # Prebuild datasets for each unique input length
    for inp in input_lens:
        set_seed(42)
        pp.windowizer = SampleWindowizer(lookback=inp, horizon=H)
        dataset_cache[inp] = pp.build_patch_train(df_full)

    # Iterate over grid while reusing cached datasets
    for inp in input_lens:
        X, S, y, series_ids, label_dates = dataset_cache[inp]
        for patch, lr, scaler in itertools.product(patch_lens, lrs, scalers):
            if inp % patch != 0:
                continue
            try:
                set_seed(42)
                params = PatchTSTParams(
                    patch_len=patch,
                    stride=patch,
                    lr=lr,
                    scaler=scaler,
                    num_workers=PATCH_PARAMS.get("num_workers", 0),
                )
                trainer = PatchTSTTrainer(
                    params=params, L=inp, H=H, model_dir=cfg.model_dir, device=device
                )
                trainer.train(X, S, y, series_ids, label_dates, cfg)
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
        tuner = PatchTSTTuner(pp, df_full, cfg)
        tuner.run(n_trials=args.n_trials, force=False)
        tuner.best_params()

    if not args.patch:
        demo_study(args.n_trials)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
