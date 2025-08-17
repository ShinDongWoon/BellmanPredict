
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
import pandas as pd

from LGHackerton.preprocess import Preprocessor, DATE_COL, SERIES_COL, SALES_COL, L, H
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.models.lgbm_trainer import LGBMParams, LGBMTrainer
from LGHackerton.models.patchtst_trainer import PatchTSTParams, PatchTSTTrainer, TORCH_OK
from LGHackerton.utils.device import select_device
from LGHackerton.config.default import (
    TRAIN_PATH,
    ARTIFACTS_PATH,
    LGBM_PARAMS,
    PATCH_PARAMS,
    TRAIN_CFG,
    ARTIFACTS_DIR,
    ENSEMBLE_CFG,
    OOF_LGBM_OUT,
    OOF_PATCH_OUT,
    SHOW_PROGRESS,
    OPTUNA_DIR,
)
from LGHackerton.tune import tune_lgbm, tune_patchtst
from LGHackerton.utils.seed import set_seed


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def main(show_progress: bool | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="show preprocessing progress")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="hide preprocessing progress")
    parser.add_argument("--tune-lgbm", action="store_true", help="tune LightGBM hyperparameters")
    parser.add_argument("--tune-patch", action="store_true", help="tune PatchTST hyperparameters")
    parser.add_argument("--trials", type=int, default=20, help="number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="time limit for tuning (seconds)")
    parser.set_defaults(show_progress=SHOW_PROGRESS)

    args = parser.parse_args()
    if show_progress is None:
        show_progress = args.show_progress

    device = select_device()  # ask user for compute environment

    df_train_raw = _read_table(TRAIN_PATH)
    pp = Preprocessor(show_progress=show_progress)
    df_full = pp.fit_transform_train(df_train_raw)
    pp.save(ARTIFACTS_PATH)

    lgbm_train = pp.build_lgbm_train(df_full)
    X_train, y_train, series_ids, label_dates = pp.build_patch_train(df_full)

    if args.tune_lgbm:
        tune_lgbm(args.trials, args.timeout)
        study_path = Path(OPTUNA_DIR) / "lgbm_study.json"
        with study_path.open("r", encoding="utf-8") as f:
            trials = json.load(f)
        best = min(trials, key=lambda t: t["value"])
        lgbm_params_dict = {**LGBM_PARAMS, **best["params"]}
    else:
        lgbm_params_dict = LGBM_PARAMS
    lgb_params = LGBMParams(**lgbm_params_dict)

    cfg = TrainConfig(**TRAIN_CFG)
    cfg.n_trials = args.trials
    cfg.timeout = args.timeout
    set_seed(cfg.seed)
    lgb_tr = LGBMTrainer(params=lgb_params, features=pp.feature_cols, model_dir=cfg.model_dir, device=device)
    lgb_tr.train(lgbm_train, cfg)
    lgb_tr.get_oof().to_csv(OOF_LGBM_OUT, index=False)

    if TORCH_OK:
        if args.tune_patch:
            tune_patchtst(X_train, y_train, series_ids, label_dates, cfg)
            best_path = Path(OPTUNA_DIR) / "patchtst_best.json"
            with best_path.open("r", encoding="utf-8") as f:
                patch_best = json.load(f)
            patch_params_dict = {**PATCH_PARAMS, **patch_best}
        else:
            patch_params_dict = PATCH_PARAMS
        patch_params = PatchTSTParams(**patch_params_dict)
        pt_tr = PatchTSTTrainer(params=patch_params, L=L, H=H, model_dir=cfg.model_dir, device=device)
        pt_tr.train(X_train, y_train, series_ids, label_dates, cfg)
        pt_tr.get_oof().to_csv(OOF_PATCH_OUT, index=False)
    elif args.tune_patch:
        raise RuntimeError("PyTorch not available; cannot tune PatchTST")

if __name__ == "__main__":
    main()
