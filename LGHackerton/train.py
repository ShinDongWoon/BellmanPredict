
from __future__ import annotations
import argparse
import os
import json
import logging
from pathlib import Path
import pandas as pd

from LGHackerton.preprocess import Preprocessor, DATE_COL, SERIES_COL, SALES_COL, L, H
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.models.lgbm_trainer import LGBMParams, LGBMTrainer
from LGHackerton.models.patchtst_trainer import PatchTSTParams, PatchTSTTrainer, TORCH_OK
from LGHackerton.utils.device import select_device
from LGHackerton.utils.diagnostics import (
    compute_acf,
    compute_pacf,
    ljung_box_test,
    white_test,
    plot_residuals,
)
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


def load_best_lgbm_params() -> dict:
    """Load best LightGBM params from Optuna study if available."""
    study_path = Path(OPTUNA_DIR) / "lgbm_study.json"
    try:
        with study_path.open("r", encoding="utf-8") as f:
            trials = json.load(f)
        best = min(trials, key=lambda t: t["value"])
        return {**LGBM_PARAMS, **best["params"]}
    except Exception as e:  # pragma: no cover - best effort
        logging.warning("Failed to load LGBM params from %s: %s", study_path, e)
        return LGBM_PARAMS


def load_best_patch_params() -> dict:
    """Load best PatchTST params from Optuna results if available."""
    best_path = Path(OPTUNA_DIR) / "patchtst_best.json"
    try:
        with best_path.open("r", encoding="utf-8") as f:
            patch_best = json.load(f)
        return {**PATCH_PARAMS, **patch_best}
    except Exception as e:  # pragma: no cover - best effort
        logging.warning("Failed to load PatchTST params from %s: %s", best_path, e)
        return PATCH_PARAMS


def main(show_progress: bool | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="show preprocessing progress")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="hide preprocessing progress")
    parser.add_argument("--skip-tune", action="store_true", help="skip hyperparameter tuning")
    parser.add_argument("--force-tune", action="store_true", help="re-run tuning even if artifacts exist")
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

    if not args.skip_tune:
        study_file = Path(OPTUNA_DIR) / "lgbm_study.json"
        if args.force_tune or not study_file.exists():
            tune_lgbm(args.trials, args.timeout)
    lgbm_params_dict = load_best_lgbm_params()
    lgb_params = LGBMParams(**lgbm_params_dict)

    cfg = TrainConfig(**TRAIN_CFG)
    cfg.n_trials = args.trials
    cfg.timeout = args.timeout
    set_seed(cfg.seed)
    lgb_tr = LGBMTrainer(params=lgb_params, features=pp.feature_cols, model_dir=cfg.model_dir, device=device)
    lgb_tr.train(lgbm_train, cfg)
    oof_lgbm = lgb_tr.get_oof()
    oof_lgbm.to_csv(OOF_LGBM_OUT, index=False)

    # diagnostics for LGBM
    try:
        res = oof_lgbm["y"] - oof_lgbm["yhat"]
        diag_dir = ARTIFACTS_DIR / "diagnostics" / "lgbm" / "oof"
        diag_dir.mkdir(parents=True, exist_ok=True)
        acf_df = compute_acf(res)
        pacf_df = compute_pacf(res)
        lb_df = ljung_box_test(res, lags=[10, 20, 30])
        wt_df = white_test(res)
        acf_df.to_csv(diag_dir / "acf.csv", index=False)
        pacf_df.to_csv(diag_dir / "pacf.csv", index=False)
        lb_df.to_csv(diag_dir / "ljung_box.csv", index=False)
        wt_df.to_csv(diag_dir / "white_test.csv", index=False)
        plot_residuals(res, diag_dir)
        logging.info("LGBM Ljung-Box p-values: %s", lb_df["pvalue"].tolist())
        logging.info("LGBM White test p-value: %s", wt_df["lm_pvalue"].iloc[0])
    except Exception as e:  # pragma: no cover - best effort
        logging.warning("LGBM diagnostics failed: %s", e)

    if TORCH_OK and not args.skip_tune:
        patch_file = Path(OPTUNA_DIR) / "patchtst_best.json"
        if args.force_tune or not patch_file.exists():
            tune_patchtst(X_train, y_train, series_ids, label_dates, cfg)

    if TORCH_OK:
        patch_params_dict = load_best_patch_params()
        patch_params = PatchTSTParams(**patch_params_dict)
        pt_tr = PatchTSTTrainer(params=patch_params, L=L, H=H, model_dir=cfg.model_dir, device=device)
        pt_tr.train(X_train, y_train, series_ids, label_dates, cfg)
        oof_patch = pt_tr.get_oof()
        oof_patch.to_csv(OOF_PATCH_OUT, index=False)

        # diagnostics for PatchTST
        try:
            res_p = oof_patch["y"] - oof_patch["yhat"]
            diag_dir = ARTIFACTS_DIR / "diagnostics" / "patchtst" / "oof"
            diag_dir.mkdir(parents=True, exist_ok=True)
            acf_df = compute_acf(res_p)
            pacf_df = compute_pacf(res_p)
            lb_df = ljung_box_test(res_p, lags=[10, 20, 30])
            wt_df = white_test(res_p)
            acf_df.to_csv(diag_dir / "acf.csv", index=False)
            pacf_df.to_csv(diag_dir / "pacf.csv", index=False)
            lb_df.to_csv(diag_dir / "ljung_box.csv", index=False)
            wt_df.to_csv(diag_dir / "white_test.csv", index=False)
            plot_residuals(res_p, diag_dir)
            logging.info("PatchTST Ljung-Box p-values: %s", lb_df["pvalue"].tolist())
            logging.info("PatchTST White test p-value: %s", wt_df["lm_pvalue"].iloc[0])
        except Exception as e:  # pragma: no cover - best effort
            logging.warning("PatchTST diagnostics failed: %s", e)

if __name__ == "__main__":
    main()
