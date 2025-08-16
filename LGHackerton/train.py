
from __future__ import annotations
import argparse
import os
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
)
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
    parser.set_defaults(show_progress=SHOW_PROGRESS)
    if show_progress is None:
        args = parser.parse_args()
        show_progress = args.show_progress

    device = select_device()  # ask user for compute environment

    df_train_raw = _read_table(TRAIN_PATH)
    pp = Preprocessor(show_progress=show_progress)
    df_full = pp.fit_transform_train(df_train_raw)
    pp.save(ARTIFACTS_PATH)

    lgbm_train = pp.build_lgbm_train(df_full)
    X_train, y_train, series_ids, label_dates = pp.build_patch_train(df_full)

    lgb_params = LGBMParams(**LGBM_PARAMS)
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)
    lgb_tr = LGBMTrainer(params=lgb_params, features=pp.feature_cols, model_dir=cfg.model_dir, device=device)
    lgb_tr.train(lgbm_train, cfg)
    lgb_tr.get_oof().to_csv(OOF_LGBM_OUT, index=False)

    if TORCH_OK:
        patch_params = PatchTSTParams(**PATCH_PARAMS)
        pt_tr = PatchTSTTrainer(params=patch_params, L=L, H=H, model_dir=cfg.model_dir, device=device)
        pt_tr.train(X_train, y_train, series_ids, label_dates, cfg)
        pt_tr.get_oof().to_csv(OOF_PATCH_OUT, index=False)

if __name__ == "__main__":
    main()
