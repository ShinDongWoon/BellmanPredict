
from __future__ import annotations
import os, sys
import pandas as pd

try:
    from preprocess_pipeline_v1_1 import Preprocessor, DATE_COL, SERIES_COL, SALES_COL, L, H
except Exception as e:
    raise RuntimeError("preprocess_pipeline_v1_1.py를 /mnt/data에 두세요.") from e

from models.base_trainer import TrainConfig
from models.lgbm_trainer import LGBMParams, LGBMTrainer
from models.patchtst_trainer import PatchTSTParams, PatchTSTTrainer, TORCH_OK
from config.default import (TRAIN_PATH, ARTIFACTS_PATH, LGBM_PARAMS, PATCH_PARAMS, TRAIN_CFG)

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

def main():
    df_train_raw = _read_table(TRAIN_PATH)
    pp = Preprocessor()
    df_full = pp.fit_transform_train(df_train_raw)
    pp.save(ARTIFACTS_PATH)

    lgbm_train = pp.build_lgbm_train(df_full)
    X_train, y_train, label_dates = pp.build_patch_train(df_full)

    lgb_params = LGBMParams(**LGBM_PARAMS)
    cfg = TrainConfig(**TRAIN_CFG)
    lgb_tr = LGBMTrainer(params=lgb_params, features=pp.feature_cols, model_dir=cfg.model_dir)
    lgb_tr.train(lgbm_train, cfg)

    if TORCH_OK:
        patch_params = PatchTSTParams(**PATCH_PARAMS)
        pt_tr = PatchTSTTrainer(params=patch_params, L=L, H=H, model_dir=cfg.model_dir)
        pt_tr.train(X_train, y_train, label_dates, cfg)

if __name__ == "__main__":
    main()
