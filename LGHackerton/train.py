
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd

from .preprocess import Preprocessor, DATE_COL, SERIES_COL, SALES_COL, L, H
from .models.base_trainer import TrainConfig
from .models.lgbm_trainer import LGBMParams, LGBMTrainer
from .models.patchtst_trainer import PatchTSTParams, PatchTSTTrainer, TORCH_OK
from .config.default import (
    TRAIN_PATH,
    ARTIFACTS_PATH,
    LGBM_PARAMS,
    PATCH_PARAMS,
    TRAIN_CFG,
    ARTIFACTS_DIR,
    ENSEMBLE_CFG,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

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
    X_train, y_train, series_ids, label_dates = pp.build_patch_train(df_full)

    lgb_params = LGBMParams(**LGBM_PARAMS)
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)
    lgb_tr = LGBMTrainer(params=lgb_params, features=pp.feature_cols, model_dir=cfg.model_dir)
    lgb_tr.train(lgbm_train, cfg)
    lgb_tr.get_oof().to_csv(ARTIFACTS_DIR / "oof_lgbm.csv", index=False)

    if TORCH_OK:
        patch_params = PatchTSTParams(**PATCH_PARAMS)
        pt_tr = PatchTSTTrainer(params=patch_params, L=L, H=H, model_dir=cfg.model_dir)
        pt_tr.train(X_train, y_train, series_ids, label_dates, cfg)
        pt_tr.get_oof().to_csv(ARTIFACTS_DIR / "oof_patch.csv", index=False)

if __name__ == "__main__":
    main()
