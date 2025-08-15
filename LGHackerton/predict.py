
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.append("/mnt/data")
try:
    from preprocess_pipeline_v1_1 import Preprocessor, L, H
except Exception as e:
    raise RuntimeError("preprocess_pipeline_v1_1.py를 /mnt/data에 두세요.") from e

from models.lgbm_trainer import LGBMTrainer, LGBMParams
from models.patchtst_trainer import PatchTSTTrainer, PatchTSTParams, TORCH_OK
from config.default import (EVAL_PATH, ARTIFACTS_PATH, LGBM_EVAL_OUT, PATCH_EVAL_OUT,
                            LGBM_PARAMS, PATCH_PARAMS, TRAIN_CFG)

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

def main():
    pp = Preprocessor(); pp.load(ARTIFACTS_PATH)
    df_eval_raw = _read_table(EVAL_PATH)
    df_eval_full = pp.transform_eval(df_eval_raw)

    lgbm_eval = pp.build_lgbm_eval(df_eval_full)
    X_eval, meta = pp.build_patch_eval(df_eval_full)

    from models.base_trainer import TrainConfig
    cfg = TrainConfig(**TRAIN_CFG)
    lgb = LGBMTrainer(params=LGBMParams(**LGBM_PARAMS), features=pp.feature_cols, model_dir=cfg.model_dir)
    lgb.load(os.path.join(cfg.model_dir, "lgbm_models.json"))
    df_lgb = lgb.predict(lgbm_eval)

    y_patch=None
    if TORCH_OK and os.path.exists(os.path.join(cfg.model_dir,"patchtst.pt")):
        pt = PatchTSTTrainer(params=PatchTSTParams(**PATCH_PARAMS), L=L, H=H, model_dir=cfg.model_dir)
        pt.load(os.path.join(cfg.model_dir,"patchtst.pt"))
        y_patch = pt.predict(X_eval)

    out = df_lgb.copy()
    if y_patch is not None:
        sids = [sid for sid,_ in meta]
        reps = np.repeat(sids, H); hs = np.tile(np.arange(1,H+1), len(sids))
        dfp = pd.DataFrame({"series_id": reps, "h": hs, "yhat_patch": y_patch.reshape(-1)})
        out = out.merge(dfp, on=["series_id","h"], how="left")
        out["yhat_ens_median"] = np.where(out["yhat_patch"].notna(),
                                          np.median(np.stack([out["yhat_lgbm"].values, out["yhat_patch"].values],0),0),
                                          out["yhat_lgbm"].values)
    else:
        out["yhat_patch"]=np.nan; out["yhat_ens_median"]=out["yhat_lgbm"]

    os.makedirs(os.path.dirname(LGBM_EVAL_OUT), exist_ok=True)
    out.to_csv(os.path.join(os.path.dirname(LGBM_EVAL_OUT), "predictions.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
