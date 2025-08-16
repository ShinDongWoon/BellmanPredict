
from __future__ import annotations
import os
import random
import glob
import re
import numpy as np
import pandas as pd

from .preprocess import Preprocessor, L, H
from .models.lgbm_trainer import LGBMTrainer, LGBMParams
from .models.patchtst_trainer import PatchTSTTrainer, PatchTSTParams, TORCH_OK
from .utils.ensemble_manager import EnsembleManager
from .config.default import (
    TEST_GLOB,
    ARTIFACTS_PATH,
    LGBM_EVAL_OUT,
    SAMPLE_SUB_PATH,
    SUBMISSION_OUT,
    LGBM_PARAMS,
    PATCH_PARAMS,
    TRAIN_CFG,
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


def convert_to_submission(pred_df: pd.DataFrame, sample_path: str) -> pd.DataFrame:
    sample_df = _read_table(sample_path)
    pred_dict = {(row.date, row.series_id): row.yhat_ens for row in pred_df.itertuples()}
    date_col = sample_df.columns[0]
    out_df = sample_df.copy()
    for idx, row in out_df.iterrows():
        date = row[date_col]
        for col in out_df.columns[1:]:
            out_df.at[idx, col] = pred_dict.get((date, col), 0.0)
    return out_df

def main():
    pp = Preprocessor(); pp.load(ARTIFACTS_PATH)

    from .models.base_trainer import TrainConfig
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)

    lgb = LGBMTrainer(params=LGBMParams(**LGBM_PARAMS), features=pp.feature_cols, model_dir=cfg.model_dir)
    lgb.load(os.path.join(cfg.model_dir, "lgbm_models.json"))

    pt = None
    if TORCH_OK and os.path.exists(os.path.join(cfg.model_dir, "patchtst.pt")):
        pt = PatchTSTTrainer(params=PatchTSTParams(**PATCH_PARAMS), L=L, H=H, model_dir=cfg.model_dir)
        pt.load(os.path.join(cfg.model_dir, "patchtst.pt"))

    ens = EnsembleManager()
    ens.load(os.path.join(cfg.model_dir, "ensemble_meta.json"))

    all_outputs = []

    for path in sorted(glob.glob(TEST_GLOB)):
        df_eval_raw = _read_table(path)
        df_eval_full = pp.transform_eval(df_eval_raw)

        lgbm_eval = pp.build_lgbm_eval(df_eval_full)
        X_eval, sids, _ = pp.build_patch_eval(df_eval_full)

        df_lgb = lgb.predict(lgbm_eval)

        y_patch = None
        if pt is not None:
            sid_idx = np.array([pt.id2idx[sid] for sid in sids])
            y_patch = pt.predict(X_eval, sid_idx)

        out = df_lgb.copy()
        if y_patch is not None:
            reps = np.repeat(sids, H)
            hs = np.tile(np.arange(1, H + 1), len(sids))
            dfp = pd.DataFrame({"series_id": reps, "h": hs, "yhat_patch": y_patch.reshape(-1)})
            out = out.merge(dfp, on=["series_id", "h"], how="left")
        else:
            out["yhat_patch"] = np.nan

        yhat_ens = ens.predict(
            df_lgb["yhat_lgbm"].values,
            out.get("yhat_patch").values,
        )
        out["yhat_ens"] = np.where(out["yhat_patch"].notna(), yhat_ens, df_lgb["yhat_lgbm"].values)

        prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        out["test_id"] = prefix
        out["date"] = out["h"].map(lambda h: f"{prefix}+{h}일")
        all_outputs.append(out)

    os.makedirs(os.path.dirname(LGBM_EVAL_OUT), exist_ok=True)
    all_pred = pd.concat(all_outputs, ignore_index=True)
    all_pred.to_csv(LGBM_EVAL_OUT, index=False, encoding="utf-8-sig")

    submission_df = convert_to_submission(all_pred, SAMPLE_SUB_PATH)
    submission_df.to_csv(SUBMISSION_OUT, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
