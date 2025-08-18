
from __future__ import annotations
import os
import glob
import re
import numpy as np
import pandas as pd
import logging

from LGHackerton.preprocess import Preprocessor, L, H
from LGHackerton.models.lgbm_trainer import LGBMTrainer, LGBMParams
from LGHackerton.models.patchtst_trainer import PatchTSTTrainer, PatchTSTParams, TORCH_OK
from LGHackerton.utils.ensemble_manager import EnsembleManager
from LGHackerton.utils.device import select_device
from LGHackerton.config.default import (
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
from LGHackerton.utils.seed import set_seed

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def convert_to_submission(pred_df: pd.DataFrame, sample_path: str) -> pd.DataFrame:
    sample_df = _read_table(sample_path)

    pred_df = pred_df.copy()
    pred_df["series_id"] = pred_df["series_id"].str.replace("::", "_", n=1)

    wide = pred_df.pivot(index="date", columns="series_id", values="yhat_ens")
    wide = wide.reindex(sample_df.iloc[:, 0]).reindex(columns=sample_df.columns[1:], fill_value=0.0)

    missing_dates = set(sample_df.iloc[:, 0]) - set(pred_df["date"])
    missing_cols = set(sample_df.columns[1:]) - set(pred_df["series_id"].unique())

    if missing_dates:
        logging.warning("Missing dates in predictions: %s", sorted(missing_dates))
    if missing_cols:
        logging.warning("Missing columns in predictions: %s", sorted(missing_cols))

    out_df = sample_df.copy()
    out_df.iloc[:, 1:] = wide.to_numpy()
    assert list(out_df.columns) == list(sample_df.columns)
    return out_df

def main():
    device = select_device()

    pp = Preprocessor(); pp.load(ARTIFACTS_PATH)

    from LGHackerton.models.base_trainer import TrainConfig
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)

    lgb = LGBMTrainer(params=LGBMParams(**LGBM_PARAMS), features=pp.feature_cols, model_dir=cfg.model_dir, device=device)
    lgb.load(os.path.join(cfg.model_dir, "lgbm_models.json"))

    pt = None
    if TORCH_OK and os.path.exists(os.path.join(cfg.model_dir, "patchtst.pt")):
        pt = PatchTSTTrainer(params=PatchTSTParams(**PATCH_PARAMS), L=L, H=H, model_dir=cfg.model_dir, device=device)
        pt.load(os.path.join(cfg.model_dir, "patchtst.pt"))

    ens = EnsembleManager()
    ens.load(os.path.join(cfg.model_dir, "ensemble_meta.json"))

    all_outputs = []

    for path in sorted(glob.glob(TEST_GLOB)):
        df_eval_raw = _read_table(path)
        df_eval_full = pp.transform_eval(df_eval_raw)

        lgbm_eval = pp.build_lgbm_eval(df_eval_full)
        df_lgb = lgb.predict(lgbm_eval)

        out = df_lgb.copy()
        if pt is not None:
            X_eval, sids, _ = pp.build_patch_eval(df_eval_full)
            sid_idx = np.array([pt.id2idx.get(sid, 0) for sid in sids])
            y_patch = pt.predict(X_eval, sid_idx)
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
