
from __future__ import annotations
import os
import glob
import re
import numpy as np
import pandas as pd
import argparse

from LGHackerton.preprocess import Preprocessor, L, H
from LGHackerton.models import ModelRegistry
from LGHackerton.models.patchtst.trainer import PatchTSTParams
from LGHackerton.utils.device import select_device
from LGHackerton.config.default import (
    TEST_GLOB,
    ARTIFACTS_PATH,
    PATCH_PRED_OUT,
    SUBMISSION_OUT,
    PATCH_PARAMS,
    TRAIN_CFG,
)
from LGHackerton.utils.seed import set_seed
from src.data.preprocess import inverse_symmetric_transform
from LGHackerton.postprocess import aggregate_predictions, convert_to_submission

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def main():
    parser = argparse.ArgumentParser()
    available = ", ".join(ModelRegistry.available())
    parser.add_argument("--model", default="patchtst", help=f"model name ({available})")
    args = parser.parse_args()
    try:
        trainer_cls = ModelRegistry.get(args.model)
    except KeyError as e:
        parser.error(str(e))

    device = select_device()

    pp = Preprocessor(); pp.load(ARTIFACTS_PATH)

    from LGHackerton.models.base_trainer import TrainConfig
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)

    pt = trainer_cls(params=PatchTSTParams(**PATCH_PARAMS), L=L, H=H, model_dir=cfg.model_dir, device=device)
    pt.load(os.path.join(cfg.model_dir, f"{args.model}.pt"))

    model_outputs = []

    patch_outputs = []
    for path in sorted(glob.glob(TEST_GLOB)):
        df_eval_raw = _read_table(path)
        df_eval_full = pp.transform_eval(df_eval_raw)

        X_eval, sids, _ = pp.build_patch_eval(df_eval_full)
        sid_idx = np.array([pt.id2idx.get(sid, 0) for sid in sids])
        y_patch = pt.predict(X_eval, sid_idx)
        reps = np.repeat(sids, H)
        hs = np.tile(np.arange(1, H + 1), len(sids))
        out = pd.DataFrame({"series_id": reps, "h": hs, "yhat_patch": y_patch.reshape(-1)})

        out["yhat_patch"] = inverse_symmetric_transform(out["yhat_patch"].values)

        prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        out["test_id"] = prefix
        out["date"] = out["h"].map(lambda h: f"{prefix}+{h}일")
        patch_outputs.append(out)

    # Combine outputs from all test files for the PatchTST model
    patch_df = pd.concat(patch_outputs, ignore_index=True)
    model_outputs.append(patch_df)

    # Example: to ensemble another model, append its dataframe here.
    # lgbm_df = pd.read_csv("artifacts/eval_lgbm.csv")
    # model_outputs.append(lgbm_df)

    os.makedirs(os.path.dirname(PATCH_PRED_OUT), exist_ok=True)
    patch_df.to_csv(PATCH_PRED_OUT, index=False, encoding="utf-8-sig")

    all_pred = aggregate_predictions(model_outputs)
    submission_df = convert_to_submission(all_pred)
    submission_df.to_csv(SUBMISSION_OUT, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
