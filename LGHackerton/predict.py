
from __future__ import annotations
import os
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from LGHackerton.preprocess import Preprocessor, L, H
from LGHackerton.models import ModelRegistry
from LGHackerton.utils.device import select_device
from LGHackerton.config.default import (
    TEST_GLOB,
    ARTIFACTS_PATH,
    PRED_OUT,
    SUBMISSION_OUT,
    PATCH_PARAMS,
    TRAIN_CFG,
)
from LGHackerton.utils.seed import set_seed
from src.data.preprocess import inverse_symmetric_transform
from LGHackerton.postprocess import aggregate_predictions, convert_to_submission
import importlib, inspect

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def main():
    parser = argparse.ArgumentParser()
    available = ", ".join(ModelRegistry.available())
    parser.add_argument(
        "--model",
        default=None,
        help=f"model name ({available})",
    )  # omit to use PatchTST
    args = parser.parse_args()
    model_name = args.model or ModelRegistry.DEFAULT_MODEL  # default to PatchTST
    try:
        trainer_cls = ModelRegistry.get(model_name)
    except ValueError as e:
        parser.error(str(e))

    device = select_device()

    pp = Preprocessor(); pp.load(ARTIFACTS_PATH)

    from LGHackerton.models.base_trainer import TrainConfig
    cfg = TrainConfig(**TRAIN_CFG)
    set_seed(cfg.seed)

    module = importlib.import_module(trainer_cls.__module__)
    params_cls = None
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.endswith("Params"):
            params_cls = obj
            break
    if params_cls is None:
        raise RuntimeError(f"No Params dataclass found in {module.__name__}")

    params = params_cls(**PATCH_PARAMS)
    pt = trainer_cls(params=params, L=L, H=H, model_dir=cfg.model_dir, device=device)
    pt.load(os.path.join(cfg.model_dir, f"{model_name}.pt"))

    model_outputs = []

    single_outputs = []
    col_suffix = getattr(trainer_cls, "prediction_column_name", model_name)
    y_col = f"yhat_{col_suffix}"
    for path in sorted(glob.glob(TEST_GLOB)):
        df_eval_raw = _read_table(path)
        df_eval_full = pp.transform_eval(df_eval_raw)

        X_eval, sids, _ = trainer_cls.build_eval_dataset(pp, df_eval_full)
        sid_idx = np.array([pt.id2idx.get(sid, 0) for sid in sids])
        y_pred = pt.predict(X_eval, sid_idx)
        reps = np.repeat(sids, H)
        hs = np.tile(np.arange(1, H + 1), len(sids))
        out = pd.DataFrame({"series_id": reps, "h": hs, y_col: y_pred.reshape(-1)})

        out[y_col] = inverse_symmetric_transform(out[y_col].values)

        prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        out["test_id"] = prefix
        out["date"] = out["h"].map(lambda h: f"{prefix}+{h}일")
        single_outputs.append(out)

    pred_df = pd.concat(single_outputs, ignore_index=True)
    model_outputs.append(pred_df)

    # Example: to ensemble another model, append its dataframe here.
    # lgbm_df = pd.read_csv("artifacts/eval_lgbm.csv")
    # model_outputs.append(lgbm_df)

    os.makedirs(os.path.dirname(PRED_OUT), exist_ok=True)
    pred_df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")

    all_pred = aggregate_predictions(model_outputs)
    submission_df = convert_to_submission(all_pred)
    submission_df.to_csv(SUBMISSION_OUT, index=False, encoding="utf-8-sig")

    sub_path = Path(SUBMISSION_OUT)
    if not sub_path.exists():
        raise RuntimeError("submission file missing")
    out_df = pd.read_csv(sub_path)
    if not {"id", "y"}.issubset(out_df.columns):
        raise RuntimeError("submission columns missing")

if __name__ == "__main__":
    main()
