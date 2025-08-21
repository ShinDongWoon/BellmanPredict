
from __future__ import annotations
import os
import glob
import re
from pathlib import Path
import pandas as pd
import argparse

from LGHackerton.preprocess import Preprocessor
from LGHackerton.models import ModelRegistry
from LGHackerton.utils.device import select_device
from LGHackerton.config.default import (
    TEST_GLOB,
    ARTIFACTS_PATH,
    PRED_OUT,
    SUBMISSION_OUT,
    TRAIN_CFG,
)
from LGHackerton.utils.seed import set_seed
from LGHackerton.postprocess import aggregate_predictions, convert_to_submission
from LGHackerton.utils.io import read_table
from LGHackerton.utils.params import load_best_params
import importlib, inspect


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

    params_dict, _ = load_best_params(model_name)
    params = params_cls(**params_dict)
    pt = trainer_cls(params=params, model_dir=cfg.model_dir, device=device)
    pt.load(os.path.join(cfg.model_dir, f"{model_name}.pt"))

    model_outputs = []

    single_outputs = []
    for path in sorted(glob.glob(TEST_GLOB)):
        df_eval_raw = read_table(path)
        df_eval_full = pp.transform_eval(df_eval_raw)

        eval_df = trainer_cls.build_eval_dataset(pp, df_eval_full)
        pred_df = pt.predict_df(eval_df)

        prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        pred_df["test_id"] = prefix
        pred_df["date"] = pred_df["h"].map(lambda h: f"{prefix}+{h}일")
        single_outputs.append(pred_df)

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
