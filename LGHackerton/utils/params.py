import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from LGHackerton.config.default import PATCH_PARAMS, ARTIFACTS_DIR, OPTUNA_DIR


def _load_patchtst_params() -> Tuple[dict, int | None]:
    """Load PatchTST parameters from tuning artifacts.

    Preference order:
    1. Grid search results saved in ``artifacts/patchtst_search.csv``. The
       combination with the lowest ``val_wsmape`` is selected and its
       ``input_len`` is returned separately to adjust window sizing.
    2. Optuna best parameters stored in ``OPTUNA_DIR/patchtst_best.json``. The
       JSON may also include ``input_len`` which is returned separately.
    3. Default ``PATCH_PARAMS`` when no artifacts are available.

    Returns
    -------
    params : dict
        Parameters for :class:`PatchTSTParams`.
    input_len : int | None
        Lookback length if provided by artifacts, otherwise ``None``.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    search_path = ARTIFACTS_DIR / "patchtst_search.csv"
    if search_path.exists():
        try:
            df = pd.read_csv(search_path)
            if "val_wsmape" in df.columns:
                df = df[df["val_wsmape"].notna()]
                if not df.empty:
                    row = df.loc[df["val_wsmape"].idxmin()]
                    params = {
                        **PATCH_PARAMS,
                        "patch_len": int(row["patch_len"]),
                        "stride": int(row["patch_len"]),
                        "lr": float(row["lr"]),
                        "scaler": row["scaler"],
                    }
                    return params, int(row["input_len"])
        except Exception as e:  # pragma: no cover - best effort
            logging.warning("Failed to parse %s: %s", search_path, e)

    best_path = Path(OPTUNA_DIR) / "patchtst_best.json"
    if not best_path.exists():
        alt_path = ARTIFACTS_DIR / "patchtst" / "best_params.json"
        if alt_path.exists():
            best_path = alt_path
    try:
        with best_path.open("r", encoding="utf-8") as f:
            patch_best = json.load(f)
        patch_best.setdefault("stride", patch_best.get("patch_len", PATCH_PARAMS["patch_len"]))
        input_len = patch_best.pop("input_len", None)
        return {**PATCH_PARAMS, **patch_best}, input_len
    except Exception as e:  # pragma: no cover - best effort
        logging.warning("Failed to load PatchTST params from %s: %s", best_path, e)
        return PATCH_PARAMS, None


def load_best_params(model_name: str) -> Tuple[dict, int | None]:
    """Dispatch to a model-specific parameter loader."""

    loaders = {
        "patchtst": _load_patchtst_params,
    }
    loader = loaders.get(model_name)
    if loader is None:
        logging.info("No specialized loader for %s; using empty parameters", model_name)
        return {}, None
    return loader()
