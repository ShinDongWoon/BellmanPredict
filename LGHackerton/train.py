
from __future__ import annotations
import argparse
import importlib
import json
import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import asdict
from datetime import datetime

from LGHackerton.preprocess import Preprocessor, L, H
from LGHackerton.preprocess.preprocess_pipeline_v1_1 import SampleWindowizer
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.models.patchtst.trainer import PatchTSTTrainer, TORCH_OK
from LGHackerton.models import ModelRegistry
from LGHackerton.utils.device import select_device
from LGHackerton.utils.diagnostics import (
    compute_acf,
    compute_pacf,
    ljung_box_test,
    white_test,
    plot_residuals,
)
from LGHackerton.utils.metrics import compute_oof_metrics
from LGHackerton.config.default import (
    TRAIN_PATH,
    ARTIFACTS_PATH,
    PATCH_PARAMS,
    TRAIN_CFG,
    ARTIFACTS_DIR,
    OOF_PATCH_OUT,
    SHOW_PROGRESS,
    OPTUNA_DIR,
)
from LGHackerton.tuning import TunerRegistry
from LGHackerton.utils.seed import set_seed


def _log_fold_start(seed: int, fold_idx: int, tr_mask: np.ndarray, va_mask: np.ndarray, cfg: TrainConfig, prefix: str) -> None:
    """Persist fold details to a timestamped JSON file under artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "seed": seed,
        "fold": fold_idx,
        "train_indices": np.where(tr_mask)[0].tolist(),
        "val_indices": np.where(va_mask)[0].tolist(),
        "config": asdict(cfg),
    }
    out = ARTIFACTS_DIR / f"{prefix}_fold{fold_idx}_{timestamp}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _patch_patchtst_logging(cfg: TrainConfig) -> None:
    """Attach fold logging callbacks to ``PatchTSTTrainer``.

    The registered callback runs once per ROCV fold with signature
    ``(seed, fold_idx, train_mask, val_mask, cfg)``. ``PatchTSTTrainer``
    isolates failures in callbacks by converting exceptions into warnings.

    Fallback order:
    1) ``PatchTSTTrainer.register_rocv_callback`` if available.
    2) Wrap legacy ``_make_rocv_slices`` and warn once.
    3) Otherwise warn that fold logging is disabled.

    Private helpers such as ``_make_rocv_slices`` are intentionally not part
    of the public surface; wrapping them only serves legacy versions. Training
    proceeds even when hooks are absent. At most one warning is emitted to
    avoid noisy logs.
    """

    # Import class and module to inspect available hooks at runtime.
    from LGHackerton.models.patchtst.trainer import PatchTSTTrainer
    import LGHackerton.models.patchtst.trainer as pt

    if hasattr(PatchTSTTrainer, "register_rocv_callback"):
        # Preferred modern API: register a callback that logs each fold.
        def _cb(seed, fold_idx, tr_mask, va_mask, cfg_inner):
            _log_fold_start(seed, fold_idx, tr_mask, va_mask, cfg_inner, "train_patchtst")
        # ``register_rocv_callback`` consumes (seed, fold_idx, train_mask,
        # val_mask, cfg) and shields training from callback errors.
        PatchTSTTrainer.register_rocv_callback(_cb)
    elif hasattr(pt, "_make_rocv_slices"):
        # Legacy fallback: temporarily wrap the private helper. This keeps
        # compatibility with older releases while avoiding permanent exposure.
        warnings.warn(
            "PatchTSTTrainer.register_rocv_callback not found; wrapping _make_rocv_slices for fold logging",
            stacklevel=2,
        )
        original = pt._make_rocv_slices

        def _wrapped(label_dates, n_folds, stride, span, purge):
            slices = original(label_dates, n_folds, stride, span, purge)
            for i, (tr_mask, va_mask) in enumerate(slices):
                _log_fold_start(cfg.seed, i, tr_mask, va_mask, cfg, "train_patchtst")
            return slices

        pt._make_rocv_slices = _wrapped
    else:  # pragma: no cover - defensive fallback
        warnings.warn(
            "No PatchTST fold logging hooks found; fold information will not be logged",
            stacklevel=2,
        )

def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def load_best_patch_params() -> tuple[dict, int | None]:
    """
    Determine PatchTST hyperparameters.

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

    # 1) Grid search CSV
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

    # 2) Optuna artifact or tuner output
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


def report_oof_metrics(oof_df, model_name: str) -> None:
    """Compute and log OOF metrics for a model."""
    metrics = compute_oof_metrics(oof_df)
    for name, value in metrics.items():
        logging.info("%s %s: %s", model_name, name, value)


def run_preprocess(show_progress: bool) -> tuple[Preprocessor, pd.DataFrame]:
    """Run preprocessing and return the fitted preprocessor and full dataframe."""
    df_train_raw = _read_table(TRAIN_PATH)
    pp = Preprocessor(show_progress=show_progress)
    df_full = pp.fit_transform_train(df_train_raw)
    pp.save(ARTIFACTS_PATH)
    return pp, df_full


def run_tuning(
    model_name: str,
    pp: Preprocessor,
    df_full: pd.DataFrame,
    cfg: TrainConfig,
    patch_params: dict,
    patch_input_len: int | None,
    skip_tune: bool,
    force_tune: bool,
) -> tuple[dict, int | None]:
    """Optionally run hyperparameter tuning and update PatchTST params."""
    if skip_tune:
        return patch_params, patch_input_len
    try:
        tuner_cls = TunerRegistry.get(model_name)
    except ValueError:  # pragma: no cover - user facing
        logging.warning("Unknown tuner for model")
        return patch_params, patch_input_len
    tuner = tuner_cls(pp, df_full, cfg)
    tuned = tuner.run(n_trials=cfg.n_trials, force=force_tune)
    patch_input_len = tuned.pop("input_len", patch_input_len)
    patch_params.update(tuned)
    return patch_params, patch_input_len


def run_training(
    trainer_cls,
    pp: Preprocessor,
    df_full: pd.DataFrame,
    cfg: TrainConfig,
    patch_params: dict,
    patch_input_len: int | None,
    device: str,
) -> pd.DataFrame:
    """Train model and return out-of-fold predictions."""
    if patch_input_len is not None:
        pp.windowizer = SampleWindowizer(lookback=patch_input_len, horizon=H)
    X_train, y_train, series_ids, label_dates = pp.build_patch_train(df_full)
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not available")
    patch_params.pop("input_len", None)
    params_module = importlib.import_module(trainer_cls.__module__)
    params_cls_name = trainer_cls.__name__.replace("Trainer", "Params")
    params_cls = getattr(params_module, params_cls_name)
    patch_params_obj = params_cls(**patch_params)
    L_used = patch_input_len if patch_input_len is not None else L
    trainer = trainer_cls(params=patch_params_obj, L=L_used, H=H, model_dir=cfg.model_dir, device=device)
    trainer.train(X_train, y_train, series_ids, label_dates, cfg)
    return trainer.get_oof()


def run_oof_prediction(oof_df: pd.DataFrame, model_name: str) -> None:
    """Persist OOF predictions and diagnostics."""
    oof_df.to_csv(OOF_PATCH_OUT, index=False)
    report_oof_metrics(oof_df, model_name)

    # diagnostics for PatchTST and similar models
    try:
        res_p = oof_df["y"] - oof_df["yhat"]
        diag_dir = ARTIFACTS_DIR / "diagnostics" / model_name / "oof"
        diag_dir.mkdir(parents=True, exist_ok=True)
        acf_df = compute_acf(res_p)
        pacf_df = compute_pacf(res_p)
        lb_df, res_used = ljung_box_test(res_p, lags=[10, 20, 30], return_residuals=True)
        wt_df = white_test(res_p)
        acf_df.to_csv(diag_dir / "acf.csv", index=False)
        pacf_df.to_csv(diag_dir / "pacf.csv", index=False)
        lb_df.to_csv(diag_dir / "ljung_box.csv", index=False)
        res_used.rename("residual").to_csv(diag_dir / "ljung_box_input.csv", index=False)
        wt_df.to_csv(diag_dir / "white_test.csv", index=False)
        plot_residuals(res_p, diag_dir)
        logging.info("%s Ljung-Box p-values: %s", model_name, lb_df["pvalue"].tolist())
        logging.info("%s Ljung-Box residual sample: %s", model_name, res_used.head().tolist())
        logging.info("%s White test p-value: %s", model_name, wt_df["lm_pvalue"].iloc[0])
    except Exception as e:  # pragma: no cover - best effort
        logging.warning("%s diagnostics failed: %s", model_name, e)


def main(show_progress: bool | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="show preprocessing progress")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="hide preprocessing progress")
    parser.add_argument("--skip-tune", action="store_true", help="skip hyperparameter tuning")
    parser.add_argument("--force-tune", action="store_true", help="re-run tuning even if artifacts exist")
    parser.add_argument("--trials", type=int, default=20, help="number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="time limit for tuning (seconds)")
    available = ", ".join(ModelRegistry.available())
    parser.add_argument("--model", default="patchtst", help=f"model name ({available})")
    parser.set_defaults(show_progress=SHOW_PROGRESS)

    args = parser.parse_args()
    try:
        trainer_cls = ModelRegistry.get(args.model)
    except KeyError as e:
        parser.error(str(e))
    if show_progress is None:
        show_progress = args.show_progress

    device = select_device()  # ask user for compute environment

    pp, df_full = run_preprocess(show_progress)

    patch_params_dict, patch_input_len = load_best_patch_params()
    cfg = TrainConfig(**TRAIN_CFG)
    cfg.n_trials = args.trials
    cfg.timeout = args.timeout
    if trainer_cls is PatchTSTTrainer:
        _patch_patchtst_logging(cfg)
    set_seed(cfg.seed)

    patch_params_dict, patch_input_len = run_tuning(
        args.model, pp, df_full, cfg, patch_params_dict, patch_input_len, args.skip_tune, args.force_tune
    )

    oof_df = run_training(
        trainer_cls, pp, df_full, cfg, patch_params_dict, patch_input_len, device
    )
    run_oof_prediction(oof_df, args.model)

if __name__ == "__main__":
    main()
