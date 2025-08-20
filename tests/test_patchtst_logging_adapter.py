import numpy as np
import pathlib
import sys
import pytest

# Ensure project root is on sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from LGHackerton.models.base_trainer import TrainConfig
import LGHackerton.models.patchtst.trainer as pt
from LGHackerton.train import _patch_patchtst_logging


def test_fallback_wraps_make_rocv_slices(monkeypatch):
    """Without ``register_rocv_callback`` fall back to wrapping ``_make_rocv_slices``."""
    cfg = TrainConfig(seed=5)
    calls = []

    # Remove modern callback API
    monkeypatch.delattr(pt.PatchTSTTrainer, "register_rocv_callback", raising=False)

    # Capture fold logging
    import LGHackerton.train as train_mod

    def fake_log(seed, fold_idx, tr_mask, va_mask, cfg_inner, prefix):
        calls.append((seed, fold_idx))

    monkeypatch.setattr(train_mod, "_log_fold_start", fake_log)

    with pytest.warns(UserWarning, match="register_rocv_callback not found"):
        _patch_patchtst_logging(cfg)

    dates = np.array("2024-01-01", dtype="datetime64[D]") + np.arange(20)
    pt._make_rocv_slices(
        dates,
        n_folds=2,
        stride=3,
        span=7,
        purge=np.timedelta64(0, "D"),
    )
    assert len(calls) == 2


def test_no_hooks_warns(monkeypatch):
    """Warn when neither callback nor slice hook exists."""
    cfg = TrainConfig()
    monkeypatch.delattr(pt.PatchTSTTrainer, "register_rocv_callback", raising=False)
    monkeypatch.delattr(pt, "_make_rocv_slices", raising=False)

    with pytest.warns(UserWarning, match="No PatchTST fold logging hooks found"):
        _patch_patchtst_logging(cfg)
