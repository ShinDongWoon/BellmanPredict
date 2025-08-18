import numpy as np
import pathlib
import sys

# Add project root to sys.path for module imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from LGHackerton.models.patchtst_trainer import PatchTSTTrainer, _make_rocv_slices
from LGHackerton.models.base_trainer import TrainConfig


def test_register_rocv_callback_invokes_for_each_fold():
    """Registered ROCV callbacks execute once per generated fold."""
    dates = np.array("2024-01-01", dtype="datetime64[D]") + np.arange(30)
    folds = _make_rocv_slices(
        dates,
        n_folds=2,
        stride=3,
        span=7,
        purge=np.timedelta64(0, "D"),
    )
    cfg = TrainConfig(seed=123)
    calls = []

    def cb(seed, fold_idx, tr_mask, va_mask, cfg_inner):
        calls.append((seed, fold_idx, tr_mask.copy(), va_mask.copy()))

    original_callbacks = list(PatchTSTTrainer._rocv_callbacks)
    PatchTSTTrainer._rocv_callbacks = []
    try:
        PatchTSTTrainer.register_rocv_callback(cb)
        PatchTSTTrainer._notify_rocv_callbacks(cfg.seed, folds, cfg)
    finally:
        PatchTSTTrainer._rocv_callbacks = original_callbacks

    assert len(calls) == len(folds)
    for call, (tr, va) in zip(calls, folds):
        assert np.array_equal(call[2], tr)
        assert np.array_equal(call[3], va)
