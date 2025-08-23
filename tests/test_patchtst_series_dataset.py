import numpy as np
import pytest

from LGHackerton.models.patchtst.trainer import _SeriesDataset


@pytest.mark.parametrize("dyn_idx", [None, []])
def test_series_dataset_requires_dynamic_channel(dyn_idx):
    X = np.zeros((2, 5, 1), dtype=np.float32)
    y = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="dynamic channel indices required"):
        _SeriesDataset(X, y, dyn_idx=dyn_idx, static_idx=[0])


def test_revin_normalizes_all_dynamic_channels():
    """Ensure RevIN scaling applies to all dynamic channels."""
    X = np.array([
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    ], dtype=np.float32)
    y = np.zeros((1,), dtype=np.float32)
    ds = _SeriesDataset(X, y, dyn_idx=[0, 1], scaler="revin")
    x, *_ = ds[0]

    tgt = X[0, :, 0]
    expected0 = (tgt - tgt.mean()) / tgt.std()
    assert np.allclose(x[:, 0], expected0)
    dyn2 = X[0, :, 1]
    expected1 = (dyn2 - dyn2.mean()) / dyn2.std()
    assert np.allclose(x[:, 1], expected1)
