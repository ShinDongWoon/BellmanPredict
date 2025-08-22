import numpy as np
import pytest

from LGHackerton.models.patchtst.trainer import _SeriesDataset


@pytest.mark.parametrize("dyn_idx", [None, []])
def test_series_dataset_requires_dynamic_channel(dyn_idx):
    X = np.zeros((2, 5, 1), dtype=np.float32)
    y = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="dynamic channel indices required"):
        _SeriesDataset(X, y, dyn_idx=dyn_idx, static_idx=[0])
