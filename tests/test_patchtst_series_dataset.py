import numpy as np
import pytest

from LGHackerton.models.patchtst.trainer import _SeriesDataset


def test_series_dataset_requires_dynamic_channel():
    X = np.zeros((2, 5, 1), dtype=np.float32)
    y = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one dynamic channel required"):
        _SeriesDataset(X, y, dyn_idx=[], static_idx=[0])
