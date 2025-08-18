import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from LGHackerton.models.patchtst_trainer import _make_rocv_slices


def test_rocv_validation_spans_are_disjoint():
    """Validation masks produced by ROCV should not overlap."""
    dates = np.array('2024-01-01', dtype='datetime64[D]') + np.arange(30)
    folds = _make_rocv_slices(dates, n_folds=3, stride=3, span=7, purge=np.timedelta64(0, 'D'))
    assert len(folds) == 3
    # ensure validation masks do not overlap
    for i in range(len(folds)):
        va_i = folds[i][1]
        for j in range(i + 1, len(folds)):
            va_j = folds[j][1]
            assert not np.any(va_i & va_j)
