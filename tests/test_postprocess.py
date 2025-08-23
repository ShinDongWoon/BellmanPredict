import pandas as pd
import numpy as np

from LGHackerton.postprocess import aggregate_predictions, convert_to_submission
from LGHackerton.config.default import SAMPLE_SUB_PATH


def _make_df(col_name: str, values):
    return pd.DataFrame({
        "series_id": ["A::0", "A::0"],
        "date": ["TEST_00+1일", "TEST_00+2일"],
        "h": [1, 2],
        col_name: values,
    })


def test_single_model_identity():
    df = _make_df("yhat_patch", [1.0, 2.0])
    agg = aggregate_predictions([df])
    expected = df.rename(columns={"yhat_patch": "yhat_ens"})[
        ["series_id", "date", "yhat_ens"]
    ]
    expected["series_id"] = expected["series_id"].str.replace("::", "_", n=1)
    pd.testing.assert_frame_equal(agg.sort_index(axis=1), expected.sort_index(axis=1))


def test_weighted_mean():
    df1 = _make_df("yhat_a", [1.0, 3.0])
    df2 = _make_df("yhat_b", [3.0, 1.0])
    agg = aggregate_predictions([df1, df2], weights=[0.7, 0.3])
    expected_vals = np.array([1.0*0.7 + 3.0*0.3, 3.0*0.7 + 1.0*0.3])
    np.testing.assert_allclose(agg["yhat_ens"].values, expected_vals)


def test_convert_submission_structure():
    sample_df = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    df = pd.DataFrame({
        "series_id": [sample_df.columns[1]],
        "date": [sample_df.iloc[0, 0]],
        "h": [1],
        "yhat_patch": [1.0],
    })
    agg = aggregate_predictions([df])
    out = convert_to_submission(agg)
    assert list(out.columns) == list(sample_df.columns)
    assert len(out) == len(sample_df)
