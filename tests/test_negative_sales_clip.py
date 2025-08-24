import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LGHackerton.preprocess.preprocess_pipeline_v1_1 import (
    Preprocessor,
    RAW_DATE,
    RAW_KEY,
    RAW_QTY,
    SALES_COL,
    SALES_FILLED_COL,
)


def test_negative_sales_clipped():
    df = pd.DataFrame({
        RAW_DATE: pd.to_datetime(["2024-01-01", "2024-01-01"]),
        RAW_KEY: ["shop1_menu", "shop2_menu"],
        RAW_QTY: [-5, 3],
    })
    pp = Preprocessor()
    out = pp.fit_transform_train(df)
    assert (out[SALES_COL] >= 0).all()
    assert (out[SALES_FILLED_COL] >= 0).all()

