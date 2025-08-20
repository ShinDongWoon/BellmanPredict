from __future__ import annotations
import pandas as pd
import logging
from LGHackerton.config.default import SAMPLE_SUB_PATH


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, encoding='utf-8-sig')
    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    raise ValueError('Unsupported file type. Use .csv or .xlsx')


def convert_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form predictions to submission format."""
    sample_df = _read_table(SAMPLE_SUB_PATH)

    pred_df = df.copy()
    pred_df['series_id'] = pred_df['series_id'].str.replace('::', '_', n=1)

    wide = pred_df.pivot(index='date', columns='series_id', values='yhat_ens')
    wide = wide.reindex(sample_df.iloc[:, 0]).reindex(columns=sample_df.columns[1:], fill_value=0.0)

    missing_dates = set(sample_df.iloc[:, 0]) - set(pred_df['date'])
    missing_cols = set(sample_df.columns[1:]) - set(pred_df['series_id'].unique())

    if missing_dates:
        logging.warning('Missing dates in predictions: %s', sorted(missing_dates))
    if missing_cols:
        logging.warning('Missing columns in predictions: %s', sorted(missing_cols))

    out_df = sample_df.copy()
    out_df.iloc[:, 1:] = wide.to_numpy()
    assert list(out_df.columns) == list(sample_df.columns)
    return out_df
