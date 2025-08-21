"""Input/output helper utilities.

This module centralizes basic table-loading logic used across the
project. It exposes :func:`read_table` which can handle both CSV and
Excel files, automatically selecting the appropriate pandas reader based
on the file extension. The underlying implementation lives in the
private :func:`_read_table` helper to allow reuse inside the package
without polluting the public API with the underscore-prefixed function.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _read_table(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV or Excel file into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to the input file. Supported extensions are ``.csv``,
        ``.xls`` and ``.xlsx``.
    **kwargs : Any
        Additional keyword arguments passed to the underlying pandas
        reader.

    Returns
    -------
    pandas.DataFrame
        Parsed dataframe containing the file's contents.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported types.
    """

    lower = path.lower()
    if lower.endswith(".csv"):
        # ``utf-8-sig`` gracefully handles files with or without BOM.
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path, **kwargs)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def read_table(path: str, **kwargs: Any) -> pd.DataFrame:
    """Public wrapper around :func:`_read_table`.

    This helper simply forwards all arguments to the private
    implementation. It mirrors pandas' behaviour while constraining the
    file formats we support within the project.
    """

    return _read_table(path, **kwargs)


__all__ = ["read_table"]

