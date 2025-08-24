"""
Preprocessing Pipeline v1.1 (AI Execution Ready)
D&O 곤지암 리조트 식음업장 수요예측 해커톤
- Strict 28-day lookback
- Sample isolation
- No external network or dynamic installs
- Optional Korean holidays provider, with safe fallback
Author: ChatGPT
"""

from __future__ import annotations

import os
import re
import json
import math
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Iterable, Literal

import numpy as np
import pandas as pd
from .transforms import symmetric_transform
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(x, *args, **kwargs):  # type: ignore
        return x
try:
    from ..utils.progress import get_progress_bar
except ImportError:  # pragma: no cover - fallback when executed as script
    from utils.progress import get_progress_bar

# ------------------------------
# Constants
# ------------------------------
RAW_DATE = "영업일자"
RAW_KEY = "영업장명_메뉴명"
RAW_QTY = "매출수량"

DATE_COL = "date"
SHOP_COL = "shop"
MENU_COL = "menu"
SERIES_COL = "series_id"
SALES_COL = "sales"
SALES_FILLED_COL = "sales_filled"

UNK_TOKEN = "<UNK>"
UNK_CODE = -1

PRIORITY_OUTLETS = {"담하", "미라시아"}

# Horizon and lookback
L = 28  # lookback window
H = 7  # forecast horizon
# ----- Verbose print helper -----
VERBOSE = False  # 필요시 False


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


# ------------------------------
# Utility
# ------------------------------
def slugify(text: str) -> str:
    if text is None:
        return UNK_TOKEN
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "-").replace("\\", "-")
    return s


def ensure_datetime_series(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) else s


def warn(msg: str):
    vprint(f"[WARN] {msg}")


# ------------------------------
# Leak Guard
# ------------------------------
class LeakGuard:
    """Scope-aware guard to prevent cross-sample leakage."""

    def __init__(self):
        self.scope = None  # 'train' | 'eval'

    def set_scope(self, scope: str):
        assert scope in ("train", "eval")
        self.scope = scope

    def assert_scope(self, allowed: Iterable[str]):
        if self.scope not in allowed:
            raise RuntimeError(f"Operation not allowed in scope={self.scope}. Allowed={allowed}")


# ------------------------------
# Optional holiday provider
# ------------------------------
class HolidayProvider:
    """
    Try to use `holidayskr`. If unavailable, provide safe fallback: weekend-based only.
    The fallback keeps `is_holiday` = 0 and relies on `is_weekend` feature.
    """

    def __init__(self):
        try:
            from holidayskr import year_holidays  # type: ignore
            self._year_holidays = year_holidays
            self._available = True
        except Exception:
            self._year_holidays = None
            self._available = False
            warn("holidayskr not available. Using weekend-only features.")

    def compute(self, years: Iterable[int]) -> set:
        if not self._available:
            return set()
        all_days = set()
        for y in sorted(set(years)):
            try:
                entries = self._year_holidays(str(y))
                all_days.update([d for d, _ in entries])
            except Exception:
                continue
        return all_days


# ------------------------------
# Components
# ------------------------------
class SchemaNormalizer:
    """
    - Validates raw schema
    - Splits composite key RAW_KEY into shop/menu
    - Normalizes to internal columns and dtypes
    - Deduplicates by (date, series_id) via sum
    - Builds and persists category vocabularies
    """

    def __init__(self):
        self.shop_vocab: Dict[str, int] = {}
        self.menu_vocab: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        self._validate_columns(df)
        ndf = self._normalize(df.copy())
        vprint(f"[SchemaNormalizer.fit] rows={len(ndf)}  series={ndf[SERIES_COL].nunique()}")

        # build vocabs from training only
        for v in ndf[SHOP_COL].dropna().unique():
            self._add_vocab(self.shop_vocab, v)
        for v in ndf[MENU_COL].dropna().unique():
            self._add_vocab(self.menu_vocab, v)
        return self

    def transform(self, df: pd.DataFrame, allow_new=False) -> pd.DataFrame:
        self._validate_columns(df)
        ndf = self._normalize(df.copy())

        # map unknowns
        def map_or_unk(vocab: Dict[str, int], v: str) -> int:
            if v in vocab:
                return vocab[v]
            if allow_new:
                self._add_vocab(vocab, v)
                return vocab[v]
            return UNK_CODE

        ndf["shop_code"] = ndf[SHOP_COL].map(lambda v: map_or_unk(self.shop_vocab, v))
        ndf["menu_code"] = ndf[MENU_COL].map(lambda v: map_or_unk(self.menu_vocab, v))
        # int codes are for potential downstream usage; keep original strings as canonical IDs
        return ndf

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in [RAW_DATE, RAW_KEY, RAW_QTY] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # date
        df[DATE_COL] = ensure_datetime_series(df[RAW_DATE]).dt.normalize()
        # split key -> shop, menu
        keys = df[RAW_KEY].astype(str).fillna(UNK_TOKEN)
        # Split at the first "_" (as per provided examples)
        shop_vals, menu_vals = [], []
        for s in keys:
            if "_" in s:
                a, b = s.split("_", 1)
            else:
                a, b = s, UNK_TOKEN
            shop_vals.append(slugify(a))
            menu_vals.append(slugify(b))

        df[SHOP_COL] = pd.Series(shop_vals, index=df.index)
        df[MENU_COL] = pd.Series(menu_vals, index=df.index)
        df[SERIES_COL] = (df[SHOP_COL] + "::" + df[MENU_COL]).astype(str)

        # sales numeric
        df[SALES_COL] = pd.to_numeric(df[RAW_QTY], errors="coerce")

        # Deduplicate by (date, series)
        df = (
            df[[DATE_COL, SHOP_COL, MENU_COL, SERIES_COL, SALES_COL]]
            .groupby([SERIES_COL, DATE_COL, SHOP_COL, MENU_COL], as_index=False)
            .agg({SALES_COL: "sum"})
            .sort_values([SERIES_COL, DATE_COL])
            .reset_index(drop=True)
        )
        return df

    def _add_vocab(self, vocab: Dict[str, int], token: str):
        if token not in vocab:
            vocab[token] = len(vocab)


class DateContinuityFixer:
    """Ensures daily continuity per series. Inserts missing dates with NaN sales."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([SERIES_COL, DATE_COL])
        vprint(f"[DateContinuityFixer] start  series={df[SERIES_COL].nunique()}  rows={len(df)}")

        parts = []
        for sid, g in df.groupby(SERIES_COL, sort=False):
            g = g.sort_values(DATE_COL)
            idx = pd.date_range(g[DATE_COL].min(), g[DATE_COL].max(), freq="D")
            g = g.set_index(DATE_COL).reindex(idx)
            g.index.name = DATE_COL
            g[SERIES_COL] = sid
            g[SHOP_COL] = g[SHOP_COL].ffill().bfill()
            g[MENU_COL] = g[MENU_COL].ffill().bfill()
            parts.append(g.reset_index())
        out = pd.concat(parts, ignore_index=True)
        vprint(f"[DateContinuityFixer] done   rows={len(out)}  unique_dates={out[DATE_COL].nunique()}")

        return out[[DATE_COL, SHOP_COL, MENU_COL, SERIES_COL, SALES_COL]]


class CalendarFeatureMaker:
    """Adds calendar features and optional Korean holidays if provider available."""

    def __init__(
        self,
        holiday_provider: Optional[HolidayProvider] = None,
        cyclical: bool = True,
        keep_months: Optional[Iterable[int]] = None,
        keep_woys: Optional[Iterable[int]] = None,
        dow_mode: Literal["integer", "cyclical", "embed"] = "cyclical",
    ):
        self.holiday_provider = holiday_provider or HolidayProvider()
        self.cyclical = cyclical
        self.keep_months = set(keep_months) if keep_months is not None else None
        self.keep_woys = set(keep_woys) if keep_woys is not None else None
        self.dow_mode = dow_mode
        self._holiday_cache: set = set()
        self._woy_cols: List[str] = []
        self._month_cols: List[str] = []
        self._promo_col: Optional[str] = None
        self._feature_names: List[str] = []

    def fit(self, df: pd.DataFrame):
        years = df[DATE_COL].dt.year.unique().tolist()
        self._holiday_cache = self.holiday_provider.compute(years)
        # store columns to align dummies at transform (only for non-cyclical)
        if not self.cyclical:
            weeks = df[DATE_COL].dt.isocalendar().week.unique().tolist()
            months = df[DATE_COL].dt.month.unique().tolist()
            if self.keep_woys is not None:
                weeks = [w for w in weeks if w in self.keep_woys]
            if self.keep_months is not None:
                months = [m for m in months if m in self.keep_months]
            self._woy_cols = [f"woy_{w}" for w in sorted(weeks)]
            self._month_cols = [f"month_{m}" for m in sorted(months)]
        else:
            self._woy_cols = []
            self._month_cols = []
        promo_candidates = [
            c
            for c in ["is_promo", "promo", "promotion", "promotion_flag"]
            if c in df.columns
        ]
        self._promo_col = promo_candidates[0] if promo_candidates else None
        base = ["year"]
        if self.dow_mode == "cyclical":
            base.extend(["dow_sin", "dow_cos"])
        else:
            base.append("dow")
        base.extend(
            [
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "is_holiday",
                "days_since_holiday",
                "days_to_next_holiday",
                "is_priority_outlet",
            ]
        )
        if self.cyclical:
            cyc_cols = ["month_sin", "month_cos", "woy_sin", "woy_cos"]
            self._feature_names = [*base, *cyc_cols]
        else:
            self._feature_names = [*base, *self._month_cols, *self._woy_cols]
        if self._promo_col is not None:
            self._feature_names.append("is_promo")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["year"] = d[DATE_COL].dt.year
        d["dow"] = d[DATE_COL].dt.weekday  # 0=Mon
        d["is_weekend"] = d["dow"].isin([5, 6]).astype(np.int8)
        d["is_month_start"] = d[DATE_COL].dt.is_month_start.astype(np.int8)
        d["is_month_end"] = d[DATE_COL].dt.is_month_end.astype(np.int8)

        if self.dow_mode == "cyclical":
            d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
            d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)

        # base calendar categories
        if self.cyclical:
            month = d[DATE_COL].dt.month
            weekofyear = d[DATE_COL].dt.isocalendar().week.astype(int)
            d["month_sin"] = np.sin(2 * np.pi * month / 12)
            d["month_cos"] = np.cos(2 * np.pi * month / 12)
            d["woy_sin"] = np.sin(2 * np.pi * weekofyear / 52)
            d["woy_cos"] = np.cos(2 * np.pi * weekofyear / 52)

        if not self.cyclical:
            d["month"] = d[DATE_COL].dt.month
            d["weekofyear"] = d[DATE_COL].dt.isocalendar().week.astype(int)
            month_dum = pd.get_dummies(d["month"], prefix="month", dtype=np.int8)
            month_dum = month_dum.reindex(columns=self._month_cols, fill_value=0)
            woy_dum = pd.get_dummies(d["weekofyear"], prefix="woy", dtype=np.int8)
            woy_dum = woy_dum.reindex(columns=self._woy_cols, fill_value=0)
            d = pd.concat(
                [d.drop(columns=["month", "weekofyear"]), month_dum, woy_dum],
                axis=1,
            )

        d["is_holiday"] = d[DATE_COL].dt.date.isin(self._holiday_cache).astype(np.int8)

        # days since/next holiday
        d_sort = d.sort_values(DATE_COL)
        holiday_dates = d_sort[DATE_COL].where(d_sort["is_holiday"].eq(1))
        prev_hol = holiday_dates.ffill()
        next_hol = holiday_dates.bfill()
        d_sort["days_since_holiday"] = (d_sort[DATE_COL] - prev_hol).dt.days
        d_sort["days_to_next_holiday"] = (next_hol - d_sort[DATE_COL]).dt.days
        _max = np.int16(9999)
        d_sort["days_since_holiday"] = (
            d_sort["days_since_holiday"].fillna(_max).astype(np.int16)
        )
        d_sort["days_to_next_holiday"] = (
            d_sort["days_to_next_holiday"].fillna(_max).astype(np.int16)
        )
        d = d_sort.sort_index()

        if self._promo_col is not None and self._promo_col in df.columns:
            d["is_promo"] = df[self._promo_col].astype(np.int8)

        d["is_priority_outlet"] = d[SHOP_COL].isin(PRIORITY_OUTLETS).astype(np.int8)
        return d


class MissingAndOutlierHandler:
    """Handle missing values and outliers with symmetric transforms.

    - Target NA handling policy for features: ``ffill(limit=3)`` then remaining
      to 0 into :data:`SALES_FILLED_COL`
    - Outlier capping with per-series thresholds computed from TRAIN only
      (Hampel-like robust cap: ``min(Q3+1.5*IQR, Q99)``)
    - Negative sales are clipped to zero prior to capping and transformation.
    - Applies an ``arcsinh`` transformation so negative values are preserved
      rather than clipped to zero
    """

    def __init__(self):
        self.caps: Dict[str, float] = {}
        import pandas as _pd  # 중복 임포트 안전
        _v = _pd.Series(self.caps).apply(lambda x: not _pd.isna(x)).sum()
        vprint(f"[OutlierCap.fit] series={len(self.caps)}  with_caps={_v}")

    def _clip_negative(self, x: pd.Series) -> pd.Series:
        cnt = (x < 0).sum()
        if cnt > 0:
            vprint(f"[MissingAndOutlierHandler] clipped {cnt} negative sales to zero")
        return x.clip(lower=0)

    def fit(self, df: pd.DataFrame):
        caps = {}
        for sid, g in df.groupby(SERIES_COL, sort=False):
            x = self._clip_negative(g[SALES_COL].dropna()).values
            if len(x) == 0:
                caps[sid] = np.nan
                continue
            q1, q3 = np.quantile(x, [0.25, 0.75])
            iqr = max(q3 - q1, 1e-12)
            cap_iqr = q3 + 1.5 * iqr
            cap_q99 = np.quantile(x, 0.99)
            caps[sid] = float(min(cap_iqr, cap_q99))
        self.caps = caps
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d[SALES_COL] = self._clip_negative(d[SALES_COL])

        # apply caps while preserving negatives
        def cap_series(sid: str, x: pd.Series) -> pd.Series:
            c = self.caps.get(sid, np.nan)
            if np.isnan(c):
                return x
            return x.astype(float).clip(upper=c)

        d[SALES_COL] = d.groupby(SERIES_COL)[SALES_COL].transform(
            lambda x: cap_series(x.name, x)
        )

        # filled series for feature calculations
        def fill_series(x: pd.Series) -> pd.Series:
            y = x.copy()
            y = y.ffill(limit=3)
            y = y.fillna(0.0)
            return y

        d[SALES_FILLED_COL] = d.groupby(SERIES_COL)[SALES_COL].transform(fill_series)

        # symmetric transformation to handle negative values
        d[SALES_COL] = symmetric_transform(d[SALES_COL])
        d[SALES_FILLED_COL] = symmetric_transform(d[SALES_FILLED_COL])
        return d


class StrictFeatureMaker:
    """
    Strict features computed only from past values within max window size L=28.
    - lags: 1..(2*H) plus longer-term anchors
    - rolling stats on shifted series
    - intermittent-demand markers
    """

    def __init__(self, lookback: int = L):
        self.lookback = lookback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values([SERIES_COL, DATE_COL]).copy()
        gb = d.groupby(SERIES_COL, group_keys=False, sort=False)

        # Shift base for rolling
        s_shift = gb[SALES_FILLED_COL].shift(1)

        # Lags up to 2*H plus anchors to preserve 28-day history
        lags = list(range(1, 2 * H + 1)) + [21, 27, 28]
        for lag in sorted(set(lags)):
            d[f"lag_{lag}"] = gb[SALES_FILLED_COL].shift(lag)
        vprint("[Strict] lag features done")

        # Rolling on shifted
        def roll_feat(series: pd.Series, window: int, func: str):
            return series.rolling(window=window, min_periods=1).agg(func)

        d["roll_mean_7"] = roll_feat(s_shift, 7, "mean")
        d["roll_mean_14"] = roll_feat(s_shift, 14, "mean")
        d["roll_mean_28"] = roll_feat(s_shift, 28, "mean")
        d["roll_std_7"] = roll_feat(s_shift, 7, "std")
        d["roll_std_28"] = roll_feat(s_shift, 28, "std")
        d["roll_min_7"] = roll_feat(s_shift, 7, "min")
        d["roll_max_7"] = roll_feat(s_shift, 7, "max")

        # Intermittency features
        is_zero = (d[SALES_FILLED_COL] == 0).astype(np.int8)
        d["zero_ratio_28"] = gb[SALES_FILLED_COL].apply(
            lambda s: (s == 0).astype(np.int8).rolling(28, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        # days_since_last_sale
        def days_since_last_sale(arr: np.ndarray) -> np.ndarray:
            res = np.zeros_like(arr, dtype=np.int16)
            last = -1_0000
            for i, v in enumerate(arr):
                if v > 0:
                    last = i
                res[i] = min(28 + 1, i - last)  # cap at 29
            return res

        # Use explicit index alignment to avoid misalignment-induced NaNs
        d["days_since_last_sale"] = gb[SALES_FILLED_COL].transform(
            lambda s: pd.Series(days_since_last_sale(s.values), index=s.index)
        )

        # zero run length (current run)
        def zero_run_len(arr: np.ndarray) -> np.ndarray:
            cnt = 0
            out = np.zeros_like(arr, dtype=np.int16)
            for i, v in enumerate(arr):
                if v == 0:
                    cnt += 1
                else:
                    cnt = 0
                out[i] = cnt
            return out

        d["zero_run_len"] = gb[SALES_FILLED_COL].transform(
            lambda s: pd.Series(zero_run_len(s.values), index=s.index)
        )
        vprint(f"[Strict] rolling+intermittency done  rows={len(d)}")

        return d



class RichLookupBuilder:
    """
    Precompute per-series statistics from TRAIN folds only, then attach via
    lookup in any scope.
    - ``series_base_mean``: ``log1p`` of mean clipped at ``>=1e-3``
    - ``series_cv``: ``log1p`` of coefficient of variation clipped at ``<=1e6``
    - ``series_dow_mean``: per-series weekday mean; centered at transform time
    """

    def __init__(self):
        self.base_table: Optional[pd.DataFrame] = None
        self.dow_table: Optional[pd.DataFrame] = None

    def fit(self, df_train: pd.DataFrame):
        # NOTE: fit on training folds only to avoid leakage
        # base: per-series mean/std -> coefficient of variation
        g = (
            df_train.groupby(SERIES_COL)[SALES_COL]
            .agg(series_base_mean="mean", series_std="std")
            .reset_index()
        )
        g["series_base_raw"] = g["series_base_mean"]  # store raw mean for later centering
        g["series_cv"] = g["series_std"] / (g["series_base_raw"] + 1e-9)
        g["series_base_mean"] = np.log1p(g["series_base_raw"].clip(lower=1e-3))  # log1p + lower clip
        # Coefficient of variation can be negative if the series mean is negative.
        # ``np.log1p`` receives values clipped to ``[0, 1e6]`` to avoid
        # ``RuntimeWarning: invalid value encountered in log1p`` and ensure the
        # resulting feature is finite.
        g["series_cv"] = np.log1p(g["series_cv"].clip(lower=0, upper=1e6))  # log1p + bounds
        self.base_table = g[[SERIES_COL, "series_base_raw", "series_base_mean", "series_cv"]].copy()

        # dow lookup: per (series_id, dow) mean if available
        if "dow" in df_train.columns:
            gd = (
                df_train.groupby([SERIES_COL, "dow"])[SALES_COL]
                .mean()
                .reset_index()
                .rename(columns={SALES_COL: "series_dow_mean"})
            )
            self.dow_table = gd
        else:
            self.dow_table = None

        vprint(
            f"[Rich.fit] base_rows={len(self.base_table)}  dow_rows={len(self.dow_table) if self.dow_table is not None else 0}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_table is None:
            raise RuntimeError("RichLookupBuilder not fit.")
        d = df.merge(self.base_table, on=SERIES_COL, how="left")
        if self.dow_table is not None and "dow" in df.columns:
            d = d.merge(self.dow_table, on=[SERIES_COL, "dow"], how="left")
            d["dow_effect"] = d["series_dow_mean"] - d["series_base_raw"]  # center weekday mean
            d = d.drop(columns=["series_dow_mean", "series_base_raw"])  # drop raw means post-centering
        else:
            d = d.drop(columns=["series_base_raw"], errors="ignore")  # remove raw base mean
        return d


class Encoder:
    """Integer encoding for series_id. Unknown -> UNK_CODE"""

    def __init__(self):
        self.series2code: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        uniq = df[SERIES_COL].dropna().unique().tolist()
        self.series2code = {sid: i for i, sid in enumerate(uniq)}
        vprint(f"[Encoder.fit] series_codes={len(self.series2code)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["series_code"] = d[SERIES_COL].map(lambda x: self.series2code.get(x, UNK_CODE)).astype(np.int32)
        return d


class SampleWindowizer:
    """
    Build LGBM direct-prediction dataset and PatchTST tensors from fully-featured panel.
    - For LGBM: one row per origin-date per horizon h in {1..7}. Features at time t, target y at t+h.
    - For Eval: produce 7 rows per series at last available date.
    - For PatchTST: X windows of shape (28, 1) and y of shape (7,)
    """

    def __init__(
            self, lookback: int = L, horizon: int = H, guard: Optional[LeakGuard] = None
    ):
        self.L = lookback
        self.H = horizon
        self.guard = guard
        self.cat_indices: Dict[str, int] = {}
        # Channel indices for PatchTST tensors
        self.dynamic_idx: Dict[str, int] = {}
        self.static_idx: Dict[str, int] = {}

    def build_lgbm_train(
            self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Vectorized construction of the direct-forecast dataset for LGBM.

        For each horizon ``h`` in ``{1..H}`` the target ``y_h`` is generated via
        a grouped forward shift. Rows lacking a full 28-day lookback are
        discarded, the target columns are unpivoted into a long format and the
        temporary ``y_h`` columns removed.
        """
        if self.guard:
            self.guard.assert_scope({"train"})

        d = df.sort_values([SERIES_COL, DATE_COL]).copy()
        # Need at least lag_28 available to respect 28-day features
        if "lag_28" in d.columns:
            lag_col = "lag_28"
        elif "lag_27" in d.columns:
            lag_col = "lag_27"
        else:
            raise ValueError("Strict features required before building LGBM train set.")

        # Track categorical feature indices (e.g., for embeddings)
        self.cat_indices = {c: i for i, c in enumerate(feature_cols) if c == "dow"}

        # --- Vectorised target generation -------------------------------------------------
        for h in range(1, self.H + 1):
            d[f"y_{h}"] = d.groupby(SERIES_COL)[SALES_COL].shift(-h)

        # Optional downcast of feature columns to save memory
        for c in feature_cols:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].astype("float32")

        # --- Filter base rows once --------------------------------------------------------
        # Require full lookback history for all features
        base = d[d[lag_col].notna()]
        base = base.dropna(
            subset=[f"y_{h}" for h in range(1, self.H + 1)], how="all"
        )

        # --- Unpivot horizons to long format ----------------------------------------------
        long = base.melt(
            id_vars=[SERIES_COL, DATE_COL, *feature_cols],
            value_vars=[f"y_{h}" for h in range(1, self.H + 1)],
            var_name="h",
            value_name="y",
        )
        long = long.dropna(subset=["y"])
        long["h"] = long["h"].str.extract(r"(\d+)").astype(int)

        # No longer need the temporary y_h columns
        base.drop(columns=[f"y_{h}" for h in range(1, self.H + 1)], inplace=True, errors="ignore")

        long = long.reset_index(drop=True)
        long = long[[SERIES_COL, DATE_COL, *feature_cols, "h", "y"]]

        vprint(f"[LGBM/TRAIN] rows={len(long)}  feats={len(feature_cols)}")

        return long

    def build_lgbm_eval(
            self, df_eval: pd.DataFrame, feature_cols: List[str], show_progress: bool = False
    ) -> pd.DataFrame:
        if self.guard:
            self.guard.assert_scope({"eval"})
        d = df_eval.sort_values([SERIES_COL, DATE_COL]).copy()

        # Track categorical feature indices (e.g., for embeddings)
        self.cat_indices = {c: i for i, c in enumerate(feature_cols) if c == "dow"}
        out_rows = []
        gb = d.groupby(SERIES_COL, sort=False)
        total = d[SERIES_COL].nunique()
        for sid, g in get_progress_bar(gb, total=total, disable=not show_progress):
            g = g.reset_index(drop=True)
            if len(g) == 0:
                continue
            t = len(g) - 1  # last date
            if pd.isna(g.loc[t, "lag_27"]):
                # Not enough lookback provided in eval sample
                warn(f"Eval series {sid}: insufficient lookback for strict features.")
                continue
            base = {
                SERIES_COL: sid,
                DATE_COL: g.loc[t, DATE_COL],
            }
            feat_dict = g.loc[t, feature_cols].to_dict()
            for h in range(1, self.H + 1):
                row = {**base, **feat_dict, "h": h}
                out_rows.append(row)
        if not out_rows:
            raise RuntimeError("No eval rows produced.")

        out = pd.DataFrame(out_rows).reset_index(drop=True)
        vprint(f"[LGBM/EVAL] rows={len(out)}  feats={len(feature_cols)}")
        return out

    def build_patch_train(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            static_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
        """Build PatchTST training windows with identifiers.

        Returns
        -------
        Tuple
            ``(X_dyn, X_stat, y, series_ids, label_dates, dynamic_idx, static_idx)``
            where ``dynamic_idx`` maps dynamic column names to channel indices
            in ``X_dyn`` and ``static_idx`` maps static column names to indices
            in ``X_stat``.
        """
        if self.guard:
            self.guard.assert_scope({"train"})
        d = df.sort_values([SERIES_COL, DATE_COL]).copy()

        if SALES_FILLED_COL not in d.columns:
            raise ValueError(f"missing required column: {SALES_FILLED_COL}")

        dynamic_cols = [SALES_FILLED_COL] + [f for f in feature_cols if f not in static_cols]
        dynamic_cols = [c for c in dict.fromkeys(dynamic_cols) if c]
        if len(dynamic_cols) < 1:
            raise ValueError("no dynamic features: ensure 'sales_filled' exists")
        self.dynamic_idx = {c: i for i, c in enumerate(dynamic_cols)}
        # Static features are kept separate from dynamic channels
        self.static_idx = {c: i for i, c in enumerate(static_cols)}

        X_dyn_list, X_stat_list, Y_list, S_list, D_list = [], [], [], [], []
        for sid, g in d.groupby(SERIES_COL, sort=False):
            g = g.reset_index(drop=True)
            # windows
            for t in range(self.L - 1, len(g) - self.H):
                dyn_window = g.loc[t - self.L + 1: t, dynamic_cols].values.astype(float)
                # Require label availability
                y_vec = g.loc[t + 1: t + self.H, SALES_COL].values.astype(float)
                if np.any(np.isnan(y_vec)):
                    continue
                if static_cols:
                    stat_vals = g.loc[t, static_cols].values.astype(np.int64)
                    stat_vals = np.clip(stat_vals + 1, 0, None)
                else:
                    stat_vals = np.empty((0,), dtype=np.int64)
                X_dyn_list.append(dyn_window)
                X_stat_list.append(stat_vals)
                Y_list.append(y_vec)
                S_list.append(sid)
                D_list.append(g.loc[t + self.H, DATE_COL])
        if not X_dyn_list:
            raise RuntimeError("No PatchTST training windows produced.")
        vprint(f"[PATCH/TRAIN] windows={len(X_dyn_list)}")

        dates = np.array(D_list, dtype="datetime64[ns]")
        sids = np.array(S_list, dtype=object)
        order = np.argsort(dates)
        X_dyn_arr = np.stack(X_dyn_list, axis=0)[order]
        X_stat_arr = (
            np.stack(X_stat_list, axis=0)[order]
            if static_cols
            else np.empty((len(X_dyn_arr), 0), dtype=np.int64)
        )
        Y_arr = np.stack(Y_list, axis=0)[order]
        sids = sids[order]
        dates = dates[order]

        return (
            X_dyn_arr,
            X_stat_arr,
            Y_arr,
            sids,
            dates,
            self.dynamic_idx,
            self.static_idx,
        )

    def build_patch_eval(
            self,
            df_eval: pd.DataFrame,
            feature_cols: List[str],
            static_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
        if self.guard:
            self.guard.assert_scope({"eval"})
        d = df_eval.sort_values([SERIES_COL, DATE_COL]).copy()

        if SALES_FILLED_COL not in d.columns:
            raise ValueError(f"missing required column: {SALES_FILLED_COL}")

        dynamic_cols = [SALES_FILLED_COL] + [f for f in feature_cols if f not in static_cols]
        dynamic_cols = [c for c in dict.fromkeys(dynamic_cols) if c]
        if len(dynamic_cols) < 1:
            raise ValueError("no dynamic features: ensure 'sales_filled' exists")
        self.dynamic_idx = {c: i for i, c in enumerate(dynamic_cols)}
        self.static_idx = {c: i for i, c in enumerate(static_cols)}

        X_dyn_list, X_stat_list, S_list, D_list = [], [], [], []
        for sid, g in d.groupby(SERIES_COL, sort=False):
            g = g.reset_index(drop=True)
            if len(g) < self.L:
                warn(f"Eval series {sid}: insufficient length {len(g)} for L={self.L}.")
                continue
            dyn_window = g.loc[len(g) - self.L: len(g) - 1, dynamic_cols].values.astype(float)
            if static_cols:
                stat_vals = g.loc[len(g) - 1, static_cols].values.astype(np.int64)
                stat_vals = np.clip(stat_vals + 1, 0, None)
            else:
                stat_vals = np.empty((0,), dtype=np.int64)
            X_dyn_list.append(dyn_window)
            X_stat_list.append(stat_vals)
            S_list.append(sid)
            D_list.append(g.loc[len(g) - 1, DATE_COL])
        if not X_dyn_list:
            raise RuntimeError("No PatchTST eval windows produced.")
        vprint(f"[PATCH/EVAL] windows={len(X_dyn_list)}")

        sids = np.array(S_list, dtype=object)
        dates = np.array(D_list, dtype="datetime64[ns]")
        X_dyn_arr = np.stack(X_dyn_list, axis=0)
        X_stat_arr = (
            np.stack(X_stat_list, axis=0)
            if static_cols
            else np.empty((len(X_dyn_arr), 0), dtype=np.int64)
        )
        return X_dyn_arr, X_stat_arr, sids, dates, self.dynamic_idx, self.static_idx


# ------------------------------
# Orchestrator
# ------------------------------
@dataclass
class PreprocessorArtifacts:
    schema_normalizer: SchemaNormalizer
    outlier_handler: MissingAndOutlierHandler
    rich_lookup: RichLookupBuilder
    encoder: Encoder
    calendar_maker: CalendarFeatureMaker
    leak_guard: LeakGuard
    feature_cols: List[str]
    low_var_cols: List[str]
    static_cols: List[str]
    static_feature_cols: List[str]
    dynamic_feature_cols: List[str]


class Preprocessor:
    """
    High-level API to fit on train and transform eval with strict leakage control.
    """

    def __init__(self, *, cyclical: bool = True, show_progress: bool = False):
        self.guard = LeakGuard()
        self.schema = SchemaNormalizer()
        self.cont_fix = DateContinuityFixer()
        self.calendar = CalendarFeatureMaker(cyclical=cyclical)
        self.missing_outlier = MissingAndOutlierHandler()
        self.strict_feats = StrictFeatureMaker()
        self.rich = RichLookupBuilder()
        self.encoder = Encoder()
        self.windowizer = SampleWindowizer(guard=self.guard)
        self.feature_cols: List[str] = []
        self.low_var_cols: List[str] = []
        self.static_cols: List[str] = []
        # Feature column subsets for PatchTST
        self.static_feature_cols: List[str] = []
        self.dynamic_feature_cols: List[str] = []
        # PatchTST-specific filtered feature columns
        self.patch_feature_cols: List[str] = []
        self.patch_static_feature_cols: List[str] = []
        # Channel index maps produced by the windowizer
        self.patch_dynamic_idx: Dict[str, int] = {}
        self.patch_static_idx: Dict[str, int] = {}
        self.show_progress = show_progress

    def _compute_patch_features(self, df: Optional[pd.DataFrame] = None):
        drop = {
            "is_priority_outlet",
            "is_promo",
            "zero_ratio_28",
            "days_since_last_sale",
            "series_code",
            "series_cv",
        }
        self.patch_feature_cols = [
            c
            for c in self.feature_cols
            if c not in drop and not c.startswith("lag_") and not c.startswith("roll_")
        ]
        self.patch_static_feature_cols = [
            c for c in self.static_feature_cols if c in self.patch_feature_cols
        ]
        return df

    # --------------------------
    # Train
    # --------------------------
    def fit_transform_train(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        self.guard.set_scope("train")
        self.guard.assert_scope({"train"})

        def _schema(df: pd.DataFrame) -> pd.DataFrame:
            df = self.schema.fit(df).transform(df, allow_new=True)
            vprint(f"[FIT] schema: rows={len(df)}  series={df[SERIES_COL].nunique()}")
            return df

        def _continuity(df: pd.DataFrame) -> pd.DataFrame:
            df = self.cont_fix.transform(df)
            vprint(f"[FIT] continuity: rows={len(df)}")
            return df

        def _calendar(df: pd.DataFrame) -> pd.DataFrame:
            self.calendar.fit(df)
            df = self.calendar.transform(df)
            vprint(f"[FIT] calendar added: cols={len(df.columns)}")
            return df

        def _missing_outlier(df: pd.DataFrame) -> pd.DataFrame:
            self.missing_outlier.fit(df)
            df = self.missing_outlier.transform(df)
            vprint(f"[FIT] outlier/na: nonnull_sales={df[SALES_COL].notna().sum()}")
            return df

        def _strict_feats(df: pd.DataFrame) -> pd.DataFrame:
            df = self.strict_feats.transform(df)
            vprint(f"[FIT] strict feats: cols={len(df.columns)}")
            return df

        def _rich_lookup(df: pd.DataFrame) -> pd.DataFrame:
            self.rich.fit(df)
            return self.rich.transform(df)

        def _encode(df: pd.DataFrame) -> pd.DataFrame:
            self.encoder.fit(df)
            return self.encoder.transform(df)

        def _drop_low_var(df: pd.DataFrame) -> pd.DataFrame:
            self.low_var_cols = [c for c in df.columns if df[c].nunique() <= 1]
            if self.low_var_cols:
                logging.warning(
                    "Dropping %d low variance columns: %s",
                    len(self.low_var_cols),
                    self.low_var_cols,
                )
                df = df.drop(columns=self.low_var_cols)
            return df

        def _feature_cols(df: pd.DataFrame) -> pd.DataFrame:
            self.feature_cols = self._select_feature_columns(df)
            assert set(self.low_var_cols).isdisjoint(self.feature_cols)
            gb = df.groupby(SERIES_COL)
            self.static_cols = [
                c for c in self.feature_cols if gb[c].nunique().le(1).all()
            ]
            # Track explicit static feature columns for PatchTST
            self.static_feature_cols = [
                c for c in ["shop_code", "menu_code"] if c in self.feature_cols
            ]
            self.dynamic_feature_cols = [
                c for c in self.feature_cols if c not in self.static_feature_cols
            ]
            vprint(
                f"[FIT] rich+encode: feature_cols={len(self.feature_cols)}  total_cols={len(df.columns)}  static_cols={len(self.static_cols)}"
            )
            return df

        steps = [
            ("schema", _schema),
            ("continuity", _continuity),
            ("calendar", _calendar),
            ("missing_outlier", _missing_outlier),
            ("strict_feats", _strict_feats),
            ("rich_lookup", _rich_lookup),
            ("encoder", _encode),
            ("drop_low_var", _drop_low_var),
            ("feature_cols", _feature_cols),
            ("patch_features", self._compute_patch_features),
        ]

        df = df_raw
        with get_progress_bar(total=len(steps), disable=not self.show_progress) as pbar:
            for name, func in steps:
                df = func(df)
                pbar.set_description(name)
                pbar.update(1)
        return df

    # --------------------------
    # Eval
    # --------------------------
    def transform_eval(self, df_eval_raw: pd.DataFrame) -> pd.DataFrame:
        self.guard.set_scope("eval")
        self.guard.assert_scope({"eval"})

        steps = [
            ("schema", lambda df: self.schema.transform(df, allow_new=False)),
            ("continuity", self.cont_fix.transform),
            ("calendar", self.calendar.transform),
            ("missing_outlier", self.missing_outlier.transform),
            ("strict_feats", self.strict_feats.transform),
            ("rich_lookup", self.rich.transform),
            ("encoder", self.encoder.transform),
            ("drop_low_var", lambda df: df.drop(columns=self.low_var_cols, errors="ignore")),
        ]

        df = df_eval_raw
        with get_progress_bar(total=len(steps), disable=not self.show_progress) as pbar:
            for name, func in steps:
                df = func(df)
                pbar.set_description(name)
                pbar.update(1)

        # no feature_cols change
        vprint(f"[EVAL] transformed: rows={len(df)}  series={df[SERIES_COL].nunique()}  cols={len(df.columns)}")

        return df

    # --------------------------
    # Build datasets
    # --------------------------
    def build_lgbm_train(self, df_full: pd.DataFrame) -> pd.DataFrame:
        self.guard.assert_scope({"train"})
        return self.windowizer.build_lgbm_train(df_full, self.feature_cols)

    def build_lgbm_eval(self, df_eval_full: pd.DataFrame) -> pd.DataFrame:
        self.guard.assert_scope({"eval"})
        return self.windowizer.build_lgbm_eval(
            df_eval_full, self.feature_cols, show_progress=self.show_progress
        )

    def build_patch_train(
        self, df_full: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.guard.assert_scope({"train"})
        X_dyn, X_stat, Y, sids, dates, dyn_idx, stat_idx = self.windowizer.build_patch_train(
            df_full, self.patch_feature_cols, self.patch_static_feature_cols
        )
        self.patch_dynamic_idx = dyn_idx
        self.patch_static_idx = stat_idx
        return X_dyn, X_stat, Y, sids, dates

    def build_patch_eval(
        self, df_eval_full: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.guard.assert_scope({"eval"})
        X_dyn, X_stat, sids, dates, dyn_idx, stat_idx = self.windowizer.build_patch_eval(
            df_eval_full, self.patch_feature_cols, self.patch_static_feature_cols
        )
        self.patch_dynamic_idx = dyn_idx
        self.patch_static_idx = stat_idx
        return X_dyn, X_stat, sids, dates

    # --------------------------
    # Artifacts I/O
    # --------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifacts = PreprocessorArtifacts(
            schema_normalizer=self.schema,
            outlier_handler=self.missing_outlier,
            rich_lookup=self.rich,
            encoder=self.encoder,
            calendar_maker=self.calendar,
            leak_guard=self.guard,
            feature_cols=self.feature_cols,
            low_var_cols=self.low_var_cols,
            static_cols=self.static_cols,
            static_feature_cols=self.static_feature_cols,
            dynamic_feature_cols=self.dynamic_feature_cols,
        )
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            art: PreprocessorArtifacts = pickle.load(f)
        self.schema = art.schema_normalizer
        self.missing_outlier = art.outlier_handler
        self.rich = art.rich_lookup
        self.encoder = art.encoder
        self.calendar = art.calendar_maker
        self.guard = art.leak_guard
        self.feature_cols = art.feature_cols
        self.low_var_cols = art.low_var_cols
        self.static_cols = art.static_cols
        self.static_feature_cols = art.static_feature_cols
        self.dynamic_feature_cols = art.dynamic_feature_cols
        self._compute_patch_features()
        # re-wire
        self.cont_fix = DateContinuityFixer()
        self.strict_feats = StrictFeatureMaker()
        self.windowizer = SampleWindowizer(guard=self.guard)

    # --------------------------
    # Helpers
    # --------------------------
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        # Exclude target and identifiers
        exclude = {
            SALES_COL, SALES_FILLED_COL, DATE_COL, SHOP_COL, MENU_COL, SERIES_COL
        }
        # Keep numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in num_cols if c not in exclude]
        # Safety: ensure deterministic ordering
        cols = sorted(cols)
        for c in ["days_since_holiday", "days_to_next_holiday"]:
            if c in df.columns and c not in cols:
                cols.append(c)
        cols = sorted(cols)
        return cols


# ------------------------------
# Weighted SMAPE for local validation (optional)
# ------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def weighted_smape(
        y_true: np.ndarray, y_pred: np.ndarray, outlet_names: List[str]
) -> float:
    weights = np.array([3.0 if o in PRIORITY_OUTLETS else 1.0 for o in outlet_names], dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.average(np.abs(y_true[mask] - y_pred[mask]) / denom[mask], weights=weights[mask]))


# ------------------------------
# CLI (optional)
# ------------------------------
def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing Pipeline v1.1")
    sub = parser.add_subparsers(dest="cmd")

    fitp = sub.add_parser("fit", help="Fit preprocessor on train data and save artifacts")
    fitp.add_argument("--train_path", required=True, help="Path to TRAIN file (.csv/.xlsx)")
    fitp.add_argument("--artifacts", required=True, help="Path to save artifacts .pkl")
    fitp.add_argument("--train_out", required=False, help="Optional path to save full processed TRAIN parquet")

    evp = sub.add_parser("eval", help="Transform eval data with saved artifacts")
    evp.add_argument("--eval_path", required=True, help="Path to EVAL file (.csv/.xlsx)")
    evp.add_argument("--artifacts", required=True, help="Path to load artifacts .pkl")
    evp.add_argument("--eval_out_full", required=False, help="Optional path to save processed EVAL parquet")
    evp.add_argument("--lgbm_eval_csv", required=False, help="Optional path to save LGBM eval features CSV")
    evp.add_argument("--patch_eval_npy", required=False,
                     help="Optional base path to save PatchTST X numpy (adds _X.npy and _meta.json)")

    args = parser.parse_args()

    if args.cmd == "fit":
        df_train_raw = _read_table(args.train_path)
        pp = Preprocessor()
        df_full = pp.fit_transform_train(df_train_raw)
        vprint(f"[CLI/FIT] processed_train: rows={len(df_full)}  cols={len(df_full.columns)}")

        if args.train_out:
            os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
            df_full.to_parquet(args.train_out, index=False)

        pp.save(args.artifacts)
        vprint(f"Artifacts saved to {args.artifacts}")
        vprint(
            f"Feature columns ({len(pp.feature_cols)}): {pp.feature_cols[:10]}{'...' if len(pp.feature_cols) > 10 else ''}")
        return

    if args.cmd == "eval":
        pp = Preprocessor()
        pp.load(args.artifacts)

        df_eval_raw = _read_table(args.eval_path)
        df_eval_full = pp.transform_eval(df_eval_raw)
        vprint(f"[CLI/EVAL] processed_eval: rows={len(df_eval_full)}  cols={len(df_eval_full.columns)}")

        if args.eval_out_full:
            os.makedirs(os.path.dirname(args.eval_out_full), exist_ok=True)
            df_eval_full.to_parquet(args.eval_out_full, index=False)

        # Build LGBM eval rows
        if args.lgbm_eval_csv:
            lgbm_eval = pp.build_lgbm_eval(df_eval_full)
            os.makedirs(os.path.dirname(args.lgbm_eval_csv), exist_ok=True)
            lgbm_eval.to_csv(args.lgbm_eval_csv, index=False, encoding="utf-8-sig")
            vprint(f"LGBM eval rows saved: {args.lgbm_eval_csv}")

        # Build PatchTST eval windows
        if args.patch_eval_npy:
            X_eval, sids, dates = pp.build_patch_eval(df_eval_full)
            x_path = args.patch_eval_npy + "_X.npy"
            m_path = args.patch_eval_npy + "_meta.json"
            os.makedirs(os.path.dirname(x_path), exist_ok=True)
            np.save(x_path, X_eval)
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump([{"series_id": sid, "asof": str(ts.date())} for sid, ts in zip(sids, dates)], f, ensure_ascii=False,
                          indent=2)
            vprint(f"PatchTST eval windows saved: {x_path}, meta: {m_path}")
        return

    parser.print_help()

# ----- Hardcoded runner (no-CLI) -----
# 이 블록을 켜면 아래 지정 경로로 즉시 실행된다.
HARDCODED = False  # 사용 안 하려면 False
if HARDCODED and __name__ == "__main__":
    import sys

    # TODO: 아래 경로를 실제 파일로 교체
    TRAIN_PATH = r"/Users/castorp/Downloads/open (1)/train/train.csv"      # 학습 원본
    EVAL_PATH = r'/Users/castorp/Downloads/open (1)/test/TEST_00.csv'        # 평가 원본(선택)
    ARTIFACTS_PATH = r'/Users/castorp/Downloads/open (1)/artifacts.pkl'  # 아티팩트 저장
    LGBM_EVAL_OUT = r'/Users/castorp/Downloads/open (1)/eval_lgbm.csv'   # LGBM용 평가 피처 출력
    PATCH_EVAL_OUT = r'/Users/castorp/Downloads/open (1)'     # PatchTST X/메타 저장 prefix

    pp = Preprocessor()

    # 1) 학습 전처리 + 아티팩트 저장
    df_train_raw = _read_table(TRAIN_PATH)
    df_full = pp.fit_transform_train(df_train_raw)
    pp.save(ARTIFACTS_PATH)
    vprint(f"[HARDCODED] train processed: rows={len(df_full)}, cols={len(df_full.columns)}")
    vprint(f"[HARDCODED] artifacts saved: {ARTIFACTS_PATH}")

    # 2) 평가 전처리 + 산출물 저장(선택)
    if EVAL_PATH:
        df_eval_raw = _read_table(EVAL_PATH)
        df_eval_full = pp.transform_eval(df_eval_raw)

        # LGBM용 평가 피처
        lgbm_eval = pp.build_lgbm_eval(df_eval_full)
        lgbm_eval.to_csv(LGBM_EVAL_OUT, index=False, encoding="utf-8-sig")
        vprint(f"[HARDCODED] LGBM eval saved: {LGBM_EVAL_OUT}  rows={len(lgbm_eval)}")

        # PatchTST용 평가 윈도
        X_eval, sids, dates = pp.build_patch_eval(df_eval_full)
        np.save(PATCH_EVAL_OUT + "_X.npy", X_eval)
        with open(PATCH_EVAL_OUT + "_meta.json", "w", encoding="utf-8") as f:
            json.dump([{"series_id": sid, "asof": str(ts.date())} for sid, ts in zip(sids, dates)], f, ensure_ascii=False, indent=2)
        vprint(f"[HARDCODED] Patch eval saved: {PATCH_EVAL_OUT}_X.npy, {PATCH_EVAL_OUT}_meta.json, windows={len(X_eval)}")

    sys.exit(0)

if __name__ == "__main__":
    main()
