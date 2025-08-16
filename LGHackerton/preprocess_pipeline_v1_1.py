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
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(x, *args, **kwargs):  # type: ignore
        return x

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
VERBOSE = True  # 필요시 False


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
    print(f"[WARN] {msg}")


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

    def __init__(self, holiday_provider: Optional[HolidayProvider] = None):
        self.holiday_provider = holiday_provider or HolidayProvider()
        self._holiday_cache: set = set()

    def fit(self, df: pd.DataFrame):
        years = df[DATE_COL].dt.year.unique().tolist()
        self._holiday_cache = self.holiday_provider.compute(years)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["year"] = d[DATE_COL].dt.year
        d["month"] = d[DATE_COL].dt.month
        d["day"] = d[DATE_COL].dt.day
        d["dow"] = d[DATE_COL].dt.weekday  # 0=Mon
        d["weekofyear"] = d[DATE_COL].dt.isocalendar().week.astype(int)
        d["is_weekend"] = d["dow"].isin([5, 6]).astype(np.int8)
        d["is_month_start"] = d[DATE_COL].dt.is_month_start.astype(np.int8)
        d["is_month_end"] = d[DATE_COL].dt.is_month_end.astype(np.int8)
        if self._holiday_cache:
            d["is_holiday"] = d[DATE_COL].dt.date.isin(self._holiday_cache).astype(np.int8)
        else:
            d["is_holiday"] = 0
        d["is_priority_outlet"] = d[SHOP_COL].isin(PRIORITY_OUTLETS).astype(np.int8)
        return d


class MissingAndOutlierHandler:
    """
    - Target NA handling policy for features: ffill(limit=3) then remaining to 0 into SALES_FILLED_COL
    - Outlier capping with per-series thresholds computed from TRAIN only
      Hampel-like robust cap: min( Q3+1.5*IQR, Q99 )
    """

    def __init__(self):
        self.caps: Dict[str, float] = {}
        import pandas as _pd  # 중복 임포트 안전
        _v = _pd.Series(self.caps).apply(lambda x: not _pd.isna(x)).sum()
        vprint(f"[OutlierCap.fit] series={len(self.caps)}  with_caps={_v}")

    def fit(self, df: pd.DataFrame):
        caps = {}
        for sid, g in df.groupby(SERIES_COL, sort=False):
            x = g[SALES_COL].dropna().values
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

        # apply caps
        def cap_series(sid: str, x: pd.Series) -> pd.Series:
            c = self.caps.get(sid, np.nan)
            if np.isnan(c):
                return x
            return x.astype(float).clip(lower=0, upper=c)


        d[SALES_COL] = d.groupby(SERIES_COL)[SALES_COL].transform(lambda x: cap_series(x.name, x))

        # filled series for feature calculations
        def fill_series(x: pd.Series) -> pd.Series:
            y = x.copy()
            y = y.ffill(limit=3)
            y = y.fillna(0.0)
            return y

        d[SALES_FILLED_COL] = d.groupby(SERIES_COL)[SALES_COL].transform(fill_series)
        return d


class StrictFeatureMaker:
    """
    Strict features computed only from past values within max window size L=28.
    - lags: 1,2,7,14,21,28
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

        # Lags
        for lag in [1, 2, 7, 14, 21, 27, 28]:
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

        d["days_since_last_sale"] = gb[SALES_FILLED_COL].transform(lambda s: pd.Series(days_since_last_sale(s.values)))

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

        d["zero_run_len"] = gb[SALES_FILLED_COL].transform(lambda s: pd.Series(zero_run_len(s.values)))
        vprint(f"[Strict] rolling+intermittency done  rows={len(d)}")

        return d


class RichLookupBuilder:
    """
    Precompute per-series statistics from TRAIN only, then attach via lookup in any scope.
    - series_base_mean
    - series_cv
    - series_dow_mean (by series_id, dow)
    """

    def __init__(self):
        self.base_table: Optional[pd.DataFrame] = None
        self.dow_table: Optional[pd.DataFrame] = None

    def fit(self, df_train: pd.DataFrame):
        # base: per-series mean/std -> cv
        g = (
            df_train.groupby(SERIES_COL)[SALES_COL]
            .agg(series_base_mean="mean", series_std="std")
            .reset_index()
        )
        g["series_cv"] = (g["series_std"] / (g["series_base_mean"] + 1e-9)).astype(float)
        self.base_table = g[[SERIES_COL, "series_base_mean", "series_cv"]].copy()

        # dow lookup: per (series_id, dow) mean
        if "dow" not in df_train.columns:
            raise ValueError("Calendar features required before RichLookupBuilder.fit")
        gd = (
            df_train.groupby([SERIES_COL, "dow"])[SALES_COL]
            .mean()
            .reset_index()
            .rename(columns={SALES_COL: "series_dow_mean"})
        )
        self.dow_table = gd

        vprint(f"[Rich.fit] base_rows={len(self.base_table)}  dow_rows={len(self.dow_table)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_table is None or self.dow_table is None:
            raise RuntimeError("RichLookupBuilder not fit.")
        d = df.merge(self.base_table, on=SERIES_COL, how="left")
        d = d.merge(self.dow_table, on=[SERIES_COL, "dow"], how="left")
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

    def __init__(self, lookback: int = L, horizon: int = H):
        self.L = lookback
        self.H = horizon

    def build_lgbm_train(
            self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        d = df.sort_values([SERIES_COL, DATE_COL]).copy()
        # Need at least lag_28 available to respect 28-day features
        if "lag_27" not in d.columns:
            raise ValueError("Strict features required before building LGBM train set.")
        rows = []
        gb = d.groupby(SERIES_COL, sort=False)
        for sid, g in tqdm(gb, total=d[SERIES_COL].nunique(), desc="lgbm-train"):
            g = g.reset_index(drop=True)
            # candidate t where we have full lookback features
            valid_mask = g["lag_27"].notna()
            idxs = np.where(valid_mask.values)[0]
            for t in idxs:
                # ensure labels availability for all horizons
                for h in range(1, self.H + 1):
                    if t + h >= len(g):
                        break
                    y = g.loc[t + h, SALES_COL]
                    if pd.isna(y):
                        # training target must exist
                        continue
                    row = {
                        SERIES_COL: sid,
                        DATE_COL: g.loc[t, DATE_COL],
                        "h": h,
                        "y": float(y),
                    }
                    row.update(g.loc[t, feature_cols].to_dict())
                    rows.append(row)
        if not rows:
            raise RuntimeError("No training rows produced. Check input coverage.")
        out = pd.DataFrame(rows)
        vprint(f"[LGBM/TRAIN] rows={len(out)}  feats={len(feature_cols)}")

        return out.reset_index(drop=True)

    def build_lgbm_eval(
            self, df_eval: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        d = df_eval.sort_values([SERIES_COL, DATE_COL]).copy()
        out_rows = []
        for sid, g in d.groupby(SERIES_COL, sort=False):
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
            feat = g.loc[t, feature_cols]
            for h in range(1, self.H + 1):
                row = pd.concat([pd.Series({**base, "h": h}), feat])
                out_rows.append(row)
        if not out_rows:
            raise RuntimeError("No eval rows produced.")

        _out = pd.DataFrame(out_rows).reset_index(drop=True)
        vprint(f"[LGBM/EVAL] rows={len(_out)}  feats={len(feature_cols)}")
        return _out


        return pd.DataFrame(out_rows).reset_index(drop=True)

    def build_patch_train(
            self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build PatchTST training windows with identifiers."""
        d = df.sort_values([SERIES_COL, DATE_COL]).copy()
        X_list, Y_list, S_list, D_list = [], [], [], []
        for sid, g in d.groupby(SERIES_COL, sort=False):
            g = g.reset_index(drop=True)
            # windows
            for t in range(self.L - 1, len(g) - self.H):
                x_window = g.loc[t - self.L + 1: t, SALES_FILLED_COL].values.astype(float)
                # Require label availability
                y_vec = g.loc[t + 1: t + self.H, SALES_COL].values.astype(float)
                if np.any(np.isnan(y_vec)):
                    continue
                X_list.append(x_window.reshape(self.L, 1))
                Y_list.append(y_vec)
                S_list.append(sid)
                D_list.append(g.loc[t + self.H, DATE_COL])
        if not X_list:
            raise RuntimeError("No PatchTST training windows produced.")
        vprint(f"[PATCH/TRAIN] windows={len(X_list)}")

        dates = np.array(D_list, dtype="datetime64[ns]")
        sids = np.array(S_list, dtype=object)
        order = np.argsort(dates)
        X_arr = np.stack(X_list, axis=0)[order]
        Y_arr = np.stack(Y_list, axis=0)[order]
        sids = sids[order]
        dates = dates[order]

        return X_arr, Y_arr, sids, dates

    def build_patch_eval(
            self, df_eval: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = df_eval.sort_values([SERIES_COL, DATE_COL]).copy()
        X_list, S_list, D_list = [], [], []
        for sid, g in d.groupby(SERIES_COL, sort=False):
            g = g.reset_index(drop=True)
            if len(g) < self.L:
                warn(f"Eval series {sid}: insufficient length {len(g)} for L={self.L}.")
                continue
            x_window = g.loc[len(g) - self.L: len(g) - 1, SALES_FILLED_COL].values.astype(float)
            X_list.append(x_window.reshape(self.L, 1))
            S_list.append(sid)
            D_list.append(g.loc[len(g) - 1, DATE_COL])
        if not X_list:
            raise RuntimeError("No PatchTST eval windows produced.")
        vprint(f"[PATCH/EVAL] windows={len(X_list)}")

        sids = np.array(S_list, dtype=object)
        dates = np.array(D_list, dtype="datetime64[ns]")
        return np.stack(X_list, axis=0), sids, dates


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


class Preprocessor:
    """
    High-level API to fit on train and transform eval with strict leakage control.
    """

    def __init__(self):
        self.guard = LeakGuard()
        self.schema = SchemaNormalizer()
        self.cont_fix = DateContinuityFixer()
        self.calendar = CalendarFeatureMaker()
        self.missing_outlier = MissingAndOutlierHandler()
        self.strict_feats = StrictFeatureMaker()
        self.rich = RichLookupBuilder()
        self.encoder = Encoder()
        self.windowizer = SampleWindowizer()
        self.feature_cols: List[str] = []

    # --------------------------
    # Train
    # --------------------------
    def fit_transform_train(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        self.guard.set_scope("train")

        # Schema normalize
        df = self.schema.fit(df_raw).transform(df_raw, allow_new=True)
        vprint(f"[FIT] schema: rows={len(df)}  series={df[SERIES_COL].nunique()}")

        # Daily continuity
        df = self.cont_fix.transform(df)
        vprint(f"[FIT] continuity: rows={len(df)}")
        # Calendar
        self.calendar.fit(df)
        df = self.calendar.transform(df)
        vprint(f"[FIT] calendar added: cols={len(df.columns)}")
        # Missing + Outlier
        self.missing_outlier.fit(df)
        df = self.missing_outlier.transform(df)
        vprint(f"[FIT] outlier/na: nonnull_sales={df[SALES_COL].notna().sum()}")

        # Strict features
        df = self.strict_feats.transform(df)
        vprint(f"[FIT] strict feats: cols={len(df.columns)}")
        # Rich lookup from TRAIN only
        self.rich.fit(df)
        df = self.rich.transform(df)

        # Encoder
        self.encoder.fit(df)
        df = self.encoder.transform(df)

        # Feature columns for LGBM
        self.feature_cols = self._select_feature_columns(df)
        vprint(f"[FIT] rich+encode: feature_cols={len(self.feature_cols)}  total_cols={len(df.columns)}")
        return df

    # --------------------------
    # Eval
    # --------------------------
    def transform_eval(self, df_eval_raw: pd.DataFrame) -> pd.DataFrame:
        self.guard.set_scope("eval")

        # Transform-only path. No fitting.
        df = self.schema.transform(df_eval_raw, allow_new=False)
        df = self.cont_fix.transform(df)
        df = self.calendar.transform(df)
        df = self.missing_outlier.transform(df)
        df = self.strict_feats.transform(df)
        df = self.rich.transform(df)
        df = self.encoder.transform(df)
        # no feature_cols change
        vprint(f"[EVAL] transformed: rows={len(df)}  series={df[SERIES_COL].nunique()}  cols={len(df.columns)}")

        return df

    # --------------------------
    # Build datasets
    # --------------------------
    def build_lgbm_train(self, df_full: pd.DataFrame) -> pd.DataFrame:
        return self.windowizer.build_lgbm_train(df_full, self.feature_cols)

    def build_lgbm_eval(self, df_eval_full: pd.DataFrame) -> pd.DataFrame:
        return self.windowizer.build_lgbm_eval(df_eval_full, self.feature_cols)

    def build_patch_train(self, df_full: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.windowizer.build_patch_train(df_full)

    def build_patch_eval(self, df_eval_full: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.windowizer.build_patch_eval(df_eval_full)

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
        # re-wire
        self.cont_fix = DateContinuityFixer()
        self.strict_feats = StrictFeatureMaker()
        self.windowizer = SampleWindowizer()

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
        print(f"Artifacts saved to {args.artifacts}")
        print(
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
            print(f"LGBM eval rows saved: {args.lgbm_eval_csv}")

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
            print(f"PatchTST eval windows saved: {x_path}, meta: {m_path}")
        return

    parser.print_help()

# ----- Hardcoded runner (no-CLI) -----
# 이 블록을 켜면 아래 지정 경로로 즉시 실행된다.
HARDCODED = True  # 사용 안 하려면 False
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
    print(f"[HARDCODED] train processed: rows={len(df_full)}, cols={len(df_full.columns)}")
    print(f"[HARDCODED] artifacts saved: {ARTIFACTS_PATH}")

    # 2) 평가 전처리 + 산출물 저장(선택)
    if EVAL_PATH:
        df_eval_raw = _read_table(EVAL_PATH)
        df_eval_full = pp.transform_eval(df_eval_raw)

        # LGBM용 평가 피처
        lgbm_eval = pp.build_lgbm_eval(df_eval_full)
        lgbm_eval.to_csv(LGBM_EVAL_OUT, index=False, encoding="utf-8-sig")
        print(f"[HARDCODED] LGBM eval saved: {LGBM_EVAL_OUT}  rows={len(lgbm_eval)}")

        # PatchTST용 평가 윈도
        X_eval, sids, dates = pp.build_patch_eval(df_eval_full)
        np.save(PATCH_EVAL_OUT + "_X.npy", X_eval)
        with open(PATCH_EVAL_OUT + "_meta.json", "w", encoding="utf-8") as f:
            json.dump([{"series_id": sid, "asof": str(ts.date())} for sid, ts in zip(sids, dates)], f, ensure_ascii=False, indent=2)
        print(f"[HARDCODED] Patch eval saved: {PATCH_EVAL_OUT}_X.npy, {PATCH_EVAL_OUT}_meta.json, windows={len(X_eval)}")

    sys.exit(0)

if __name__ == "__main__":
    main()
