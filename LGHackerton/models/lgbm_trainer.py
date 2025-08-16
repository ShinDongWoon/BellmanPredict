
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import lightgbm as lgb
from LGHackerton.models.base_trainer import BaseModel, TrainConfig
from LGHackerton.utils.metrics import lgbm_weighted_smape
from LGHackerton.preprocess import L, H
PRIORITY_OUTLETS = {"담하", "미라시아"}

@dataclass
class LGBMParams:
    objective: str = "tweedie"       # "tweedie" or "mae"
    tweedie_variance_power: float = 1.3
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_estimators: int = 3000
    early_stopping_rounds: int = 200

class LGBMTrainer(BaseModel):
    def __init__(self, params: LGBMParams, features: List[str], model_dir: str):
        super().__init__(model_params=asdict(params), model_dir=model_dir)
        self.params = params
        self.features = features
        # for each horizon keep separate lists for regressor and classifier boosters
        self.models: Dict[int, Dict[str, List[lgb.Booster]]] = {
            h: {"reg": [], "clf": []} for h in range(1, 8)
        }
        self.use_asinh_target = False
        self.use_hurdle = False
        self.oof_records: List[Dict[str, Any]] = []

    @staticmethod
    def _compute_label_date(df: pd.DataFrame, date_col: str, h_col: str) -> pd.Series:
        return pd.to_datetime(df[date_col]) + pd.to_timedelta(df[h_col].astype(int), unit="D")

    def _make_cv_slices(self, df_h: pd.DataFrame, cfg: TrainConfig, date_col: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(df_h[date_col]).sort_values().unique()
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        if len(dates) == 0:
            return folds
        purge_days = cfg.purge_days if getattr(cfg, "purge_days", 0) > 0 else (L + H if cfg.purge_mode == "L+H" else L)
        purge = np.timedelta64(purge_days, 'D')
        for i in range(cfg.n_folds):
            end = dates[-1] - np.timedelta64(i * cfg.cv_stride, 'D')
            start = end - np.timedelta64(cfg.cv_stride - 1, 'D')
            val_mask = (df_h[date_col] >= start) & (df_h[date_col] <= end)
            train_mask = df_h[date_col] < (start - purge)
            if val_mask.sum() > 0 and train_mask.sum() > 0:
                assert df_h.loc[train_mask, date_col].max() < df_h.loc[val_mask, date_col].min() - np.timedelta64(purge_days, "D")
                folds.append((train_mask.values, val_mask.values))
        assert folds, "No valid folds generated"
        return folds

    def train(self, df_train: pd.DataFrame, cfg: TrainConfig) -> None:
        self.use_asinh_target = cfg.use_asinh_target
        self.use_hurdle = getattr(cfg, "use_hurdle", False)
        os.makedirs(self.model_dir, exist_ok=True)

        df = df_train.copy()
        df["label_date"] = self._compute_label_date(df, "date", "h")
        self.oof_records = []

        for h in range(1, 8):
            dfh = df[df["h"] == h].reset_index(drop=True)
            if dfh.empty:
                continue
            feat_cols = [c for c in self.features if c in dfh.columns]
            X = dfh[feat_cols].fillna(0).astype("float32").values
            y = dfh["y"].astype("float32").values
            # classification label
            z = (y > 0).astype("float32")
            if cfg.use_asinh_target:
                if self.params.objective != "mae":
                    self.params.objective = "mae"
                y_tr = np.arcsinh(y)
            else:
                y_tr = y
            shops = dfh["series_id"].str.split("::").str[0].values
            w = np.where(np.isin(shops, list(PRIORITY_OUTLETS)), cfg.priority_weight, 1.0).astype("float32")

            folds = self._make_cv_slices(dfh, cfg, "label_date")
            self.models[h] = {"reg": [], "clf": []}
            for i, (tr_mask, va_mask) in enumerate(folds):
                X_tr, y_tr_f = X[tr_mask], y_tr[tr_mask]
                X_va, y_va_f = X[va_mask], y_tr[va_mask]
                y_va = y[va_mask]
                z_tr, z_va = z[tr_mask], z[va_mask]
                w_tr, w_va = w[tr_mask], w[va_mask]

                if self.use_hurdle:
                    # classifier
                    dtrain_clf = lgb.Dataset(X_tr, label=z_tr, weight=w_tr, free_raw_data=False)
                    dvalid_clf = lgb.Dataset(X_va, label=z_va, weight=w_va, reference=dtrain_clf, free_raw_data=False)
                    clf_params = dict(
                        objective="binary",
                        learning_rate=self.params.learning_rate,
                        num_leaves=self.params.num_leaves,
                        max_depth=self.params.max_depth,
                        min_data_in_leaf=self.params.min_data_in_leaf,
                        subsample=self.params.subsample,
                        colsample_bytree=self.params.colsample_bytree,
                        reg_alpha=self.params.reg_alpha,
                        reg_lambda=self.params.reg_lambda,
                        metric="binary_logloss",
                    )
                    callbacks = []
                    if self.params.early_stopping_rounds and self.params.early_stopping_rounds > 0:
                        callbacks.append(
                            lgb.early_stopping(stopping_rounds=self.params.early_stopping_rounds, verbose=False))
                    clf_booster = lgb.train(
                        params=clf_params,
                        train_set=dtrain_clf,
                        num_boost_round=self.params.n_estimators,
                        valid_sets=[dvalid_clf],
                        callbacks=callbacks,
                    )
                    self.models[h]["clf"].append(clf_booster)

                    # regressor on positive samples only
                    pos_tr = z_tr > 0
                    pos_va = z_va > 0
                    if pos_tr.sum() == 0 or pos_va.sum() == 0:
                        continue
                    dtrain_reg = lgb.Dataset(X_tr[pos_tr], label=y_tr_f[pos_tr], weight=w_tr[pos_tr], free_raw_data=False)
                    dvalid_reg = lgb.Dataset(X_va[pos_va], label=y_va_f[pos_va], weight=w_va[pos_va], reference=dtrain_reg, free_raw_data=False)

                    obj = self.params.objective
                    if cfg.use_asinh_target:
                        obj = "regression_l1"  # 'mae' 아님

                    reg_params = dict(
                        objective=obj,
                        learning_rate=self.params.learning_rate,
                        num_leaves=self.params.num_leaves,
                        max_depth=self.params.max_depth,
                        min_data_in_leaf=self.params.min_data_in_leaf,
                        subsample=self.params.subsample,
                        colsample_bytree=self.params.colsample_bytree,
                        reg_alpha=self.params.reg_alpha,
                        reg_lambda=self.params.reg_lambda,
                        metric=None,
                    )
                    if obj == "tweedie":
                        reg_params["tweedie_variance_power"] = self.params.tweedie_variance_power

                    callbacks_reg = []
                    if self.params.early_stopping_rounds and self.params.early_stopping_rounds > 0:
                        callbacks_reg.append(
                            lgb.early_stopping(stopping_rounds=self.params.early_stopping_rounds, verbose=False))

                    reg_booster = lgb.train(
                        params=reg_params,
                        train_set=dtrain_reg,
                        num_boost_round=self.params.n_estimators,
                        valid_sets=[dvalid_reg],
                        feval=lambda preds, ds: lgbm_weighted_smape(preds, ds, use_asinh_target=cfg.use_asinh_target),
                        callbacks=callbacks_reg,
                    )
                    self.models[h]["reg"].append(reg_booster)

                    # OOF prediction for this fold
                    prob = clf_booster.predict(X_va, num_iteration=clf_booster.best_iteration)
                    reg_pred = reg_booster.predict(X_va, num_iteration=reg_booster.best_iteration)
                    if cfg.use_asinh_target:
                        reg_pred = np.sinh(reg_pred)
                    reg_pred = np.clip(reg_pred, 0.0, None)
                    yhat = np.clip(prob * reg_pred, 0.0, None)
                    oof_df = dfh.loc[va_mask, ["series_id", "h"]].copy()
                    oof_df["y"] = y_va
                    oof_df["yhat"] = yhat
                    self.oof_records.extend(oof_df.to_dict("records"))
                else:
                    dtrain = lgb.Dataset(X_tr, label=y_tr_f, weight=w_tr, free_raw_data=False)
                    dvalid = lgb.Dataset(X_va, label=y_va_f, weight=w_va, reference=dtrain, free_raw_data=False)

                    obj = self.params.objective
                    if cfg.use_asinh_target:
                        obj = "regression_l1"  # 'mae' 아님

                    lgb_params = dict(
                        objective=obj,
                        learning_rate=self.params.learning_rate,
                        num_leaves=self.params.num_leaves,
                        max_depth=self.params.max_depth,
                        min_data_in_leaf=self.params.min_data_in_leaf,
                        subsample=self.params.subsample,
                        colsample_bytree=self.params.colsample_bytree,
                        reg_alpha=self.params.reg_alpha,
                        reg_lambda=self.params.reg_lambda,
                        metric=None,
                    )
                    if obj == "tweedie":
                        lgb_params["tweedie_variance_power"] = self.params.tweedie_variance_power

                    callbacks = []
                    if self.params.early_stopping_rounds and self.params.early_stopping_rounds > 0:
                        callbacks.append(
                            lgb.early_stopping(stopping_rounds=self.params.early_stopping_rounds, verbose=False))

                    booster = lgb.train(
                        params=lgb_params,
                        train_set=dtrain,
                        num_boost_round=self.params.n_estimators,
                        valid_sets=[dvalid],
                        feval=lambda preds, ds: lgbm_weighted_smape(preds, ds, use_asinh_target=cfg.use_asinh_target),
                        callbacks=callbacks,
                    )
                    self.models[h]["reg"].append(booster)

                    # OOF prediction for this fold
                    yhat = booster.predict(X_va, num_iteration=booster.best_iteration)
                    if cfg.use_asinh_target:
                        yhat = np.sinh(yhat)
                    yhat = np.clip(yhat, 0.0, None)
                    oof_df = dfh.loc[va_mask, ["series_id", "h"]].copy()
                    oof_df["y"] = y_va
                    oof_df["yhat"] = yhat
                    self.oof_records.extend(oof_df.to_dict("records"))

        self.save(os.path.join(self.model_dir, "lgbm_models.json"))

    def predict(self, df_eval: pd.DataFrame) -> pd.DataFrame:
        if not any(v["reg"] or v["clf"] for v in self.models.values()):
            raise RuntimeError("Models not trained/loaded.")
        dfe = df_eval.copy()
        feat_cols = [c for c in self.features if c in dfe.columns]
        feats = dfe[feat_cols].fillna(0).astype("float32").values

        preds = np.zeros(len(dfe), dtype="float32")
        for h in range(1, 8):
            mask = (dfe["h"] == h).values
            if not mask.any():
                continue
            Xh = feats[mask]
            reg_list = []
            for booster in self.models.get(h, {}).get("reg", []):
                yhat = booster.predict(Xh, num_iteration=booster.best_iteration)
                reg_list.append(yhat)
            if not reg_list:
                continue
            reg_mean = np.mean(np.stack(reg_list, axis=0), axis=0)
            if self.use_asinh_target:
                reg_mean = np.sinh(reg_mean)
            reg_mean = np.clip(reg_mean, 0.0, None)

            if self.use_hurdle:
                clf_list = []
                for booster in self.models.get(h, {}).get("clf", []):
                    ph = booster.predict(Xh, num_iteration=booster.best_iteration)
                    clf_list.append(ph)
                if clf_list:
                    prob_mean = np.mean(np.stack(clf_list, axis=0), axis=0)
                else:
                    prob_mean = 1.0
                preds[mask] = np.clip(prob_mean * reg_mean, 0.0, None)
            else:
                preds[mask] = reg_mean

        out = dfe[["series_id", "h"]].copy()
        out["yhat_lgbm"] = preds
        return out

    def get_oof(self) -> pd.DataFrame:
        return pd.DataFrame(self.oof_records)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index = {}
        for h, comps in self.models.items():
            index[str(h)] = {"reg": [], "clf": []}
            for i, booster in enumerate(comps.get("reg", [])):
                fpath = os.path.join(self.model_dir, f"lgbm_reg_h{h}_fold{i}.txt")
                booster.save_model(fpath, num_iteration=booster.best_iteration)
                index[str(h)]["reg"].append(os.path.basename(fpath))
            for i, booster in enumerate(comps.get("clf", [])):
                fpath = os.path.join(self.model_dir, f"lgbm_clf_h{h}_fold{i}.txt")
                booster.save_model(fpath, num_iteration=booster.best_iteration)
                index[str(h)]["clf"].append(os.path.basename(fpath))
        meta = {
            "params": self.model_params,
            "use_asinh_target": self.use_asinh_target,
            "use_hurdle": self.use_hurdle,
            "index": index,
            "features": self.features,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_params = meta["params"]
        self.use_asinh_target = bool(meta.get("use_asinh_target", False))
        self.use_hurdle = bool(meta.get("use_hurdle", False))
        self.features = list(meta.get("features", []))
        index = meta["index"]
        self.models = {h: {"reg": [], "clf": []} for h in range(1,8)}
        for h_str, comp in index.items():
            h = int(h_str)
            reg_files = comp.get("reg", [])
            clf_files = comp.get("clf", [])
            for fname in reg_files:
                self.models[h]["reg"].append(lgb.Booster(model_file=os.path.join(self.model_dir, fname)))
            for fname in clf_files:
                self.models[h]["clf"].append(lgb.Booster(model_file=os.path.join(self.model_dir, fname)))
