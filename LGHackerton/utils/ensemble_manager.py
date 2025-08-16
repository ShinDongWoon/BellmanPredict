from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np


class EnsembleManager:
    """Utility for combining model predictions with learned weights."""

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.meta_model: Any = None
        self.cfg: Dict[str, Any] = {}
        self.fallback: str = "median"

    def fit(
        self,
        oof_lgb: np.ndarray,
        oof_patch: np.ndarray,
        y_true: np.ndarray,
        cfg: Dict[str, Any],
    ) -> "EnsembleManager":
        """Fit ensemble weights or meta model based on configuration.

        Parameters
        ----------
        oof_lgb, oof_patch : np.ndarray
            Out-of-fold predictions from the base models.
        y_true : np.ndarray
            Ground truth targets.
        cfg : dict
            Configuration containing at least ``type`` and optionally
            ``fallback`` and ``params``.
        """
        self.cfg = dict(cfg)
        self.fallback = cfg.get("fallback", "median")
        X = np.column_stack([oof_lgb, oof_patch])
        y = np.asarray(y_true)
        self.weights = None
        self.meta_model = None

        try:
            if cfg.get("type") == "nnls":
                from scipy.optimize import nnls

                w, _ = nnls(X, y)
                if w.sum() > 0:
                    self.weights = w / w.sum()
                else:
                    raise ValueError("Degenerate NNLS solution")
            elif cfg.get("type") == "lgbm":
                import lightgbm as lgb

                params = cfg.get("params", {})
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y)
                # store booster for easier saving/loading
                self.meta_model = model.booster_
            else:
                raise ValueError(f"Unknown ensemble type: {cfg.get('type')}")
        except Exception:
            # training failed; fall back later
            self.weights = None
            self.meta_model = None

        return self

    def predict(self, pred_lgb: np.ndarray, pred_patch: np.ndarray) -> np.ndarray:
        """Combine predictions using learned weights or meta model."""
        p1 = np.asarray(pred_lgb)
        p2 = np.asarray(pred_patch)
        if self.weights is not None:
            w1, w2 = self.weights
            return w1 * p1 + w2 * p2
        if self.meta_model is not None:
            X = np.column_stack([p1, p2])
            return self.meta_model.predict(X)
        if self.fallback == "median":
            return np.median(np.vstack([p1, p2]), axis=0)
        if self.fallback == "lgb":
            return p1
        if self.fallback == "patch":
            return p2
        return p1

    def save(self, path: str = "artifacts/ensemble_meta.json") -> None:
        """Save ensemble configuration and weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data: Dict[str, Any] = {
            "cfg": self.cfg,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "model_file": None,
        }
        if self.meta_model is not None:
            model_file = path.replace(".json", "_lgbm.txt")
            self.meta_model.save_model(model_file)
            data["model_file"] = model_file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str = "artifacts/ensemble_meta.json") -> "EnsembleManager":
        """Load ensemble configuration and weights if available."""
        if not os.path.exists(path):
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cfg = data.get("cfg", {})
        self.fallback = self.cfg.get("fallback", "median")
        w = data.get("weights")
        if w is not None:
            self.weights = np.asarray(w, dtype=float)
        model_file = data.get("model_file")
        if model_file and os.path.exists(model_file):
            import lightgbm as lgb

            self.meta_model = lgb.Booster(model_file=model_file)
        return self

