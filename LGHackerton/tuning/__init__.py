"""Hyperparameter tuning utilities and registry."""

from .registry import TunerRegistry
from .patchtst import PatchTSTTuner

TunerRegistry.register("patchtst", PatchTSTTuner)

try:  # optional tuners
    from .lgbm import LGBMTuner
    TunerRegistry.register("lgbm", LGBMTuner)
except Exception:  # pragma: no cover - optional
    LGBMTuner = None  # type: ignore

try:
    from .tft import TFTTuner  # type: ignore
    TunerRegistry.register("tft", TFTTuner)
except Exception:  # pragma: no cover - optional
    TFTTuner = None  # type: ignore

__all__ = ["TunerRegistry", "PatchTSTTuner", "LGBMTuner", "TFTTuner"]
