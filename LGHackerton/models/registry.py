"""Simple registry for model trainers."""
from __future__ import annotations
from typing import Type, Dict


class ModelRegistry:
    """Registry mapping model names to trainer classes."""
    _REGISTRY: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, trainer_cls: Type) -> None:
        cls._REGISTRY[name] = trainer_cls

    @classmethod
    def get(cls, name: str):
        if name not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )
        return cls._REGISTRY[name]

    @classmethod
    def available(cls):
        return sorted(cls._REGISTRY)


# register known trainers
from LGHackerton.models.patchtst.trainer import PatchTSTTrainer
ModelRegistry.register("patchtst", PatchTSTTrainer)

try:  # optional registrations
    from LGHackerton.models.lgbm_trainer import LGBMTrainer
    ModelRegistry.register("lgbm", LGBMTrainer)
except Exception:  # pragma: no cover - optional
    pass

try:
    from LGHackerton.models.tft_trainer import TFTTrainer  # type: ignore
    ModelRegistry.register("tft", TFTTrainer)
except Exception:  # pragma: no cover - optional
    pass
