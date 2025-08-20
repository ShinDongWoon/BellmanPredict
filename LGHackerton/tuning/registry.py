from __future__ import annotations

"""Registry for hyperparameter tuner classes."""

from typing import Dict, Type

from LGHackerton.tuning.base import HyperparameterTuner


class TunerRegistry:
    """Simple mapping from model name to its tuner class."""

    _REGISTRY: Dict[str, Type[HyperparameterTuner]] = {}

    @classmethod
    def register(cls, name: str, tuner_cls: Type[HyperparameterTuner]) -> None:
        """Register a tuner class under a given name."""
        cls._REGISTRY[name] = tuner_cls

    @classmethod
    def get(cls, name: str) -> Type[HyperparameterTuner]:
        """Retrieve the tuner class for ``name``.

        Parameters
        ----------
        name : str
            Name of the model/tuner.

        Returns
        -------
        Type[HyperparameterTuner]
            The registered tuner class.

        Raises
        ------
        ValueError
            If ``name`` has not been registered.
        """
        try:
            return cls._REGISTRY[name]
        except KeyError as e:  # pragma: no cover - user facing
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unknown tuner '{name}'. Available tuners: {available}"
            ) from e
