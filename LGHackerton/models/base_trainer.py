
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TrainConfig:
    seed:int=42
    n_folds:int=3
    cv_stride:int=7
    priority_weight:float=3.0
    use_weighted_loss:bool=False
    use_asinh_target:bool=False
    model_dir:str="./artifacts"
    val_ratio:float=0.2
    min_val_days:int=28
    purge_mode:str="L"

class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any], model_dir: str):
        self.model_params = model_params
        self.model_dir = model_dir

    @abstractmethod
    def train(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def predict(self, *args, **kwargs): ...

    @abstractmethod
    def save(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def load(self, *args, **kwargs) -> None: ...
