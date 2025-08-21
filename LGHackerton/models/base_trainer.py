
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class TrainConfig:
    seed:int=42
    n_folds:int=3
    cv_stride:int=7
    priority_weight:float=3.0
    use_weighted_loss:bool=False
    non_zero_weight:float=2.0
    use_asinh_target:bool=False
    use_hurdle:bool=False
    model_dir:str="./artifacts"
    # validation control
    val_policy:str="ratio"          # "ratio", "span", "rocv"
    val_ratio:float=0.2              # used when val_policy == "ratio"
    val_span_days:int=28             # used when val_policy == "span"
    rocv_n_folds:int=3               # rolling-origin CV folds
    rocv_stride_days:int=7           # step between ROCV folds
    rocv_val_span_days:int=7         # validation span for each ROCV fold
    purge_days:int=0                 # explicit purge gap (0 -> derive from purge_mode)
    min_val_samples:int=28           # minimum validation samples per fold
    purge_mode:str="L"               # fallback for legacy behaviour
    input_lens: List[int] | None = None

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

    @staticmethod
    @abstractmethod
    def build_eval_dataset(pp, df_eval_full):
        """Build evaluation dataset for prediction."""

    @abstractmethod
    def predict_df(self, eval_df):
        """Generate prediction dataframe from evaluation dataset."""
