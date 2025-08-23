import pandas as pd
import pytest

from LGHackerton.preprocess import Preprocessor
from LGHackerton.models.base_trainer import TrainConfig
from LGHackerton.tuning.patchtst import PatchTSTTuner
from LGHackerton.models.patchtst.trainer import PatchTSTParams
from LGHackerton.tuning.lgbm import LGBMTuner
from LGHackerton.models.lgbm_trainer import LGBMParams
from LGHackerton.tuning.tft import TFTTuner


@pytest.fixture
def dummy_ctx():
    pp = Preprocessor()
    df = pd.DataFrame()
    cfg = TrainConfig()
    return pp, df, cfg


def test_patchtst_validate_params(dummy_ctx):
    pp, df, cfg = dummy_ctx
    tuner = PatchTSTTuner(pp, df, cfg)
    params = PatchTSTParams()
    params_dict = params.__dict__.copy()
    # Ensure parameters fall within the search space for fields not under test
    s = tuner.search_space
    stride_val = s.stride[0] if isinstance(s.stride, tuple) else s.stride
    params_dict.update(
        {
            "d_model": s.d_model[0],
            "n_heads": s.n_heads[0],
            "patch_len": s.patch_len[0],
            "stride": stride_val,
            "id_embed_dim": s.id_embed_dim[0],
            "batch_size": s.batch_size[0],
        }
    )
    params_missing = params_dict.copy()
    params_missing.pop("patch_len")
    with pytest.raises(TypeError, match="Missing field: patch_len"):
        tuner.validate_params(params_missing)
    params_bad = params_dict.copy()
    params_bad["patch_len"] = 1
    params_bad["stride"] = 1
    with pytest.raises(ValueError, match="patch_len out of range: 1"):
        tuner.validate_params(params_bad)


def test_lgbm_validate_params(dummy_ctx):
    pp, df, cfg = dummy_ctx
    tuner = LGBMTuner(pp, df, cfg)
    params = LGBMParams()
    params_dict = params.__dict__.copy()
    params_missing = params_dict.copy()
    params_missing.pop("num_leaves")
    with pytest.raises(TypeError, match="Missing field: num_leaves"):
        tuner.validate_params(params_missing)
    params_bad = params_dict.copy()
    params_bad["num_leaves"] = 0
    with pytest.raises(ValueError, match="num_leaves out of range: 0"):
        tuner.validate_params(params_bad)


def test_tft_validate_params(dummy_ctx):
    pp, df, cfg = dummy_ctx
    tuner = TFTTuner(pp, df, cfg)
    params = {
        "hidden_size": 64,
        "lstm_layers": 1,
        "dropout": 0.1,
        "attention_heads": 1,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "max_epochs": 50,
        "patience": 5,
    }
    params_missing = params.copy()
    params_missing.pop("hidden_size")
    with pytest.raises(TypeError, match="Missing field: hidden_size"):
        tuner.validate_params(params_missing)
    params_bad = params.copy()
    params_bad["hidden_size"] = 0
    with pytest.raises(ValueError, match="hidden_size out of range: 0"):
        tuner.validate_params(params_bad)

