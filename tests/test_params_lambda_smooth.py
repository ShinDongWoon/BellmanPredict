from LGHackerton.utils import params as params_mod
from LGHackerton.models.patchtst.trainer import PatchTSTParams


def test_load_best_params_includes_lambda_smooth(tmp_path, monkeypatch):
    monkeypatch.setattr(params_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(params_mod, "OPTUNA_DIR", tmp_path / "optuna")

    params_dict, input_len = params_mod.load_best_params("patchtst")

    assert input_len is None
    assert "lambda_smooth" in params_dict
    assert params_dict["lambda_smooth"] == 0.0

    params_obj = PatchTSTParams(**params_dict)
    assert params_obj.lambda_smooth == params_dict["lambda_smooth"]
