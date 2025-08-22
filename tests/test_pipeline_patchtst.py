import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_pipeline_patchtst(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "repo"
    shutil.copytree(
        repo_root,
        workdir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".git"),
    )

    cfg_path = workdir / "LGHackerton" / "config" / "default.py"
    with cfg_path.open("a", encoding="utf-8") as f:
        f.write(
            "\nPATCH_PARAMS['max_epochs']=1\nPATCH_PARAMS['patience']=0\nTRAIN_CFG['n_folds']=1\n"
        )

    pt_path = workdir / "LGHackerton" / "models" / "patchtst" / "trainer.py"
    orig_pt = pt_path.read_text()
    stub = orig_pt + (
        "\n"
        "def _train(self, X_train, y_train, series_ids, label_dates, cfg, preprocessors=None):\n"
        "    import os\n"
        "    os.makedirs(self.model_dir, exist_ok=True)\n"
        "    open(os.path.join(self.model_dir, 'patchtst.pt'), 'wb').close()\n"
        "    self.oof_records = [{'series_id': 'A::0', 'y': 0.0, 'yhat': 0.0}]\n"
        "PatchTSTTrainer.train=_train\n"
        "\n"
        "def _get_oof(self):\n"
        "    import pandas as pd\n"
        "    return pd.DataFrame(self.oof_records)\n"
        "PatchTSTTrainer.get_oof=_get_oof\n"
        "\n"
        "PatchTSTTrainer.load=lambda self, path: None\n"
        "\n"
        "def _predict(self, X, sid_idx):\n"
        "    k = 1.0\n"
        "    eps = self.params.epsilon_leaky\n"
        "    import numpy as np\n"
        "    mu = np.full((len(X), self.H), 2.0, dtype=float)\n"
        "    prob = np.full((len(X), self.H), 0.5, dtype=float)\n"
        "    p0 = np.power(k / (k + mu), k)\n"
        "    cond_mean = mu / np.maximum(1.0 - p0, 1e-6)\n"
        "    return ((1 - eps) * prob + eps) * cond_mean\n"
        "PatchTSTTrainer.predict=_predict\n"
    )
    pt_path.write_text(stub)

    conv_path = workdir / "LGHackerton" / "postprocess" / "convert.py"
    conv_path.write_text(
        "import pandas as pd\n"
        "from LGHackerton.config.default import SAMPLE_SUB_PATH\n\n"
        "def aggregate_predictions(pred_dfs, weights=None, how='mean'):\n"
        "    df = pred_dfs[0][['series_id','yhat_patch']].copy()\n"
        "    df.rename(columns={'series_id':'id','yhat_patch':'y'}, inplace=True)\n"
        "    return df\n\n"
        "def convert_to_submission(pred_df, weights=None, how='mean'):\n"
        "    sample_cols = pd.read_csv(SAMPLE_SUB_PATH, nrows=0).columns\n"
        "    return pd.DataFrame([[0]*len(sample_cols)], columns=sample_cols)\n"
    )

    env = {**os.environ, "PYTHONPATH": str(workdir)}
    subprocess.run(
        [sys.executable, "LGHackerton/train.py", "--skip-tune"],
        cwd=workdir,
        env=env,
        check=True,
        input="cpu\n",
        text=True,
    )

    artifacts_dir = workdir / "LGHackerton" / "artifacts"
    model_path = artifacts_dir / "models" / "patchtst.pt"
    assert model_path.exists()
    assert (artifacts_dir / "preprocess_artifacts.pkl").exists()

    subprocess.run(
        [sys.executable, "LGHackerton/predict.py"],
        cwd=workdir,
        env=env,
        check=True,
        input="cpu\n",
        text=True,
    )

    sub_csv = artifacts_dir / "submission.csv"
    assert sub_csv.exists()

    sub_df = pd.read_csv(sub_csv)
    sample_cols = pd.read_csv(
        workdir / "LGHackerton" / "data" / "sample_submission.csv", nrows=0
    ).columns
    assert list(sub_df.columns) == list(sample_cols)

    pred_csv = artifacts_dir / "eval_pred.csv"
    assert pred_csv.exists()
    pred_df = pd.read_csv(pred_csv)
    kappa = 1.0
    eps = 0.0
    mu = 2.0
    prob = 0.5
    p0 = (kappa / (kappa + mu)) ** kappa
    cond_mean = mu / (1 - p0)
    expected = ((1 - eps) * prob + eps) * cond_mean
    assert pred_df["yhat_patch"].iloc[0] == pytest.approx(expected)
