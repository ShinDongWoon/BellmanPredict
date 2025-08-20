import os
import subprocess
import sys
import shutil
from pathlib import Path

import pandas as pd


def test_patchtst_pipeline(tmp_path):
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

    pt_path = workdir / "LGHackerton" / "models" / "patchtst_trainer.py"
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
        "    import numpy as np\n"
        "    return np.zeros((len(X), self.H))\n"
        "PatchTSTTrainer.predict=_predict\n"
    )
    pt_path.write_text(stub)

    env = {**os.environ, "PYTHONPATH": str(workdir)}
    subprocess.run(
        [sys.executable, "LGHackerton/train.py", "--model", "patchtst", "--skip-tune"],
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
        [sys.executable, "LGHackerton/predict.py", "--model", "patchtst"],
        cwd=workdir,
        env=env,
        check=True,
        input="cpu\n",
        text=True,
    )

    eval_csv = artifacts_dir / "eval_patch.csv"
    sub_csv = artifacts_dir / "submission.csv"
    assert eval_csv.exists() and sub_csv.exists()

    eval_df = pd.read_csv(eval_csv)
    assert list(eval_df.columns) == [
        "series_id",
        "h",
        "yhat_patch",
        "yhat_ens",
        "test_id",
        "date",
    ]

    sub_df = pd.read_csv(sub_csv)
    sample_df = pd.read_csv(workdir / "LGHackerton" / "data" / "sample_submission.csv")
    assert list(sub_df.columns) == list(sample_df.columns)
    assert len(sub_df) == len(sample_df)
