import subprocess
import sys
import shutil
from pathlib import Path


def test_patchtst_pipeline(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "repo"
    shutil.copytree(repo_root, workdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".git"))

    cfg_path = workdir / "LGHackerton" / "config" / "default.py"
    with cfg_path.open("a", encoding="utf-8") as f:
        f.write("\nPATCH_PARAMS['max_epochs']=1\nPATCH_PARAMS['patience']=0\nTRAIN_CFG['n_folds']=1\n")
    pt_path = workdir / "LGHackerton" / "models" / "patchtst_trainer.py"
    orig_pt = pt_path.read_text()
    with pt_path.open("a", encoding="utf-8") as f:
        f.write("\nTORCH_OK = False\n")
    subprocess.run(
        [sys.executable, "-m", "LGHackerton.train", "--skip-tune"],
        cwd=workdir,
        check=True,
        input="cpu\n",
        text=True,
    )
    stub = orig_pt + (
        "\nPatchTSTTrainer.load=lambda self, path: None\n"
        "def _pred(self, X, sid_idx):\n"
        "    import numpy as np\n    return np.zeros((len(X), self.H))\n"
        "PatchTSTTrainer.predict=_pred\n"
    )
    pt_path.write_text(stub)

    artifacts_dir = workdir / "LGHackerton" / "artifacts"
    assert (artifacts_dir / "preprocess_artifacts.pkl").exists()

    subprocess.run(
        [sys.executable, "-m", "LGHackerton.predict"],
        cwd=workdir,
        check=True,
        input="cpu\n",
        text=True,
    )
    assert (artifacts_dir / "eval_patch.csv").exists()
    assert (artifacts_dir / "submission.csv").exists()





