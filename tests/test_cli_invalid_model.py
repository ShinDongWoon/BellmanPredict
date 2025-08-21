import os
import shutil
import subprocess
import sys
from pathlib import Path


def _copy_repo(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "repo"
    shutil.copytree(
        repo_root,
        workdir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".git"),
    )
    return workdir


def test_train_unknown_model(tmp_path):
    workdir = _copy_repo(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(workdir)}
    proc = subprocess.run(
        [sys.executable, "LGHackerton/train.py", "--model", "unknown", "--skip-tune"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unknown model 'unknown'" in proc.stderr


def test_predict_unknown_model(tmp_path):
    workdir = _copy_repo(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(workdir)}
    proc = subprocess.run(
        [sys.executable, "LGHackerton/predict.py", "--model", "unknown"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unknown model 'unknown'" in proc.stderr

