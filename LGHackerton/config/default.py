# config/default.py
from __future__ import annotations
from pathlib import Path

# 프로젝트 루트 = 이 파일(config/)의 한 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 데이터 파일 위치(배포시 함께 넣는 상대 경로)
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = str((DATA_DIR / "train.csv").resolve())
TEST_DIR = DATA_DIR
TEST_GLOB = str((TEST_DIR / "TEST_*.csv").resolve())
# EVAL_PATH  = str((DATA_DIR / "TEST_00.csv").resolve())
SAMPLE_SUB_PATH = str((DATA_DIR / "sample_submission.csv").resolve())

# 산출물 저장 루트(자동 생성)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Optuna 결과 저장 디렉터리
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

# 모델/피처 산출물 경로
MODEL_DIR       = str((ARTIFACTS_DIR / "models").resolve())
ARTIFACTS_PATH  = str((ARTIFACTS_DIR / "preprocess_artifacts.pkl").resolve())
SUBMISSION_OUT  = str((ARTIFACTS_DIR / "submission.csv").resolve())
# PatchTST는 prefix를 요구하므로 확장자 없이 파일명만 제공
PATCH_EVAL_OUT  = str((ARTIFACTS_DIR / "patch_eval").resolve())
# PatchTST 예측 결과 저장 경로
PATCH_PRED_OUT  = str((ARTIFACTS_DIR / "eval_patch.csv").resolve())
OOF_PATCH_OUT = str((ARTIFACTS_DIR / "oof_patch.csv").resolve())

# preprocessing options
SHOW_PROGRESS = True

# 하이퍼파라미터(필요 시 그대로 유지)
PATCH_PARAMS = dict(
    d_model=128, n_heads=8, depth=4,
    patch_len=4, stride=1, dropout=0.1,
    id_embed_dim=16,
    lr=1e-3, weight_decay=1e-4, batch_size=256,
    max_epochs=200, patience=20,
    num_workers=0,
)
TRAIN_CFG = dict(
    seed=42, n_folds=3, cv_stride=7,
    priority_weight=3.0,
    use_weighted_loss=True,
    model_dir=MODEL_DIR,
    val_policy="ratio",
    val_ratio=0.2,
    val_span_days=28,
    rocv_n_folds=3,
    rocv_stride_days=7,
    rocv_val_span_days=7,
    purge_days=28,
    min_val_samples=28,
    purge_mode="L",
)
