# config/default.py
from __future__ import annotations
from pathlib import Path

# 프로젝트 루트 = 이 파일(config/)의 한 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 데이터 파일 위치(배포시 함께 넣는 상대 경로)
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = str((DATA_DIR / "train.csv").resolve())
EVAL_PATH  = str((DATA_DIR / "TEST_00.csv").resolve())

# 산출물 저장 루트(자동 생성)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 모델/피처 산출물 경로
MODEL_DIR       = str((ARTIFACTS_DIR / "models").resolve())
ARTIFACTS_PATH  = str((ARTIFACTS_DIR / "preprocess_artifacts.pkl").resolve())
LGBM_EVAL_OUT   = str((ARTIFACTS_DIR / "eval_lgbm.csv").resolve())
# PatchTST는 prefix를 요구하므로 확장자 없이 파일명만 제공
PATCH_EVAL_OUT  = str((ARTIFACTS_DIR / "patch_eval").resolve())

# 하이퍼파라미터(필요 시 그대로 유지)
LGBM_PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=1.3,
    num_leaves=63, max_depth=-1, min_data_in_leaf=50,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    n_estimators=3000, early_stopping_rounds=200,
)
PATCH_PARAMS = dict(
    d_model=128, n_heads=8, depth=4,
    patch_len=4, stride=1, dropout=0.1,
    lr=1e-3, weight_decay=1e-4, batch_size=256,
    max_epochs=200, patience=20,
)
TRAIN_CFG = dict(
    seed=42, n_folds=3, cv_stride=7,
    priority_weight=3.0,
    use_asinh_target=False,
    model_dir=MODEL_DIR,
)
