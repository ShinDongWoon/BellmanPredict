
from __future__ import annotations
import os
from typing import Dict, Any

TRAIN_PATH = r"/Users/castorp/Downloads/open (1)/train/train.csv"
EVAL_PATH = r"/Users/castorp/Downloads/open (1)/test/TEST_00.csv"
ARTIFACTS_PATH = r"/Users/castorp/Downloads/open (1)/artifacts.pkl"
LGBM_EVAL_OUT = r"/Users/castorp/Downloads/open (1)/eval_lgbm.csv"
PATCH_EVAL_OUT = r"/Users/castorp/Downloads/open (1)"

MODEL_DIR = os.path.join(os.path.dirname(PATCH_EVAL_OUT), "models")

LGBM_PARAMS: Dict[str, Any] = dict(
    objective="tweedie",
    tweedie_variance_power=1.3,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_estimators=3000,
    early_stopping_rounds=200,
)

PATCH_PARAMS: Dict[str, Any] = dict(
    d_model=128,
    n_heads=8,
    depth=4,
    patch_len=4,
    stride=1,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=200,
    patience=20,
)

TRAIN_CFG: Dict[str, Any] = dict(
    seed=42,
    n_folds=3,
    cv_stride=7,
    priority_weight=3.0,
    use_asinh_target=False,
    model_dir=MODEL_DIR,
)
