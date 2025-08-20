# BellmanPredict

## Hurdle Model Combination

Throughout the project, binary classification probabilities and regression
forecasts are combined using probability multiplication. The final demand
estimate for each horizon is computed as ``p * \hat{y}``, where ``p`` is the
predicted probability of non-zero demand. This convention is implemented in
`LGBMTrainer`, the standalone LightGBM utilities, and the Optuna tuning
objective.

## Preprocessing

`SampleWindowizer.build_lgbm_train` now vectorizes target generation and
unpivoting, producing the same dataset as the previous row-wise
implementation but more efficiently.

## Baseline Forecasting

Run baseline models with the provided configuration:

```bash
python LGHackerton/train_baseline.py --config configs/baseline.yaml --model naive
```

## PatchTST Grid Search

Run a grid search over PatchTST settings:

```bash
python LGHackerton/tune.py --task patchtst_grid --config configs/patchtst.yaml
```

## PatchTST Hyperparameter Selection

`train.py` determines PatchTST settings using the following order:

1. If `artifacts/patchtst_search.csv` exists (created by the grid-search task),
   the combination with the lowest `val_wsmape` is chosen. The associated
   `input_len` is applied to window generation and to the model.
2. Otherwise, Optuna results from `artifacts/optuna/patchtst_best.json` are
   used when available.
3. If neither artifact is present, default parameters from `PATCH_PARAMS` are
   used.

This avoids confusion when both grid-search and Optuna artifacts may exist.

## LightGBM 튜닝 강화 로직

`LGHackerton/tune.py` 의 `objective_lgbm` 함수는 각 horizon 별로 데이터 품질을
검증한 후 학습을 진행하도록 개선되었습니다. 샘플 수가 100건 미만이거나
80% 이상이 상수 피처인 경우, 유효 피처가 5개 미만인 경우, 날짜 범위가
10일 미만인 경우에는 해당 horizon 을 건너뜁니다. 양성 비율이 1% 미만일 때도
학습을 수행하지 않습니다. 또한 분산이 낮은 피처는 제거되며 양성 샘플에는
가중치 2.0 이 적용됩니다. 이러한 검증 로직은 불필요한 경고를 줄이고,
하이퍼파라미터 탐색의 안정성을 높이기 위해 도입되었습니다.

## Skipping LightGBM training

Pass ``--skip-lgbm`` to ``train.py`` to bypass LightGBM tuning, training and diagnostics:

```bash
python LGHackerton/train.py --skip-lgbm
```

