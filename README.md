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

## Combining Predictions

After generating predictions from different models, use the postprocessing
utilities to ensemble them and create a submission file:

```bash
python LGHackerton/predict.py --model patchtst --out patch.csv
python LGHackerton/predict.py --model lgbm --out lgbm.csv

python - <<'PY'
import pandas as pd
from LGHackerton.postprocess import aggregate_predictions, convert_to_submission

preds = [pd.read_csv('patch.csv'), pd.read_csv('lgbm.csv')]
ens = aggregate_predictions(preds, weights=[0.7, 0.3])
convert_to_submission(ens).to_csv('submission.csv', index=False, encoding='utf-8-sig')
PY
```

The helper functions `aggregate_predictions` and `convert_to_submission` ensure
consistent formatting and handle any missing or duplicate entries.

