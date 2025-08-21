# BellmanPredict

## Hurdle Model Combination

Binary classification probabilities and regression forecasts are blended using
the conditional mean of a zero-truncated distribution. Given a classifier
probability ``p`` and an unconditional mean forecast ``\mu_u``, the zero
probability is ``P0 = (\kappa / (\kappa + \mu_u))^\kappa``. The conditional
mean ``\mu_c`` is then ``\mu_u / (1 - P0)`` and the final prediction becomes::

    y_hat = ((1 - \epsilon_{leaky}) * p + \epsilon_{leaky}) * \mu_c

The shape parameter ``kappa`` and the leakage term ``epsilon_leaky`` can be
configured via ``PATCH_PARAMS`` and control the zero-probability assumption and
stability of the combination respectively.

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

## Model and Tuner Registry

Training scripts resolve trainer classes and Optuna-based tuners through
lightweight registries. Providing an unknown name raises a `ValueError` listing
available options. See [docs/registry_tuner.md](docs/registry_tuner.md) for full
usage and error-handling rules.

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

## Documentation

- [Callbacks](docs/callbacks.md)
- [Hyperparameter Tuning](docs/tuning.md)
- [Model & Tuner Registries](docs/registry_tuner.md)

