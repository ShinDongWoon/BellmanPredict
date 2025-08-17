# BellmanPredict

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
