# Hyperparameter Tuning

This project provides Optuna-based tuners for supported models. Invoke a tuner
through the training script:

```bash
python LGHackerton/train.py --model patchtst --trials 20
```

Use `--force-tune` to ignore any cached studies and rerun the search from
scratch. Without this flag, existing results in `ARTIFACTS_DIR` are reused to
save time.

## Interruptions

If the search terminates without completing any trials, a user-friendly warning
is emitted and no `best_params.json` file is written. Successful runs save the
best configuration to `ARTIFACTS_DIR/<model_name>/best_params.json`.

## Exceptions

Tuners validate their inputs:

- Missing required fields raise `TypeError` with a message `Missing field: <name>`.
- Values outside allowed ranges raise `ValueError` formatted as
  `"<field> out of range: <value>"`.
- Requesting a tuner for an unsupported model prints `Unknown tuner for model`.

## Caching

Tuning results are cached under the artifacts directory. Use `--force-tune` to
recompute parameters even when cached results exist.

## Default Search Spaces

Each tuner exposes a dataclass describing the search space explored by Optuna.
The following tables list the default ranges or choices used during tuning.

### PatchTST

| Parameter | Range/Choices |
|-----------|---------------|
| `d_model` | 64, 128, 256 |
| `n_heads` | 4, 8 |
| `depth` | 2–6 |
| `patch_len`/`stride` | 8, 12, 14, 16, 24 |
| `dropout` | 0.0–0.5 |
| `lr` | 1e-4–1e-2 (log) |
| `weight_decay` | 1e-6–1e-3 (log) |
| `id_embed_dim` | 0, 16 |
| `batch_size` | 64, 128, 256 |
| `max_epochs` | 50–200 |
| `patience` | 5–30 |

### LightGBM

| Parameter | Range/Choices |
|-----------|---------------|
| `num_leaves` | 31–255 |
| `max_depth` | 3–16 |
| `learning_rate` | 1e-3–0.3 (log) |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_data_in_leaf` | 10–200 |
| `reg_alpha` | 1e-8–1.0 (log) |
| `reg_lambda` | 1e-8–1.0 (log) |
| `n_estimators` | 500–4000 |

### Temporal Fusion Transformer

| Parameter | Range/Choices |
|-----------|---------------|
| `hidden_size` | 64–512 |
| `lstm_layers` | 1–4 |
| `dropout` | 0.0–0.5 |
| `attention_heads` | 1–8 |
| `learning_rate` | 1e-4–1e-2 (log) |
| `batch_size` | 32–256 |
| `max_epochs` | 50–200 |
| `patience` | 5–30 |
