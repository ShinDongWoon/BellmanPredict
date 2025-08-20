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
