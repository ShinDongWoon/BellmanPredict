# ROCV Callbacks

`PatchTSTTrainer` provides a callback hook that fires after rolling-origin
cross-validation (ROCV) slices are generated and before training begins.
Callbacks receive `(seed, fold_idx, train_mask, val_mask, cfg)` and run once
for every fold.

```python
from LGHackerton.models.patchtst_trainer import PatchTSTTrainer

def log_fold(seed, fold_idx, tr_mask, va_mask, cfg):
    print(f"fold {fold_idx} -> {tr_mask.sum()} train / {va_mask.sum()} val")

PatchTSTTrainer.register_rocv_callback(log_fold)
```

Failures inside the callback are isolated from training: the trainer converts
exceptions into warnings so that logging issues do not interrupt model fitting.

The public callback API intentionally avoids exposing private helpers.
Older versions without ``register_rocv_callback`` required wrapping the
private ``_make_rocv_slices`` function; ``train.py`` still supports this
as a legacy fallback.
