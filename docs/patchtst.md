# PatchTST Model

PatchTST consumes tensors shaped `(B, L, C)` where `B` is the batch size, `L` the input length and `C` the number of channels. Each channel is unfolded into patches of length `patch_len` and passed through a linear layer `nn.Linear(patch_len, d_model)` so patches are projected independently before fusion.

## Channel fusion

The `channel_fusion` hyperparameter controls how channel embeddings are merged:

- `attention` – a learned query attends across channels.
- `linear` – a linear layer produces softmax weights for each channel.
- `mean` – simple arithmetic mean over channel embeddings.

## Late fusion example

```python
import torch
from LGHackerton.models.patchtst.trainer import PatchTSTTrainer

net = PatchTSTTrainer.PatchTSTNet(
    L=96, H=24, d_model=128, n_heads=8, depth=2,
    patch_len=16, stride=8, dropout=0.2,
    channel_fusion="attention",
)

x = torch.randn(4, 96, 5)  # (B, L, C)
logits, mu_raw, kappa_raw = net(x)
prob = torch.sigmoid(logits)
```

Each channel is patched and projected separately; the resulting embeddings are fused
late using the configured strategy before producing classification and regression outputs.
