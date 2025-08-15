
from __future__ import annotations
import os, json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Iterable, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception as _e:
    TORCH_OK = False
    _TORCH_ERR = _e

from models.base_trainer import BaseModel, TrainConfig
from utils.metrics import smape, weighted_smape_np, PRIORITY_OUTLETS

@dataclass
class PatchTSTParams:
    d_model:int=128
    n_heads:int=8
    depth:int=4
    patch_len:int=4
    stride:int=1
    dropout:float=0.1
    lr:float=1e-3
    weight_decay:float=1e-4
    batch_size:int=256
    max_epochs:int=200
    patience:int=20

class _SeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, series_ids: Optional[Iterable[str]] = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        if series_ids is None:
            series_ids = ["" for _ in range(len(X))]
        self.sids = np.array(list(series_ids))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]
        mu = x.mean()
        std = x.std()
        if std == 0:
            std = 1.0
        x = (x - mu) / std
        return x, self.y[idx], self.sids[idx]

if TORCH_OK:
    class PatchTSTBlock(nn.Module):
        def __init__(self, d_model, n_heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
            )
            self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        def forward(self, x):
            a,_ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
            x = x + a
            x = x + self.ff(self.n2(x))
            return x

    class PatchTSTNet(nn.Module):
        def __init__(self, L:int, H:int, d_model:int, n_heads:int, depth:int, patch_len:int, stride:int, dropout:float):
            super().__init__()
            self.L, self.H = L, H
            self.unfold = nn.Unfold(kernel_size=(patch_len,1), stride=(stride,1))
            self.proj = nn.Linear(patch_len, d_model)
            self.blocks = nn.ModuleList([PatchTSTBlock(d_model, n_heads, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, H))
        def forward(self, x):
            B,L,C = x.shape
            x2 = x.view(B,1,L,1)
            p = self.unfold(x2).squeeze(1).transpose(1,2)  # (B, n_patches, patch_len)
            z = self.proj(p)
            for blk in self.blocks: z = blk(z)
            z = self.norm(z).mean(1)
            return self.head(z)

class PatchTSTTrainer(BaseModel):
    def __init__(self, params: PatchTSTParams, L:int, H:int, model_dir: str):
        super().__init__(model_params=asdict(params), model_dir=model_dir)
        self.params = params; self.L=L; self.H=H
        self.model=None; self.device="cpu"
    def _ensure_torch(self):
        if not TORCH_OK: raise RuntimeError(f"PyTorch not available: {_TORCH_ERR}")
    def train(self, X_train: np.ndarray, y_train: np.ndarray, series_ids: np.ndarray, label_dates: np.ndarray, cfg: TrainConfig) -> None:
        self._ensure_torch(); import torch
        os.makedirs(self.model_dir, exist_ok=True)
        order = np.argsort(label_dates)
        X_train = X_train[order]
        y_train = y_train[order]
        series_ids = np.array(series_ids)[order]
        label_dates = label_dates[order]
        n = len(label_dates)
        idx = np.arange(n)
        purge_win = self.L + self.H if cfg.purge_mode == "L+H" else self.L
        n_val = max(cfg.min_val_days, int(cfg.val_ratio * n))
        n_tr = max(1, n - n_val - purge_win)
        tr_mask = idx < n_tr
        va_mask = idx >= n_tr + purge_win
        assert tr_mask.sum() > 0 and va_mask.sum() > 0
        assert label_dates[tr_mask].max() < label_dates[va_mask].min() - np.timedelta64(purge_win, "D")
        tr_ds = _SeriesDataset(X_train[tr_mask], y_train[tr_mask], series_ids[tr_mask])
        va_ds = _SeriesDataset(X_train[va_mask], y_train[va_mask], series_ids[va_mask])
        tr_loader = DataLoader(tr_ds, batch_size=self.params.batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=self.params.batch_size, shuffle=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = PatchTSTNet(self.L,self.H,self.params.d_model,self.params.n_heads,self.params.depth,
                          self.params.patch_len,self.params.stride,self.params.dropout).to(self.device)
        self.model = net
        opt = torch.optim.AdamW(net.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        loss_fn = torch.nn.L1Loss()
        best=float("inf"); best_state=None; bad=0
        for ep in range(self.params.max_epochs):
            net.train()
            for xb,yb,sb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = net(xb)
                if cfg.use_weighted_loss:
                    outlets = [sid.split("::")[0] for sid in sb]
                    w = torch.tensor([cfg.priority_weight if o in PRIORITY_OUTLETS else 1.0 for o in outlets], device=self.device).view(-1,1)
                    loss = (torch.abs(pred - yb) * w).mean()
                else:
                    loss = loss_fn(pred, yb)
                loss.backward(); opt.step()
            # val
            net.eval(); P=[]; T=[]; S=[]
            with torch.no_grad():
                for xb,yb,sb in va_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = net(xb)
                    P.append(out.cpu().numpy()); T.append(yb.cpu().numpy()); S.extend(sb)
            y_pred = np.clip(np.concatenate(P,0),0,None); y_true = np.concatenate(T,0)
            outlets = [sid.split("::")[0] for sid in S]
            eps = 1e-8
            w_val = weighted_smape_np(y_true.ravel(), y_pred.ravel(), outlets, cfg.priority_weight, eps)
            s_val = smape(y_true.ravel(), y_pred.ravel(), eps)
            if w_val + 1e-8 < best:
                best = w_val; best_state = net.state_dict(); bad = 0
            else:
                bad += 1
            print(f"Epoch {ep}: wSMAPE={w_val:.4f} SMAPE={s_val:.4f}")
            if bad >= self.params.patience:
                break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.save(os.path.join(self.model_dir,"patchtst.pt"))
    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        self._ensure_torch(); import torch
        if self.model is None: raise RuntimeError("Model not loaded.")
        self.model.eval()
        ds = _SeriesDataset(X_eval, np.zeros((X_eval.shape[0], self.H), dtype=np.float32))
        loader = torch.utils.data.DataLoader(ds, batch_size=self.params.batch_size, shuffle=False)
        outs=[]
        with torch.no_grad():
            for xb,_,_ in loader:
                xb = xb.to(self.device); outs.append(self.model(xb).cpu().numpy())
        yhat = np.clip(np.concatenate(outs,0),0,None)
        return yhat
    def save(self, path:str)->None:
        if not TORCH_OK: return
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path.replace(".pt",".json"),"w",encoding="utf-8") as f:
            json.dump({"params":self.model_params,"L":self.L,"H":self.H},f,ensure_ascii=False,indent=2)
        if self.model is not None: torch.save(self.model.state_dict(), path)
    def load(self, path:str)->None:
        self._ensure_torch(); import torch, json, os
        meta=path.replace(".pt",".json")
        if os.path.exists(meta):
            with open(meta,"r",encoding="utf-8") as f:
                m=json.load(f); self.model_params=m.get("params",self.model_params); self.L=int(m.get("L",self.L)); self.H=int(m.get("H",self.H))
        from types import SimpleNamespace
        p=SimpleNamespace(**self.model_params)
        self.model = PatchTSTNet(self.L,self.H,getattr(p,"d_model",128),getattr(p,"n_heads",8),getattr(p,"depth",4),
                                 getattr(p,"patch_len",4),getattr(p,"stride",1),getattr(p,"dropout",0.1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load(path, map_location=self.device)); self.model.to(self.device)
