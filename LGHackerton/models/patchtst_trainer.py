from __future__ import annotations
import os, json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Iterable, Optional, List, Any

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
    # validation settings (mirrors TrainConfig)
    val_policy:str="ratio"
    val_ratio:float=0.2
    val_span_days:int=28
    rocv_n_folds:int=3
    rocv_stride_days:int=7
    rocv_val_span_days:int=7
    purge_days:int=0
    min_val_samples:int=28

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
        mu = x.mean(); std = x.std()
        if std == 0: std = 1.0
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
            p = self.unfold(x2).squeeze(1).transpose(1,2)
            z = self.proj(p)
            for blk in self.blocks: z = blk(z)
            z = self.norm(z).mean(1)
            return self.head(z)

class PatchTSTTrainer(BaseModel):
    def __init__(self, params: PatchTSTParams, L:int, H:int, model_dir: str):
        super().__init__(model_params=asdict(params), model_dir=model_dir)
        self.params = params; self.L=L; self.H=H
        self.models: List[Any] = []
        self.device="cpu"
    def _ensure_torch(self):
        if not TORCH_OK: raise RuntimeError(f"PyTorch not available: {_TORCH_ERR}")
    def train(self, X_train: np.ndarray, y_train: np.ndarray, series_ids: np.ndarray, label_dates: np.ndarray, cfg: TrainConfig) -> None:
        """Train PatchTST models under various validation policies."""
        self._ensure_torch(); import torch
        os.makedirs(self.model_dir, exist_ok=True)
        order = np.argsort(label_dates)
        X_train = X_train[order]; y_train = y_train[order]
        series_ids = np.array(series_ids)[order]
        label_dates = label_dates[order]
        n = len(label_dates)
        min_samples = max(cfg.min_val_samples, 5 * self.params.batch_size)
        purge_days = cfg.purge_days if cfg.purge_days > 0 else (self.L + self.H if cfg.purge_mode == "L+H" else self.L)
        purge = np.timedelta64(purge_days, 'D')
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        def _make_rocv_slices(n_folds:int, stride:int, span:int):
            slices=[]; used = np.zeros(n, dtype=bool)
            dates = np.sort(np.unique(label_dates))
            for i in range(n_folds):
                end = dates[-1] - np.timedelta64(i*stride, 'D')
                start = end - np.timedelta64(span-1, 'D')
                va_mask = (label_dates >= start) & (label_dates <= end) & (~used)
                tr_mask = label_dates < (start - purge)
                if va_mask.sum() == 0 or tr_mask.sum() == 0:
                    continue
                assert label_dates[tr_mask].max() < label_dates[va_mask].min() - purge
                slices.append((tr_mask, va_mask))
                used |= va_mask
            return slices
        if cfg.val_policy == "ratio":
            n_val = max(min_samples, int(cfg.val_ratio * n))
            n_val = min(n_val, n-1)
            cutoff_date = label_dates[-n_val]
            va_mask = label_dates >= cutoff_date
            tr_mask = label_dates < (cutoff_date - purge)
            if tr_mask.sum() > 0 and va_mask.sum() >= min_samples:
                folds = [(tr_mask, va_mask)]
        elif cfg.val_policy == "span":
            end = label_dates[-1]
            start = end - np.timedelta64(cfg.val_span_days-1, 'D')
            va_mask = label_dates >= start
            if va_mask.sum() < min_samples:
                start = label_dates[-min_samples]
                va_mask = label_dates >= start
            tr_mask = label_dates < (start - purge)
            if tr_mask.sum() > 0 and va_mask.sum() >= min_samples:
                folds = [(tr_mask, va_mask)]
        elif cfg.val_policy == "rocv":
            span = cfg.rocv_val_span_days
            n_folds = cfg.rocv_n_folds
            while True:
                folds = _make_rocv_slices(n_folds, cfg.rocv_stride_days, span)
                if folds and min(f.sum() for _,f in folds) >= min_samples:
                    break
                if span > 1:
                    span = max(1, span // 2)
                    continue
                if n_folds > 1:
                    n_folds -= 1
                    continue
                folds = _make_rocv_slices(1, cfg.rocv_stride_days, span)
                break
        else:
            raise ValueError(f"Unknown val_policy: {cfg.val_policy}")
        assert folds, "No valid folds generated"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for i,(tr_mask,va_mask) in enumerate(folds):
            tr_ds = _SeriesDataset(X_train[tr_mask], y_train[tr_mask], series_ids[tr_mask])
            va_ds = _SeriesDataset(X_train[va_mask], y_train[va_mask], series_ids[va_mask])
            tr_loader = DataLoader(tr_ds, batch_size=self.params.batch_size, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=self.params.batch_size, shuffle=False)
            net = PatchTSTNet(self.L,self.H,self.params.d_model,self.params.n_heads,self.params.depth,
                              self.params.patch_len,self.params.stride,self.params.dropout).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
            loss_fn = torch.nn.L1Loss()
            best=float("inf"); best_state=None; bad=0
            for ep in range(self.params.max_epochs):
                net.train()
                for xb,yb,sb in tr_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad(); pred = net(xb)
                    if cfg.use_weighted_loss:
                        outlets = [sid.split("::")[0] for sid in sb]
                        w = torch.tensor([cfg.priority_weight if o in PRIORITY_OUTLETS else 1.0 for o in outlets], device=self.device).view(-1,1)
                        loss = (torch.abs(pred - yb) * w).mean()
                    else:
                        loss = loss_fn(pred, yb)
                    loss.backward(); opt.step()
                net.eval(); P=[]; T=[]; S=[]
                with torch.no_grad():
                    for xb,yb,sb in va_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        out = net(xb)
                        P.append(out.cpu().numpy()); T.append(yb.cpu().numpy()); S.extend(sb)
                y_pred = np.clip(np.concatenate(P,0),0,None)
                y_true = np.concatenate(T,0)
                series_ids_val = np.array(S)
                outlets = [sid.split("::")[0] for sid in series_ids_val]
                eps = 1e-8
                w_val = weighted_smape_np(
                    y_true.ravel(),
                    y_pred.ravel(),
                    np.repeat(outlets, self.H),
                    cfg.priority_weight,
                    eps,
                )
                s_val = smape(y_true.ravel(), y_pred.ravel(), eps)
                mae_val = float(np.mean(np.abs(y_true - y_pred)))
                if w_val + 1e-8 < best:
                    best = w_val; best_state = net.state_dict(); bad = 0
                else:
                    bad += 1
                print(
                    f"Fold {i} Epoch {ep}: wSMAPE={w_val:.4f} SMAPE={s_val:.4f} MAE={mae_val:.4f}"
                )
                if bad >= self.params.patience:
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            self.models.append(net)
        self.save(os.path.join(self.model_dir,"patchtst.pt"))
    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        self._ensure_torch(); import torch
        if not self.models:
            raise RuntimeError("Model not loaded.")
        ds = _SeriesDataset(X_eval, np.zeros((X_eval.shape[0], self.H), dtype=np.float32))
        loader = torch.utils.data.DataLoader(ds, batch_size=self.params.batch_size, shuffle=False)
        outs=[]
        with torch.no_grad():
            for xb,_,_ in loader:
                xb = xb.to(self.device)
                preds = [m(xb).cpu().numpy() for m in self.models]
                outs.append(np.mean(preds, axis=0))
        yhat = np.clip(np.concatenate(outs,0),0,None)
        return yhat
    def save(self, path:str)->None:
        if not TORCH_OK: return
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index=[]
        for i, m in enumerate(self.models):
            fpath = os.path.join(self.model_dir, f"patchtst_fold{i}.pt")
            torch.save(m.state_dict(), fpath)
            index.append(os.path.basename(fpath))
        with open(path.replace(".pt",".json"),"w",encoding="utf-8") as f:
            json.dump({"params":self.model_params,"L":self.L,"H":self.H,"index":index},f,ensure_ascii=False,indent=2)
    def load(self, path:str)->None:
        self._ensure_torch(); import torch, json, os
        meta=path.replace(".pt",".json")
        if os.path.exists(meta):
            with open(meta,"r",encoding="utf-8") as f:
                m=json.load(f); self.model_params=m.get("params",self.model_params); self.L=int(m.get("L",self.L)); self.H=int(m.get("H",self.H)); index=m.get("index",[])
        else:
            index=[]
        self.params = PatchTSTParams(**self.model_params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models=[]
        for fname in index:
            net = PatchTSTNet(self.L,self.H,self.params.d_model,self.params.n_heads,self.params.depth,
                               self.params.patch_len,self.params.stride,self.params.dropout)
            net.load_state_dict(torch.load(os.path.join(self.model_dir,fname), map_location=self.device))
            net.to(self.device)
            self.models.append(net)
