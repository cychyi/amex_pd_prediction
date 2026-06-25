"""
STAGE 5B - MODELLING: DEEP-LEARNING MODELS
============================================================================
Trains the sequential models on the (N, 13, F) tensor from Stage 3:
  * LSTM                (slide)
  * GRU                 (added per request - cheaper recurrent baseline)
  * Transformer Encoder (slide)

Device is auto-selected: CUDA (Kaggle GPU) > MPS (Mac Mini M4) > CPU.
Training uses class-weighted BCE (the data is ~26% bad), early stopping on
validation AUC, and the SAME metrics module as the ML models for a fair
comparison. Per-customer predicted PDs are saved for the Stage 5C ensembles.

Run:
    python src/05b_modelling_dl.py
Outputs:
    data/models/{lstm,gru,transformer}.pt
    data/features/preds_dl.parquet
    data/reports/metrics_dl.csv
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths
bootstrap_paths()
import config as C
from metrics import evaluate, metrics_row

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():           # Apple Silicon (M4)
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class RecurrentClassifier(nn.Module):
    """Shared body for LSTM and GRU; `cell` picks the recurrent unit."""
    def __init__(self, n_feat, cell="LSTM", hidden=128, layers=2, dropout=0.3):
        super().__init__()
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(n_feat, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask):
        out, _ = self.rnn(x)                       # (B, L, H)
        lengths = mask.sum(1).long().clamp(min=1)  # last real timestep index+1
        last = out[torch.arange(out.size(0)), lengths - 1]   # gather last real step
        return self.head(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, hidden=128, layers=2, heads=8, ff=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = PositionalEncoding(hidden, max_len=C.MAX_SEQ_LEN + 1)
        enc = nn.TransformerEncoderLayer(hidden, heads, ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask):
        h = self.pos(self.proj(x))
        pad = mask == 0                            # True where padded -> ignored
        h = self.encoder(h, src_key_padding_mask=pad)
        # masked mean over real timesteps
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training / prediction loop
# ---------------------------------------------------------------------------
def make_loaders(X, M, y, tr_idx, va_idx, batch):
    def loader(idx, shuffle):
        ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(M[idx]),
                           torch.tensor(y[idx], dtype=torch.float32))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)
    return loader(tr_idx, True), loader(va_idx, False)


def predict(model, X, M, device, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            mb = torch.tensor(M[i:i+batch]).to(device)
            out.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
    return np.concatenate(out)


def train_model(name, model, X, M, y, tr_idx, va_idx, device):
    p = C.DL_PARAMS
    model = model.to(device)
    pos_weight = torch.tensor([(y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    tr_loader, va_loader = make_loaders(X, M, y, tr_idx, va_idx, p["batch_size"])

    best_auc, best_state, wait = -1.0, None, 0
    for epoch in range(p["epochs"]):
        model.train(); t0 = time.time()
        for xb, mb, yb in tr_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, mb), yb)
            loss.backward(); opt.step()
        va_pred = predict(model, X[va_idx], M[va_idx], device)
        auc = roc_auc_score(y[va_idx], va_pred)
        print(f"   [{name}] epoch {epoch+1:>2}/{p['epochs']} "
              f"valid AUC={auc:.4f} ({time.time()-t0:.1f}s)")
        if auc > best_auc:
            best_auc, best_state, wait = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= p["patience"]:
                print(f"   [{name}] early stop"); break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    C.banner("STAGE 5B - DEEP-LEARNING MODELS")
    device = get_device()
    print(f"   device: {device}")

    data = np.load(C.FEATURE_DIR / "sequential.npz")
    X, M, ids, y = data["X"], data["mask"], data["ids"], data["y"].astype(int)
    split = np.load(C.FEATURE_DIR / "split.npz")
    train_set = set(split["train_ids"].tolist())
    tr_idx = np.where(np.isin(ids, list(train_set)))[0]
    va_idx = np.where(~np.isin(ids, list(train_set)))[0]
    F = X.shape[2]
    print(f"   X{X.shape}  train {len(tr_idx):,} | valid {len(va_idx):,}")

    p = C.DL_PARAMS
    builders = {
        "LSTM":        lambda: RecurrentClassifier(F, "LSTM", p["hidden_size"], p["num_layers"], p["dropout"]),
        "GRU":         lambda: RecurrentClassifier(F, "GRU",  p["hidden_size"], p["num_layers"], p["dropout"]),
        "Transformer": lambda: TransformerClassifier(F, p["hidden_size"], p["num_layers"], p["n_heads"], p["ff_dim"], p["dropout"]),
    }

    rows = []
    preds = {C.ID_COL: ids, "split": np.where(np.isin(ids, list(train_set)), "train", "valid")}
    for name, build in builders.items():
        torch.manual_seed(C.RANDOM_STATE)
        model = train_model(name, build(), X, M, y, tr_idx, va_idx, device)
        torch.save(model.state_dict(), C.MODEL_DIR / f"{name.lower()}.pt")
        p_tr = predict(model, X[tr_idx], M[tr_idx], device)
        p_va = predict(model, X[va_idx], M[va_idx], device)
        full = np.empty(len(ids)); full[tr_idx] = p_tr; full[va_idx] = p_va
        preds[name] = full
        rows.append(metrics_row(name, evaluate(y[tr_idx], p_tr), evaluate(y[va_idx], p_va)))
        print(f"   {name} valid Gini={rows[-1]['Valid_Gini']:.4f}")

    pd.DataFrame(preds).to_parquet(C.FEATURE_DIR / "preds_dl.parquet", index=False)
    pd.DataFrame(rows).to_csv(C.REPORT_DIR / "metrics_dl.csv", index=False)
    print(f"\n   metrics -> {C.REPORT_DIR / 'metrics_dl.csv'}")
    C.banner("STAGE 5B COMPLETE")


if __name__ == "__main__":
    main()
