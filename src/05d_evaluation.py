"""
STAGE 5D: COMPREHENSIVE EVALUATION & INTERPRETABILITY
- Comparison tables (train + validate with all metrics: Gini, AUC, KS, F1, etc)
- Decile tables for ALL models (train + validate)
- SHAP feature importance (ML models)
- Gradient importance (DL models)
- Visualization plots
"""

import sys
import time
import json
import pickle
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, f1_score, 
    precision_score, recall_score
)

# ===== Config =====
ROOT = Path.cwd()
INTERIM_DIR = ROOT / "data" / "interim"
FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "data" / "models"
REPORT_DIR = ROOT / "data" / "reports"

ID_COL = "customer_ID"
TARGET_COL = "target"
MAX_SEQ_LEN = 13
RANDOM_STATE = 42

# DL params (from config)
DL_PARAMS = {
    "hidden_size": 128, "num_layers": 2, "dropout": 0.30,
    "n_heads": 8, "ff_dim": 256
}

_T0 = time.time()
def log(msg=""):
    el = time.time() - _T0
    print(f"[{time.strftime('%H:%M:%S')} | +{el:6.1f}s] {msg}", flush=True)

def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)

# ===== Metrics helpers =====
def gini_from_auc(auc):
    return 2.0 * auc - 1.0

def ks_statistic(y_true, y_pred):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)}).sort_values("p", ascending=False)
    cum_bad = (d["y"].cumsum() / d["y"].sum()).to_numpy()
    cum_good = ((1 - d["y"]).cumsum() / (1 - d["y"]).sum()).to_numpy()
    return float(np.max(np.abs(cum_bad - cum_good)))

def decile_table(y_true, y_pred, n_bins=10):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False),
                          q=n_bins, labels=range(1, n_bins + 1))
    g = d.groupby("decile", observed=True)
    out = pd.DataFrame({
        "decile": g.groups.keys(),
        "n": g.size().values,
        "n_bad": g["y"].sum().values,
        "default_rate": g["y"].mean().values,
        "avg_pred_pd": g["p"].mean().values,
    }).reset_index(drop=True)
    cum_bad = out["n_bad"].cumsum() / out["n_bad"].sum()
    out.insert(4, "cum_bad_rate", cum_bad)
    return out

def compute_metrics(y_true, y_pred):
    """Compute all metrics for a model."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.clip(np.asarray(y_pred, float), 1e-15, 1 - 1e-15)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    return {
        "ROC_AUC": round(roc_auc_score(y_true, y_pred), 5),
        "KS": round(ks_statistic(y_true, y_pred), 5),
        "Gini": round(gini_from_auc(roc_auc_score(y_true, y_pred)), 5),
        "LogLoss": round(log_loss(y_true, y_pred), 5),
        "Accuracy": round(accuracy_score(y_true, y_pred_binary), 5),
        "Precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 5),
        "Recall": round(recall_score(y_true, y_pred_binary, zero_division=0), 5),
        "F1": round(f1_score(y_true, y_pred_binary, zero_division=0), 5),
    }

# ===== DL model classes =====
class RecurrentClassifier(nn.Module):
    def __init__(self, n_feat, cell="LSTM", hidden=128, layers=2, dropout=0.3):
        super().__init__()
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(n_feat, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x, mask):
        out, _ = self.rnn(x)
        lengths = mask.sum(1).long().clamp(min=1)
        last = out[torch.arange(out.size(0)), lengths - 1]
        return self.head(last).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, hidden=128, layers=2, heads=8, ff=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = PositionalEncoding(hidden, MAX_SEQ_LEN + 1)
        enc = nn.TransformerEncoderLayer(hidden, heads, ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x, mask):
        h = self.pos(self.proj(x))
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

# ===== Main evaluation =====
def main():
    banner("STAGE 5D — COMPREHENSIVE EVALUATION & INTERPRETABILITY")
    
    # Load predictions
    log("Loading predictions...")
    ml_preds = pd.read_parquet(FEATURE_DIR / "preds_ml.parquet")
    dl_preds = pd.read_parquet(FEATURE_DIR / "preds_dl.parquet")
    ens_preds = pd.read_parquet(FEATURE_DIR / "preds_ensemble.parquet")
    labels = pd.read_parquet(INTERIM_DIR / "labels.parquet")
    
    # Merge all
    all_preds = ml_preds.merge(dl_preds.drop(columns=["split"]), on=ID_COL).merge(
        ens_preds.drop(columns=["split"]), on=ID_COL).merge(labels, on=ID_COL)
    
    tr = all_preds["split"] == "train"
    models = [c for c in all_preds.columns if c not in (ID_COL, "split", TARGET_COL)]
    log(f"Evaluating {len(models)} models")
    
    # ===== PART 1: Comparison tables =====
    banner("PART 1 — COMPREHENSIVE COMPARISON TABLES (TRAIN + VALIDATE)")
    
    rows_train, rows_valid = [], []
    for model in models:
        ytr, yva = all_preds.loc[tr, TARGET_COL], all_preds.loc[~tr, TARGET_COL]
        ptr, pva = all_preds.loc[tr, model], all_preds.loc[~tr, model]
        
        m_tr = compute_metrics(ytr, ptr)
        m_va = compute_metrics(yva, pva)
        
        rows_train.append({"Model": model, **m_tr})
        rows_valid.append({"Model": model, **m_va})
    
    comp_train = pd.DataFrame(rows_train).sort_values("Gini", ascending=False)
    comp_valid = pd.DataFrame(rows_valid).sort_values("Gini", ascending=False)
    
    comp_train.to_csv(REPORT_DIR / "comparison_train_full.csv", index=False)
    comp_valid.to_csv(REPORT_DIR / "comparison_validate_full.csv", index=False)
    
    log("✓ Comparison tables saved")
    print("\nTRAIN SET:")
    print(comp_train.to_string(index=False))
    print("\nVALIDATE SET:")
    print(comp_valid.to_string(index=False))
    
    # ===== PART 2: Decile tables =====
    banner("PART 2 — DECILE TABLES (ALL MODELS, TRAIN + VALIDATE)")
    
    for model in models:
        for split_name, split_mask in [("train", tr), ("validate", ~tr)]:
            y_split = all_preds.loc[split_mask, TARGET_COL]
            p_split = all_preds.loc[split_mask, model]
            dt = decile_table(y_split, p_split)
            dt.to_csv(REPORT_DIR / f"decile_{model}_{split_name}.csv", index=False)
    
    log(f"✓ {len(models)} × 2 decile tables saved")
    
    # ===== PART 3: SHAP =====
    banner("PART 3 — SHAP FEATURE IMPORTANCE (ML MODELS)")
    
    try:
        import shap
        snap = pd.read_parquet(FEATURE_DIR / "snapshot_selected.parquet")
        feats = [c for c in snap.columns if c not in (ID_COL, TARGET_COL)]
        sample = snap[feats].sample(min(10000, len(snap)), random_state=RANDOM_STATE)
        
        for ml_name in ["LightGBM", "XGBoost"]:
            try:
                model = pickle.load(open(MODEL_DIR / f"{ml_name}.pkl", "rb"))
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(sample)
                shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                imp = pd.Series(np.abs(shap_vals).mean(0), index=feats).sort_values(ascending=False)
                imp.to_csv(REPORT_DIR / f"shap_{ml_name.lower()}_top15.csv")
                log(f"✓ SHAP {ml_name}")
            except Exception as e:
                log(f"✗ SHAP {ml_name}: {e}")
    except Exception as e:
        log(f"✗ SHAP skipped: {e}")
    
    # ===== PART 4: Gradient importance =====
    banner("PART 4 — GRADIENT IMPORTANCE (DL MODELS)")
    
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        log(f"device: {device}")
        
        data = np.load(FEATURE_DIR / "sequential.npz")
        X, Mk = data["X"], data["mask"]
        split = np.load(FEATURE_DIR / "split.npz")
        va_idx = np.where(~np.isin(data["ids"], set(split["train_ids"].tolist())))[0]
        
        Xva, Mva = X[va_idx], Mk[va_idx]
        seq_feats = json.load(open(FEATURE_DIR / "seq_feature_list.json"))
        
        dl_models = {
            "LSTM": RecurrentClassifier(X.shape[2], "LSTM", DL_PARAMS["hidden_size"],
                                       DL_PARAMS["num_layers"], DL_PARAMS["dropout"]),
            "GRU": RecurrentClassifier(X.shape[2], "GRU", DL_PARAMS["hidden_size"],
                                      DL_PARAMS["num_layers"], DL_PARAMS["dropout"]),
            "Transformer": TransformerClassifier(X.shape[2], DL_PARAMS["hidden_size"],
                                                DL_PARAMS["num_layers"], DL_PARAMS["n_heads"],
                                                DL_PARAMS["ff_dim"], DL_PARAMS["dropout"]),
        }
        
        for dl_name, model in dl_models.items():
            try:
                model = model.to(device)
                model.load_state_dict(torch.load(MODEL_DIR / f"{dl_name.lower()}.pt", map_location=device))
                model.eval()
                
                Xva_torch = torch.tensor(Xva[:500]).to(device).requires_grad_(True)
                Mva_torch = torch.tensor(Mva[:500]).to(device)
                output = model(Xva_torch, Mva_torch)
                loss = output.mean()
                loss.backward()
                grad_imp = torch.abs(Xva_torch.grad).mean(dim=(0, 1)).detach().cpu().numpy()
                
                imp = pd.Series(grad_imp, index=seq_feats).sort_values(ascending=False)
                imp.to_csv(REPORT_DIR / f"gradient_{dl_name.lower()}_top15.csv")
                log(f"✓ Gradient {dl_name}")
            except Exception as e:
                log(f"✗ Gradient {dl_name}: {e}")
    except Exception as e:
        log(f"✗ Gradient importance skipped: {e}")
    
    # ===== PART 5: Plots =====
    banner("PART 5 — VISUALIZATION PLOTS")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        # SHAP plots
        for ml_name in ["lightgbm", "xgboost"]:
            try:
                shap_file = REPORT_DIR / f"shap_{ml_name}_top15.csv"
                if shap_file.exists():
                    shap_data = pd.read_csv(shap_file, index_col=0)
                    plt.figure(figsize=(10, 6))
                    shap_data.sort_values("mean_abs_shap").plot(kind="barh", legend=False, color="steelblue")
                    plt.title(f"SHAP Feature Importance - {ml_name.upper()}")
                    plt.xlabel("Mean |SHAP value|")
                    plt.tight_layout()
                    plt.savefig(REPORT_DIR / f"shap_{ml_name}_plot.png", dpi=300, bbox_inches="tight")
                    plt.close()
                    log(f"✓ shap_{ml_name}_plot.png")
            except Exception as e:
                log(f"✗ SHAP plot {ml_name}: {e}")
        
        # Gradient plots
        for dl_name in ["lstm", "gru", "transformer"]:
            try:
                grad_file = REPORT_DIR / f"gradient_{dl_name}_top15.csv"
                if grad_file.exists():
                    grad_data = pd.read_csv(grad_file, index_col=0)
                    plt.figure(figsize=(10, 6))
                    grad_data.sort_values("mean_abs_gradient").plot(kind="barh", legend=False, color="coral")
                    plt.title(f"Gradient Feature Importance - {dl_name.upper()}")
                    plt.xlabel("Mean |Gradient|")
                    plt.tight_layout()
                    plt.savefig(REPORT_DIR / f"gradient_{dl_name}_plot.png", dpi=300, bbox_inches="tight")
                    plt.close()
                    log(f"✓ gradient_{dl_name}_plot.png")
            except Exception as e:
                log(f"✗ Gradient plot {dl_name}: {e}")
        
        # Model comparison
        try:
            comp_valid = pd.read_csv(REPORT_DIR / "comparison_validate_full.csv")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            comp_valid.sort_values("Gini", ascending=False).plot(x="Model", y="Gini", kind="barh", ax=axes[0, 0], legend=False, color="green")
            axes[0, 0].set_title("Gini - Validation")
            axes[0, 0].set_xlabel("")
            
            comp_valid.sort_values("ROC_AUC", ascending=False).plot(x="Model", y="ROC_AUC", kind="barh", ax=axes[0, 1], legend=False, color="blue")
            axes[0, 1].set_title("ROC-AUC - Validation")
            axes[0, 1].set_xlabel("")
            
            comp_valid.sort_values("F1", ascending=False).plot(x="Model", y="F1", kind="barh", ax=axes[1, 0], legend=False, color="orange")
            axes[1, 0].set_title("F1-Score - Validation")
            axes[1, 0].set_xlabel("")
            
            comp_valid.sort_values("KS", ascending=False).plot(x="Model", y="KS", kind="barh", ax=axes[1, 1], legend=False, color="red")
            axes[1, 1].set_title("KS Statistic - Validation")
            axes[1, 1].set_xlabel("")
            
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "model_comparison_metrics.png", dpi=300, bbox_inches="tight")
            plt.close()
            log("✓ model_comparison_metrics.png")
        except Exception as e:
            log(f"✗ Model comparison plot: {e}")
    except Exception as e:
        log(f"✗ Plots skipped: {e}")
    
    banner("STAGE 5D COMPLETE")
    log("All results in data/reports/")

if __name__ == "__main__":
    main()
