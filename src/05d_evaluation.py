"""
STAGE 5D - MODEL EVALUATION, COMPARISON & INTERPRETABILITY
============================================================================
Consolidates every model into the two tables shown in the Evaluation slides:

  Table 1 (Training set):  Model | ROC-AUC | KS | Gini | LogLoss
  Table 2 (Validate set):  Model | ROC-AUC | KS | Gini | LogLoss | Diff Gini (%)

plus:
  * the overfit flag (Diff Gini > 30%),
  * a decile risk-ranking table for the best validation model,
  * interpretability (Objective 2):
       - SHAP top-10 features for the best ML model,
       - Integrated Gradients top-10 features for a DL model,
       - attention weights from the Transformer (saved for plotting).

Interpretability is heavy, so it is guarded by flags - run the table build
quickly, then add --shap / --ig when you want the explanations.

Run:
    python src/05d_evaluation.py                 # tables only
    python src/05d_evaluation.py --shap --ig     # + interpretability
Outputs:
    data/reports/comparison_train.csv
    data/reports/comparison_validate.csv
    data/reports/best_model_deciles.csv
    data/reports/shap_top10.csv          (with --shap)
    data/reports/ig_top10.csv            (with --ig)
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths
bootstrap_paths()
import config as C
from metrics import evaluate, decile_table

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def build_comparison() -> pd.DataFrame:
    parts = []
    for f in ["metrics_ml.csv", "metrics_dl.csv", "metrics_ensemble.csv"]:
        p = C.REPORT_DIR / f
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError("Run the modelling stages (05a/05b/05c) first.")
    allm = pd.concat(parts, ignore_index=True)

    train_tbl = allm[["Model", "Train_AUC", "Train_KS", "Train_Gini", "Train_LogLoss"]].copy()
    train_tbl.columns = ["Model", "ROC_AUC", "KS", "Gini", "LogLoss"]

    valid_tbl = allm[["Model", "Valid_AUC", "Valid_KS", "Valid_Gini",
                      "Valid_LogLoss", "DiffGini_%", "Overfit", "Valid_RiskRanked"]].copy()
    valid_tbl.columns = ["Model", "ROC_AUC", "KS", "Gini", "LogLoss",
                         "DiffGini_%", "Overfit", "RiskRanked"]

    train_tbl = train_tbl.sort_values("Gini", ascending=False).reset_index(drop=True)
    valid_tbl = valid_tbl.sort_values("Gini", ascending=False).reset_index(drop=True)
    train_tbl.to_csv(C.REPORT_DIR / "comparison_train.csv", index=False)
    valid_tbl.to_csv(C.REPORT_DIR / "comparison_validate.csv", index=False)

    C.banner("TABLE 1 - TRAINING SET")
    print(train_tbl.to_string(index=False))
    C.banner("TABLE 2 - VALIDATE SET (with overfit check)")
    print(valid_tbl.to_string(index=False))
    return valid_tbl, allm


def best_model_deciles(valid_tbl: pd.DataFrame) -> None:
    best = valid_tbl.iloc[0]["Model"]
    C.banner(f"DECILE RISK-RANKING - best validation model: {best}")
    # locate predictions for the best model
    for f in ["preds_ml.parquet", "preds_dl.parquet", "preds_ensemble.parquet"]:
        df = pd.read_parquet(C.FEATURE_DIR / f)
        if best in df.columns:
            break
    y = pd.read_parquet(C.INTERIM_DIR / "labels.parquet")
    df = df.merge(y, on=C.ID_COL, how="left")
    va = df[df["split"] == "valid"]
    dt = decile_table(va[C.TARGET_COL].to_numpy(), va[best].to_numpy())
    dt.to_csv(C.REPORT_DIR / "best_model_deciles.csv", index=False)
    print(dt.to_string(index=False))
    print(f"   monotonic risk-ranking: {evaluate(va[C.TARGET_COL], va[best])['RiskRanked']}")


# ---------------------------------------------------------------------------
def run_shap() -> None:
    """SHAP top-10 features for the best tree model (Objective 2)."""
    try:
        import shap
    except ImportError:
        print("   [skip] pip install shap to enable SHAP."); return
    C.banner("SHAP - top 10 features (best tree model)")
    snap = pd.read_parquet(C.FEATURE_DIR / "snapshot_selected.parquet")
    feats = [c for c in snap.columns if c not in (C.ID_COL, C.TARGET_COL)]
    with open(C.MODEL_DIR / "LightGBM.pkl", "rb") as f:
        model = pickle.load(f)
    sample = snap[feats].sample(min(5000, len(snap)), random_state=C.RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = (pd.Series(np.abs(sv).mean(0), index=feats)
           .sort_values(ascending=False).head(10))
    imp.to_csv(C.REPORT_DIR / "shap_top10.csv")
    print(imp.to_string())


def run_ig() -> None:
    """Integrated Gradients top-10 features for the LSTM (Objective 2)."""
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ImportError:
        print("   [skip] pip install captum torch to enable Integrated Gradients."); return
    C.banner("INTEGRATED GRADIENTS - top 10 features (LSTM)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dl", Path(__file__).resolve().parents[1] / "src" / "05b_modelling_dl.py")
    dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)

    data = np.load(C.FEATURE_DIR / "sequential.npz")
    X, M = data["X"], data["mask"]
    import json
    feats = json.load(open(C.FEATURE_DIR / "seq_feature_list.json"))
    device = dl.get_device()
    model = dl.RecurrentClassifier(X.shape[2], "LSTM",
                                   C.DL_PARAMS["hidden_size"], C.DL_PARAMS["num_layers"],
                                   C.DL_PARAMS["dropout"]).to(device)
    model.load_state_dict(torch.load(C.MODEL_DIR / "lstm.pt", map_location=device))
    model.eval()

    idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X), size=min(256, len(X)), replace=False)
    xb = torch.tensor(X[idx]).to(device); mb = torch.tensor(M[idx]).to(device)
    ig = IntegratedGradients(model)
    attr = ig.attribute(xb, baselines=torch.zeros_like(xb),
                        additional_forward_args=(mb,), n_steps=16,
                        internal_batch_size=32)
    # aggregate |attribution| over batch & time -> per-feature importance
    imp = (pd.Series(attr.abs().mean(dim=(0, 1)).detach().cpu().numpy(), index=feats)
           .sort_values(ascending=False).head(10))
    imp.to_csv(C.REPORT_DIR / "ig_top10.csv")
    print(imp.to_string())


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="run SHAP on best tree model")
    ap.add_argument("--ig", action="store_true", help="run Integrated Gradients on LSTM")
    args = ap.parse_args()

    valid_tbl, _ = build_comparison()
    best_model_deciles(valid_tbl)
    if args.shap:
        run_shap()
    if args.ig:
        run_ig()
    C.banner("STAGE 5D COMPLETE")


if __name__ == "__main__":
    main()
