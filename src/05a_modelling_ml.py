"""
STAGE 5A - MODELLING: MACHINE-LEARNING MODELS
============================================================================
Trains the three snapshot-based models from the slide:
  * Logistic Regression  (baseline, interpretable)
  * XGBoost
  * LightGBM

Each model is trained on the TRAIN customers and scored on TRAIN + VALIDATE
using the shared metrics module (ROC-AUC, KS, Gini, log-loss, decile ranking,
overfit check). Per-customer predicted PDs are saved so Stage 5C can build the
ML+DL ensembles, and the trained models are saved for the SHAP step.

Run:
    python src/05a_modelling_ml.py
Outputs:
    data/models/{logreg,xgboost,lightgbm}.pkl
    data/features/preds_ml.parquet      (customer_ID, split, model -> pd)
    data/reports/metrics_ml.csv
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths
bootstrap_paths()
import config as C
from metrics import evaluate, metrics_row

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb


def load_data():
    snap = pd.read_parquet(C.FEATURE_DIR / "snapshot_selected.parquet")
    split = np.load(C.FEATURE_DIR / "split.npz")
    feats = [c for c in snap.columns if c not in (C.ID_COL, C.TARGET_COL)]
    tr = snap[C.ID_COL].isin(set(split["train_ids"].tolist()))
    return snap, feats, tr


def main() -> None:
    C.banner("STAGE 5A - MACHINE-LEARNING MODELS")
    snap, feats, tr = load_data()
    Xtr, ytr = snap.loc[tr, feats], snap.loc[tr, C.TARGET_COL]
    Xva, yva = snap.loc[~tr, feats], snap.loc[~tr, C.TARGET_COL]
    print(f"   train {len(Xtr):,} | valid {len(Xva):,} | features {len(feats)}")

    rows, preds = [], {C.ID_COL: snap[C.ID_COL], "split": np.where(tr, "train", "valid")}

    def record(name, p_tr, p_va, model):
        ev_tr = evaluate(ytr, p_tr)
        ev_va = evaluate(yva, p_va)
        rows.append(metrics_row(name, ev_tr, ev_va))
        full = np.empty(len(snap)); full[tr.to_numpy()] = p_tr; full[~tr.to_numpy()] = p_va
        preds[name] = full
        with open(C.MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"   {name:<20} valid AUC={ev_va['ROC_AUC']:.4f} "
              f"Gini={ev_va['Gini']:.4f} KS={ev_va['KS']:.4f}")

    # --- Logistic Regression (impute -> scale -> logit) --------------------
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
    ])
    logreg.fit(Xtr, ytr)
    record("LogisticRegression",
           logreg.predict_proba(Xtr)[:, 1], logreg.predict_proba(Xva)[:, 1], logreg)

    # --- XGBoost -----------------------------------------------------------
    xgb_clf = xgb.XGBClassifier(**C.XGB_PARAMS)
    xgb_clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    record("XGBoost",
           xgb_clf.predict_proba(Xtr)[:, 1], xgb_clf.predict_proba(Xva)[:, 1], xgb_clf)

    # --- LightGBM ----------------------------------------------------------
    lgb_clf = lgb.LGBMClassifier(**C.LGB_PARAMS)
    lgb_clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    record("LightGBM",
           lgb_clf.predict_proba(Xtr)[:, 1], lgb_clf.predict_proba(Xva)[:, 1], lgb_clf)

    pd.DataFrame(preds).to_parquet(C.FEATURE_DIR / "preds_ml.parquet", index=False)
    pd.DataFrame(rows).to_csv(C.REPORT_DIR / "metrics_ml.csv", index=False)
    print(f"\n   metrics -> {C.REPORT_DIR / 'metrics_ml.csv'}")
    C.banner("STAGE 5A COMPLETE")


if __name__ == "__main__":
    main()
