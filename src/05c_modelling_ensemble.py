"""
STAGE 5C - MODELLING: ENSEMBLES (ML + DL)
============================================================================
Builds the four requested hybrid models by blending a tree model's snapshot
PD with a sequential model's PD:
  * LightGBM + GRU
  * XGBoost  + LSTM
  * LightGBM + LSTM
  * XGBoost  + GRU

Method: rank-average blend  p = w * p_tree + (1 - w) * p_seq, where the weight
w in [0, 1] is chosen to MAXIMISE validation Gini (searched on a 0.05 grid).
The same w is then applied to the train set, and both are scored with the
shared metrics module so ensembles sit in the same comparison table.

Why blend tree + sequence? The tree model captures cross-sectional snapshot
interactions; the recurrent model captures temporal repayment behaviour. The
thesis question is whether combining them beats either alone.

Run (after 05a and 05b):
    python src/05c_modelling_ensemble.py
Outputs:
    data/features/preds_ensemble.parquet
    data/reports/metrics_ensemble.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths
bootstrap_paths()
import config as C
from metrics import evaluate, metrics_row

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ENSEMBLES = [
    ("LightGBM+GRU",  "LightGBM", "GRU"),
    ("XGBoost+LSTM",  "XGBoost",  "LSTM"),
    ("LightGBM+LSTM", "LightGBM", "LSTM"),
    ("XGBoost+GRU",   "XGBoost",  "GRU"),
]


def to_rank(p: np.ndarray) -> np.ndarray:
    """Map probabilities to [0,1] ranks so the two models blend on one scale."""
    return rankdata(p) / len(p)


def best_weight(y, p_tree, p_seq):
    """Grid-search the blend weight that maximises validation Gini."""
    rt, rs = to_rank(p_tree), to_rank(p_seq)
    best_w, best_gini = 0.5, -1
    for w in np.linspace(0, 1, 21):
        gini = evaluate(y, w * rt + (1 - w) * rs)["Gini"]
        if gini > best_gini:
            best_gini, best_w = gini, w
    return best_w


def main() -> None:
    C.banner("STAGE 5C - ENSEMBLE MODELS (ML + DL)")
    ml = pd.read_parquet(C.FEATURE_DIR / "preds_ml.parquet")
    dl = pd.read_parquet(C.FEATURE_DIR / "preds_dl.parquet")
    df = ml.merge(dl.drop(columns=["split"]), on=C.ID_COL, how="inner")
    tr = df["split"].to_numpy() == "train"
    y = pd.read_parquet(C.INTERIM_DIR / "labels.parquet")
    df = df.merge(y, on=C.ID_COL, how="left")
    ytr, yva = df.loc[tr, C.TARGET_COL].to_numpy(), df.loc[~tr, C.TARGET_COL].to_numpy()
    print(f"   merged predictions: {len(df):,} customers")

    rows = []
    preds = {C.ID_COL: df[C.ID_COL], "split": df["split"]}
    for name, tree, seq in ENSEMBLES:
        # fit blend weight on validation, then apply to both splits
        w = best_weight(yva, df.loc[~tr, tree].to_numpy(), df.loc[~tr, seq].to_numpy())
        def blend(mask):
            rt = to_rank(df.loc[mask, tree].to_numpy())
            rs = to_rank(df.loc[mask, seq].to_numpy())
            return w * rt + (1 - w) * rs
        p_tr, p_va = blend(tr), blend(~tr)
        full = np.empty(len(df)); full[tr] = p_tr; full[~tr] = p_va
        preds[name] = full
        rows.append(metrics_row(name, evaluate(ytr, p_tr), evaluate(yva, p_va)))
        print(f"   {name:<16} w(tree)={w:.2f}  valid Gini={rows[-1]['Valid_Gini']:.4f}")

    pd.DataFrame(preds).to_parquet(C.FEATURE_DIR / "preds_ensemble.parquet", index=False)
    pd.DataFrame(rows).to_csv(C.REPORT_DIR / "metrics_ensemble.csv", index=False)
    print(f"\n   metrics -> {C.REPORT_DIR / 'metrics_ensemble.csv'}")
    C.banner("STAGE 5C COMPLETE")


if __name__ == "__main__":
    main()
