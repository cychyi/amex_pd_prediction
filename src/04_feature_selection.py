"""
STAGE 4 - FEATURE SELECTION
============================================================================
Applies to the SNAPSHOT (machine-learning) feature set. The deep-learning
sequential features are kept whole, because the LSTM/GRU/Transformer learn
their own representation - selecting raw timesteps would discard the temporal
signal the thesis is testing.

Three-step funnel on the ~1,000 snapshot features:
  1. Variance filter   - drop near-constant engineered columns.
  2. Correlation filter - within highly-correlated pairs (|r| > 0.95) drop one,
     keeping the column with fewer missing values.
  3. Model-based ranking - train a fast LightGBM and keep features whose gain
     importance is above a small threshold (cumulative 99% of total gain), so
     the final ML models train on a compact, informative set.

Run:
    python src/04_feature_selection.py
Outputs:
    data/features/snapshot_selected.parquet
    data/features/selected_features.json
    data/reports/feature_importance_ranking.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths
bootstrap_paths()
import config as C

import numpy as np
import pandas as pd
import lightgbm as lgb


def main() -> None:
    C.banner("STAGE 4 - FEATURE SELECTION (snapshot / ML features)")
    snap = pd.read_parquet(C.FEATURE_DIR / "snapshot.parquet")
    split = np.load(C.FEATURE_DIR / "split.npz")
    train_ids = set(split["train_ids"].tolist())

    feature_cols = [c for c in snap.columns if c not in (C.ID_COL, C.TARGET_COL)]
    X = snap[feature_cols]
    y = snap[C.TARGET_COL]
    print(f"   start: {len(feature_cols)} candidate features")

    # 1) Variance filter --------------------------------------------------
    nun = X.nunique()
    keep = nun[nun > 1].index.tolist()
    print(f"   after variance filter : {len(keep)}")

    # 2) Correlation filter ----------------------------------------------
    # Compute on a sample of rows for speed; drop one of each |r|>0.95 pair.
    sample = X[keep].sample(min(50_000, len(X)), random_state=C.RANDOM_STATE)
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    miss_rate = X[keep].isna().mean()
    to_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > C.CORR_THRESHOLD]
        for p in partners:
            # drop whichever of the pair has more missing values
            loser = p if miss_rate[p] >= miss_rate[col] else col
            to_drop.add(loser)
    keep = [c for c in keep if c not in to_drop]
    print(f"   after correlation filter: {len(keep)} (dropped {len(to_drop)})")

    # 3) Model-based ranking ---------------------------------------------
    tr = snap[C.ID_COL].isin(train_ids)
    booster = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=128,
        feature_fraction=0.5, random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    booster.fit(X.loc[tr, keep], y.loc[tr])
    imp = (pd.Series(booster.feature_importances_, index=keep, name="gain")
           .sort_values(ascending=False))
    imp.to_csv(C.REPORT_DIR / "feature_importance_ranking.csv")

    cum = imp.cumsum() / imp.sum()
    selected = cum[cum <= 0.99].index.tolist() or imp.index[:50].tolist()
    print(f"   after model-based ranking: {len(selected)} (cumulative 99% gain)")

    # Persist selected snapshot ------------------------------------------
    out_cols = [C.ID_COL] + selected + [C.TARGET_COL]
    snap[out_cols].to_parquet(C.FEATURE_DIR / "snapshot_selected.parquet", index=False)
    with open(C.FEATURE_DIR / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"   saved -> {C.FEATURE_DIR / 'snapshot_selected.parquet'}")
    C.banner("STAGE 4 COMPLETE")


if __name__ == "__main__":
    main()
