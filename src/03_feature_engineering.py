"""
STAGE 3 - FEATURE ENGINEERING
============================================================================
Methodology slide "Feature Engineering":
  * Machine-learning (snapshot) features  ~1,000 -> one row per customer.
  * Deep-learning (sequential) features    ~200  -> a (customer, time, feat) tensor.
  * After aggregation everything is at USER level: 458,913 unique customers.

--------------------------------------------------------------------------
A) SNAPSHOT FEATURES (consumed by Logistic Regression / XGBoost / LightGBM)
   For every numeric feature we compute mean, std, min, max, last, first.
   For every categorical feature we compute last, nunique, count.
   Plus behavioural deltas: (last - mean) and (last - first) capture how a
   customer's most recent statement deviates from their own history - exactly
   the "evolving repayment behaviour" the thesis motivates.

B) SEQUENTIAL FEATURES (consumed by LSTM / GRU / Transformer)
   Each customer's statements are ordered in time, right-aligned and
   padded/truncated to MAX_SEQ_LEN=13 timesteps. Output:
     X_seq : float32 (N, 13, F)
     mask  : float32 (N, 13)      1 = real statement, 0 = padding
     ids   : int64   (N,)
     y     : int8    (N,)
   Features are standardised with TRAIN-set statistics only (no leakage); the
   scaler is saved for inference.

Run:
    python src/03_feature_engineering.py
Outputs:
    data/features/snapshot.parquet                 (ML)
    data/features/sequential.npz                   (DL: X_seq, mask, ids, y)
    data/features/seq_feature_list.json
    data/features/seq_scaler.npz                   (mean/std for inference)
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths, reduce_mem_usage
bootstrap_paths()
import config as C

import numpy as np
import pandas as pd


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_parquet(C.INTERIM_DIR / "clean.parquet")
    labels = pd.read_parquet(C.INTERIM_DIR / "labels.parquet")
    with open(C.INTERIM_DIR / "kept_features.json") as f:
        feats = json.load(f)
    return df, labels, feats


# ===========================================================================
# A) SNAPSHOT FEATURES (ML)
# ===========================================================================
def build_snapshot(df: pd.DataFrame, feats: dict) -> pd.DataFrame:
    C.banner("3A - SNAPSHOT FEATURES (machine-learning models)")
    num = feats["numeric"]
    cat = feats["categorical"]

    print(f"   aggregating {len(num)} numeric features with {C.NUM_AGGS}...")
    num_agg = df.groupby(C.ID_COL)[num].agg(C.NUM_AGGS)
    num_agg.columns = [f"{c}_{a}" for c, a in num_agg.columns]

    print(f"   aggregating {len(cat)} categorical features with {C.CAT_AGGS}...")
    if cat:
        cat_agg = df.groupby(C.ID_COL)[cat].agg(C.CAT_AGGS)
        cat_agg.columns = [f"{c}_{a}" for c, a in cat_agg.columns]
    else:
        cat_agg = pd.DataFrame(index=num_agg.index)

    # Behavioural deltas: how the latest statement deviates from own history.
    print("   engineering behavioural deltas (last-mean, last-first)...")
    delta_cols = {}
    for c in num:
        delta_cols[f"{c}_last_minus_mean"]  = num_agg[f"{c}_last"]  - num_agg[f"{c}_mean"]
        delta_cols[f"{c}_last_minus_first"] = num_agg[f"{c}_last"]  - num_agg[f"{c}_first"]
    delta = pd.DataFrame(delta_cols, index=num_agg.index)

    snap = pd.concat([num_agg, cat_agg, delta], axis=1).reset_index()
    snap = reduce_mem_usage(snap)
    print(f"   snapshot matrix: {snap.shape[0]:,} customers x {snap.shape[1]-1} features")
    return snap


# ===========================================================================
# B) SEQUENTIAL FEATURES (DL)
# ===========================================================================
def build_sequential(df: pd.DataFrame, feats: dict, train_ids: np.ndarray):
    C.banner("3B - SEQUENTIAL FEATURES (deep-learning models)")
    seq_feats = feats["all"]                 # ~200 raw features per timestep
    F = len(seq_feats)
    L = C.MAX_SEQ_LEN
    print(f"   building (N, {L}, {F}) tensor, right-aligned + padded...")

    # Standardise with TRAIN statistics only (fit on rows of training customers).
    train_mask_rows = df[C.ID_COL].isin(set(train_ids.tolist()))
    means = df.loc[train_mask_rows, seq_feats].mean()
    stds  = df.loc[train_mask_rows, seq_feats].std().replace(0, 1.0)
    df_std = df[[C.ID_COL]].copy()
    df_std[seq_feats] = ((df[seq_feats] - means) / stds).fillna(0).astype(np.float32)

    ids = df[C.ID_COL].to_numpy()
    values = df_std[seq_feats].to_numpy(dtype=np.float32)

    # Build per-customer sequences. Rows are already sorted by (id, date).
    unique_ids, start_idx, counts = np.unique(ids, return_index=True, return_counts=True)
    N = len(unique_ids)
    X = np.zeros((N, L, F), dtype=np.float32)
    M = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        s = start_idx[i]
        n = min(counts[i], L)              # truncate to last L statements
        block = values[s + counts[i] - n : s + counts[i]]   # most-recent n rows
        X[i, L - n:, :] = block            # right-align (pad at the front)
        M[i, L - n:] = 1.0

    np.savez_compressed(
        C.FEATURE_DIR / "seq_scaler.npz",
        mean=means.to_numpy(np.float32), std=stds.to_numpy(np.float32),
        features=np.array(seq_feats),
    )
    with open(C.FEATURE_DIR / "seq_feature_list.json", "w") as f:
        json.dump(seq_feats, f, indent=2)

    print(f"   sequential tensor: X{X.shape}  mask{M.shape}")
    return unique_ids, X, M


# ===========================================================================
def main() -> None:
    df, labels, feats = load_clean()
    label_map = dict(zip(labels[C.ID_COL], labels[C.TARGET_COL]))

    # Stratified-ish train/valid split BY CUSTOMER (slide: 367,140 / 91,783).
    rng = np.random.RandomState(C.RANDOM_STATE)
    all_ids = labels[C.ID_COL].to_numpy()
    y_all = labels[C.TARGET_COL].to_numpy()
    perm = rng.permutation(len(all_ids))
    n_valid = int(len(all_ids) * C.VALID_SIZE)
    valid_ids = all_ids[perm[:n_valid]]
    train_ids = all_ids[perm[n_valid:]]
    np.savez(C.FEATURE_DIR / "split.npz", train_ids=train_ids, valid_ids=valid_ids)
    print(f"Split -> train {len(train_ids):,} | valid {len(valid_ids):,} customers")

    # --- A) Snapshot for ML -------------------------------------------------
    snap = build_snapshot(df, feats).copy()
    snap[C.TARGET_COL] = snap[C.ID_COL].map(label_map).astype("int8")
    snap.to_parquet(C.FEATURE_DIR / "snapshot.parquet", index=False)
    print(f"   saved -> {C.FEATURE_DIR / 'snapshot.parquet'}")

    # --- B) Sequential for DL ----------------------------------------------
    seq_ids, X, M = build_sequential(df, feats, train_ids)
    y = np.array([label_map[i] for i in seq_ids], dtype=np.int8)
    np.savez_compressed(C.FEATURE_DIR / "sequential.npz",
                        X=X, mask=M, ids=seq_ids, y=y)
    print(f"   saved -> {C.FEATURE_DIR / 'sequential.npz'}")
    C.banner("STAGE 3 COMPLETE")


if __name__ == "__main__":
    main()
