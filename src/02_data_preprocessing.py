"""
STAGE 2 - DATA PREPROCESSING
============================================================================
Methodology slide "Data Preprocessing":
  2.1 Denoise: multiply numeric features by 100 and round (recovers the
      quantised values hidden by the small uniform noise AMEX added), and
      encode the categorical variables D_63, D_64.
  2.2 EDA-driven filtering: drop features with >90% missing rate and features
      with a single unique value. The slide names D_87 as a dropped feature.
  2.3 Persist a clean, statement-level Parquet that later stages consume.

We keep the data at STATEMENT level here (one row per customer-statement),
because Stage 3 needs the raw sequence for the deep-learning features and the
per-customer aggregations for the machine-learning features.

Run:
    python src/02_data_preprocessing.py
Outputs:
    data/interim/clean.parquet          (statement-level, denoised, encoded)
    data/interim/kept_features.json     (feature lists for downstream stages)
"""
from __future__ import annotations
import sys, json, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths, reduce_mem_usage
bootstrap_paths()
import config as C

import numpy as np
import pandas as pd


def load_train_parquet() -> pd.DataFrame:
    files = sorted(glob.glob(str(C.PARQUET_DIR / "train_*.parquet")))
    if not files:
        raise FileNotFoundError("Run 01_data_landing_eda.py first.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> None:
    C.banner("STAGE 2 - DATA PREPROCESSING")
    df = load_train_parquet()
    print(f"Loaded statement-level data: {df.shape[0]:,} rows x {df.shape[1]} cols")

    feature_cols = [c for c in df.columns if c not in (C.ID_COL, C.DATE_COL)]
    num_cols = [c for c in feature_cols if c not in C.CAT_COLS]
    cat_cols = [c for c in C.CAT_COLS if c in df.columns]

    # --- 2.1a Denoise numeric features: *100 then round ---------------------
    print("2.1  Denoising numeric features (x100, round)...")
    for c in num_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = (df[c] * C.DENOISE_SCALE).round(0)

    # --- 2.1b Encode categoricals to integer codes --------------------------
    # We use integer codes (not wide one-hot) at statement level so the
    # sequential DL tensor stays compact; Stage 3 one-hot-aggregates for ML.
    print(f"2.1  Encoding categoricals {cat_cols} to integer codes...")
    cat_categories = {}
    for c in cat_cols:
        df[c] = df[c].astype("category")
        cat_categories[c] = [str(x) for x in df[c].cat.categories.tolist()]
        df[c] = df[c].cat.codes.astype("int16")   # -1 == missing

    # --- 2.2 Filter features: >90% missing, single unique value, named drops -
    print("2.2  Filtering features (missing>90% | nunique<=1 | named drops)...")
    drop = set(C.DROP_FEATURES)
    miss = df[feature_cols].isna().mean()
    drop |= set(miss[miss > C.MISSING_THRESHOLD].index)
    nun = df[feature_cols].nunique(dropna=True)
    drop |= set(nun[nun <= 1].index)
    drop = [c for c in drop if c in df.columns]
    df = df.drop(columns=drop)
    print(f"     dropped {len(drop)} features: "
          f"{sorted(drop)[:8]}{' ...' if len(drop) > 8 else ''}")

    # Recompute kept feature lists.
    kept_features = [c for c in df.columns if c not in (C.ID_COL, C.DATE_COL)]
    kept_cat = [c for c in cat_cols if c in df.columns]
    kept_num = [c for c in kept_features if c not in kept_cat]

    # Sort by customer then date so sequences are chronologically ordered.
    df = df.sort_values([C.ID_COL, C.DATE_COL]).reset_index(drop=True)
    df = reduce_mem_usage(df)

    # --- 2.3 Persist --------------------------------------------------------
    out = C.INTERIM_DIR / "clean.parquet"
    df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    with open(C.INTERIM_DIR / "kept_features.json", "w") as f:
        json.dump({
            "all": kept_features,
            "numeric": kept_num,
            "categorical": kept_cat,
            "cat_categories": cat_categories,
            "dropped": sorted(drop),
        }, f, indent=2)

    print(f"\n     clean.parquet: {df.shape[0]:,} rows x {df.shape[1]} cols -> {out}")
    print(f"     kept {len(kept_num)} numeric + {len(kept_cat)} categorical features")
    C.banner("STAGE 2 COMPLETE")


if __name__ == "__main__":
    main()
