"""
STAGE 1 - DATA LANDING & EDA
============================================================================
Methodology slide "Data Ingestion":
  * The train CSV is large (~16 GB; the test CSV ~33 GB), so reading it
    directly with pd.read_csv() is risky. Instead we stream it in chunks of
    500,000 rows and write each chunk to Parquet, then read the Parquet back.
  * customer_ID is compressed to int64, S_2 parsed as a date.

Then we run the EDA required by the slides:
  * per-feature missing rate
  * descriptive statistics (mean/std/min/max/percentiles)
  * count of unique values (to find constant columns)
  * feature-family summary (P/B/S/R/D)
All EDA artefacts are written to reports/ for the thesis appendix.

Run:
    python src/01_data_landing_eda.py
Outputs:
    data/parquet/train_*.parquet           (chunked raw, compressed)
    data/reports/eda_missing.csv
    data/reports/eda_describe.csv
    data/reports/eda_feature_families.csv
    data/reports/eda_target_balance.csv
"""
from __future__ import annotations
import sys, glob, time
from pathlib import Path

# --- path bootstrap -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from common import bootstrap_paths, reduce_mem_usage, compress_customer_id
bootstrap_paths()
import config as C

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
def land_csv_to_parquet(csv_path: Path, prefix: str) -> None:
    """Stream a big CSV to compressed Parquet chunks (slide: chunk = 500k)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the AMEX dataset into {C.RAW_DIR} "
            f"(or run on Kaggle where it is mounted at /kaggle/input)."
        )
    C.banner(f"LANDING {csv_path.name} -> Parquet (chunk={C.CHUNK_SIZE:,})")
    t0 = time.time()
    reader = pd.read_csv(csv_path, chunksize=C.CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        chunk[C.ID_COL] = compress_customer_id(chunk[C.ID_COL])
        if C.DATE_COL in chunk.columns:
            chunk[C.DATE_COL] = pd.to_datetime(chunk[C.DATE_COL])
        chunk = reduce_mem_usage(chunk, verbose=False)
        out = C.PARQUET_DIR / f"{prefix}_{i:04d}.parquet"
        chunk.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        print(f"   chunk {i:>3}: {len(chunk):>7,} rows -> {out.name}")
    print(f"   done in {time.time()-t0:.1f}s")


def load_parquet(prefix: str) -> pd.DataFrame:
    """Read all chunks for a prefix back into one DataFrame."""
    files = sorted(glob.glob(str(C.PARQUET_DIR / f"{prefix}_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files for prefix '{prefix}'. Run landing first.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df


# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, labels: pd.DataFrame) -> None:
    C.banner("EDA")
    feature_cols = [c for c in df.columns if c not in (C.ID_COL, C.DATE_COL)]

    # 1) Missing rate per feature.
    missing = (df[feature_cols].isna().mean()
               .sort_values(ascending=False)
               .rename("missing_rate").to_frame())
    missing["n_unique"] = [df[c].nunique(dropna=True) for c in missing.index]
    missing["flag_drop"] = (missing["missing_rate"] > C.MISSING_THRESHOLD) | (missing["n_unique"] <= 1)
    missing.to_csv(C.REPORT_DIR / "eda_missing.csv")
    print(f"   features with >90% missing : "
          f"{(missing['missing_rate'] > C.MISSING_THRESHOLD).sum()}")
    print(f"   constant features (nunique<=1): {(missing['n_unique'] <= 1).sum()}")
    print(f"   total flagged for drop        : {missing['flag_drop'].sum()}")

    # 2) Descriptive statistics (mean/std/min/max/percentiles).
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc.to_csv(C.REPORT_DIR / "eda_describe.csv")
    print(f"   describe() written for {len(num_cols)} numeric features")

    # 3) Feature-family summary (P / B / S / R / D).
    fam_rows = []
    for prefix, name in C.FEATURE_PREFIXES.items():
        cols = [c for c in feature_cols if c.startswith(prefix + "_")]
        if cols:
            fam_rows.append({
                "prefix": prefix, "family": name, "n_features": len(cols),
                "avg_missing_rate": df[cols].isna().mean().mean(),
            })
    pd.DataFrame(fam_rows).to_csv(C.REPORT_DIR / "eda_feature_families.csv", index=False)
    for r in fam_rows:
        print(f"   {r['prefix']} ({r['family']:<11}): {r['n_features']:>3} features, "
              f"avg missing {r['avg_missing_rate']:.2%}")

    # 4) Target balance.
    bal = labels[C.TARGET_COL].value_counts().rename("count").to_frame()
    bal["pct"] = bal["count"] / bal["count"].sum()
    bal.to_csv(C.REPORT_DIR / "eda_target_balance.csv")
    print(f"   target balance: good(0)={bal.loc[0,'count']:,} "
          f"bad(1)={bal.loc[1,'count']:,} "
          f"bad-rate={bal.loc[1,'pct']:.2%}")

    # 5) Statements-per-customer distribution (justifies MAX_SEQ_LEN=13).
    counts = df.groupby(C.ID_COL).size()
    print(f"   statements/customer: min={counts.min()} "
          f"median={int(counts.median())} max={counts.max()}")


# ---------------------------------------------------------------------------
def main() -> None:
    # Land train features + labels. (Land the test set later if you submit to Kaggle.)
    if not list(C.PARQUET_DIR.glob("train_*.parquet")):
        land_csv_to_parquet(C.TRAIN_DATA_CSV, "train")
    else:
        print("Parquet chunks already exist - skipping landing. Delete them to re-run.")

    df = load_parquet("train")
    print(f"\nLoaded train: {df.shape[0]:,} rows x {df.shape[1]} cols")

    labels = pd.read_csv(C.TRAIN_LABELS_CSV)
    labels[C.ID_COL] = compress_customer_id(labels[C.ID_COL])
    labels.to_parquet(C.INTERIM_DIR / "labels.parquet", index=False)

    run_eda(df, labels)
    C.banner("STAGE 1 COMPLETE")


if __name__ == "__main__":
    main()
