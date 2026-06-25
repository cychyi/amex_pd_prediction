"""
Central configuration for the AMEX Probability-of-Default (PD) thesis pipeline.

Edit ONLY this file when moving between environments (local Mac vs Kaggle).
Every script imports from here so paths and constants stay consistent.

Author: Cham Ying Chyi (23076054) - University of Malaya
"""
from __future__ import annotations
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 0. ENVIRONMENT DETECTION
# ----------------------------------------------------------------------------
# The pipeline auto-detects whether it is running on Kaggle or locally so you
# do not have to change paths by hand. Override with the AMEX_ENV env var.
def _detect_env() -> str:
    if os.environ.get("AMEX_ENV"):
        return os.environ["AMEX_ENV"].lower()
    if os.path.isdir("/kaggle/input"):
        return "kaggle"
    return "local"

ENV = _detect_env()

# ----------------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------------
if ENV == "kaggle":
    # The official competition dataset is mounted read-only here.
    RAW_DIR    = Path("/kaggle/input/amex-default-prediction")
    # /kaggle/working is the only writable dir and becomes the notebook output.
    OUTPUT_DIR = Path("/kaggle/working")
else:
    # Local Mac Mini M4 layout. Put the downloaded CSVs in data/raw/.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR      = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR   = PROJECT_ROOT / "data"

# Derived directories (created on demand by each script).
PARQUET_DIR   = OUTPUT_DIR / "parquet"          # chunked raw -> parquet
INTERIM_DIR   = OUTPUT_DIR / "interim"          # preprocessed parquet
FEATURE_DIR   = OUTPUT_DIR / "features"         # snapshot + sequential features
MODEL_DIR     = OUTPUT_DIR / "models"           # saved models
REPORT_DIR    = OUTPUT_DIR / "reports"          # metrics tables, plots, SHAP

for _d in (PARQUET_DIR, INTERIM_DIR, FEATURE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw file names (rename here if yours differ).
TRAIN_DATA_CSV   = RAW_DIR / "train_data.csv"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
TEST_DATA_CSV    = RAW_DIR / "test_data.csv"     # optional, ~33 GB

# ----------------------------------------------------------------------------
# 2. COLUMN / FEATURE GROUPS
# ----------------------------------------------------------------------------
ID_COL     = "customer_ID"
DATE_COL   = "S_2"            # statement date
TARGET_COL = "target"

# The slide treats D_63 and D_64 as the categorical variables to one-hot encode.
# The full official AMEX categorical list is kept for reference / extension.
CAT_COLS_SLIDE = ["D_63", "D_64"]
CAT_COLS_FULL  = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
                  "D_126", "D_63", "D_64", "D_66", "D_68"]
# Switch this if your supervisor wants the full categorical treatment.
CAT_COLS = CAT_COLS_SLIDE

# Feature-family prefixes (used for EDA grouping and reporting).
FEATURE_PREFIXES = {
    "P": "Payment",
    "B": "Balance",
    "S": "Spend",
    "R": "Risk",
    "D": "Delinquency",
}

# Feature explicitly dropped in the slide's EDA (>90% missing, low deviation).
DROP_FEATURES = ["D_87"]

# ----------------------------------------------------------------------------
# 3. PIPELINE CONSTANTS
# ----------------------------------------------------------------------------
CHUNK_SIZE        = 100_000   # rows per parquet chunk (matches the slide)
MISSING_THRESHOLD = 0.90      # drop columns with > 90% missing
DENOISE_SCALE     = 100       # multiply numeric features by 100, then round
MAX_SEQ_LEN       = 13        # statements per customer (AMEX has up to 13)
SEQ_TOP_N         = 15        # sequential tensor uses only the top-N features (memory-safe DL)
CORR_THRESHOLD    = 0.95      # drop one of any feature pair above this
RANDOM_STATE      = 42

# Train / validate split sizes from the slide (367,140 / 91,783 ~= 80/20).
VALID_SIZE = 0.20

# Overfit rule from the slide: if (train Gini - valid Gini) / train Gini > 30%.
OVERFIT_GINI_DIFF = 0.30

# Snapshot aggregations applied to numeric features (drives the ~1,000 ML cols).
NUM_AGGS = ["mean", "std", "min", "max", "last", "first"]
# Categorical aggregations (after encoding to integer codes).
CAT_AGGS = ["last", "nunique", "count"]

# ----------------------------------------------------------------------------
# 4. MODEL HYPERPARAMETERS (sensible defaults; tune for the final thesis run)
# ----------------------------------------------------------------------------
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": -1,
    "min_child_samples": 40,
    "feature_fraction": 0.30,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.80,
    "colsample_bytree": 0.40,
    "min_child_weight": 8,
    "reg_lambda": 2.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",     # use "gpu_hist" on Kaggle GPU for a big speedup
}

# Deep-learning shared settings.
DL_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.30,
    "batch_size": 1024,
    "epochs": 15,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 3,            # early-stopping patience on valid AUC
    # Transformer-specific
    "n_heads": 8,
    "ff_dim": 256,
}


def banner(title: str) -> None:
    """Pretty section header for console logs."""
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}", flush=True)
