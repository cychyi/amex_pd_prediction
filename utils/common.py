"""
Shared helper utilities used across all pipeline stages.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_paths() -> None:
    """
    Make `configs` and `utils` importable regardless of where a script is run
    from (locally or as a flat Kaggle notebook). Call this at the top of every
    stage script before importing config.
    """
    here = Path(__file__).resolve()
    root = here.parents[1]                 # repo root
    for p in (root, root / "configs", root / "utils"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest safe dtype. Critical for fitting
    the AMEX data in 16 GB of unified memory on a Mac Mini M4.
    """
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        dt = df[col].dtype
        # only touch real numeric columns; leave datetime/category/object/bool alone
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        if str(dt).startswith("category"):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(dt).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            # keep float32 - float16 loses too much precision for PD work
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"   memory: {start:6.1f} MB -> {end:6.1f} MB "
              f"({100*(start-end)/start:4.1f}% reduction)")
    return df


def compress_customer_id(s: pd.Series) -> pd.Series:
    """
    customer_ID is a 64-char hex string. Keep the last 16 hex chars and store as
    int64 - this is unique for the AMEX data and saves ~30x memory vs. the
    string, which makes groupby and merges dramatically faster.
    """
    return s.str[-16:].apply(lambda x: int(x, 16)).astype("int64")
