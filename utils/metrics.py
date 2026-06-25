"""
Evaluation metrics for the PD thesis, implemented exactly as described in the
proposal slides (Evaluation Metrics section):

  - ROC-AUC
  - Kolmogorov-Smirnov (KS) statistic
  - Gini coefficient  = 2 * AUC - 1
  - Log-loss
  - 10-decile binning to check monotonic risk-ranking
  - Train-vs-Validate Gini difference for the overfit rule (> 30% => overfit)

These functions take y_true (0/1) and y_pred (predicted PD in [0, 1]) and are
reused by every modelling script so ML and DL models are scored identically.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def gini_from_auc(auc: float) -> float:
    """Gini = 2 * AUC - 1  (slide formula)."""
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of goods (y=0) and bads (y=1) across the score range.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    total_bad = df["y"].sum()
    total_good = (1 - df["y"]).sum()
    cum_bad = (df["y"].cumsum() / total_bad).to_numpy()
    cum_good = ((1 - df["y"]).cumsum() / total_good).to_numpy()
    return float(np.max(np.abs(cum_bad - cum_good)))


def decile_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Bin customers into deciles by predicted PD (highest risk = decile 1) and
    tabulate the actual default rate per bin. Used to confirm that the model is
    risk-ranked (default rate should fall monotonically from decile 1 -> 10).
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    # qcut on rank avoids errors when many predictions tie.
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                           q=n_bins, labels=range(1, n_bins + 1))
    grp = df.groupby("decile", observed=True)
    out = pd.DataFrame({
        "n": grp.size(),
        "n_bad": grp["y"].sum(),
        "default_rate": grp["y"].mean(),
        "avg_pred_pd": grp["p"].mean(),
    }).reset_index()
    out["cum_bad_rate"] = out["n_bad"].cumsum() / out["n_bad"].sum()
    return out


def is_risk_ranked(dtable: pd.DataFrame) -> bool:
    """True if actual default_rate is (weakly) monotonically decreasing across deciles."""
    rates = dtable.sort_values("decile")["default_rate"].to_numpy()
    return bool(np.all(np.diff(rates) <= 1e-9))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the full metric dict for one dataset (train OR validate)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    auc = roc_auc_score(y_true, y_pred)
    dtable = decile_table(y_true, y_pred)
    return {
        "ROC_AUC": auc,
        "KS": ks_statistic(y_true, y_pred),
        "Gini": gini_from_auc(auc),
        "LogLoss": log_loss(y_true, y_pred),
        "RiskRanked": is_risk_ranked(dtable),
        "_decile_table": dtable,
    }


def overfit_check(train_gini: float, valid_gini: float, threshold: float = 0.30) -> tuple[float, bool]:
    """
    Slide rule: if relative Gini drop from train to validate exceeds the
    threshold (default 30%), flag the model as overfitted.

    Returns (relative_diff, is_overfit).
    """
    if train_gini == 0:
        return 0.0, False
    rel_diff = (train_gini - valid_gini) / abs(train_gini)
    return rel_diff, bool(rel_diff > threshold)


def metrics_row(name: str, train_eval: dict, valid_eval: dict) -> dict:
    """Build one row for the comparison table across all models."""
    rel_diff, overfit = overfit_check(train_eval["Gini"], valid_eval["Gini"])
    return {
        "Model": name,
        "Train_AUC": round(train_eval["ROC_AUC"], 5),
        "Valid_AUC": round(valid_eval["ROC_AUC"], 5),
        "Train_KS": round(train_eval["KS"], 5),
        "Valid_KS": round(valid_eval["KS"], 5),
        "Train_Gini": round(train_eval["Gini"], 5),
        "Valid_Gini": round(valid_eval["Gini"], 5),
        "Train_LogLoss": round(train_eval["LogLoss"], 5),
        "Valid_LogLoss": round(valid_eval["LogLoss"], 5),
        "DiffGini_%": round(100 * rel_diff, 2),
        "Overfit": overfit,
        "Valid_RiskRanked": valid_eval["RiskRanked"],
    }
