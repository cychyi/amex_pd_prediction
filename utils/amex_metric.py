"""
Official American Express default-prediction competition metric.

  M = 0.5 * (G + D)
    G = normalized Gini coefficient
    D = default rate captured at the top 4% of predictions
  (Bad rows are weighted x1, good rows x20 to mimic the 5% subsampling
   of negatives in the competition test set.)

Included as an OPTIONAL extra metric. The thesis primarily reports ROC-AUC,
KS, Gini and log-loss per the slides, but reporting the competition metric as
well lets you benchmark directly against the public Kaggle leaderboard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"target": np.asarray(y_true), "prediction": np.asarray(y_pred)})

    def _top_four_percent(d: pd.DataFrame) -> float:
        d = d.sort_values("prediction", ascending=False)
        d["weight"] = d["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * d["weight"].sum())
        d["cum_weight"] = d["weight"].cumsum()
        within = d[d["cum_weight"] <= cutoff]
        return within["target"].sum() / d["target"].sum()

    def _weighted_gini(d: pd.DataFrame) -> float:
        d = d.sort_values("prediction", ascending=False)
        d["weight"] = d["target"].apply(lambda x: 20 if x == 0 else 1)
        rand = (d["weight"] / d["weight"].sum()).cumsum()
        total_pos = (d["target"] * d["weight"]).sum()
        d["cum_pos"] = (d["target"] * d["weight"]).cumsum() / total_pos
        return (d["cum_pos"] * d["weight"]).sum() - (rand * d["weight"]).sum() * 0.5

    def _normalized_gini(d: pd.DataFrame) -> float:
        pred = _weighted_gini(d)
        perfect = _weighted_gini(d.assign(prediction=d["target"]))
        return pred / perfect

    g = _normalized_gini(df)
    d = _top_four_percent(df)
    return 0.5 * (g + d)
