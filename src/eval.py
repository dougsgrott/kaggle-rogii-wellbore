"""Local scoring: per-well RMSE and tail decomposition (issue 002 grows here)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def per_well_report(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """results: well_id -> (y_true, y_pred) over that well's eval zone.

    Returns one row per well (worst first) plus the global RMSE and each
    well's share of total squared error — the tail-decomposition view.
    """
    rows = []
    for well_id, (y_true, y_pred) in results.items():
        err2 = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
        rows.append({"well_id": well_id, "n": len(err2), "sse": err2.sum(), "rmse": np.sqrt(err2.mean())})
    df = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    df["share_of_sse"] = df["sse"] / df["sse"].sum()
    df["cum_share"] = df["share_of_sse"].cumsum()
    df.attrs["global_rmse"] = float(np.sqrt(df["sse"].sum() / df["n"].sum()))
    return df
