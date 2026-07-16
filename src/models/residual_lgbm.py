"""Baseline C: anchor + residual LightGBM (docs/issues/003-baselines.md).

Target is ``TVT - anchor`` per eval row; features use only the test schema
(MD, X, Y, Z, GR, TVT_input) so train and inference paths are identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Well
from .baselines import prefix_slope

FEATURES = [
    "dmd",
    "z_delta",
    "slope_100",
    "slope_250",
    "slope_1000",
    "drift_150",
    "gr",
    "gr_roll_mean",
    "gr_roll_std",
    "prefix_len",
    "eval_frac",
]


def build_features(test_well: Well) -> pd.DataFrame:
    """Per-eval-row features from a test-schema well."""
    h = test_well.horizontal
    mask = test_well.eval_mask
    known = test_well.known_prefix
    md_anchor = known["MD"].iloc[-1]
    z_anchor = known["Z"].iloc[-1]

    gr_filled = h["GR"].ffill()
    roll = gr_filled.rolling(100, min_periods=10)

    df = pd.DataFrame(index=np.flatnonzero(mask))
    ev = h.loc[mask]
    df["dmd"] = ev["MD"].to_numpy() - md_anchor
    df["z_delta"] = ev["Z"].to_numpy() - z_anchor
    for w in (100, 250, 1000):
        df[f"slope_{w}"] = prefix_slope(test_well, w)
    df["drift_150"] = df["slope_100"] * 150.0 * (1.0 - np.exp(-df["dmd"] / 150.0))
    df["gr"] = ev["GR"].to_numpy()
    df["gr_roll_mean"] = roll.mean()[mask].to_numpy()
    df["gr_roll_std"] = roll.std()[mask].to_numpy()
    df["prefix_len"] = len(known)
    df["eval_frac"] = mask.mean()
    return df[FEATURES]


class ResidualLGBM:
    needs_fit = True

    def __init__(self, seed: int = 0, n_estimators: int = 400):
        self.seed = seed
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, wells) -> None:
        import lightgbm as lgb

        xs, ys = [], []
        for well in wells:
            x = build_features(well.as_test())
            y = well.eval_target - well.last_known_tvt
            xs.append(x)
            ys.append(y)
        X = pd.concat(xs, ignore_index=True)
        y = np.concatenate(ys)
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
        )
        self.model.fit(X, y)

    def predict_well(self, well: Well) -> np.ndarray:
        X = build_features(well)
        return well.last_known_tvt + self.model.predict(X)
