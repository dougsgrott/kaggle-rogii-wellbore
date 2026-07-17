"""Residual GBM stage on tracker output (docs/issues/008-residual-stack-ensemble.md).

Issue 003's naive residual-lgbm (target = TVT - anchor, no tracker/alignment
signal) was a documented negative (CV 16.30, worse than the anchor):
without cross-well GR-alignment features a tabular stage fits per-well
noise that does not generalize. This version predicts the residual of
`hmm-structure` instead of the anchor, and its features are exactly the
validated cross-well signals from issues 006/007 (uncertainty score
Spearman 0.45) plus local GR-window stats.

Training data: after fitting the base model on the fold's wells, replay
`predict_well` on those same wells to get pseudo-OOF residuals. This is
leak-free for each well's own eval-zone TVT: the tracker only ever reads
a well's own known prefix, and the structure prior excludes a well (via
typewell-hash match) from acting as its own neighbor — the same procedure
that will run at real inference time on unseen test wells.

Feature-building during fit() runs serially and is deliberately *not*
parallelized:
- fork()ing a new process pool here is unsafe: this class is constructed
  inside eval.py's per-fold ThreadPool (used to parallelize well-loading),
  and fork()ing while other threads are alive risks inheriting a lock in
  a held state — the child then hangs forever the first time it touches
  that lock (observed directly: workers parked in futex_wait with ~0 CPU
  time over 2h+ wall time).
- a ThreadPool was tried instead (no fork, so safe) but measured *slower*
  than serial: the tracker's forward-backward loop is memory-bandwidth-
  bound (already documented in eval.py) and additionally does thousands
  of small scipy calls per well, so GIL handoff overhead compounds with
  bandwidth contention rather than offsetting it.
Cost is instead bounded by `gbm_train_wells`: measured ~0.8s/well serial
at full-scale (773-well) prior-pool size, so a fold subsamples down to
that many wells for residual-training-row construction rather than
replaying all ~600 fold wells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Well
from .structure import HMMStructure

FEATURES = [
    "dmd",
    "eval_frac_pos",
    "z_delta",
    "gr",
    "gr_roll_mean",
    "gr_roll_std",
    "offset_mag",
    "posterior_std",
    "gate_weight",
    "prefix_tracker_rmse",
    "prefix_center_rmse",
    "n_calibrated_neighbors",
    "neighbor_dist_mean",
    "alpha_mean",
    "prefix_len",
    "eval_frac",
]


def _row_features(well: Well, pred: np.ndarray, model: HMMStructure) -> pd.DataFrame:
    h = well.horizontal
    mask = well.eval_mask
    known = well.known_prefix
    md_anchor = known["MD"].iloc[-1]
    z_anchor = known["Z"].iloc[-1]

    gr_filled = h["GR"].ffill()
    roll = gr_filled.rolling(100, min_periods=10)

    ev = h.loc[mask]
    dmd = ev["MD"].to_numpy() - md_anchor
    n = len(dmd)

    d = model.tracker.last_diagnostics or {}
    pd_ = model.prior.last_diagnostics or {}
    centers = getattr(model.tracker, "_last_centers", None)
    offset_mag = np.abs(pred - centers) if centers is not None else np.zeros(n)
    post = d.get("posterior_std")
    post = np.asarray(post) if post is not None else np.zeros(n)

    df = pd.DataFrame(
        {
            "dmd": dmd,
            "eval_frac_pos": dmd / max(dmd[-1], 1.0),
            "z_delta": ev["Z"].to_numpy() - z_anchor,
            "gr": ev["GR"].to_numpy(),
            "gr_roll_mean": roll.mean()[mask].to_numpy(),
            "gr_roll_std": roll.std()[mask].to_numpy(),
            "offset_mag": offset_mag,
            "posterior_std": post,
            "gate_weight": d.get("gate_weight", np.nan),
            "prefix_tracker_rmse": d.get("prefix_tracker_rmse", np.nan),
            "prefix_center_rmse": d.get("prefix_anchor_rmse", np.nan),
            "n_calibrated_neighbors": pd_.get("n_calibrated_neighbors", 0),
            "neighbor_dist_mean": pd_.get("mean_nearest_dist", np.nan),
            "alpha_mean": pd_.get("mean_alpha", np.nan),
            "prefix_len": len(known),
            "eval_frac": mask.mean(),
        }
    )
    return df[FEATURES]


class ResidualStack:
    needs_fit = True

    def __init__(
        self,
        seed: int = 0,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 500,
        gbm_train_wells: int = 150,
        **structure_overrides,
    ):
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.gbm_train_wells = gbm_train_wells
        self.model = HMMStructure(**structure_overrides)
        self.gbm = None
        self.last_diagnostics: dict | None = None

    def fit(self, wells) -> None:
        import lightgbm as lgb

        wells = list(wells)
        self.model.fit(wells)

        rng = np.random.default_rng(self.seed)
        train_wells = (
            wells
            if len(wells) <= self.gbm_train_wells
            else [wells[i] for i in rng.choice(len(wells), self.gbm_train_wells, replace=False)]
        )

        feats, resids = [], []
        for well in train_wells:
            test_well = well.as_test()
            pred = self.model.predict_well(test_well)
            feats.append(_row_features(test_well, pred, self.model))
            resids.append(well.eval_target - pred)

        X = pd.concat(feats, ignore_index=True)
        resid = np.concatenate(resids)
        self.gbm = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
            # LightGBM's default n_jobs=-1 spins up a persistent OpenMP
            # thread pool that outlives fit(). eval.py's outer scoring
            # stage fork()s a process pool right after model.fit() returns
            # for each CV fold; fork()ing while those threads are still
            # alive risks the same class of hang diagnosed earlier for our
            # own ThreadPool (a lock duplicated in a held state in the
            # child). Single-threaded here is fine — the outer --jobs
            # process pool already parallelizes across wells.
            n_jobs=1,
        )
        self.gbm.fit(X, resid)

    def predict_well(self, well: Well) -> np.ndarray:
        pred = self.model.predict_well(well)
        X = _row_features(well, pred, self.model)
        correction = self.gbm.predict(X)
        self.last_diagnostics = {"mean_correction": float(np.mean(correction))}
        return pred + correction
