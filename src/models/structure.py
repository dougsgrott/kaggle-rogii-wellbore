"""Spatial datum-surface prior (docs/issues/006-structural-priors.md).

The verified structural object of this dataset is the datum field
s(X, Y) = TVT + Z: formation columns are exactly TVT + Z - const
(residual < 0.01 ft), so every train-well row is a dense sample of the
one structural surface. s varies ~223 ft along an average well — the
low-frequency dip the damped-drift center cannot see.

The datum is not globally consistent: 25% of nearby well pairs are
shifted > 30 ft (typewell TVT-origin offsets and sub-seismic faults).
Neighbors are therefore offset-calibrated against the test well's own
prefix datum before their shape is transferred, and the aggregate decays
toward the damped-drift path where no neighbor is close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..data import Well
from .baselines import prefix_slope


class StructurePrior:
    needs_fit = True

    def __init__(
        self,
        sample_step: int = 25,          # ft between datum samples per train well
        radius: float = 4000.0,         # max XY distance a sample may contribute from
        calib_k: int = 48,              # neighbor samples per prefix point (calibration)
        calib_radius: float = 1500.0,   # max XY distance for offset calibration matches
        calib_subsample: int = 10,      # ft between prefix calibration points
        min_matches: int = 8,           # matches needed to trust a neighbor's offset
        calib_window: float = 0.0,      # calibrate only within this MD distance of
                                        # the anchor (0 = whole prefix)
        match_n0: float = 20.0,         # match-count weight saturation
        mad0: float = 15.0,             # calibration-MAD penalty scale (ft)
        d0: float = 150.0,              # softening of the 1/d^2 weights (ft)
        tau_dist: float = 600.0,        # struct->drift decay scale (ft to nearest
                                        # calibrated sample; LOO optimum 500-600,
                                        # flat minimum)
        smooth_ft: int = 101,           # rolling-mean window on the datum path
        anchor_tau: float = 200.0,      # damped continuity correction at the anchor (ft)
        agg: str = "median",            # "median" (weighted; robust to one bad
                                        # neighbor) | "mean"
        drift_window_ft: float = 100.0, # fallback damped-drift parameters (baseline B)
        drift_tau: float = 150.0,
        self_test: bool = True,         # prefix self-test soft gate vs drift
                                        # (same construction as the tracker's)
        self_test_frac: float = 0.5,
        min_prefix_rows: int = 400,
    ):
        self.self_test = self_test
        self.self_test_frac = self_test_frac
        self.min_prefix_rows = min_prefix_rows
        self.sample_step = sample_step
        self.radius = radius
        self.calib_k = calib_k
        self.calib_radius = calib_radius
        self.calib_subsample = calib_subsample
        self.min_matches = min_matches
        self.calib_window = calib_window
        self.match_n0 = match_n0
        self.mad0 = mad0
        self.d0 = d0
        self.tau_dist = tau_dist
        self.smooth_ft = smooth_ft
        self.anchor_tau = anchor_tau
        self.agg = agg
        self.drift_window_ft = drift_window_ft
        self.drift_tau = drift_tau
        self._tree: cKDTree | None = None
        self._s: np.ndarray | None = None      # datum per sample
        self._wid: np.ndarray | None = None    # source-well code per sample
        self._tw_hash: dict[int, str] = {}     # source-well code -> typewell hash
        self.last_diagnostics: dict | None = None

    # -- fit ------------------------------------------------------------------

    @staticmethod
    def _typewell_hash(well: Well) -> str:
        import hashlib

        arr = np.ascontiguousarray(well.typewell[["TVT", "GR"]].to_numpy(float))
        return hashlib.md5(arr.tobytes()).hexdigest()

    def fit(self, wells) -> None:
        xs, ys, ss, ws = [], [], [], []
        for code, well in enumerate(wells):
            h = well.horizontal
            if "TVT" not in h.columns:
                continue
            sub = h.iloc[:: self.sample_step]
            v = sub["TVT"].notna() & sub["Z"].notna()
            xs.append(sub.loc[v, "X"].to_numpy())
            ys.append(sub.loc[v, "Y"].to_numpy())
            ss.append((sub.loc[v, "TVT"] + sub.loc[v, "Z"]).to_numpy())
            ws.append(np.full(int(v.sum()), code))
            self._tw_hash[code] = self._typewell_hash(well)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        self._s = np.concatenate(ss)
        self._wid = np.concatenate(ws)
        self._tree = cKDTree(np.column_stack([x, y]))

    # -- helpers ---------------------------------------------------------------

    def _neighbor_offsets(
        self, pxy: np.ndarray, s_own: np.ndarray, excluded: set[int]
    ) -> dict[int, tuple[float, int, float]]:
        """Per-neighbor-well datum offset over prefix-adjacent matches.

        Returns {well_code: (delta = median(s_own - s_w), n_matches, mad)}.
        """
        step = max(self.calib_subsample, 1)
        pxy = pxy[::step]
        s_own = s_own[::step]
        d, idx = self._tree.query(pxy, k=min(self.calib_k, len(self._s)))
        diffs: dict[int, list[float]] = {}
        for j in range(len(pxy)):
            ok = d[j] <= self.calib_radius
            for sample in idx[j][ok]:
                w = int(self._wid[sample])
                if w in excluded:
                    continue
                diffs.setdefault(w, []).append(s_own[j] - self._s[sample])
        out = {}
        for w, v in diffs.items():
            if len(v) < self.min_matches:
                continue
            med = float(np.median(v))
            mad = float(np.median(np.abs(np.asarray(v) - med)))
            out[w] = (med, len(v), mad)
        return out

    def _drift_path(self, well: Well) -> np.ndarray:
        h = well.horizontal
        mask = well.eval_mask
        anchor = well.last_known_tvt
        slope = prefix_slope(well, self.drift_window_ft)
        dmd = h.loc[mask, "MD"].to_numpy() - h.loc[~mask, "MD"].iloc[-1]
        return anchor + slope * self.drift_tau * (1.0 - np.exp(-dmd / self.drift_tau))

    # -- predict ----------------------------------------------------------------

    def _structure_path(self, well: Well) -> tuple[np.ndarray, dict]:
        h = well.horizontal
        mask = well.eval_mask
        known = well.known_prefix
        exy = h.loc[mask, ["X", "Y"]].to_numpy()
        ez = h.loc[mask, "Z"].to_numpy()
        emd = h.loc[mask, "MD"].to_numpy()
        n = len(exy)

        v = known["TVT_input"].notna() & known["Z"].notna()
        pxy = known.loc[v, ["X", "Y"]].to_numpy()
        s_own = (known.loc[v, "TVT_input"] + known.loc[v, "Z"]).to_numpy()
        pmd = known.loc[v, "MD"].to_numpy()
        if self.calib_window > 0:
            late = pmd >= pmd[-1] - self.calib_window
            cal_xy, cal_s = pxy[late], s_own[late]
        else:
            cal_xy, cal_s = pxy, s_own

        # The target well never borrows from itself or its typewell mates
        # (matters for LOO tuning and for the example test wells, which are
        # byte-copies of train wells).
        own_hash = self._typewell_hash(well)
        excluded = {c for c, h_ in self._tw_hash.items() if h_ == own_hash}

        deltas = self._neighbor_offsets(cal_xy, cal_s, excluded)
        drift = self._drift_path(well)

        # Only prefix-calibrated neighbors contribute: a neighbor's datum is
        # offset-free relative to the test well only through its delta_w, so
        # uncalibrated wells (unknown TVT origin, possible faults) are unusable.
        per_well_s, per_well_w = [], []
        d_cal = np.full(n, np.inf)
        for w, (delta, n_matches, mad) in deltas.items():
            m = self._wid == w
            tree_w = cKDTree(np.column_stack([self._tree.data[m, 0], self._tree.data[m, 1]]))
            d_w, i_w = tree_w.query(exy)
            s_w = self._s[m][i_w] + delta
            quality = (n_matches / (n_matches + self.match_n0)) / (
                1.0 + (mad / self.mad0) ** 2
            )
            wt = quality / (d_w**2 + self.d0**2)
            wt[d_w > self.radius] = 0.0
            per_well_s.append(s_w)
            per_well_w.append(wt)
            d_cal = np.minimum(d_cal, d_w)

        # The well's own prefix datum is a neighbor with delta = 0.
        step = max(self.calib_subsample, 1)
        own_tree = cKDTree(pxy[::step])
        s_own_sub = s_own[::step]
        d_o, i_o = own_tree.query(exy)
        w_o = 1.0 / (d_o**2 + self.d0**2)
        per_well_s.append(s_own_sub[i_o])
        per_well_w.append(w_o)
        d_cal = np.minimum(d_cal, d_o)

        s_all = np.column_stack(per_well_s)   # (n, wells)
        w_all = np.column_stack(per_well_w)
        if self.agg == "median":
            order = np.argsort(s_all, axis=1)
            s_sorted = np.take_along_axis(s_all, order, axis=1)
            w_sorted = np.take_along_axis(w_all, order, axis=1)
            cw = np.cumsum(w_sorted, axis=1)
            half = cw[:, -1:] / 2.0
            pick = (cw >= half).argmax(axis=1)
            s_hat = s_sorted[np.arange(len(s_all)), pick]
        else:
            s_hat = (s_all * w_all).sum(axis=1) / np.maximum(w_all.sum(axis=1), 1e-12)

        if self.smooth_ft > 1:
            s_hat = (
                pd.Series(s_hat)
                .rolling(self.smooth_ft, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
        pred = s_hat - ez

        # decay toward the damped-drift path where no calibrated sample is close
        alpha = 1.0 / (1.0 + (d_cal / self.tau_dist) ** 2)
        pred = alpha * pred + (1.0 - alpha) * drift

        diag = {
            "n_calibrated_neighbors": len(deltas),
            "mean_nearest_dist": float(np.where(np.isfinite(d_cal), d_cal, 1e9).mean()),
            "mean_alpha": float(alpha.mean()),
        }
        return pred, diag

    def predict_well(self, well: Well) -> np.ndarray:
        pred, diag = self._structure_path(well)
        drift = self._drift_path(well)

        # Prefix self-test soft gate (issue 004's construction): replay the
        # prediction with the late prefix masked, score both paths on the
        # masked segment (truth known from TVT_input), soft-blend by
        # inverse MSE. Catches wells whose neighbor transfer fails.
        w = 1.0
        t_rmse = a_rmse = float("nan")
        if self.self_test:
            known = well.known_prefix
            n_mask = min(int(len(known) * self.self_test_frac), 1500)
            if len(known) >= self.min_prefix_rows and n_mask >= 300:
                cut = len(known) - n_mask
                h2 = well.horizontal.copy()
                idx = known.index[cut:]
                truth = h2.loc[idx, "TVT_input"].to_numpy()
                h2.loc[h2.index >= idx[0], "TVT_input"] = np.nan
                pseudo = Well(well.well_id, well.split, h2, well.typewell)
                p_st = self._structure_path(pseudo)[0][: len(idx)]
                p_dr = self._drift_path(pseudo)[: len(idx)]
                t_rmse = float(np.sqrt(np.mean((p_st - truth) ** 2)))
                a_rmse = float(np.sqrt(np.mean((p_dr - truth) ** 2)))
                w = a_rmse**2 / (a_rmse**2 + t_rmse**2)
        pred = w * pred + (1.0 - w) * drift

        # damped continuity correction at the anchor
        if self.anchor_tau > 0 and len(pred):
            emd = well.horizontal.loc[well.eval_mask, "MD"].to_numpy()
            anchor = well.last_known_tvt
            pred = pred + (anchor - pred[0]) * np.exp(-(emd - emd[0]) / self.anchor_tau)

        diag.update(prefix_struct_rmse=t_rmse, prefix_drift_rmse=a_rmse, gate_weight=w)
        self.last_diagnostics = diag
        return pred


class HMMStructure:
    """HMM tracker running on the structure prior's center path."""

    needs_fit = True

    def __init__(self, **tracker_overrides):
        from .tracker import HMMTracker

        self.prior = StructurePrior()
        self.tracker = HMMTracker(
            center_mode="model", center_model=self.prior, **tracker_overrides
        )
        self.last_diagnostics: dict | None = None

    def fit(self, wells) -> None:
        self.prior.fit(wells)

    def predict_well(self, well: Well) -> np.ndarray:
        pred = self.tracker.predict_well(well)
        self.last_diagnostics = self.tracker.last_diagnostics
        return pred
