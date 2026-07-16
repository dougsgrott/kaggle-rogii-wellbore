"""Exact HMM geosteering tracker (docs/issues/004-particle-filter-tracker.md).

Model (all quantities per 1-ft MD step, measured on train prefixes):

- state: TVT offset from the dead-reckoning path center_i = anchor - cumsum(dZ),
  on a 0.5-ft grid (corr(dTVT, dZ) = -0.99 → the offset is the slowly
  drifting structure term)
- transition: offset random walk, Gaussian blur with sigma_step
- emission: GR_h[i] ~ a*GR_tw(tvt) + b + noise(sigma_gr), (a, b) fitted on
  the well's own prefix; NaN GR rows emit uniformly; robust (clipped) loss

Inference is exact forward-backward on the grid — deterministic, so there
is no seed variance to average away. Outputs posterior mean (prediction),
posterior std, and a per-row multimodality measure for the router.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..data import Well

GRID_STEP = 0.5


class HMMTracker:
    needs_fit = False

    def __init__(
        self,
        sigma_step: float = 0.2,    # TVT random-walk noise per ft of MD (ft)
        sigma_gr: float = 14.0,     # emission residual std (API units)
        half_width: float = 100.0,  # offset grid bound (ft)
        gr_smooth: int = 5,         # rolling-median window on horizontal GR
        clip_z: float = 3.0,        # robust clip on emission z-score
        center_mode: str = "drift", # "drift" | "flat" | "dz-highpass" | "dead-reckoning"
        highpass_ft: int = 200,     # rolling window for the dZ high-pass
        beta: float = 0.05,         # emission temper: GR is autocorrelated
                                    # (~15-25 ft cycles), so per-ft samples
                                    # are far from independent; beta ~ 1/L
                                    # discounts to the effective sample rate
        emission: str = "point",    # "point" | "ncc" (windowed shape match)
        ncc_window_ft: int = 60,    # MD window for NCC
        ncc_stride_ft: int = 10,    # emit every K ft (uniform in between)
        ncc_beta: float = 2.0,      # weight on Fisher-z of the NCC score
        self_test: bool = True,     # prefix self-test gate (see predict_well)
        self_test_frac: float = 0.5,
        min_prefix_rows: int = 400,
        gate_mode: str = "soft",    # "soft": w = a^2/(a^2+t^2) on the offset;
                                    # "hard": all-or-nothing at 0.9 margin
        estimator: str = "mean",    # "mean" (posterior mean) | "map" (Viterbi)
        zshape_win: int = 100,      # rolling z-score window for "zshape"
        center_model=None,          # per-well predictor supplying the center
                                    # path when center_mode == "model"
                                    # (e.g. a fitted StructurePrior)
    ):
        self.center_model = center_model
        self.estimator = estimator
        self.zshape_win = zshape_win
        self.self_test = self_test
        self.self_test_frac = self_test_frac
        self.min_prefix_rows = min_prefix_rows
        self.gate_mode = gate_mode
        self.beta = beta
        self.emission = emission
        self.ncc_window_ft = ncc_window_ft
        self.ncc_stride_ft = ncc_stride_ft
        self.ncc_beta = ncc_beta
        self.sigma_step = sigma_step
        self.sigma_gr = sigma_gr
        self.half_width = half_width
        self.gr_smooth = gr_smooth
        self.clip_z = clip_z
        self.center_mode = center_mode
        self.highpass_ft = highpass_ft
        self.last_diagnostics: dict | None = None

    # -- helpers ------------------------------------------------------------

    def _calibrate_gr(self, well: Well) -> tuple[float, float]:
        """Per-well affine map GR_tw -> GR_h fitted on the prefix.

        The prefix has known TVT (TVT_input), so typewell GR can be sampled
        at the true geological position. Median slope is ~0.79 across wells
        with 0.67-0.93 spread — well-specific calibration matters.
        """
        known = well.known_prefix
        tw = well.typewell
        v = known["GR"].notna()
        if v.sum() < 50:
            return 1.0, 0.0
        gr_h = known.loc[v, "GR"].to_numpy()
        gr_t = np.interp(known.loc[v, "TVT_input"].to_numpy(), tw["TVT"], tw["GR"])
        a, b = np.polyfit(gr_t, gr_h, 1)
        if not np.isfinite(a) or a <= 0.05:  # degenerate fit: fall back
            return 1.0, 0.0
        return float(a), float(b)

    def _smooth_gr(self, gr: np.ndarray) -> np.ndarray:
        if self.gr_smooth <= 1:
            return gr
        import pandas as pd

        return (
            pd.Series(gr).rolling(self.gr_smooth, center=True, min_periods=1).median().to_numpy()
        )

    def _ncc_loglik(
        self, well: Well, centers: np.ndarray, offsets: np.ndarray, i_anchor: int
    ) -> np.ndarray:
        """Windowed shape-match emission.

        Every `ncc_stride_ft`, correlate the horizontal GR window against the
        typewell sampled along the candidate path. Within a window the local
        TVT relief is taken from the trajectory (dTVT ~= -dZ high-frequency),
        so undulation is respected. NCC is affine-invariant per window - no
        per-well GR calibration needed. Rows between emission points stay
        uniform; the Fisher-z of the correlation enters the log-likelihood.
        """
        h = well.horizontal
        tw = well.typewell
        tw_tvt = tw["TVT"].to_numpy()
        tw_gr = tw["GR"].to_numpy()
        gr = self._smooth_gr(h["GR"].to_numpy())
        z = h["Z"].to_numpy()
        n, k = len(centers), len(offsets)
        half = self.ncc_window_ft // 2
        loglik = np.zeros((n, k))

        for i in range(half, n - half, self.ncc_stride_ft):
            gi = i_anchor + 1 + i                      # absolute row index
            lo, hi = gi - half, gi + half + 1
            gh = gr[lo:hi]
            valid = np.isfinite(gh)
            if valid.sum() < self.ncc_window_ft // 3:
                continue
            gh = gh[valid]
            gh = gh - gh.mean()
            gh_norm = np.sqrt((gh**2).sum())
            if gh_norm < 1e-6:
                continue
            # local TVT relief within the window, from trajectory undulation
            rel = -(z[lo:hi][valid] - z[gi])
            cand_tvt = (centers[i] + rel[None, :]) + offsets[:, None]  # (k, w)
            seg = np.interp(cand_tvt.ravel(), tw_tvt, tw_gr).reshape(k, -1)
            seg = seg - seg.mean(axis=1, keepdims=True)
            seg_norm = np.sqrt((seg**2).sum(axis=1))
            ncc = (seg @ gh) / np.maximum(seg_norm * gh_norm, 1e-9)
            loglik[i] = self.ncc_beta * np.arctanh(np.clip(ncc, -0.99, 0.99))
        return loglik

    def _viterbi(self, loglik: np.ndarray, sigma_cells: float, offsets: np.ndarray) -> np.ndarray:
        """MAP path (banded max-plus DP) — the DTW-style single best alignment."""
        n, k = loglik.shape
        band = max(3, int(np.ceil(4 * sigma_cells)))
        shifts = np.arange(-band, band + 1)
        cost = -(shifts.astype(float) ** 2) / (2 * max(sigma_cells, 1e-6) ** 2)
        delta = np.full(k, -np.inf)
        delta[np.argmin(np.abs(offsets))] = 0.0
        back = np.zeros((n, k), dtype=np.int8)
        for i in range(n):
            cand = np.full((len(shifts), k), -np.inf)
            for j, d in enumerate(shifts):
                if d >= 0:
                    cand[j, d:] = delta[: k - d] + cost[j] if d else delta + cost[j]
                else:
                    cand[j, :d] = delta[-d:] + cost[j]
            best = cand.argmax(axis=0)
            delta = cand[best, np.arange(k)] + loglik[i]
            back[i] = best
        path = np.empty(n, dtype=int)
        s = int(delta.argmax())
        for i in range(n - 1, -1, -1):
            path[i] = s
            s = s - shifts[back[i, s]]
        return offsets[path]

    # -- prefix self-test gate ----------------------------------------------

    def _prefix_gate(self, well: Well) -> tuple[bool, float, float]:
        """Track the well's own prefix suffix (truth known from TVT_input)
        and decide whether GR tracking can be trusted on this well.

        Returns (use_tracker, tracker_prefix_rmse, anchor_prefix_rmse).
        Fully legal at inference time: uses only TVT_input.
        """
        known = well.known_prefix
        # Mask only late-lateral prefix rows: the early prefix is the landing
        # curve (TVT moves hundreds of ft), which would swamp the comparison.
        n_mask = min(int(len(known) * self.self_test_frac), 1500)
        if len(known) < self.min_prefix_rows or n_mask < 300:
            return True, np.nan, np.nan            # too short to judge; keep tracker
        cut = len(known) - n_mask
        h2 = well.horizontal.copy()
        idx = known.index[cut:]
        truth = h2.loc[idx, "TVT_input"].to_numpy()
        h2.loc[h2.index >= idx[0], "TVT_input"] = np.nan
        pseudo = Well(well.well_id, well.split, h2, well.typewell)
        sub = self._clone(self_test=False)
        pred = sub.predict_well(pseudo)[: len(idx)]  # pseudo eval = old eval + masked prefix
        t_rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        # comparator = the actual fallback path, i.e. the center path itself
        # (w = 0 in the soft gate); identical to AnchorDrift for "drift" centers
        fallback = sub._last_centers[: len(idx)]
        a_rmse = float(np.sqrt(np.mean((fallback - truth) ** 2)))
        return t_rmse < 0.9 * a_rmse, t_rmse, a_rmse

    def _clone(self, **overrides) -> "HMMTracker":
        import inspect

        params = {
            name: getattr(self, name)
            for name in inspect.signature(HMMTracker.__init__).parameters
            if name != "self"
        }
        params.update(overrides)
        return HMMTracker(**params)

    # -- main ---------------------------------------------------------------

    def predict_well(self, well: Well) -> np.ndarray:
        h = well.horizontal
        mask = well.eval_mask
        tw = well.typewell
        anchor = well.last_known_tvt

        # Center path for the offset grid. dZ splits into high-frequency
        # undulation (maps to dTVT ~= -dZ, per-step corr -0.99) and the
        # low-frequency structural dip the well *follows* (TVT unchanged;
        # raw dead reckoning double-counts it and drifts 116-200 ft).
        z = h["Z"].to_numpy()
        i_anchor = int(np.flatnonzero(~mask)[-1])
        dz = np.diff(z)[i_anchor:]                      # one per eval row
        if self.center_mode == "dead-reckoning":
            centers = anchor - np.cumsum(dz)
        elif self.center_mode == "dz-highpass":
            import pandas as pd

            trend = pd.Series(dz).rolling(self.highpass_ft, center=True, min_periods=1).mean().to_numpy()
            centers = anchor - np.cumsum(dz - trend)
        elif self.center_mode == "drift":               # damped-drift path (baseline B)
            from .baselines import prefix_slope

            slope = prefix_slope(well, 100.0)
            dmd = h["MD"].to_numpy()[i_anchor + 1 :] - h["MD"].to_numpy()[i_anchor]
            centers = anchor + slope * 150.0 * (1.0 - np.exp(-dmd / 150.0))
        elif self.center_mode == "model":               # external prior path
            centers = np.asarray(self.center_model.predict_well(well), dtype=float)
        else:                                           # "flat"
            centers = np.full(len(dz), anchor)
        n = len(centers)
        self._last_centers = centers

        offsets = np.arange(-self.half_width, self.half_width + GRID_STEP, GRID_STEP)
        k = len(offsets)

        # emission log-likelihoods (n, k)
        a, b = 1.0, 0.0
        if self.emission == "ncc":
            loglik = self._ncc_loglik(well, centers, offsets, i_anchor)
        elif self.emission == "zshape":
            # locally standardized GR on both curves: removes baseline/scale
            # (derivative-DTW spirit); residual is in z-units, sigma_gr ~ 1
            import pandas as pd

            gh = pd.Series(self._smooth_gr(h["GR"].to_numpy()))
            r = gh.rolling(self.zshape_win, center=True, min_periods=20)
            zh = ((gh - r.mean()) / r.std().clip(lower=1e-6)).to_numpy()[i_anchor + 1 :]
            gt = pd.Series(tw["GR"].to_numpy())
            rt = gt.rolling(2 * self.zshape_win, center=True, min_periods=40)  # 0.5-ft grid
            zt = ((gt - rt.mean()) / rt.std().clip(lower=1e-6)).to_numpy()
            tvt_cand = centers[:, None] + offsets[None, :]
            zt_cand = np.interp(tvt_cand, tw["TVT"].to_numpy(), zt)
            zscore = (zh[:, None] - zt_cand) / self.sigma_gr
            np.clip(zscore, -self.clip_z, self.clip_z, out=zscore)
            loglik = -0.5 * self.beta * zscore**2
            loglik[~np.isfinite(loglik)] = 0.0          # NaN GR -> uniform
        else:
            a, b = self._calibrate_gr(well)
            gr_obs = self._smooth_gr(h["GR"].to_numpy())[i_anchor + 1 :]
            tvt_cand = centers[:, None] + offsets[None, :]
            gr_tw = a * np.interp(tvt_cand, tw["TVT"].to_numpy(), tw["GR"].to_numpy()) + b
            zscore = (gr_obs[:, None] - gr_tw) / self.sigma_gr
            np.clip(zscore, -self.clip_z, self.clip_z, out=zscore)
            loglik = -0.5 * self.beta * zscore**2
            loglik[~np.isfinite(loglik)] = 0.0          # NaN GR -> uniform

        sigma_cells = self.sigma_step / GRID_STEP

        if self.estimator == "map":
            mean_off = self._viterbi(loglik, sigma_cells, offsets)
            self.last_diagnostics = {"posterior_std": None, "gr_affine": (a, b)}
            if self.self_test:
                use_tracker, t_rmse, a_rmse = self._prefix_gate(well)
                w = 1.0
                if np.isfinite(t_rmse):
                    if self.gate_mode == "soft":
                        w = a_rmse**2 / (a_rmse**2 + t_rmse**2)
                    elif not use_tracker:
                        w = 0.0
                self.last_diagnostics.update(
                    prefix_tracker_rmse=t_rmse, prefix_anchor_rmse=a_rmse, gate_weight=w
                )
                return centers + w * mean_off
            return centers + mean_off

        # forward-backward with Gaussian-blur transitions on the offset grid
        emis = np.exp(loglik - loglik.max(axis=1, keepdims=True))

        alpha = np.zeros((n, k))
        prior = np.zeros(k)
        prior[np.argmin(np.abs(offsets))] = 1.0         # offset 0 at the anchor
        f = prior
        for i in range(n):
            f = gaussian_filter1d(f, sigma_cells, mode="constant") * emis[i]
            s = f.sum()
            f = f / s if s > 0 else np.full(k, 1.0 / k)
            alpha[i] = f

        beta = np.ones(k)
        posterior = np.empty((n, k))
        posterior[-1] = alpha[-1]
        for i in range(n - 2, -1, -1):
            beta = gaussian_filter1d(beta * emis[i + 1], sigma_cells, mode="constant")
            s = beta.sum()
            beta = beta / s if s > 0 else np.full(k, 1.0 / k)
            p = alpha[i] * beta
            s = p.sum()
            posterior[i] = p / s if s > 0 else alpha[i]

        mean_off = posterior @ offsets
        var_off = posterior @ offsets**2 - mean_off**2
        self.last_diagnostics = {
            "posterior_std": np.sqrt(np.maximum(var_off, 0.0)),
            "gr_affine": (a, b),
        }

        if self.self_test:
            use_tracker, t_rmse, a_rmse = self._prefix_gate(well)
            w = 1.0
            if np.isfinite(t_rmse):
                if self.gate_mode == "soft":
                    w = a_rmse**2 / (a_rmse**2 + t_rmse**2)
                elif not use_tracker:
                    w = 0.0
            self.last_diagnostics.update(
                prefix_tracker_rmse=t_rmse, prefix_anchor_rmse=a_rmse, gate_weight=w
            )
            return centers + w * mean_off
        return centers + mean_off

    def fit(self, wells) -> None:  # pragma: no cover - stateless
        pass
