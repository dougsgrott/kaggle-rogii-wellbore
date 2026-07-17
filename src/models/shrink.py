"""Uncertainty-gated shrink blend (docs/issues/008-residual-stack-ensemble.md).

hmm-structure already gates the GR offset against its own center path
(inverse-replay-MSE, `gate_weight`) and the structure prior already decays
toward damped drift by neighbor distance (`mean_alpha`). Both are per-well
scalars computed from a single prefix probe. This model asks whether a
*second*, coarser shrink toward the plain damped-drift path — driven by
the prior's own decay state and the tracker's posterior spread, issue
007's validated uncertainty signals (Spearman 0.45 vs realized error) —
buys anything on top of those two existing gates.
"""

from __future__ import annotations

import numpy as np

from ..data import Well
from .baselines import AnchorDrift
from .structure import HMMStructure


class ShrinkBlend:
    needs_fit = True

    def __init__(
        self,
        k_alpha: float = 0.0,       # weight on (1 - prior alpha_mean)
        k_dist: float = 0.0,        # weight on log1p(neighbor_dist_mean / d_ref)
        k_post: float = 0.0,        # weight on posterior_std_mean / s_ref
        d_ref: float = 1000.0,
        s_ref: float = 5.0,
        x0: float = 1.0,            # logistic midpoint
        slope: float = 2.0,         # logistic steepness
        max_shrink: float = 0.6,    # cap on the extra shrink toward drift
        **structure_overrides,
    ):
        self.k_alpha = k_alpha
        self.k_dist = k_dist
        self.k_post = k_post
        self.d_ref = d_ref
        self.s_ref = s_ref
        self.x0 = x0
        self.slope = slope
        self.max_shrink = max_shrink
        self.model = HMMStructure(**structure_overrides)
        self.drift = AnchorDrift()
        self.last_diagnostics: dict | None = None

    def fit(self, wells) -> None:
        self.model.fit(wells)

    def _score(self) -> float:
        d = self.model.tracker.last_diagnostics or {}
        pd_ = self.model.prior.last_diagnostics or {}
        alpha = pd_.get("mean_alpha", 1.0)
        dist = pd_.get("mean_nearest_dist", 0.0)
        post = d.get("posterior_std")
        post_mean = float(np.mean(post)) if post is not None else 0.0
        x = (
            self.k_alpha * (1.0 - alpha)
            + self.k_dist * np.log1p(dist / self.d_ref)
            + self.k_post * (post_mean / self.s_ref)
        )
        return 1.0 / (1.0 + np.exp(-self.slope * (x - self.x0)))

    def predict_well(self, well: Well) -> np.ndarray:
        pred = self.model.predict_well(well)
        shrink = self.max_shrink * self._score()
        if shrink <= 1e-9:
            self.last_diagnostics = {"shrink": 0.0}
            return pred
        fallback = self.drift.predict_well(well)
        self.last_diagnostics = {"shrink": float(shrink)}
        return (1.0 - shrink) * pred + shrink * fallback
