"""Per-well router over model paths (docs/issues/007-well-router-uncertainty.md).

OOF accounting on hmm-structure-v1 shows the tail failure mode is a
committed wrong datum (worst-30 |bias|/RMSE ~ 0.88) and that a per-well
oracle over the four existing paths scores 9.79 vs 11.94 — routing
headroom, not new-model headroom.

RouterBlend generalizes the proven pairwise inverse-MSE prefix gate to
n candidates: one late-prefix replay scores every candidate path where
the truth is known, and the eval-zone prediction is the soft-weighted
average of the candidate paths. On wells the replay cannot separate,
weights spread and the blend lands between the modes — the RMSE-optimal
midpoint policy on genuinely ambiguous wells (topics/711878.md).

Candidates run without their internal pairwise gates: the router is the
gate.
"""

from __future__ import annotations

import numpy as np

from ..data import Well
from .baselines import AnchorDrift
from .structure import StructurePrior
from .tracker import HMMTracker

PATH_NAMES = ("drift", "struct", "hmm_drift", "hmm_struct")


class RouterBlend:
    needs_fit = True

    def __init__(
        self,
        power: float = 1.0,        # weight = 1 / (replay_mse + eps)^power
        eps: float = 1.0,          # MSE floor (ft^2): guards near-zero replay MSE
        self_test_frac: float = 0.5,
        min_prefix_rows: int = 400,
        max_mask_rows: int = 1500,
    ):
        self.power = power
        self.eps = eps
        self.self_test_frac = self_test_frac
        self.min_prefix_rows = min_prefix_rows
        self.max_mask_rows = max_mask_rows
        self.prior = StructurePrior(self_test=False)
        self.drift = AnchorDrift()
        self.hmm_drift = HMMTracker(self_test=False)
        self.hmm_struct = HMMTracker(
            center_mode="model", center_model=self.prior, self_test=False
        )
        self.last_diagnostics: dict | None = None

    def fit(self, wells) -> None:
        self.prior.fit(wells)

    # -- candidate paths -------------------------------------------------------

    def _paths(self, well: Well) -> np.ndarray:
        """(4, n_eval) candidate predictions, PATH_NAMES order."""
        return np.vstack(
            [
                self.drift.predict_well(well),
                self.prior.predict_well(well),
                self.hmm_drift.predict_well(well),
                self.hmm_struct.predict_well(well),
            ]
        )

    # -- replay weights ----------------------------------------------------------

    def _replay(self, well: Well) -> tuple[np.ndarray, np.ndarray]:
        """Late-prefix replay -> (weights, per-candidate replay MSEs).

        Falls back to routing everything to hmm_struct (the model of
        record) when the prefix is too short to replay.
        """
        known = well.known_prefix
        n_mask = min(int(len(known) * self.self_test_frac), self.max_mask_rows)
        default = np.array([0.0, 0.0, 0.0, 1.0])
        if len(known) < self.min_prefix_rows or n_mask < 300:
            return default, np.full(len(PATH_NAMES), np.nan)
        cut = len(known) - n_mask
        h2 = well.horizontal.copy()
        idx = known.index[cut:]
        truth = h2.loc[idx, "TVT_input"].to_numpy()
        h2.loc[h2.index >= idx[0], "TVT_input"] = np.nan
        pseudo = Well(well.well_id, well.split, h2, well.typewell)
        paths = self._paths(pseudo)[:, : len(idx)]
        mse = ((paths - truth) ** 2).mean(axis=1)
        w = 1.0 / (mse + self.eps) ** self.power
        return w / w.sum(), mse

    # -- main ---------------------------------------------------------------------

    def predict_well(self, well: Well) -> np.ndarray:
        paths = self._paths(well)
        w, mse = self._replay(well)
        pred = w @ paths

        disagreement = float(paths.std(axis=0).mean())
        entropy = float(-(w[w > 0] * np.log(w[w > 0])).sum())
        self.last_diagnostics = {
            **{f"replay_mse_{n}": float(m) for n, m in zip(PATH_NAMES, mse)},
            **{f"weight_{n}": float(x) for n, x in zip(PATH_NAMES, w)},
            "weight_entropy": entropy,
            "path_disagreement": disagreement,
            "max_dev_from_anchor": float(np.abs(pred - well.last_known_tvt).max()),
            **{f"prior_{k}": v for k, v in (self.prior.last_diagnostics or {}).items()},
        }
        return pred
