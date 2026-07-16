"""Average of decorrelated tracker variants (docs/issues/005-dtw-alignment.md).

Members run independently (each with its own prefix self-test gate); the
blend is their unweighted mean, with pairwise disagreement recorded per well
as the multimodality signal for the router (issue 007).
"""

from __future__ import annotations

import numpy as np

from ..data import Well
from .tracker import HMMTracker


def default_members() -> list[HMMTracker]:
    return [
        HMMTracker(),                                            # posterior mean
        HMMTracker(estimator="map"),                             # DTW-style MAP path
        HMMTracker(emission="zshape", sigma_gr=1.0, beta=0.02),  # shape cost
    ]


class TrackerBlend:
    needs_fit = False

    def __init__(self, members=None):
        self.members = members if members is not None else default_members()
        self.last_diagnostics: dict | None = None

    def fit(self, wells) -> None:  # pragma: no cover - stateless
        pass

    def predict_well(self, well: Well) -> np.ndarray:
        preds = np.stack([m.predict_well(well) for m in self.members])
        dis = float(np.mean(np.std(preds, axis=0)))
        self.last_diagnostics = {"member_spread_ft": dis}
        return preds.mean(axis=0)
