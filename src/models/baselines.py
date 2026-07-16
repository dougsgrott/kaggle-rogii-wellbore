"""Reference baselines (docs/issues/003-baselines.md).

AnchorLastValue is Baseline A and doubles as the harness proof for issue
002: predict the last known TVT_input for the whole evaluation zone.
"""

from __future__ import annotations

import numpy as np

from ..data import Well


class AnchorLastValue:
    needs_fit = False

    def fit(self, wells) -> None:  # pragma: no cover - stateless
        pass

    def predict_well(self, well: Well) -> np.ndarray:
        return np.full(int(well.eval_mask.sum()), well.last_known_tvt)


def prefix_slope(well: Well, window_ft: float) -> float:
    """TVT-vs-MD slope on the last `window_ft` of the known prefix.

    Restricting to the late prefix avoids importing the landing-curve slope
    (whole-prefix mean 0.39 ft/ft vs ~0.003 on the lateral, data-notes.md).
    """
    known = well.known_prefix
    tail = known[known["MD"] >= known["MD"].iloc[-1] - window_ft]
    if len(tail) < 20:
        return 0.0
    return float(np.polyfit(tail["MD"], tail["TVT_input"], 1)[0])


class AnchorDrift:
    """Baseline B: anchor + damped linear drift.

    pred(dMD) = anchor + slope * tau * (1 - exp(-dMD / tau))
    tau -> 0 recovers Baseline A; tau -> inf is undamped extrapolation.
    """

    needs_fit = False

    def __init__(self, window_ft: float = 100.0, tau: float = 150.0):
        self.window_ft = window_ft
        self.tau = tau

    def fit(self, wells) -> None:  # pragma: no cover - stateless
        pass

    def predict_well(self, well: Well) -> np.ndarray:
        h = well.horizontal
        mask = well.eval_mask
        anchor = well.last_known_tvt
        slope = prefix_slope(well, self.window_ft)
        dmd = h.loc[mask, "MD"].to_numpy() - h.loc[~mask, "MD"].iloc[-1]
        if self.tau <= 0:
            return np.full(mask.sum(), anchor)
        return anchor + slope * self.tau * (1.0 - np.exp(-dmd / self.tau))
