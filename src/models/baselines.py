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
