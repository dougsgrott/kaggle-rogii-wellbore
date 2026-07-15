"""Build submission.csv from per-well predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import Well


def build_submission(predictions: dict[str, tuple["Well", np.ndarray]], out_path: str | Path = "submission.csv") -> pd.DataFrame:
    """predictions: well_id -> (well, tvt array over that well's eval zone)."""
    frames = []
    for well_id, (well, tvt) in predictions.items():
        ids = well.eval_ids
        if len(ids) != len(tvt):
            raise ValueError(f"{well_id}: {len(tvt)} predictions for {len(ids)} eval rows")
        frames.append(pd.DataFrame({"id": ids, "tvt": np.asarray(tvt, dtype=float)}))
    sub = pd.concat(frames, ignore_index=True)
    sub.to_csv(out_path, index=False)
    return sub
