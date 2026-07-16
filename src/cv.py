"""Fold assignment for cross-validation.

Two grouping modes (docs/issues/002-cv-harness.md):

- ``well``: random well-level folds — the everyday CV. Wells that share a
  typewell are forced into the same fold (13 typewells are shared by 34
  wells; splitting them would leak the reference log's identity).
- ``spatial``: k-means regions on mean (X, Y) — leave-one-region-out.
  Wells share one global coordinate frame (~180k ft span, see
  docs/data-notes.md), so regions are geologically meaningful. Use this to
  detect features that only work near their training wells.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Split, data_dir, list_wells

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _REPO_ROOT / "analysis"


def well_xy(split: Split = "train") -> pd.DataFrame:
    """Per-well mean X/Y/Z, cached to analysis/well_xy.csv (train only)."""
    cache = _CACHE_DIR / "well_xy.csv"
    if split == "train" and cache.exists():
        return pd.read_csv(cache)
    rows = []
    for wid in list_wells(split):
        h = pd.read_csv(
            data_dir() / split / f"{wid}__horizontal_well.csv", usecols=["X", "Y", "Z"]
        )
        rows.append(
            {"well_id": wid, "x0": h.X.iloc[0], "y0": h.Y.iloc[0], "xm": h.X.mean(), "ym": h.Y.mean(), "zm": h.Z.mean()}
        )
    df = pd.DataFrame(rows)
    if split == "train":
        _CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def typewell_groups(split: Split = "train") -> dict[str, str]:
    """well_id -> typewell content hash. Shared hash == shared typewell."""
    cache = _CACHE_DIR / "typewell_groups.csv"
    if split == "train" and cache.exists():
        df = pd.read_csv(cache)
        return dict(zip(df.well_id, df.tw_hash))
    out = {}
    for wid in list_wells(split):
        digest = hashlib.md5(
            (data_dir() / split / f"{wid}__typewell.csv").read_bytes()
        ).hexdigest()
        out[wid] = digest
    if split == "train":
        _CACHE_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"well_id": list(out), "tw_hash": list(out.values())}).to_csv(cache, index=False)
    return out


def make_folds(
    n_splits: int = 5, mode: str = "well", seed: int = 0, split: Split = "train"
) -> dict[str, int]:
    """well_id -> fold index. Wells sharing a typewell never straddle folds."""
    groups = typewell_groups(split)
    wells = sorted(groups)

    if mode == "well":
        # Shuffle typewell-groups (not wells) into folds, balancing well counts.
        rng = np.random.default_rng(seed)
        unit_wells: dict[str, list[str]] = {}
        for wid, g in groups.items():
            unit_wells.setdefault(g, []).append(wid)
        units = list(unit_wells)
        rng.shuffle(units)
        fold_sizes = np.zeros(n_splits, dtype=int)
        folds: dict[str, int] = {}
        for unit in units:
            f = int(fold_sizes.argmin())
            for wid in unit_wells[unit]:
                folds[wid] = f
            fold_sizes[f] += len(unit_wells[unit])
        return folds

    if mode == "spatial":
        from scipy.cluster.vq import kmeans2

        xy = well_xy(split).set_index("well_id").loc[wells]
        pts = xy[["xm", "ym"]].to_numpy()
        pts_std = (pts - pts.mean(0)) / pts.std(0)
        _, labels = kmeans2(pts_std, n_splits, minit="++", seed=seed)
        folds = dict(zip(wells, (int(v) for v in labels)))
        # Pull typewell-sharing wells into one (majority/first) region.
        unit_wells = {}
        for wid, g in groups.items():
            unit_wells.setdefault(g, []).append(wid)
        for ws in unit_wells.values():
            if len(ws) > 1:
                target = folds[ws[0]]
                for wid in ws:
                    folds[wid] = target
        return folds

    raise ValueError(f"unknown fold mode: {mode}")
