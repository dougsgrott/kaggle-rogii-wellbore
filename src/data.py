"""Canonical data layer for the ROGII wellbore competition.

Every model and analysis script goes through these loaders; nothing else
reads the raw CSVs directly. Paths resolve to the repo-local ``data/``
directory by default and can be pointed at the Kaggle input mount with
``ROGII_DATA_DIR`` (e.g. ``/kaggle/input/rogii-wellbore-geology-prediction``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

Split = Literal["train", "test"]

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Horizontal-well columns, per the competition data description. The
# formation-depth columns and targets are train-only; loaders must not
# assume they exist.
TRAJECTORY_COLS = ["MD", "X", "Y", "Z"]
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TARGET_COL = "TVT"
INPUT_TARGET_COL = "TVT_input"


def data_dir() -> Path:
    return Path(os.environ.get("ROGII_DATA_DIR", _REPO_ROOT / "data"))


@dataclass(frozen=True)
class Well:
    """One well: lateral trajectory/log frame plus its typewell reference."""

    well_id: str
    split: Split
    horizontal: pd.DataFrame
    typewell: pd.DataFrame

    @property
    def eval_mask(self) -> np.ndarray:
        """Boolean mask (per horizontal row) of the evaluation zone.

        The zone is defined by ``TVT_input`` being NaN. On test wells the
        hidden ``TVT`` is NaN on exactly this zone; on train wells ``TVT``
        stays fully observed, which is what makes local scoring possible.
        """
        if INPUT_TARGET_COL in self.horizontal:
            return self.horizontal[INPUT_TARGET_COL].isna().to_numpy()
        # Degenerate fallback (column missing entirely): score nothing.
        return np.zeros(len(self.horizontal), dtype=bool)

    @property
    def eval_ids(self) -> list[str]:
        """Submission ids ({well}_{row_index}) for the evaluation zone."""
        idx = np.flatnonzero(self.eval_mask)
        return [f"{self.well_id}_{i}" for i in idx]

    @property
    def known_prefix(self) -> pd.DataFrame:
        """Rows where TVT_input is observed (the steering history)."""
        return self.horizontal.loc[~self.eval_mask]

    @property
    def last_known_tvt(self) -> float:
        known = self.horizontal[INPUT_TARGET_COL].dropna()
        return float(known.iloc[-1]) if len(known) else float("nan")


def list_wells(split: Split = "train") -> list[str]:
    root = data_dir() / split
    return sorted(p.name.split("__")[0] for p in root.glob("*__horizontal_well.csv"))


def load_well(well_id: str, split: Split = "train") -> Well:
    root = data_dir() / split
    horizontal = pd.read_csv(root / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(root / f"{well_id}__typewell.csv")
    return Well(well_id=well_id, split=split, horizontal=horizontal, typewell=typewell)


def iter_wells(split: Split = "train") -> Iterator[Well]:
    for well_id in list_wells(split):
        yield load_well(well_id, split)


@lru_cache(maxsize=1)
def sample_submission() -> pd.DataFrame:
    df = pd.read_csv(data_dir() / "sample_submission.csv")
    parts = df["id"].str.rsplit("_", n=1)
    df["well_id"] = parts.str[0]
    df["row_index"] = parts.str[1].astype(int)
    return df


def validate_id_scheme(split: Split = "test") -> None:
    """Assert submission ids == {well}_{positional row index of eval zone}.

    Run once after downloading data; raises if the assumption the whole
    pipeline rests on is wrong.
    """
    sub = sample_submission()
    for well_id, grp in sub.groupby("well_id", sort=False):
        well = load_well(str(well_id), split)
        expected = np.flatnonzero(well.eval_mask)
        got = grp["row_index"].to_numpy()
        if not np.array_equal(np.sort(got), expected):
            raise AssertionError(
                f"{well_id}: sample_submission rows do not match TVT_input-NaN "
                f"rows ({len(got)} ids vs {len(expected)} eval rows)"
            )
