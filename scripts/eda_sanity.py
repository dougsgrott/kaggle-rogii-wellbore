"""Issue-001 sanity EDA. Run after the data download:

    uv run python scripts/eda_sanity.py

Prints the dataset facts destined for docs/data-notes.md and validates
the loader assumptions (id scheme, eval-zone structure).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src import data


def main() -> None:
    train_ids = data.list_wells("train")
    test_ids = data.list_wells("test")
    print(f"train wells: {len(train_ids)}, test wells (example): {len(test_ids)}")

    data.validate_id_scheme("test")
    print("id scheme validated: submission ids == positional TVT_input-NaN rows")

    zone_rows = []
    tw_hashes: dict[str, list[str]] = {}
    gr_nan_total = 0
    n_rows_total = 0
    for well in data.iter_wells("train"):
        h = well.horizontal
        mask = well.eval_mask
        idx = np.flatnonzero(mask)
        contiguous = bool(len(idx) and (np.diff(idx) == 1).all())
        at_end = bool(len(idx) and idx[-1] == len(h) - 1)
        # per-well drift: robust linear fit of TVT vs MD on the known prefix
        known = h.loc[~mask]
        slope = np.nan
        if len(known) > 10 and "TVT" in h:
            slope = np.polyfit(known["MD"], known["TVT"], 1)[0]
        zone_rows.append(
            {
                "well_id": well.well_id,
                "n_rows": len(h),
                "n_eval": int(mask.sum()),
                "eval_frac": mask.mean(),
                "zone_contiguous": contiguous,
                "zone_at_end": at_end,
                "gr_nan": int(h["GR"].isna().sum()) if "GR" in h else -1,
                "tvt_slope_per_ft": slope,
                "tw_rows": len(well.typewell),
            }
        )
        gr_nan_total += zone_rows[-1]["gr_nan"]
        n_rows_total += len(h)
        digest = hashlib.md5(
            pd.util.hash_pandas_object(well.typewell, index=False).values.tobytes()
        ).hexdigest()
        tw_hashes.setdefault(digest, []).append(well.well_id)

    df = pd.DataFrame(zone_rows)
    out = data._REPO_ROOT / "analysis" / "eda_well_summary.csv"
    df.to_csv(out, index=False)
    print(f"per-well summary -> {out}")

    print("\n-- evaluation zones --")
    print(df[["n_rows", "n_eval", "eval_frac"]].describe().round(2))
    print(f"zones contiguous: {df.zone_contiguous.mean():.1%}, at end of lateral: {df.zone_at_end.mean():.1%}")

    print("\n-- GR missingness --")
    print(f"total GR NaNs: {gr_nan_total} / {n_rows_total} rows; wells with any NaN: {(df.gr_nan > 0).sum()}")

    dupes = {h: ws for h, ws in tw_hashes.items() if len(ws) > 1}
    n_dup_wells = sum(len(ws) for ws in dupes.values())
    print("\n-- typewell duplication --")
    print(f"{len(tw_hashes)} distinct typewells for {len(train_ids)} wells; {len(dupes)} shared by {n_dup_wells} wells")

    print("\n-- per-well TVT drift slope (ft/ft of MD, known prefix) --")
    print(df.tvt_slope_per_ft.describe().round(4))

    tw = data.load_well(train_ids[0]).typewell
    print(f"\ntypewell columns: {list(tw.columns)}; Geology labels example: {tw['Geology'].dropna().unique()[:8]}")

    h = data.load_well(train_ids[0]).horizontal
    print(f"horizontal columns: {list(h.columns)}")


if __name__ == "__main__":
    main()
