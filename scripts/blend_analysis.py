"""Issue-005 analysis: full-set CVs of tracker variants, blend evaluation,
and per-well disagreement features for the router (issue 007).

    uv run python scripts/blend_analysis.py

Writes analysis/disagreement.csv and prints blend CVs + Spearman of
disagreement vs realized per-well error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import list_wells
from src.eval import per_well_report, run_cv
from src.models.tracker import HMMTracker

VARIANTS = {
    "mean": lambda: HMMTracker(),
    "map": lambda: HMMTracker(estimator="map"),
    "zshape": lambda: HMMTracker(emission="zshape", sigma_gr=1.0, beta=0.02),
}


def main() -> None:
    results = {}
    for name, factory in VARIANTS.items():
        r = run_cv(factory, verbose=False)
        results[name] = r
        print(f"{name:7} full CV {r.global_rmse:.4f}  median {r.per_well.rmse.median():.3f}", flush=True)

    base = results["mean"]
    wells = list(base.truths)

    # blends over all non-empty member subsets containing >= 2 members
    from itertools import combinations

    names = list(VARIANTS)
    best = None
    for k in (2, 3):
        for combo in combinations(names, k):
            preds = {
                w: np.mean([results[m].predictions[w] for m in combo], axis=0) for w in wells
            }
            rep = per_well_report(base.truths, preds, base.mds)
            print(f"blend {'+'.join(combo):20} -> CV {rep.global_rmse:.4f}  median {rep.per_well.rmse.median():.3f}")
            if best is None or rep.global_rmse < best[1].global_rmse:
                best = (combo, rep)
    print(f"\nbest blend: {'+'.join(best[0])} CV {best[1].global_rmse:.4f}")

    # disagreement features + validity check against realized error
    rows = []
    for w in wells:
        stack = np.stack([results[m].predictions[w] for m in names])
        err = np.sqrt(np.mean((best[1].predictions[w] - base.truths[w]) ** 2))
        rows.append(
            {
                "well_id": w,
                "spread_mean_ft": float(np.std(stack, axis=0).mean()),
                "spread_max_ft": float(np.std(stack, axis=0).max()),
                "mean_vs_map_rmse": float(np.sqrt(np.mean((stack[0] - stack[1]) ** 2))),
                "blend_rmse": float(err),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("analysis/disagreement.csv", index=False)
    for col in ["spread_mean_ft", "spread_max_ft", "mean_vs_map_rmse"]:
        rho = spearmanr(df[col], df["blend_rmse"]).statistic
        print(f"spearman({col}, blend_rmse) = {rho:.3f}")
    print("disagreement features -> analysis/disagreement.csv")


if __name__ == "__main__":
    main()
