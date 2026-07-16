"""Run the CV harness for a registered model. Examples:

    uv run python scripts/run_cv.py --model anchor
    uv run python scripts/run_cv.py --model anchor --mode spatial --limit 100

Outputs: global RMSE, tail summary, per-well CSV and plots under
analysis/cv/<name>/, and a ready-to-paste ledger row for docs/experiments.md.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.cv import make_folds
from src.eval import run_cv
from src.models import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--mode", default="well", choices=["well", "spatial"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N wells (0 = all)")
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    args = ap.parse_args()

    name = args.name or args.model
    factory = REGISTRY[args.model]
    folds = make_folds(args.n_folds, mode=args.mode, seed=args.seed)
    well_ids = sorted(folds)[: args.limit] if args.limit else None

    result = run_cv(factory, folds=folds, well_ids=well_ids)

    out_dir = Path("analysis/cv") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_well.to_csv(out_dir / "per_well.csv", index=False)

    from src.eval import report_plots

    report_plots(result, out_dir)

    df = result.per_well
    n10 = max(len(df) // 10, 1)
    print(f"\nper-well median RMSE: {df.rmse.median():.3f}")
    print(f"worst-decile wells carry {df.cum_share.iloc[n10 - 1]:.0%} of squared error")
    print(f"bimodal-flagged wells: {int(df.bimodal.sum())} ({df.bimodal.mean():.0%})")
    print(f"artifacts -> {out_dir}")

    subset = f" subset={args.limit}" if args.limit else ""
    print(
        f"\nledger row:\n| {date.today()} | {name} | {args.model} | "
        f"{args.mode}{args.n_folds}{subset} | {result.global_rmse:.4f} | — | — |"
    )


if __name__ == "__main__":
    main()
