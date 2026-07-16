"""CV harness: leak-proof scoring, per-well diagnostics, seed averaging,
and the wall test (docs/issues/002-cv-harness.md).

The contract every model implements:

    class MyModel:
        needs_fit = True          # False for pure per-well predictors
        def fit(self, wells: Iterable[Well]) -> None: ...
        def predict_well(self, well: Well) -> np.ndarray  # len == eval zone

``predict_well`` always receives ``well.as_test()`` — the horizontal frame
has no ``TVT`` column, so eval-zone leakage is structurally impossible.
Train wells' evaluation zones are the competition-defined ``TVT_input``-NaN
suffixes, i.e. local scoring replicates the hidden-test geometry exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from .data import Well, iter_wells, list_wells, load_well


class WellPredictor(Protocol):
    needs_fit: bool

    def fit(self, wells: Iterable[Well]) -> None: ...

    def predict_well(self, well: Well) -> np.ndarray: ...


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ---------------------------------------------------------------------------
# Per-well diagnostics


def _two_means_1d(x: np.ndarray) -> tuple[float, float, float, float]:
    """Exact 1-D 2-means via sorted prefix sums.

    Returns (separation, minority_weight, pooled_within_std, split_value).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 10:
        return 0.0, 0.0, float(x.std()), float("nan")
    csum = np.cumsum(x)
    csum2 = np.cumsum(x**2)
    ks = np.arange(1, n)
    sse_l = csum2[:-1] - csum[:-1] ** 2 / ks
    sse_r = (csum2[-1] - csum2[:-1]) - (csum[-1] - csum[:-1]) ** 2 / (n - ks)
    k = int(np.argmin(sse_l + sse_r)) + 1
    c1, c2 = csum[k - 1] / k, (csum[-1] - csum[k - 1]) / (n - k)
    pooled = float(np.sqrt((sse_l[k - 1] + sse_r[k - 1]) / n))
    return float(c2 - c1), float(min(k, n - k) / n), pooled, float(x[k - 1])


def bimodality_row(residuals: np.ndarray) -> dict:
    sep, minority, pooled, _ = _two_means_1d(residuals)
    return {
        "bimodal_sep": sep,
        "bimodal_minority": minority,
        # heuristic flag for ±15 ft datum wells (topics/711878.md): two
        # well-populated modes clearly wider apart than their spread
        "bimodal": bool(sep > 8.0 and minority > 0.2 and sep > 2.0 * max(pooled, 1e-9)),
    }


@dataclass
class CVResult:
    per_well: pd.DataFrame           # worst-first, with tail decomposition
    global_rmse: float
    predictions: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    truths: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    mds: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def per_well_report(
    truths: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    mds: dict[str, np.ndarray] | None = None,
) -> CVResult:
    rows = []
    for well_id, y_true in truths.items():
        res = np.asarray(predictions[well_id]) - np.asarray(y_true)
        rows.append(
            {
                "well_id": well_id,
                "n": len(res),
                "sse": float((res**2).sum()),
                "rmse": float(np.sqrt((res**2).mean())),
                "bias": float(res.mean()),
                **bimodality_row(res),
            }
        )
    df = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    df["share_of_sse"] = df["sse"] / df["sse"].sum()
    df["cum_share"] = df["share_of_sse"].cumsum()
    global_rmse = float(np.sqrt(df["sse"].sum() / df["n"].sum()))
    return CVResult(df, global_rmse, dict(predictions), dict(truths), dict(mds or {}))


# ---------------------------------------------------------------------------
# CV runner


def run_cv(
    factory: Callable[[], WellPredictor],
    folds: dict[str, int] | None = None,
    well_ids: Sequence[str] | None = None,
    verbose: bool = True,
) -> CVResult:
    """Score a model over train wells with leak-proof inputs.

    ``folds`` is required for models with ``needs_fit``; per-well predictors
    are scored in a single pass. ``well_ids`` restricts to a subset (fast
    iteration); tail numbers on subsets are indicative only.
    """
    ids = list(well_ids) if well_ids is not None else list_wells("train")
    probe = factory()
    preds: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    mds: dict[str, np.ndarray] = {}

    def _score(model: WellPredictor, wid: str) -> None:
        well = load_well(wid, "train")
        y = model.predict_well(well.as_test())
        n_eval = int(well.eval_mask.sum())
        if len(y) != n_eval:
            raise ValueError(f"{wid}: predictor returned {len(y)} values for {n_eval} eval rows")
        preds[wid] = np.asarray(y, dtype=float)
        truths[wid] = well.eval_target
        mds[wid] = well.horizontal.loc[well.eval_mask, "MD"].to_numpy()

    if not getattr(probe, "needs_fit", True):
        for wid in ids:
            _score(probe, wid)
    else:
        if folds is None:
            raise ValueError("folds required for models with needs_fit=True")
        fold_of = {w: folds[w] for w in ids}
        for f in sorted(set(fold_of.values())):
            model = factory()
            train_ids = [w for w in folds if folds[w] != f and (well_ids is None or w in fold_of)]
            model.fit(load_well(w, "train") for w in train_ids)
            for wid in [w for w, ff in fold_of.items() if ff == f]:
                _score(model, wid)
            if verbose:
                print(f"fold {f}: fitted on {len(train_ids)} wells")

    result = per_well_report(truths, preds, mds)
    if verbose:
        print(f"global RMSE: {result.global_rmse:.4f} over {len(ids)} wells")
    return result


# ---------------------------------------------------------------------------
# Seed averaging (stochastic pipelines drift ±0.1–0.4 RMSE between reruns)


def run_seeds(
    factory_of_seed: Callable[[int], WellPredictor],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    **run_cv_kwargs,
) -> dict:
    """Run the CV once per seed; report per-seed spread and the CV of the
    seed-averaged prediction (which is what a submission should use)."""
    results = [run_cv(lambda s=s: factory_of_seed(s), verbose=False, **run_cv_kwargs) for s in seeds]
    per_seed = [r.global_rmse for r in results]
    avg_preds = {
        wid: np.mean([r.predictions[wid] for r in results], axis=0)
        for wid in results[0].predictions
    }
    averaged = per_well_report(results[0].truths, avg_preds, results[0].mds)
    return {
        "per_seed_rmse": per_seed,
        "seed_mean": float(np.mean(per_seed)),
        "seed_std": float(np.std(per_seed)),
        "averaged_result": averaged,
    }


# ---------------------------------------------------------------------------
# Wall test: is a per-well feature real signal or noise/leak?
# Leave-one-group-out R² with a shuffle-null control (topics/712037.md).


def _logo_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X])
    pred = np.empty_like(y, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        beta, *_ = np.linalg.lstsq(X[~m], y[~m], rcond=None)
        pred[m] = X[m] @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def wall_test(
    feature: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict:
    """Out-of-group R² of feature→target vs a permutation null.

    Verdict "wall" == the feature does not generalize across groups. A
    feature passes only if real R² > 0 and beats the 95th-percentile null.
    """
    X = np.atleast_2d(np.asarray(feature, dtype=float))
    X = X.T if X.shape[0] == 1 else X
    y = np.asarray(target, dtype=float)
    g = np.asarray(groups)
    real = _logo_r2(X, y, g)
    rng = np.random.default_rng(seed)
    nulls = np.array([_logo_r2(X, rng.permutation(y), g) for _ in range(n_shuffles)])
    p = float((nulls >= real).mean())
    return {
        "r2": real,
        "null_mean": float(nulls.mean()),
        "null_p95": float(np.quantile(nulls, 0.95)),
        "p_value": p,
        "passes": bool(real > 0 and real > np.quantile(nulls, 0.95)),
    }


# ---------------------------------------------------------------------------
# Plots


def report_plots(result: CVResult, out_dir: str | Path, n_worst: int = 8) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = result.per_well

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.arange(1, len(df) + 1) / len(df) * 100, df["cum_share"] * 100)
    ax[0].set(xlabel="worst % of wells", ylabel="% of total squared error", title=f"Tail concentration (global RMSE {result.global_rmse:.3f})")
    ax[0].grid(alpha=0.3)
    ax[1].hist(df["rmse"], bins=50)
    ax[1].set(xlabel="per-well RMSE (ft)", ylabel="wells", title="Per-well RMSE distribution")
    fig.tight_layout()
    fig.savefig(out / "tail.png", dpi=120)
    plt.close(fig)

    worst = df.head(n_worst)["well_id"]
    fig, axes = plt.subplots((n_worst + 3) // 4, 4, figsize=(16, 3 * ((n_worst + 3) // 4)), squeeze=False)
    for ax_, wid in zip(axes.flat, worst):
        res = result.predictions[wid] - result.truths[wid]
        md = result.mds.get(wid, np.arange(len(res)))
        ax_.plot(md, res, lw=0.7)
        ax_.axhline(0, color="k", lw=0.5)
        row = df.loc[df.well_id == wid].iloc[0]
        ax_.set_title(f"{wid} rmse={row.rmse:.1f}{' BIMODAL' if row.bimodal else ''}", fontsize=9)
    fig.suptitle("Residual vs MD — worst wells")
    fig.tight_layout()
    fig.savefig(out / "worst_wells.png", dpi=120)
    plt.close(fig)
    return out
