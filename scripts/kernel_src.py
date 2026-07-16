"""Standalone submission code — the single source for the Kaggle kernel.

Self-contained on purpose (no `src` import): the kernel runs offline with
only preinstalled libs. Runs locally too:

    ROGII_INPUT=data uv run python scripts/kernel_src.py

On Kaggle, INPUT resolves to the competition mount and the hidden rerun
replaces the example test wells with ~200 real ones.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

def _find_input() -> Path:
    env = os.environ.get("ROGII_INPUT")
    if env:
        return Path(env)
    base = Path("/kaggle/input")
    for d in sorted(base.iterdir()) if base.exists() else []:
        print(f"input mount: {d} -> {sorted(p.name for p in d.iterdir())[:6]}")
    hits = sorted(base.glob("*/test"))
    if not hits:
        hits = sorted(base.glob("*/*/test"))
    if not hits:
        raise FileNotFoundError(f"no */test folder under {base}")
    print(f"using input root: {hits[0].parent}")
    return hits[0].parent


INPUT = _find_input()
MODEL = os.environ.get("ROGII_MODEL", "anchor-drift")

DRIFT_WINDOW_FT = 100.0
DRIFT_TAU = 150.0


def prefix_slope(known: pd.DataFrame, window_ft: float) -> float:
    tail = known[known["MD"] >= known["MD"].iloc[-1] - window_ft]
    if len(tail) < 20:
        return 0.0
    return float(np.polyfit(tail["MD"], tail["TVT_input"], 1)[0])


def predict_anchor_drift(h: pd.DataFrame, eval_mask: np.ndarray) -> np.ndarray:
    known = h.loc[~eval_mask]
    anchor = float(known["TVT_input"].iloc[-1])
    slope = prefix_slope(known, DRIFT_WINDOW_FT)
    dmd = h.loc[eval_mask, "MD"].to_numpy() - known["MD"].iloc[-1]
    return anchor + slope * DRIFT_TAU * (1.0 - np.exp(-dmd / DRIFT_TAU))


def build_features(h: pd.DataFrame, eval_mask: np.ndarray) -> pd.DataFrame:
    known = h.loc[~eval_mask]
    md_anchor = known["MD"].iloc[-1]
    z_anchor = known["Z"].iloc[-1]
    gr_filled = h["GR"].ffill()
    roll = gr_filled.rolling(100, min_periods=10)
    ev = h.loc[eval_mask]
    df = pd.DataFrame()
    df["dmd"] = ev["MD"].to_numpy() - md_anchor
    df["z_delta"] = ev["Z"].to_numpy() - z_anchor
    for w in (100, 250, 1000):
        df[f"slope_{w}"] = prefix_slope(known, w)
    df["drift_150"] = df["slope_100"] * 150.0 * (1.0 - np.exp(-df["dmd"] / 150.0))
    df["gr"] = ev["GR"].to_numpy()
    df["gr_roll_mean"] = roll.mean()[eval_mask].to_numpy()
    df["gr_roll_std"] = roll.std()[eval_mask].to_numpy()
    df["prefix_len"] = len(known)
    df["eval_frac"] = float(eval_mask.mean())
    return df


def well_ids(split: str) -> list[str]:
    return sorted(p.name.split("__")[0] for p in (INPUT / split).glob("*__horizontal_well.csv"))


def load(split: str, wid: str) -> tuple[pd.DataFrame, np.ndarray]:
    h = pd.read_csv(INPUT / split / f"{wid}__horizontal_well.csv")
    return h, h["TVT_input"].isna().to_numpy()


def main() -> None:
    model = None
    if MODEL == "residual-lgbm":
        import lightgbm as lgb

        xs, ys = [], []
        for wid in well_ids("train"):
            h, mask = load("train", wid)
            xs.append(build_features(h, mask))
            anchor = float(h.loc[~mask, "TVT_input"].iloc[-1])
            ys.append(h.loc[mask, "TVT"].to_numpy() - anchor)
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1,
        )
        model.fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))
        print(f"trained on {len(xs)} wells")

    frames = []
    for wid in well_ids("test"):
        h, mask = load("test", wid)
        anchor = float(h.loc[~mask, "TVT_input"].iloc[-1])
        if MODEL == "residual-lgbm":
            pred = anchor + model.predict(build_features(h, mask))
        else:
            pred = predict_anchor_drift(h, mask)
        if not np.isfinite(pred).all():  # never ship NaN: fall back to anchor
            pred = np.where(np.isfinite(pred), pred, anchor)
        ids = [f"{wid}_{i}" for i in np.flatnonzero(mask)]
        frames.append(pd.DataFrame({"id": ids, "tvt": pred}))
        print(f"{wid}: {len(ids)} predictions, anchor={anchor:.1f}")

    sub = pd.concat(frames, ignore_index=True)
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv written: {len(sub)} rows, model={MODEL}")


if __name__ == "__main__":
    main()
