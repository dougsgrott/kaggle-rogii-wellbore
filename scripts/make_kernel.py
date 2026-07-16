"""Package scripts/kernel_src.py as a Kaggle notebook kernel.

    uv run python scripts/make_kernel.py --model anchor-drift --name rogii-baseline-v0

Creates submissions/<name>/ with kernel-metadata.json + <name>.ipynb.
Push with:  kaggle kernels push -p submissions/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USERNAME = "dougsgrott"
COMPETITION = "rogii-wellbore-geology-prediction"


BUNDLE_FILES = [
    "src/data.py",
    "src/models/baselines.py",
    "src/models/tracker.py",
    "src/models/structure.py",
]

BUNDLE_MODELS = {"hmm": "HMMTracker()", "hmm-structure": "HMMStructure()"}

BUNDLE_DRIVER = '''

# ---------------------------------------------------------------- driver
def _find_input():
    env = os.environ.get("ROGII_DATA_DIR")
    if env:
        return Path(env)
    base = Path("/kaggle/input")
    hits = sorted(base.glob("*/test")) or sorted(base.glob("*/*/test"))
    if not hits:
        raise FileNotFoundError(f"no */test folder under {base}")
    return hits[0].parent


def main():
    os.environ["ROGII_DATA_DIR"] = str(_find_input())
    print(f"input root: {data_dir()}")
    model = {model_expr}
    if getattr(model, "needs_fit", False):
        print("fitting on train wells...")
        model.fit(iter_wells("train"))
    frames = []
    for wid in list_wells("test"):
        well = load_well(wid, "test")
        pred = model.predict_well(well)
        anchor = well.last_known_tvt
        if not np.isfinite(pred).all():
            pred = np.where(np.isfinite(pred), pred, anchor)
        frames.append(pd.DataFrame({"id": well.eval_ids, "tvt": pred}))
        d = model.last_diagnostics or {}
        print(f"{wid}: {len(pred)} rows, anchor={anchor:.1f}, gate_w={d.get('gate_weight')}")
    sub = pd.concat(frames, ignore_index=True)
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv written: {len(sub)} rows")


main()
'''


def bundle_source(model: str) -> str:
    """Concatenate real src modules into one flat namespace (no forks:
    the kernel runs the exact code that produced the CV number)."""
    import re

    parts = ["import os\nfrom pathlib import Path\n"]
    for rel in BUNDLE_FILES:
        code = (REPO / rel).read_text()
        code = re.sub(r"^\s*from \.[\.\w]* import .*$", "", code, flags=re.M)
        code = re.sub(r"^from __future__ import .*$", "", code, flags=re.M)
        code = code.replace('Path(__file__).resolve().parent.parent', 'Path(".")')  # no __file__ in notebooks
        parts.append(f"\n# ===== {rel} =====\n{code}")
    return "\n".join(parts) + BUNDLE_DRIVER.replace("{model_expr}", BUNDLE_MODELS[model])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        choices=["anchor-drift", "residual-lgbm", *BUNDLE_MODELS],
    )
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.model in BUNDLE_MODELS:
        src = bundle_source(args.model)
    else:
        src = (REPO / "scripts" / "kernel_src.py").read_text()
        # Freeze the model choice into the kernel (no env vars on Kaggle).
        src = src.replace(
            'MODEL = os.environ.get("ROGII_MODEL", "anchor-drift")',
            f'MODEL = os.environ.get("ROGII_MODEL", "{args.model}")',
        )

    out = REPO / "submissions" / args.name
    out.mkdir(parents=True, exist_ok=True)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            }
        ],
    }
    (out / f"{args.name}.ipynb").write_text(json.dumps(notebook))
    meta = {
        "id": f"{USERNAME}/{args.name}",
        "title": args.name,
        "code_file": f"{args.name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel packaged -> {out}")
    print(f"push:   kaggle kernels push -p {out.relative_to(REPO)}")
    print(f"status: kaggle kernels status {USERNAME}/{args.name}")


if __name__ == "__main__":
    main()
