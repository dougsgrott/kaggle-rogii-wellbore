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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["anchor-drift", "residual-lgbm"])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

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
