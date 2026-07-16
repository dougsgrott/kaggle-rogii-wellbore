"""Model registry: name -> zero-arg factory (used by scripts/run_cv.py)."""

from .baselines import AnchorLastValue

REGISTRY = {
    "anchor": AnchorLastValue,
}
