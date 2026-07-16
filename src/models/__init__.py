"""Model registry: name -> zero-arg factory (used by scripts/run_cv.py)."""

from .baselines import AnchorDrift, AnchorLastValue
from .residual_lgbm import ResidualLGBM

REGISTRY = {
    "anchor": AnchorLastValue,
    "anchor-drift": AnchorDrift,
    "residual-lgbm": ResidualLGBM,
}
