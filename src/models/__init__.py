"""Model registry: name -> zero-arg factory (used by scripts/run_cv.py)."""

from .baselines import AnchorDrift, AnchorLastValue
from .residual_lgbm import ResidualLGBM
from .router import RouterBlend
from .structure import HMMStructure, StructurePrior
from .tracker import HMMTracker

REGISTRY = {
    "anchor": AnchorLastValue,
    "anchor-drift": AnchorDrift,
    "residual-lgbm": ResidualLGBM,
    "hmm": HMMTracker,
    "structure-prior": StructurePrior,
    "hmm-structure": HMMStructure,
    "router": RouterBlend,
}
