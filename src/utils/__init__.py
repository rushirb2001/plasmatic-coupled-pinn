"""
Utilities package for CCP-II PINN model.
"""

from .physics import (
    PhysicalConstants,
    DomainParameters,
    PlasmaParameters,
    ScalingParameters,
    ParameterSpace,
)
from .nondim import (
    DimensionlessCoefficients,
    NonDimensionalizer,
)
from .seed import (
    seed_everything,
    get_device,
    get_accelerator,
)
from .gradients import (
    AdaptiveLossBalancer,
    GradientMonitor,
    compute_gradient_norm,
    compute_top_hessian_eigenvalue,
    extract_gradients,
)

__all__ = [
    # Physics
    "PhysicalConstants",
    "DomainParameters",
    "PlasmaParameters",
    "ScalingParameters",
    "ParameterSpace",
    # Non-dimensionalization
    "DimensionlessCoefficients",
    "NonDimensionalizer",
    # Reproducibility
    "seed_everything",
    "get_device",
    "get_accelerator",
    # Gradients
    "AdaptiveLossBalancer",
    "GradientMonitor",
    "compute_gradient_norm",
    "compute_top_hessian_eigenvalue",
    "extract_gradients",
]
