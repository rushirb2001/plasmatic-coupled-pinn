"""
Architecture components for PINN models.

Exports:
    - Activation functions: PTanh, PExp
    - Fourier feature mappings: FourierFeatureMapping, FourierFeatureMapping2D
    - Networks: MLP, SequentialModel, GatedSequentialModel, ModulatedSequentialModel,
                ModulatedPINN, FourierMLP, DensityNetwork, PotentialNetwork, TwoNetworkModel
    - Poisson solvers: PoissonSolverCPU, PoissonSolverGPU
    - Interpolators: PoissonInterpolator, DensityInterpolator, FieldCache
"""

from .activations import PTanh, PExp
from .fourier import FourierFeatureMapping, FourierFeatureMapping2D
from .networks import (
    MLP,
    SequentialModel,
    GatedSequentialModel,
    ModulatedSequentialModel,
    ModulatedPINN,
    FourierMLP,
    DensityNetwork,
    PotentialNetwork,
    TwoNetworkModel,
)
from .solvers import PoissonSolverCPU, PoissonSolverGPU
from .interpolators import PoissonInterpolator, DensityInterpolator, FieldCache

__all__ = [
    # Activations
    "PTanh",
    "PExp",
    # Fourier features
    "FourierFeatureMapping",
    "FourierFeatureMapping2D",
    # Networks
    "MLP",
    "SequentialModel",
    "GatedSequentialModel",
    "ModulatedSequentialModel",
    "ModulatedPINN",
    "FourierMLP",
    "DensityNetwork",
    "PotentialNetwork",
    "TwoNetworkModel",
    # Solvers
    "PoissonSolverCPU",
    "PoissonSolverGPU",
    # Interpolators
    "PoissonInterpolator",
    "DensityInterpolator",
    "FieldCache",
]
