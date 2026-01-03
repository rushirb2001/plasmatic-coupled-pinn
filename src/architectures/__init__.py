"""
Architecture components for PINN models.

Exports:
    - Activation functions: PTanh, PExp
    - Fourier feature mappings: FourierFeatureMapping, FourierFeatureMapping1D,
                                FourierFeatureMapping2D, PeriodicTimeEmbedding
    - Networks: MLP, SequentialModel, SequentialModelPeriodic, GatedSequentialModel,
                ModulatedSequentialModel, ModulatedPINN, FourierMLP,
                DensityNetwork, PotentialNetwork, TwoNetworkModel
    - Poisson solvers: PoissonSolverCPU, PoissonSolverGPU
    - Interpolators: PoissonInterpolator, DensityInterpolator, FieldCache
"""

from .activations import PTanh, PExp
from .fourier import (
    FourierFeatureMapping,
    FourierFeatureMapping1D,
    FourierFeatureMapping2D,
    PeriodicTimeEmbedding,
)
from .networks import (
    MLP,
    SequentialModel,
    SequentialModelPeriodic,
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
    "FourierFeatureMapping1D",
    "FourierFeatureMapping2D",
    "PeriodicTimeEmbedding",
    # Networks
    "MLP",
    "SequentialModel",
    "SequentialModelPeriodic",
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
