"""
CCP-II Physics-Informed Neural Network Package.

Main modules:
- model: PINN models (BasePINN, SequentialPINN, GatedPINN, etc.)
- trainer: Training utilities and Lightning CLI
- architectures: Neural network architectures
- data: Collocation samplers and FDM solver
- utils: Physics parameters and utilities
"""

from src.model import (
    BasePINN,
    SequentialPINN,
    GatedPINN,
    ModulatedPINNModel,
    FourierPINN,
    TwoNetworkPINN,
    HybridPINN,
    NonDimPINN,
    MODEL_REGISTRY,
    get_model_class,
    create_model,
    list_models,
)

__all__ = [
    # Models
    "BasePINN",
    "SequentialPINN",
    "GatedPINN",
    "ModulatedPINNModel",
    "FourierPINN",
    "TwoNetworkPINN",
    "HybridPINN",
    "NonDimPINN",
    # Registry
    "MODEL_REGISTRY",
    "get_model_class",
    "create_model",
    "list_models",
]
