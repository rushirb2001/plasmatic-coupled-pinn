"""
Non-dimensionalization utilities for CCP-II PINN model.

Provides:
- DimensionlessCoefficients: Computed PDE coefficients in dimensionless form
- NonDimensionalizer: Scaling/unscaling utilities for physical quantities
"""

from dataclasses import dataclass
from typing import Union
import torch
import numpy as np

from .physics import ParameterSpace


@dataclass
class DimensionlessCoefficients:
    """
    Computed dimensionless coefficients for the CCP-II PDE system.

    These appear in the non-dimensionalized PDEs (matching archive formula):

    Continuity: dn'/dt' + alpha * d2n'/dx'2 + beta * (n' * d2phi'/dx'2 + dphi'/dx' * dn'/dx') - gamma * R = 0

    Poisson: d2phi'/dx'2 - delta * (n' - n'_io) = 0

    Note: alpha is NEGATIVE for correct diffusion physics.
    """
    alpha: float   # Diffusion scale: -D * t_ref / x_ref^2 (NEGATIVE)
    beta: float    # Drift scale: mu * phi_ref * t_ref / x_ref^2
    gamma: float   # Reaction scale: t_ref / n_ref (R_0 applied separately in residual)
    delta: float   # Poisson scale: e * n_ref * x_ref^2 / (epsilon_0 * phi_ref)

    def to_dict(self) -> dict:
        """Export coefficients for logging"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }


class NonDimensionalizer:
    """
    Handles scaling between physical and dimensionless quantities.

    Usage:
        params = ParameterSpace()
        nondim = NonDimensionalizer(params)

        # Get PDE coefficients
        coeffs = nondim.coeffs

        # Scale physical values
        x_dim = nondim.scale_x(x_physical)  # x' = x / x_ref

        # Unscale to physical values
        ne_physical = nondim.unscale_ne(ne_dim)  # n_e = n_e' * n_ref
    """

    def __init__(self, params: ParameterSpace):
        self.params = params
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Compute dimensionless PDE coefficients matching archive implementation"""
        p = self.params
        s = p.scales
        c = p.constants
        plasma = p.plasma

        # Diffusion coefficient: -D * t_ref / x_ref^2 (NEGATIVE to match archive)
        # Archive: alpha = -(D*T_ref)/(L_ref ** 2)
        alpha = -(plasma.D * s.t_ref / s.x_ref**2)

        # Drift coefficient: mu * phi_ref * t_ref / x_ref^2 (POSITIVE to match archive)
        # Archive: beta = (mu*phi_ref*T_ref)/(L_ref ** 2)
        beta = plasma.mu * s.phi_ref * s.t_ref / s.x_ref**2

        # Reaction coefficient: t_ref / n_ref (matches archive formula)
        # Archive: gamma = T_ref/N_ref
        # Note: R_val in residual is multiplied by R0, so gamma doesn't include it
        gamma = s.t_ref / s.n_ref

        # Poisson coefficient: e * n_ref * x_ref^2 / (epsilon_0 * phi_ref)
        # Archive: delta = ((e*N_ref)/epsilon_0) * (L_ref**2)/phi_ref
        delta = (c.e * s.n_ref * s.x_ref**2) / (c.epsilon_0 * s.phi_ref)

        self.coeffs = DimensionlessCoefficients(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta
        )

    # Scaling functions (physical -> dimensionless)

    def scale_x(self, x_phys: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Scale spatial coordinate: x' = x / x_ref"""
        return x_phys / self.params.scales.x_ref

    def scale_t(self, t_phys: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Scale temporal coordinate: t' = t / t_ref"""
        return t_phys / self.params.scales.t_ref

    def scale_ne(self, ne_phys: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Scale electron density: n_e' = n_e / n_ref"""
        return ne_phys / self.params.scales.n_ref

    def scale_phi(self, phi_phys: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Scale electric potential: phi' = phi / phi_ref"""
        return phi_phys / self.params.scales.phi_ref

    # Unscaling functions (dimensionless -> physical)

    def unscale_x(self, x_dim: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Unscale spatial coordinate: x = x' * x_ref"""
        return x_dim * self.params.scales.x_ref

    def unscale_t(self, t_dim: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Unscale temporal coordinate: t = t' * t_ref"""
        return t_dim * self.params.scales.t_ref

    def unscale_ne(self, ne_dim: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Unscale electron density: n_e = n_e' * n_ref"""
        return ne_dim * self.params.scales.n_ref

    def unscale_phi(self, phi_dim: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Unscale electric potential: phi = phi' * phi_ref"""
        return phi_dim * self.params.scales.phi_ref

    # Derived quantities

    def compute_normalized_n_io(self) -> float:
        """
        Compute normalized background ion density.

        n_io = R0 * (x2 - x1) * sqrt(m_i / (e * T_e))
        n_io' = n_io / n_ref
        """
        return self.params.compute_n_io()

    def compute_normalized_voltage(self, t_dim: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute normalized driving voltage at dimensionless time t'.

        V(t) = V0 * sin(2*pi*f*t)
        V'(t') = sin(2*pi*t')  (since t' = t * f)

        Returns: phi' = V/phi_ref = (V0/phi_ref) * sin(2*pi*t')
        """
        # With standard scaling, V0 = phi_ref, so this simplifies
        if isinstance(t_dim, torch.Tensor):
            return torch.sin(2 * 3.141592653589793 * t_dim)
        else:
            return np.sin(2 * 3.141592653589793 * t_dim)

    def __repr__(self) -> str:
        return (
            f"NonDimensionalizer(\n"
            f"  alpha={self.coeffs.alpha:.4e} (diffusion)\n"
            f"  beta={self.coeffs.beta:.4e} (drift)\n"
            f"  gamma={self.coeffs.gamma:.4e} (reaction)\n"
            f"  delta={self.coeffs.delta:.4e} (Poisson)\n"
            f")"
        )
