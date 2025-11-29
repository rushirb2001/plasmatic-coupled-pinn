"""
Interpolators for cached field values.

Used to interpolate pre-computed fields (phi, n_e) to arbitrary query points.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


class PoissonInterpolator:
    """Interpolate pre-computed Poisson solution."""

    def __init__(
        self,
        phi: torch.Tensor,
        dphi_dx: torch.Tensor,
        dphi_dxx: torch.Tensor,
        device: str = "cpu"
    ):
        """
        Args:
            phi, dphi_dx, dphi_dxx: Tensors of shape [Nx, Nt] on uniform grid
        """
        self.device = device
        self.phi_img = phi.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        self.dphi_dx_img = dphi_dx.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        self.dphi_dxx_img = dphi_dxx.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        self.nx, self.nt = phi.shape

    def interpolate(
        self,
        x_t_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpolate to query points.

        Args:
            x_t_batch: Tensor of shape [B, 2] with values in [0, 1]

        Returns:
            phi, dphi_dx, dphi_dxx: All shape [B, 1]
        """
        B = x_t_batch.shape[0]

        x_norm = 2.0 * x_t_batch[:, 0:1] - 1.0
        t_norm = 2.0 * x_t_batch[:, 1:2] - 1.0
        grid = torch.stack([t_norm, x_norm], dim=-1).view(1, B, 1, 2)

        phi_vals = F.grid_sample(
            self.phi_img, grid, mode='bilinear', align_corners=True
        )[0, 0, :, 0].unsqueeze(1)

        dphi_dx_vals = F.grid_sample(
            self.dphi_dx_img, grid, mode='bilinear', align_corners=True
        )[0, 0, :, 0].unsqueeze(1)

        dphi_dxx_vals = F.grid_sample(
            self.dphi_dxx_img, grid, mode='bilinear', align_corners=True
        )[0, 0, :, 0].unsqueeze(1)

        return phi_vals, dphi_dx_vals, dphi_dxx_vals


class DensityInterpolator:
    """Interpolate pre-computed electron density."""

    def __init__(self, ne: torch.Tensor, device: str = "cpu"):
        """
        Args:
            ne: Tensor of shape [Nx, Nt] on uniform grid
        """
        self.device = device
        self.ne_img = ne.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        self.nx, self.nt = ne.shape

    def interpolate(self, x_t_batch: torch.Tensor) -> torch.Tensor:
        """
        Interpolate to query points.

        Args:
            x_t_batch: Tensor of shape [B, 2] with values in [0, 1]

        Returns:
            ne: Shape [B, 1]
        """
        B = x_t_batch.shape[0]

        x_norm = 2.0 * x_t_batch[:, 0:1] - 1.0
        t_norm = 2.0 * x_t_batch[:, 1:2] - 1.0
        grid = torch.stack([t_norm, x_norm], dim=-1).view(1, B, 1, 2)

        ne_vals = F.grid_sample(
            self.ne_img, grid, mode='bilinear', align_corners=True
        )[0, 0, :, 0].unsqueeze(1)

        return ne_vals


class FieldCache:
    """Cache for pre-computed fields with lazy interpolation."""

    def __init__(
        self,
        phi: torch.Tensor = None,
        dphi_dx: torch.Tensor = None,
        dphi_dxx: torch.Tensor = None,
        ne: torch.Tensor = None,
        device: str = "cpu"
    ):
        self.device = device
        self._phi_interp = None
        self._ne_interp = None

        if phi is not None:
            self._phi_interp = PoissonInterpolator(phi, dphi_dx, dphi_dxx, device)
        if ne is not None:
            self._ne_interp = DensityInterpolator(ne, device)

    def get_phi(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get interpolated phi and derivatives."""
        if self._phi_interp is None:
            raise ValueError("Phi cache not initialized")
        return self._phi_interp.interpolate(x_t)

    def get_ne(self, x_t: torch.Tensor) -> torch.Tensor:
        """Get interpolated electron density."""
        if self._ne_interp is None:
            raise ValueError("Density cache not initialized")
        return self._ne_interp.interpolate(x_t)

    def update_phi(
        self,
        phi: torch.Tensor,
        dphi_dx: torch.Tensor,
        dphi_dxx: torch.Tensor
    ):
        """Update phi cache."""
        self._phi_interp = PoissonInterpolator(phi, dphi_dx, dphi_dxx, self.device)

    def update_ne(self, ne: torch.Tensor):
        """Update density cache."""
        self._ne_interp = DensityInterpolator(ne, self.device)
