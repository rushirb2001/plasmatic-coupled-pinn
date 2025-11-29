"""
Poisson solvers for hybrid PINN approaches.

These solve the Poisson equation given n_e from the neural network,
enabling hybrid NN + numerical solver approaches.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, splu
from scipy.interpolate import griddata
from typing import Tuple


class PoissonSolverCPU:
    """CPU-based Poisson solver using sparse LU decomposition."""

    def __init__(
        self,
        nx: int = 50,
        nt: int = 20000,
        t_max: float = 0.3,
        delta: float = 1.0,
        n_io: float = 0.0,
        V0: float = 1.0,
        device: str = "cpu"
    ):
        self.nx = nx
        self.nt = nt
        self.delta = delta
        self.n_io = n_io
        self.V0 = V0
        self.device = device

        self.x_uniform = torch.linspace(0, 1, nx)
        self.t_uniform = torch.linspace(0, t_max, nt)
        self.X, self.T = torch.meshgrid(self.x_uniform, self.t_uniform, indexing='ij')
        self.x_t_uniform = torch.stack([self.X.flatten(), self.T.flatten()], dim=1).to(device)

        self._build_laplace_solver()

    def _build_laplace_solver(self):
        dx = 1.0 / (self.nx - 1)
        e_vec = np.ones(self.nx)
        laplace_mat = diags([e_vec, -2 * e_vec, e_vec], offsets=[-1, 0, 1], shape=(self.nx, self.nx)) / dx**2
        laplace_mat = laplace_mat.tolil()
        laplace_mat[0, :] = laplace_mat[-1, :] = 0
        laplace_mat[0, 0] = laplace_mat[-1, -1] = 1
        laplace_mat = laplace_mat.tocsr()
        self.lu_obj = splu(laplace_mat)

    def solve(self, n_e_uniform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Poisson equation given electron density.

        Args:
            n_e_uniform: Electron density on uniform grid (nx, nt)

        Returns:
            phi, dphi_dx, dphi_dxx: Potential and derivatives (nx, nt)
        """
        dx = 1.0 / (self.nx - 1)
        rhs_all = self.delta * (n_e_uniform - self.n_io)

        t_vals = self.t_uniform.cpu().numpy()
        rhs_all[0, :] = 0
        rhs_all[-1, :] = self.V0 * np.sin(2 * np.pi * t_vals)

        phi = self.lu_obj.solve(rhs_all)

        dphi_dx = np.zeros_like(phi)
        dphi_dxx = np.zeros_like(phi)
        dphi_dx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
        dphi_dxx[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / dx**2
        dphi_dx[0, :] = (phi[1, :] - phi[0, :]) / dx
        dphi_dx[-1, :] = (phi[-1, :] - phi[-2, :]) / dx

        return phi, dphi_dx, dphi_dxx

    def interpolate(
        self,
        x_t_rand: torch.Tensor,
        phi: np.ndarray,
        dphi_dx: np.ndarray,
        dphi_dxx: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate solution to random points using griddata."""
        points_rand = x_t_rand.detach().cpu().numpy()
        x_grid_np = self.X.cpu().numpy().flatten()
        t_grid_np = self.T.cpu().numpy().flatten()
        points_uniform = np.stack([x_grid_np, t_grid_np], axis=1)

        phi_rand = griddata(points_uniform, phi.flatten(), points_rand, method='cubic')
        dphi_dx_rand = griddata(points_uniform, dphi_dx.flatten(), points_rand, method='cubic')
        dphi_dxx_rand = griddata(points_uniform, dphi_dxx.flatten(), points_rand, method='cubic')

        return (
            torch.tensor(phi_rand, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.tensor(dphi_dx_rand, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.tensor(dphi_dxx_rand, dtype=torch.float32, device=self.device).unsqueeze(-1),
        )


class PoissonSolverGPU:
    """GPU-based Poisson solver using batched torch.linalg.solve."""

    def __init__(
        self,
        nx: int = 50,
        nt: int = 20000,
        t_max: float = 0.3,
        delta: float = 1.0,
        n_io: float = 0.0,
        V0: float = 1.0,
        device: str = "cuda"
    ):
        self.nx = nx
        self.nt = nt
        self.delta = delta
        self.n_io = n_io
        self.V0 = V0
        self.device = device

        self.x_uniform = torch.linspace(0, 1, nx, device=device)
        self.t_uniform = torch.linspace(0, t_max, nt, device=device)
        self.X, self.T = torch.meshgrid(self.x_uniform, self.t_uniform, indexing='ij')
        self.x_t_uniform = torch.stack([self.X.flatten(), self.T.flatten()], dim=1)

        self._build_laplace_matrix()

    def _build_laplace_matrix(self):
        dx = 1.0 / (self.nx - 1)
        e = torch.ones(self.nx, device=self.device)
        L = torch.diag(-2 * e) + torch.diag(e[:-1], diagonal=1) + torch.diag(e[:-1], diagonal=-1)
        L = L / dx**2

        L[0, :] = 0
        L[-1, :] = 0
        L[0, 0] = 1
        L[-1, -1] = 1

        self.laplace_mat = L

    def solve(self, n_e_uniform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve Poisson equation on GPU."""
        dx = 1.0 / (self.nx - 1)
        rhs_all = self.delta * (n_e_uniform - self.n_io)

        rhs_all[0, :] = 0
        rhs_all[-1, :] = self.V0 * torch.sin(2 * torch.pi * self.t_uniform) / dx**2

        L = self.laplace_mat.unsqueeze(0).expand(self.nt, -1, -1)
        B = rhs_all.T.unsqueeze(-1)
        phi = torch.linalg.solve(L, B).squeeze(-1).T

        dphi_dx = torch.zeros_like(phi)
        dphi_dxx = torch.zeros_like(phi)
        dphi_dx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
        dphi_dxx[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / dx**2
        dphi_dx[0, :] = (phi[1, :] - phi[0, :]) / dx
        dphi_dx[-1, :] = (phi[-1, :] - phi[-2, :]) / dx

        return phi, dphi_dx, dphi_dxx

    def interpolate_bilinear(
        self,
        x_t_rand: torch.Tensor,
        phi: torch.Tensor,
        dphi_dx: torch.Tensor,
        dphi_dxx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU bilinear interpolation using grid_sample."""
        phi_img = phi.unsqueeze(0).unsqueeze(0)
        dphi_dx_img = dphi_dx.unsqueeze(0).unsqueeze(0)
        dphi_dxx_img = dphi_dxx.unsqueeze(0).unsqueeze(0)

        x = x_t_rand[:, 0] * 2 - 1
        t = (x_t_rand[:, 1] / 0.3) * 2 - 1
        grid = torch.stack((t, x), dim=-1).unsqueeze(0).unsqueeze(0)

        phi_rand = F.grid_sample(phi_img, grid, mode='bilinear', align_corners=True).squeeze()
        dphi_dx_rand = F.grid_sample(dphi_dx_img, grid, mode='bilinear', align_corners=True).squeeze()
        dphi_dxx_rand = F.grid_sample(dphi_dxx_img, grid, mode='bilinear', align_corners=True).squeeze()

        return (
            phi_rand.unsqueeze(-1),
            dphi_dx_rand.unsqueeze(-1),
            dphi_dxx_rand.unsqueeze(-1),
        )
