"""
Finite Difference Method (FDM) solver for CCP-II system.

Generates reference data for supervised PINN training.
Solves the coupled continuity-Poisson system using explicit time-stepping
for continuity and sparse matrix solve for Poisson.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import torch
from torch.utils.data import Dataset

from ..utils.physics import ParameterSpace, PhysicalConstants


@dataclass
class FDMConfig:
    """FDM solver configuration"""
    nx: int = 50                      # Spatial mesh points
    n_steps_per_cycle: int = 100000   # Time steps per RF cycle
    n_cycles: float = 1.0             # Number of RF cycles to simulate
    save_every: int = 1               # Save interval (1 = save every step)
    enforce_positivity: bool = True   # Clamp negative densities to zero
    verbose: bool = True              # Print progress


class FDMSolver:
    """
    Finite Difference Method solver for CCP-II system.

    Solves the coupled system:
    - Continuity: dn_e/dt + d(Gamma_e)/dx = R(x)
    - Poisson: d2phi/dx2 = -(e/epsilon_0)(n_e - n_i0)

    Uses explicit Euler for continuity and sparse direct solve for Poisson.
    """

    def __init__(self, params: ParameterSpace, config: Optional[FDMConfig] = None):
        self.params = params
        self.config = config or FDMConfig()
        self._setup_grid()
        self._setup_physics()
        self._setup_operators()

    def _setup_grid(self):
        """Initialize spatial and temporal grids"""
        cfg = self.config
        p = self.params

        self.nx = cfg.nx
        self.dx = p.domain.L / cfg.nx
        self.dt = 1.0 / (p.plasma.f * cfg.n_steps_per_cycle)
        self.n_total_steps = int(cfg.n_cycles * cfg.n_steps_per_cycle)

        # Spatial grid (cell centers)
        self.x = np.linspace(0, p.domain.L, cfg.nx)

    def _setup_physics(self):
        """Setup physics coefficients and initial conditions"""
        p = self.params
        c = p.constants

        # Transport coefficients
        self.mu_coef = c.e / (c.m_e * p.plasma.nu_m)  # Mobility
        self.diff_coef = self.mu_coef * p.plasma.T_e_eV  # Diffusion

        # Poisson coefficient: e/epsilon_0
        self.e_epsilon0 = c.e / c.epsilon_0

        # Background ion density from Boltzmann relation
        term = np.sqrt(p.plasma.m_i / (c.e * p.plasma.T_e_eV))
        self.ni0 = p.plasma.R0 * p.domain.reaction_zone_width * term

        # Reaction rate profile R(x)
        self.reac = np.zeros(self.nx, dtype=np.float64)
        x1_idx = int(p.domain.x1 / self.dx)
        x2_idx = int(p.domain.x2 / self.dx)
        L_x2_idx = int((p.domain.L - p.domain.x2) / self.dx)
        L_x1_idx = int((p.domain.L - p.domain.x1) / self.dx)

        self.reac[x1_idx:x2_idx] = p.plasma.R0
        self.reac[L_x2_idx:L_x1_idx] = p.plasma.R0

    def _setup_operators(self):
        """Build sparse Laplacian operator for Poisson equation"""
        nx = self.nx
        dx = self.dx

        # Tridiagonal Laplacian: [1, -2, 1] / dx^2
        e = np.ones(nx, dtype=np.float64)
        diags = np.array([-1, 0, 1])
        vals = np.vstack((e, -2 * e, e))
        Lmtx = sp.spdiags(vals, diags, nx, nx)
        Lmtx = sp.lil_matrix(Lmtx)
        Lmtx /= dx**2

        self.laplacian = sp.csr_matrix(Lmtx)

    def _driving_voltage(self, t: float) -> float:
        """Compute driving voltage V(t) = V0 * sin(2*pi*f*t)"""
        return self.params.plasma.V0 * np.sin(2.0 * np.pi * self.params.plasma.f * t)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run FDM simulation.

        Returns:
            ne: Electron density array (nt_saved, nx) - physical units (m^-3)
            phi: Electric potential array (nt_saved, nx) - physical units (V)
            t_array: Time array (nt_saved,) - physical units (s)
        """
        cfg = self.config
        p = self.params

        # Initialize electron density with background ion density
        ne = np.ones(self.nx, dtype=np.float64) * self.ni0

        # Storage for saved snapshots
        save_interval = max(1, cfg.save_every)
        n_saved = self.n_total_steps // save_interval + 1
        ne_container = []
        phi_container = []
        t_container = []

        t = 0.0
        dx = self.dx
        dt = self.dt

        if cfg.verbose:
            print(f"FDM Solver: {self.n_total_steps} steps, dt={dt:.2e}s, dx={dx:.4e}m")

        for step in range(self.n_total_steps):
            # Solve Poisson equation: Laplacian @ phi = -e/eps0 * (ni0 - ne)
            rhs = -self.e_epsilon0 * (self.ni0 - ne)
            v_source = self._driving_voltage(t)
            rhs[-1] -= v_source / dx**2  # Dirichlet BC at right boundary

            phi = spsolve(self.laplacian, rhs, permc_spec="MMD_AT_PLUS_A")

            # Store old density for explicit update
            ne_old = np.copy(ne)

            # Update boundary cells (x=0)
            ne[0] += dt * (
                self.reac[0]
                + self.diff_coef * (-2.0 * ne_old[0] + ne_old[1]) / dx**2
                + (-self.mu_coef * 0.5 / dx**2) * (
                    (ne_old[1] + ne_old[0]) * (phi[1] - phi[0])
                    - ne_old[0] * phi[0]
                )
            )

            # Update boundary cells (x=L)
            ne[-1] += dt * (
                self.reac[-1]
                + self.diff_coef * (ne_old[-2] - 2.0 * ne_old[-1]) / dx**2
                + (-self.mu_coef * 0.5 / dx**2) * (
                    ne_old[-1] * (v_source - phi[-1])
                    - (ne_old[-2] + ne_old[-1]) * (phi[-1] - phi[-2])
                )
            )

            # Update interior cells
            ne[1:-1] += dt * (
                self.reac[1:-1]
                + self.diff_coef * (
                    ne_old[:-2] - 2.0 * ne_old[1:-1] + ne_old[2:]
                ) / dx**2
                + (-self.mu_coef * 0.5 / dx**2) * (
                    (ne_old[2:] + ne_old[1:-1]) * (phi[2:] - phi[1:-1])
                    - (ne_old[:-2] + ne_old[1:-1]) * (phi[1:-1] - phi[:-2])
                )
            )

            # Enforce positivity
            if cfg.enforce_positivity:
                ne[ne < 0.0] = 0.0

            # Save snapshot
            if step % save_interval == 0:
                ne_container.append(np.copy(ne))
                phi_container.append(np.copy(phi))
                t_container.append(t)

            t += dt

            # Progress reporting
            if cfg.verbose and (step + 1) % (self.n_total_steps // 10) == 0:
                print(f"  Step {step + 1}/{self.n_total_steps} ({100*(step+1)/self.n_total_steps:.0f}%)")

        # Convert to arrays: (nt, nx)
        ne_array = np.array(ne_container)
        phi_array = np.array(phi_container)
        t_array = np.array(t_container)

        if cfg.verbose:
            print(f"FDM complete. Output shape: ne={ne_array.shape}, phi={phi_array.shape}")

        return ne_array, phi_array, t_array

    def save(self, path: str, ne: np.ndarray, phi: np.ndarray, t_array: np.ndarray):
        """Save solution to NPZ file"""
        np.savez(
            path,
            ne=ne,
            phi=phi,
            t=t_array,
            x=self.x,
            nx=self.nx,
            dt=self.dt,
            dx=self.dx,
            params=self.params.to_dict()
        )
        print(f"Saved FDM solution to {path}")

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load pre-computed FDM solution.

        Returns:
            ne: Electron density (nt, nx)
            phi: Electric potential (nt, nx)
            t: Time array (nt,)
            x: Spatial array (nx,)
        """
        data = np.load(path, allow_pickle=True)
        return data['ne'], data['phi'], data['t'], data['x']


class FDMDataset(Dataset):
    """
    PyTorch Dataset wrapping FDM reference data for supervised training.

    Provides (x, t) -> (n_e, phi) pairs from FDM solution.
    """

    def __init__(
        self,
        fdm_path: str,
        normalize: bool = True,
        n_ref: float = 1e14,
        phi_ref: float = 100.0,
        x_ref: float = 0.025,
        t_ref: Optional[float] = None,
        device: str = "cpu"
    ):
        """
        Args:
            fdm_path: Path to FDM solution NPZ file
            normalize: Whether to normalize values to [0, 1] range
            n_ref: Reference density for normalization
            phi_ref: Reference potential for normalization
            x_ref: Reference length for normalization
            t_ref: Reference time for normalization (uses max t if None)
            device: Device to store tensors on
        """
        ne, phi, t, x = FDMSolver.load(fdm_path)

        self.normalize = normalize
        self.n_ref = n_ref
        self.phi_ref = phi_ref
        self.x_ref = x_ref
        self.t_ref = t_ref if t_ref is not None else t.max()

        # Get grid dimensions
        nt, nx = ne.shape

        # Create coordinate meshgrid
        x_vals = x / self.x_ref if normalize else x
        t_vals = t / self.t_ref if normalize else t
        X, T = np.meshgrid(x_vals, t_vals, indexing='ij')

        # Normalize fields
        if normalize:
            ne = ne / n_ref
            phi = phi / phi_ref

        # Flatten and convert to tensors
        # Note: ne and phi are (nt, nx), need to transpose for (nx, nt) meshgrid
        self.coords = torch.tensor(
            np.stack([X.T.flatten(), T.T.flatten()], axis=1),
            dtype=torch.float32,
            device=device
        )
        self.ne = torch.tensor(ne.flatten(), dtype=torch.float32, device=device)
        self.phi = torch.tensor(phi.flatten(), dtype=torch.float32, device=device)

        self.nt = nt
        self.nx = nx

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        """Return (coords, ne, phi) tuple"""
        return self.coords[idx], self.ne[idx], self.phi[idx]

    def get_full_solution(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full solution as 2D tensors (nx, nt)"""
        return (
            self.coords.reshape(self.nt, self.nx, 2).permute(1, 0, 2),
            self.ne.reshape(self.nt, self.nx).T,
            self.phi.reshape(self.nt, self.nx).T
        )


def generate_fdm_reference(
    params: Optional[ParameterSpace] = None,
    config: Optional[FDMConfig] = None,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to generate FDM reference data.

    Args:
        params: Physics parameters (uses defaults if None)
        config: FDM config (uses defaults if None)
        output_path: Path to save NPZ file (optional)

    Returns:
        ne, phi, t arrays
    """
    params = params or ParameterSpace()
    config = config or FDMConfig()

    solver = FDMSolver(params, config)
    ne, phi, t = solver.solve()

    if output_path:
        solver.save(output_path, ne, phi, t)

    return ne, phi, t


if __name__ == "__main__":
    # Quick test
    params = ParameterSpace()
    config = FDMConfig(n_cycles=0.1, n_steps_per_cycle=10000, verbose=True)

    ne, phi, t = generate_fdm_reference(params, config, "fdm_test.npz")
    print(f"Generated: ne={ne.shape}, phi={phi.shape}, t={t.shape}")
