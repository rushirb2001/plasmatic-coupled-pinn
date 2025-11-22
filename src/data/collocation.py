"""
Collocation point sampling for PINN training.

Provides a hierarchy of sampler classes:
- CollocationSampler: Abstract base class
- UniformSampler: Uniform random sampling (pre-generated)
- UniformLazySampler: Uniform random sampling (per-sample generation)
- PhysicalUnitsSampler: Sampling in physical units (not normalized)
- BetaSampler: Beta distribution sampling for boundary concentration
- LatinHypercubeSampler: LHS for stratified coverage
- GridSampler: Regular mesh grid sampling
- BoundarySampler: Samples boundary points for BC enforcement
- MeshgridSampler: Dense meshgrid for visualization/validation
- MeshgridRandomTSampler: Structured x-grid with random t
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List
import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class SamplingConfig:
    """Configuration for collocation sampling"""
    num_samples: int = 10000
    x_range: Tuple[float, float] = (0.0, 1.0)
    t_range: Tuple[float, float] = (0.0, 1.0)
    clamp_x: bool = True
    eps: float = 1e-8
    device: str = "cpu"
    dtype: torch.dtype = field(default=torch.float32)

    def __post_init__(self):
        # Convert string dtype if needed
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)


class CollocationSampler(ABC, Dataset):
    """
    Base class for all collocation point samplers.

    Implements both the sampler interface and PyTorch Dataset interface
    for seamless integration with DataLoader.
    """

    def __init__(self, config: SamplingConfig):
        self.config = config
        self._samples: Optional[torch.Tensor] = None

    @abstractmethod
    def _generate(self) -> torch.Tensor:
        """Generate collocation points - implement in subclass"""
        pass

    @property
    def samples(self) -> torch.Tensor:
        """Lazy generation of samples"""
        if self._samples is None:
            self._samples = self._generate()
        return self._samples

    def resample(self) -> torch.Tensor:
        """Force regeneration of samples"""
        self._samples = self._generate()
        return self._samples

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]

    def to_device(self, device: str) -> "CollocationSampler":
        """Move samples to specified device"""
        self._samples = self.samples.to(device)
        self.config.device = device
        return self

    def get_x_t(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get x and t coordinates as separate tensors"""
        samples = self.samples
        return samples[:, 0:1], samples[:, 1:2]


class UniformSampler(CollocationSampler):
    """
    Uniform random sampling over domain (pre-generated batch).

    All samples are generated at initialization and cached.
    Corresponds to original RandomCollocations2.
    """

    def _generate(self) -> torch.Tensor:
        cfg = self.config
        n = cfg.num_samples

        x = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)
        x = x * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]

        t = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)
        t = t * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        if cfg.clamp_x:
            x = x.clamp(cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        return torch.cat([x, t], dim=1)


class UniformLazySampler(CollocationSampler):
    """
    Uniform random sampling with per-sample generation.

    Each __getitem__ call generates a fresh random sample.
    Corresponds to original RandomCollocations1.
    Useful for truly random batches during training.
    """

    def _generate(self) -> torch.Tensor:
        # This won't be used for __getitem__, but needed for interface
        return self._generate_batch(self.config.num_samples)

    def _generate_batch(self, n: int) -> torch.Tensor:
        cfg = self.config
        x = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)
        x = x * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]

        t = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)
        t = t * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        if cfg.clamp_x:
            x = x.clamp(cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        return torch.cat([x, t], dim=1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate fresh random sample for each call"""
        cfg = self.config
        x = torch.rand(1, device=cfg.device, dtype=cfg.dtype)
        x = x * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]

        t = torch.rand(1, device=cfg.device, dtype=cfg.dtype)
        t = t * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        if cfg.clamp_x:
            x = x.clamp(cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        return torch.cat([x, t], dim=0)


class PhysicalUnitsSampler(CollocationSampler):
    """
    Uniform sampling in physical units (not normalized).

    Samples x in [0, L] meters and t in [0, T] seconds.
    Corresponds to original RandomCollocations_wo_non_dim.

    Args:
        config: SamplingConfig (x_range and t_range should be physical values)
        L: Domain length in meters (default: 0.025m)
        T: Time span in seconds (default: 1/13.56MHz)
    """

    def __init__(
        self,
        config: SamplingConfig,
        L: float = 0.025,
        T: float = 1.0 / 13.56e6
    ):
        super().__init__(config)
        self.L = L
        self.T = T

    def _generate(self) -> torch.Tensor:
        cfg = self.config
        n = cfg.num_samples

        # Sample in physical units
        x = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype) * self.L
        t = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype) * self.T

        if cfg.clamp_x:
            x = x.clamp(cfg.eps, self.L - cfg.eps)

        return torch.cat([x, t], dim=1)


class BetaSampler(CollocationSampler):
    """
    Beta distribution sampling for importance weighting.

    Useful for concentrating samples near boundaries where gradients are steep.
    Uses Beta distribution for both x and t coordinates.
    Corresponds to original BetaNormalCollocations.
    """

    def __init__(
        self,
        config: SamplingConfig,
        alpha_x: float = 1.0,
        beta_x: float = 1.0,
        alpha_t: float = 1.0,
        beta_t: float = 1.0
    ):
        super().__init__(config)
        self.alpha_x = alpha_x
        self.beta_x = beta_x
        self.alpha_t = alpha_t
        self.beta_t = beta_t

    def update_parameters(
        self,
        alpha_x: Optional[float] = None,
        beta_x: Optional[float] = None,
        alpha_t: Optional[float] = None,
        beta_t: Optional[float] = None
    ):
        """Update distribution parameters (useful for curriculum learning)"""
        if alpha_x is not None:
            self.alpha_x = alpha_x
        if beta_x is not None:
            self.beta_x = beta_x
        if alpha_t is not None:
            self.alpha_t = alpha_t
        if beta_t is not None:
            self.beta_t = beta_t
        self._samples = None  # Invalidate cache

    def _generate(self) -> torch.Tensor:
        cfg = self.config
        n = cfg.num_samples

        x_dist = torch.distributions.Beta(self.alpha_x, self.beta_x)
        t_dist = torch.distributions.Beta(self.alpha_t, self.beta_t)

        x = x_dist.sample((n, 1))
        x = x * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]

        t = t_dist.sample((n, 1))
        t = t * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        if cfg.clamp_x:
            x = x.clamp(cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        return torch.cat([x, t], dim=1).to(cfg.device)


class LatinHypercubeSampler(CollocationSampler):
    """
    Latin Hypercube sampling for stratified coverage.

    Provides better space-filling properties than random sampling.
    Corresponds to original LatinHyperCubeDataset.
    Requires pyDOE package.
    """

    def _generate(self) -> torch.Tensor:
        try:
            from pyDOE import lhs
        except ImportError:
            raise ImportError("pyDOE is required for Latin Hypercube sampling. Install with: pip install pyDOE")

        cfg = self.config
        n = cfg.num_samples

        # Generate LHS samples in [0, 1]^2
        samples = lhs(2, samples=n).astype(np.float32)

        # Scale to domain
        x = samples[:, 0:1] * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]
        t = samples[:, 1:2] * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        if cfg.clamp_x:
            x = np.clip(x, cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        data = np.concatenate([x, t], axis=1)
        return torch.tensor(data, dtype=cfg.dtype, device=cfg.device)


class GridSampler(CollocationSampler):
    """
    Regular mesh grid sampling.

    Useful for validation/test sets and visualization.
    Corresponds to original MeshgridCollocations.
    """

    def __init__(self, config: SamplingConfig, nx: int = 100, nt: int = 100):
        super().__init__(config)
        self.nx = nx
        self.nt = nt
        # Update num_samples to match grid size
        self.config.num_samples = nx * nt

    def _generate(self) -> torch.Tensor:
        cfg = self.config

        x_vals = torch.linspace(
            cfg.x_range[0], cfg.x_range[1], self.nx,
            device=cfg.device, dtype=cfg.dtype
        )
        t_vals = torch.linspace(
            cfg.t_range[0], cfg.t_range[1], self.nt,
            device=cfg.device, dtype=cfg.dtype
        )

        X, T = torch.meshgrid(x_vals, t_vals, indexing='ij')
        return torch.stack([X.flatten(), T.flatten()], dim=1)

    def get_meshgrid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get 2D meshgrid tensors for visualization"""
        cfg = self.config
        x_vals = torch.linspace(
            cfg.x_range[0], cfg.x_range[1], self.nx,
            device=cfg.device, dtype=cfg.dtype
        )
        t_vals = torch.linspace(
            cfg.t_range[0], cfg.t_range[1], self.nt,
            device=cfg.device, dtype=cfg.dtype
        )
        return torch.meshgrid(x_vals, t_vals, indexing='ij')


class MeshgridRandomTSampler(CollocationSampler):
    """
    Structured spatial grid with random temporal sampling.

    Useful for problems with smooth spatial gradients but
    needing coverage across time.
    Corresponds to original Meshgrid_t_random_Collocations.
    """

    def __init__(self, config: SamplingConfig, nx: int = 50, nt: int = 100000):
        super().__init__(config)
        self.nx = nx
        self.nt = nt
        self.config.num_samples = nx * nt

    def _generate(self) -> torch.Tensor:
        cfg = self.config

        x_vals = torch.linspace(
            cfg.x_range[0], cfg.x_range[1], self.nx,
            device=cfg.device, dtype=cfg.dtype
        )
        t_vals = torch.rand(self.nt, device=cfg.device, dtype=cfg.dtype)
        t_vals = t_vals * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]

        X, T = torch.meshgrid(x_vals, t_vals, indexing='ij')
        return torch.stack([X.flatten(), T.flatten()], dim=1)


class BoundarySampler(CollocationSampler):
    """
    Samples boundary points for BC enforcement.

    Generates points at x=0 and x=L boundaries.
    """

    def _generate(self) -> torch.Tensor:
        cfg = self.config
        n = cfg.num_samples
        k0, k1 = n // 2, n - n // 2

        # x=0 boundary (left)
        t0 = torch.rand(k0, 1, device=cfg.device, dtype=cfg.dtype)
        t0 = t0 * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]
        x0 = torch.full((k0, 1), cfg.x_range[0], device=cfg.device, dtype=cfg.dtype)

        # x=L boundary (right)
        t1 = torch.rand(k1, 1, device=cfg.device, dtype=cfg.dtype)
        t1 = t1 * (cfg.t_range[1] - cfg.t_range[0]) + cfg.t_range[0]
        x1 = torch.full((k1, 1), cfg.x_range[1], device=cfg.device, dtype=cfg.dtype)

        left = torch.cat([x0, t0], dim=1)
        right = torch.cat([x1, t1], dim=1)

        return torch.cat([left, right], dim=0)

    def get_left_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get left and right boundary samples separately"""
        samples = self.samples
        n = len(samples)
        k0 = n // 2
        return samples[:k0], samples[k0:]


class InitialConditionSampler(CollocationSampler):
    """
    Samples initial condition points at t=0.

    For time-dependent problems where IC needs enforcement.
    """

    def _generate(self) -> torch.Tensor:
        cfg = self.config
        n = cfg.num_samples

        x = torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)
        x = x * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]

        if cfg.clamp_x:
            x = x.clamp(cfg.x_range[0] + cfg.eps, cfg.x_range[1] - cfg.eps)

        t = torch.full((n, 1), cfg.t_range[0], device=cfg.device, dtype=cfg.dtype)

        return torch.cat([x, t], dim=1)


# Factory function for YAML config
def create_sampler(
    name: str,
    config: Optional[SamplingConfig] = None,
    **kwargs
) -> CollocationSampler:
    """
    Factory to create sampler from config.

    Args:
        name: Sampler type ('uniform', 'uniform_lazy', 'physical', 'beta', 'lhs',
              'grid', 'boundary', 'ic', 'meshgrid_random_t')
        config: SamplingConfig instance (uses defaults if None)
        **kwargs: Additional sampler-specific arguments

    Returns:
        CollocationSampler instance
    """
    config = config or SamplingConfig()

    samplers = {
        'uniform': UniformSampler,
        'uniform_lazy': UniformLazySampler,
        'physical': PhysicalUnitsSampler,
        'physical_units': PhysicalUnitsSampler,
        'beta': BetaSampler,
        'lhs': LatinHypercubeSampler,
        'latin_hypercube': LatinHypercubeSampler,
        'grid': GridSampler,
        'meshgrid': GridSampler,
        'boundary': BoundarySampler,
        'ic': InitialConditionSampler,
        'initial_condition': InitialConditionSampler,
        'meshgrid_random_t': MeshgridRandomTSampler,
    }

    if name.lower() not in samplers:
        raise ValueError(f"Unknown sampler type: {name}. Available: {list(samplers.keys())}")

    return samplers[name.lower()](config, **kwargs)


def create_sampler_from_yaml(config_dict: dict) -> CollocationSampler:
    """
    Create sampler from YAML config dict.

    Expected format:
        sampling:
          type: uniform
          num_samples: 10000
          x_range: [0.0, 1.0]
          t_range: [0.0, 1.0]
          clamp_x: true
          # Additional sampler-specific params
          alpha_x: 1.0  # for beta sampler
          nx: 100       # for grid sampler
    """
    sampling_cfg = dict(config_dict.get('sampling', config_dict))

    sampler_type = sampling_cfg.pop('type', 'uniform')

    # Extract SamplingConfig fields
    config_fields = {
        'num_samples', 'x_range', 't_range', 'clamp_x', 'eps', 'device', 'dtype'
    }
    config_kwargs = {k: v for k, v in sampling_cfg.items() if k in config_fields}

    # Convert lists to tuples for ranges
    if 'x_range' in config_kwargs and isinstance(config_kwargs['x_range'], list):
        config_kwargs['x_range'] = tuple(config_kwargs['x_range'])
    if 't_range' in config_kwargs and isinstance(config_kwargs['t_range'], list):
        config_kwargs['t_range'] = tuple(config_kwargs['t_range'])

    config = SamplingConfig(**config_kwargs)

    # Remaining kwargs are sampler-specific
    sampler_kwargs = {k: v for k, v in sampling_cfg.items() if k not in config_fields}

    return create_sampler(sampler_type, config, **sampler_kwargs)
