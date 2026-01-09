"""
Physics module for CCP-II PINN model.

Provides nested dataclass hierarchy for physics parameters:
- PhysicalConstants: Immutable SI constants
- DomainParameters: Spatial domain configuration
- PlasmaParameters: Plasma physics parameters with derived properties
- ScalingParameters: Non-dimensionalization reference scales
- ParameterSpace: Container class with YAML loading support
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import yaml
import itertools


@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants (SI units)"""
    e: float = 1.60217663e-19      # Elementary charge (C)
    m_e: float = 9.10938356e-31    # Electron mass (kg)
    epsilon_0: float = 8.85418781e-12  # Vacuum permittivity (F/m)
    k_b: float = 1.380649e-23      # Boltzmann constant (J/K)
    amu: float = 1.66054e-27       # Atomic mass unit (kg)


@dataclass
class DomainParameters:
    """Spatial and temporal domain configuration"""
    L: float = 0.025              # Domain length (m)
    x1: float = 0.005             # Reaction zone start (m)
    x2: float = 0.01              # Reaction zone end (m)

    @property
    def reaction_zone_width(self) -> float:
        """Width of reaction zone"""
        return self.x2 - self.x1

    @property
    def x1_normalized(self) -> float:
        """Normalized reaction zone start"""
        return self.x1 / self.L

    @property
    def x2_normalized(self) -> float:
        """Normalized reaction zone end"""
        return self.x2 / self.L


@dataclass
class PlasmaParameters:
    """Plasma physics parameters with derived transport coefficients"""
    f: float = 13.56e6            # RF frequency (Hz)
    V0: float = 40.0              # Voltage amplitude (V)
    R0: float = 2.3e20            # Reaction rate (m^-3 s^-1)
    T_e_eV: float = 3.0           # Electron temperature (eV)
    m_i_amu: float = 40.0         # Ion mass (amu)
    nu_m: float = 1e8             # Collision frequency (s^-1)

    # Reference to constants (set by ParameterSpace)
    _constants: Optional[PhysicalConstants] = field(default=None, repr=False)

    @property
    def constants(self) -> PhysicalConstants:
        """Get physical constants, using defaults if not set"""
        return self._constants or PhysicalConstants()

    @property
    def T_e(self) -> float:
        """Electron temperature in Kelvin"""
        return self.T_e_eV * self.constants.e / self.constants.k_b

    @property
    def m_i(self) -> float:
        """Ion mass in kg"""
        return self.m_i_amu * self.constants.amu

    @property
    def omega(self) -> float:
        """Angular frequency (rad/s)"""
        return 2.0 * 3.141592653589793 * self.f

    @property
    def period(self) -> float:
        """RF period (s)"""
        return 1.0 / self.f

    @property
    def D(self) -> float:
        """Electron diffusion coefficient (m^2/s)"""
        c = self.constants
        return (c.e * self.T_e_eV) / (c.m_e * self.nu_m)

    @property
    def mu(self) -> float:
        """Electron mobility (m^2/V/s)"""
        c = self.constants
        return c.e / (c.m_e * self.nu_m)


@dataclass
class ScalingParameters:
    """Non-dimensionalization reference scales"""
    x_ref: float = 0.025          # Reference length (m) - typically L
    t_ref: float = 7.374e-8       # Reference time (s) - typically 1/f
    n_ref: float = 1.0e14         # Reference density (m^-3)
    phi_ref: float = 40.0         # Reference potential (V) - typically V0


@dataclass
class FDMParameters:
    """FDM solver resolution parameters for reference data generation"""
    nx: int = 1024                      # Spatial mesh points
    n_steps_per_cycle: int = 1400000    # Time steps per RF cycle
    save_every: int = 100               # Save interval (reduces output size)

    @classmethod
    def benchmark(cls) -> "FDMParameters":
        """Benchmark resolution matching colleague's best model"""
        return cls(nx=1024, n_steps_per_cycle=1400000, save_every=100)

    @classmethod
    def medium(cls) -> "FDMParameters":
        """Medium resolution for faster iteration"""
        return cls(nx=256, n_steps_per_cycle=500000, save_every=10)

    @classmethod
    def low(cls) -> "FDMParameters":
        """Low resolution for quick testing"""
        return cls(nx=50, n_steps_per_cycle=100000, save_every=1)

    @classmethod
    def from_physics(cls, domain: DomainParameters, plasma: PlasmaParameters) -> "ScalingParameters":
        """Create scaling parameters from physics parameters"""
        return cls(
            x_ref=domain.L,
            t_ref=1.0 / plasma.f,
            n_ref=1.0e16,  # Typical plasma density
            phi_ref=plasma.V0
        )


@dataclass
class AdaptiveScalingParameters:
    """
    Adaptive non-dimensionalization reference scales.

    Unlike ScalingParameters which uses fixed n_ref, this class computes
    n_ref = n_io (background ion density) to ensure normalized electron
    density n_e' is O(1) across all parameter regimes.

    This improves training stability when R0 or V0 vary significantly.
    """
    x_ref: float
    t_ref: float
    n_ref: float
    phi_ref: float

    @classmethod
    def from_physics(
        cls,
        domain: DomainParameters,
        plasma: PlasmaParameters,
        constants: Optional[PhysicalConstants] = None
    ) -> "AdaptiveScalingParameters":
        """
        Compute regime-appropriate reference scales.

        Key difference from ScalingParameters: n_ref is computed from physics
        as n_io = R0 * (x2-x1) * sqrt(m_i / (e * T_e)), ensuring normalized
        electron density is O(1) regardless of R0.

        Args:
            domain: Spatial domain parameters
            plasma: Plasma physics parameters
            constants: Physical constants (uses defaults if None)

        Returns:
            AdaptiveScalingParameters with n_ref = n_io
        """
        c = constants or PhysicalConstants()

        # Standard scales
        x_ref = domain.L
        t_ref = 1.0 / plasma.f
        phi_ref = plasma.V0

        # Adaptive n_ref: compute n_io from Boltzmann relation
        # n_io = R0 * (x2 - x1) * sqrt(m_i / (e * T_e))
        m_i = plasma.m_i_amu * c.amu
        n_io_physical = (
            plasma.R0 *
            domain.reaction_zone_width *
            (m_i / (c.e * plasma.T_e_eV)) ** 0.5
        )

        # Use n_io as n_ref so that normalized n_io = 1.0
        n_ref = n_io_physical

        return cls(x_ref=x_ref, t_ref=t_ref, n_ref=n_ref, phi_ref=phi_ref)


class ParameterSpace:
    """
    Nested container for all physics parameters.
    Supports YAML loading and discrete parameter configurations.
    """

    def __init__(
        self,
        domain: Optional[DomainParameters] = None,
        plasma: Optional[PlasmaParameters] = None,
        scales: Optional[ScalingParameters] = None,
        constants: Optional[PhysicalConstants] = None,
        fdm: Optional[FDMParameters] = None
    ):
        self.constants = constants or PhysicalConstants()
        self.domain = domain or DomainParameters()
        self.plasma = plasma or PlasmaParameters()
        self.scales = scales or ScalingParameters()
        self.fdm = fdm or FDMParameters.benchmark()

        # Link constants to plasma parameters
        object.__setattr__(self.plasma, '_constants', self.constants)

    @classmethod
    def from_yaml(cls, path: str) -> "ParameterSpace":
        """Load parameters from YAML config file"""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        physics_cfg = cfg.get('physics', cfg)  # Support nested or flat structure

        # Load FDM parameters if present, otherwise use benchmark defaults
        fdm_cfg = physics_cfg.get('fdm', {})
        fdm = FDMParameters(**fdm_cfg) if fdm_cfg else FDMParameters.benchmark()

        return cls(
            domain=DomainParameters(**physics_cfg.get('domain', {})),
            plasma=PlasmaParameters(**{
                k: v for k, v in physics_cfg.get('plasma', {}).items()
                if k != '_constants'
            }),
            scales=ScalingParameters(**physics_cfg.get('scales', {})),
            fdm=fdm
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export to dict for WandB logging"""
        return {
            'constants': {
                'e': self.constants.e,
                'm_e': self.constants.m_e,
                'epsilon_0': self.constants.epsilon_0,
                'k_b': self.constants.k_b,
            },
            'domain': {
                'L': self.domain.L,
                'x1': self.domain.x1,
                'x2': self.domain.x2,
                'reaction_zone_width': self.domain.reaction_zone_width,
            },
            'plasma': {
                'f': self.plasma.f,
                'V0': self.plasma.V0,
                'R0': self.plasma.R0,
                'T_e_eV': self.plasma.T_e_eV,
                'm_i_amu': self.plasma.m_i_amu,
                'nu_m': self.plasma.nu_m,
                'D': self.plasma.D,
                'mu': self.plasma.mu,
            },
            'scales': {
                'x_ref': self.scales.x_ref,
                't_ref': self.scales.t_ref,
                'n_ref': self.scales.n_ref,
                'phi_ref': self.scales.phi_ref,
            },
            'fdm': {
                'nx': self.fdm.nx,
                'n_steps_per_cycle': self.fdm.n_steps_per_cycle,
                'save_every': self.fdm.save_every,
            }
        }

    def with_override(self, **overrides) -> "ParameterSpace":
        """
        Create new ParameterSpace with specific parameter overrides.

        Args:
            **overrides: Dot-notation parameter overrides, e.g.:
                - plasma.V0=150
                - domain.L=0.03

        Returns:
            New ParameterSpace with overrides applied
        """
        import copy

        new_domain = copy.copy(self.domain)
        new_plasma = copy.copy(self.plasma)
        new_scales = copy.copy(self.scales)

        for key, value in overrides.items():
            parts = key.split('.')
            if len(parts) == 2:
                category, param = parts
                if category == 'domain':
                    setattr(new_domain, param, value)
                elif category == 'plasma':
                    setattr(new_plasma, param, value)
                elif category == 'scales':
                    setattr(new_scales, param, value)
            else:
                raise ValueError(f"Invalid override key: {key}. Use 'category.param' format.")

        return ParameterSpace(
            domain=new_domain,
            plasma=new_plasma,
            scales=new_scales,
            constants=self.constants
        )

    @classmethod
    def create_discrete_configs(
        cls,
        base: Optional["ParameterSpace"] = None,
        variations: Optional[Dict[str, List]] = None
    ) -> List["ParameterSpace"]:
        """
        Generate discrete parameter configurations (not continuous sweeps).

        Args:
            base: Base ParameterSpace to start from
            variations: Dict mapping parameter paths to lists of values, e.g.:
                {
                    'plasma.V0': [100, 150, 200],
                    'plasma.f': [13.56e6, 27.12e6]
                }

        Returns:
            List of ParameterSpace objects for each combination
        """
        base = base or cls()
        variations = variations or {}

        if not variations:
            return [base]

        # Generate all combinations
        keys = list(variations.keys())
        value_lists = [variations[k] for k in keys]

        configs = []
        for combo in itertools.product(*value_lists):
            overrides = dict(zip(keys, combo))
            configs.append(base.with_override(**overrides))

        return configs

    def compute_n_io(self) -> float:
        """
        Compute background ion density from Boltzmann relation.

        n_io = R0 * (x2 - x1) * sqrt(m_i / (e * T_e_eV))

        Returns normalized value (divided by n_ref).
        """
        c = self.constants
        p = self.plasma
        d = self.domain

        term = (p.m_i / (c.e * p.T_e_eV)) ** 0.5
        n_io_physical = p.R0 * d.reaction_zone_width * term
        return n_io_physical / self.scales.n_ref

    def __repr__(self) -> str:
        return (
            f"ParameterSpace(\n"
            f"  domain: L={self.domain.L}m, x1={self.domain.x1}m, x2={self.domain.x2}m\n"
            f"  plasma: f={self.plasma.f/1e6}MHz, V0={self.plasma.V0}V, R0={self.plasma.R0:.2e}\n"
            f"  scales: x_ref={self.scales.x_ref}, t_ref={self.scales.t_ref:.2e}, n_ref={self.scales.n_ref:.2e}\n"
            f")"
        )

    def get_fdm_hash(self, nx: int = None, n_steps: int = None) -> str:
        """
        Generate a unique hash based on physics parameters for FDM file naming.

        Args:
            nx: Spatial resolution (uses self.fdm.nx if None)
            n_steps: Temporal steps per cycle (uses self.fdm.n_steps_per_cycle if None)

        Returns:
            8-character hash string identifying this parameter configuration
        """
        # Use stored FDM config if not explicitly provided
        nx = nx if nx is not None else self.fdm.nx
        n_steps = n_steps if n_steps is not None else self.fdm.n_steps_per_cycle

        # Include all physics-relevant parameters that affect FDM solution
        key_params = (
            f"L={self.domain.L:.6e}_"
            f"x1={self.domain.x1:.6e}_"
            f"x2={self.domain.x2:.6e}_"
            f"f={self.plasma.f:.6e}_"
            f"V0={self.plasma.V0:.6e}_"
            f"R0={self.plasma.R0:.6e}_"
            f"Te={self.plasma.T_e_eV:.6e}_"
            f"mi={self.plasma.m_i_amu:.6e}_"
            f"nu={self.plasma.nu_m:.6e}_"
            f"nx={nx}_"
            f"nt={n_steps}"
        )
        return hashlib.md5(key_params.encode()).hexdigest()[:8]

    def get_fdm_filename(self, nx: int = None, n_steps: int = None) -> str:
        """
        Generate FDM filename based on physics parameters and resolution.

        Args:
            nx: Spatial resolution (uses self.fdm.nx if None)
            n_steps: Temporal steps per cycle (uses self.fdm.n_steps_per_cycle if None)

        Returns:
            Filename like 'fdm_a1b2c3d4.npz'
        """
        return f"fdm_{self.get_fdm_hash(nx=nx, n_steps=n_steps)}.npz"
