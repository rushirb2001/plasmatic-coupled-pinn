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
    V0: float = 100.0             # Voltage amplitude (V)
    R0: float = 2.7e20            # Reaction rate (m^-3 s^-1)
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
    n_ref: float = 1.0e16         # Reference density (m^-3)
    phi_ref: float = 100.0        # Reference potential (V) - typically V0

    @classmethod
    def from_physics(cls, domain: DomainParameters, plasma: PlasmaParameters) -> "ScalingParameters":
        """Create scaling parameters from physics parameters"""
        return cls(
            x_ref=domain.L,
            t_ref=1.0 / plasma.f,
            n_ref=1.0e16,  # Typical plasma density
            phi_ref=plasma.V0
        )


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
        constants: Optional[PhysicalConstants] = None
    ):
        self.constants = constants or PhysicalConstants()
        self.domain = domain or DomainParameters()
        self.plasma = plasma or PlasmaParameters()
        self.scales = scales or ScalingParameters()

        # Link constants to plasma parameters
        object.__setattr__(self.plasma, '_constants', self.constants)

    @classmethod
    def from_yaml(cls, path: str) -> "ParameterSpace":
        """Load parameters from YAML config file"""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        physics_cfg = cfg.get('physics', cfg)  # Support nested or flat structure

        return cls(
            domain=DomainParameters(**physics_cfg.get('domain', {})),
            plasma=PlasmaParameters(**{
                k: v for k, v in physics_cfg.get('plasma', {}).items()
                if k != '_constants'
            }),
            scales=ScalingParameters(**physics_cfg.get('scales', {}))
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
