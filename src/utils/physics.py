
from dataclasses import dataclass
import numpy as np

@dataclass
class PhysicalConstants:
    e: float = 1.60217663e-19  # Elementary charge (C)
    m_e: float = 9.10938356e-31  # Electron mass (kg)
    epsilon_0: float = 8.85418781e-12  # Vacuum permittivity (F/m)
    k_b: float = 1.380649e-23 # Boltzmann constant (J/K)

@dataclass
class DefaultParameters:
    L: float = 0.025  # Domain length (m)
    f: float = 13.56e6  # Driving frequency (Hz)
    V0: float = 100.0  # Driving voltage amplitude (V)
    R0: float = 2.7e20  # Reaction rate coefficient
    x1: float = 0.005  # Reaction zone start (m)
    x2: float = 0.01  # Reaction zone width (m)
    T_e_eV: float = 3.0  # Electron temperature (eV)
    m_i_amu: float = 40.0  # Ion mass (amu)
    nu_m: float = 1e8  # Collision frequency (s^-1)

    @property
    def T_e(self) -> float:
        """Electron temperature in Kelvin"""
        return self.T_e_eV * 1.60218e-19 / 1.380649e-23

    @property
    def m_i(self) -> float:
        """Ion mass in kg"""
        return self.m_i_amu * 1.66054e-27

    @property
    def D(self) -> float:
        """Electron diffusion coefficient"""
        return (PhysicalConstants.e * self.T_e_eV) / (PhysicalConstants.m_e * self.nu_m)

    @property
    def mu(self) -> float:
        """Electron mobility"""
        return PhysicalConstants.e / (PhysicalConstants.m_e * self.nu_m)

@dataclass
class ScalingParameters:
    # Reference scales for non-dimensionalization
    x_ref: float = 0.025 # L
    t_ref: float = 1.0 / 13.56e6 # 1/f
    n_ref: float = 1.0e16 # Typical density
    phi_ref: float = 100.0 # V0
    
    # Derived scales for derivatives
    # d/dx = (1/x_ref) * d/dx'
    # d/dt = (1/t_ref) * d/dt'
