"""
Neural network architectures for PINN models.

All networks are pure nn.Module - no Lightning logic here.
They take (x, t) concatenated as input and return (n_e, phi) tuple.
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from .fourier import (
    FourierFeatureMapping,
    FourierFeatureMapping1D,
    FourierFeatureMapping2D,
    PeriodicTimeEmbedding,
)
from .activations import PTanh, PExp


class MLP(nn.Module):
    """Basic multi-layer perceptron."""

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dims: List[int] = None,
        out_dim: int = 2,
        activation: str = 'tanh'
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]

        act_fn = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }.get(activation, nn.Tanh())

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


class SequentialModel(nn.Module):
    """FFM + MLP with optional exact BC enforcement."""

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64, 2]

        self.ffm = FourierFeatureMapping2D(num_frequencies=num_ffm_frequencies)
        self.activation = nn.Tanh()
        self.exact_bc = exact_bc

        input_dim = self.ffm.out_dim
        layer_dims = [input_dim] + layers

        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_bc = x
        x = self.ffm(x)

        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        output = self.linears[-1](x)

        n_e = output[:, 0:1]
        phi = output[:, 1:2]

        if self.exact_bc:
            x_coord = x_bc[:, 0:1]
            t_coord = x_bc[:, 1:2]
            n_e = x_coord * (1 - x_coord) * n_e
            phi = x_coord * torch.sin(2 * math.pi * t_coord) + x_coord * (1 - x_coord) * phi

        return n_e, phi


class SequentialModelPeriodic(nn.Module):
    """
    FFM (spatial) + Periodic Time Embedding + MLP with exact BC enforcement.

    This architecture guarantees exact 1-periodicity in time by encoding
    time using only sin/cos harmonics. This is essential for RF-driven
    plasma simulations where the solution must repeat every RF cycle.

    Key differences from SequentialModel:
    - Spatial x uses FourierFeatureMapping1D (dyadic frequencies)
    - Time t uses PeriodicTimeEmbedding (pure harmonics, no raw t)
    - Applies exp() to n_e for positivity before BC enforcement

    Args:
        layers: Hidden layer sizes + output size (e.g., [256, 256, 256, 2])
        num_ffm_frequencies: Number of dyadic frequencies for spatial FFM
        max_t_harmonic: Number of time harmonics (k=1..max_t_harmonic)
        exact_bc: Whether to enforce exact boundary conditions
        use_exp_ne: Whether to apply exp() to n_e for positivity
    """

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        max_t_harmonic: int = 4,
        exact_bc: bool = True,
        use_exp_ne: bool = True,
    ):
        super().__init__()
        if layers is None:
            layers = [256, 256, 256, 2]

        self.ffm_x = FourierFeatureMapping1D(num_frequencies=num_ffm_frequencies)
        self.t_periodic = PeriodicTimeEmbedding(max_harmonic=max_t_harmonic)
        self.activation = nn.Tanh()
        self.exact_bc = exact_bc
        self.use_exp_ne = use_exp_ne

        # Input dim = FFM(x) features + periodic time features
        input_dim = self.ffm_x.out_dim + self.t_periodic.out_dim
        layer_dims = [input_dim] + layers

        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        # Xavier initialization with tanh gain
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = x[:, 0:1]
        t_raw = x[:, 1:2]

        # Encode spatial with FFM, time with periodic harmonics
        x_feat = self.ffm_x(x_raw)
        t_feat = self.t_periodic(t_raw)

        h = torch.cat([x_feat, t_feat], dim=-1)

        # MLP forward pass
        for i in range(len(self.linears) - 1):
            h = self.activation(self.linears[i](h))
        output = self.linears[-1](h)

        n_e_head = output[:, 0:1]
        phi_head = output[:, 1:2]

        # Apply exp for positivity (matches archive script)
        if self.use_exp_ne:
            n_e = torch.exp(n_e_head)
        else:
            n_e = n_e_head

        if self.exact_bc:
            # n_e(0,t) = n_e(1,t) = 0, enforced via x(1-x) multiplier
            n_e = x_raw * (1 - x_raw) * n_e

            # phi(0,t) = sin(2πt), phi(1,t) = 0
            # The sin(2πt) term is 1-periodic, and phi_head depends only on
            # periodic features, so the full phi is exactly 1-periodic
            phi = x_raw * torch.sin(2 * math.pi * t_raw) + x_raw * (1 - x_raw) * phi_head
        else:
            phi = phi_head

        return n_e, phi


class ModulatedSequentialModel(nn.Module):
    """
    FFM + MLP with modulation encoders (Modified MLP).

    Uses two encoder pathways and interpolates based on hidden activations.
    This is NOT a gated architecture - it's a modulation/interpolation approach
    similar to Wang et al. (2021) Modified MLP.
    """

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64, 2]

        self.ffm = FourierFeatureMapping2D(num_frequencies=num_ffm_frequencies)
        self.activation = nn.Tanh()
        self.exact_bc = exact_bc

        input_dim = self.ffm.out_dim
        layer_dims = [input_dim] + layers

        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        hidden_dim = layers[0]
        self.encoder_1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.encoder_2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_bc = x
        x = self.ffm(x)
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(x)

        for i in range(len(self.linears) - 1):
            h = self.activation(self.linears[i](x))
            x = h * enc1 + (1 - h) * enc2

        output = self.linears[-1](x)
        n_e = output[:, 0:1]
        phi = output[:, 1:2]

        if self.exact_bc:
            x_coord = x_bc[:, 0:1]
            t_coord = x_bc[:, 1:2]
            n_e = x_coord * (1 - x_coord) * n_e
            phi = x_coord * torch.sin(2 * math.pi * t_coord) + x_coord * (1 - x_coord) * phi

        return n_e, phi


class GatedSequentialModel(nn.Module):
    """
    FFM + Gated MLP with true sigmoid gating (GLU-style).

    Each layer uses Gated Linear Units:
        output = tanh(W_v @ x + b_v) * sigmoid(W_g @ x + b_g)

    The sigmoid gate controls information flow, helping with:
    - Gradient flow through deep networks
    - Selective activation for multi-scale physics
    - Better handling of coupled PDE dynamics

    For the coupled continuity-Poisson system, this architecture allows
    the network to learn which features are relevant for each output.
    """

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64, 2]

        self.ffm = FourierFeatureMapping2D(num_frequencies=num_ffm_frequencies)
        self.exact_bc = exact_bc

        input_dim = self.ffm.out_dim
        self.num_layers = len(layers)

        # Value and gate pathways for GLU
        self.value_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()

        prev_dim = input_dim
        for i, h_dim in enumerate(layers[:-1]):
            self.value_layers.append(nn.Linear(prev_dim, h_dim))
            self.gate_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, layers[-1])

        # Initialize weights
        for layer in self.value_layers:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)
        for layer in self.gate_layers:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_bc = x
        x = self.ffm(x)

        # GLU layers: output = tanh(value) * sigmoid(gate)
        for value_layer, gate_layer in zip(self.value_layers, self.gate_layers):
            value = torch.tanh(value_layer(x))
            gate = torch.sigmoid(gate_layer(x))
            x = value * gate

        output = self.output_layer(x)
        n_e = output[:, 0:1]
        phi = output[:, 1:2]

        if self.exact_bc:
            x_coord = x_bc[:, 0:1]
            t_coord = x_bc[:, 1:2]
            n_e = x_coord * (1 - x_coord) * n_e
            phi = x_coord * torch.sin(2 * math.pi * t_coord) + x_coord * (1 - x_coord) * phi

        return n_e, phi


# Backward compatibility alias
GatedSequentialModelLegacy = ModulatedSequentialModel


class ModulatedPINN(nn.Module):
    """Modulated PINN with gated encoders."""

    def __init__(self, layers: List[int] = None):
        super().__init__()
        if layers is None:
            layers = [2, 64, 64, 64, 2]

        self.depth = len(layers) - 1
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(self.depth)
        ])

        input_dim = layers[0]
        hidden_dim = layers[1]
        self.encoder_1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.encoder_2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(x)

        out = x
        for i in range(self.depth - 1):
            h = torch.tanh(self.linears[i](out))
            out = h * enc1 + (1 - h) * enc2

        output = self.linears[-1](out)
        return output[:, 0:1], output[:, 1:2]


class FourierMLP(nn.Module):
    """MLP with random Fourier features."""

    def __init__(
        self,
        hidden_dims: List[int] = None,
        mapping_size: int = 256,
        sigma: float = 10.0,
        activation: str = 'tanh'
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]

        self.ffm = FourierFeatureMapping(in_features=2, mapping_size=mapping_size, sigma=sigma)

        act_fn = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU()}.get(activation, nn.Tanh())

        layers = []
        prev_dim = self.ffm.out_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ffm(x)
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


class DensityNetwork(nn.Module):
    """Network for electron density only (n_e)."""

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        use_exp: bool = False,
        parametrized_exp: bool = False
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64, 1]

        self.ffm = FourierFeatureMapping2D(num_frequencies=num_ffm_frequencies)
        self.activation = nn.Tanh()
        self.use_exp = use_exp

        if parametrized_exp:
            self.exp_activation = PExp()
        else:
            self.exp_activation = None

        input_dim = self.ffm.out_dim
        layer_dims = [input_dim] + layers

        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bc = x
        x = self.ffm(x)

        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        n_e = self.linears[-1](x)

        if self.use_exp:
            if self.exp_activation is not None:
                n_e = self.exp_activation(n_e)
            else:
                n_e = torch.exp(n_e)

        x_coord = x_bc[:, 0:1]
        n_e = x_coord * (1 - x_coord) * n_e

        return n_e


class PotentialNetwork(nn.Module):
    """Network for electric potential only (phi)."""

    def __init__(
        self,
        layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        use_tanh: bool = False,
        parametrized_tanh: bool = False
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64, 1]

        self.ffm = FourierFeatureMapping2D(num_frequencies=num_ffm_frequencies)
        self.activation = nn.Tanh()
        self.use_tanh = use_tanh

        if parametrized_tanh:
            self.output_activation = PTanh()
        else:
            self.output_activation = None

        input_dim = self.ffm.out_dim
        layer_dims = [input_dim] + layers

        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bc = x
        x = self.ffm(x)

        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        phi = self.linears[-1](x)

        if self.use_tanh:
            if self.output_activation is not None:
                phi = self.output_activation(phi)
            else:
                phi = torch.tanh(phi)

        x_coord = x_bc[:, 0:1]
        t_coord = x_bc[:, 1:2]
        adf = x_coord * (1 - x_coord)
        phi = x_coord * torch.sin(2 * math.pi * t_coord) + adf * phi

        return phi


class TwoNetworkModel(nn.Module):
    """Separate networks for n_e and phi."""

    def __init__(
        self,
        ne_layers: List[int] = None,
        phi_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        use_exp_ne: bool = True,
        use_tanh_phi: bool = False
    ):
        super().__init__()
        self.ne_net = DensityNetwork(
            layers=ne_layers,
            num_ffm_frequencies=num_ffm_frequencies,
            use_exp=use_exp_ne
        )
        self.phi_net = PotentialNetwork(
            layers=phi_layers,
            num_ffm_frequencies=num_ffm_frequencies,
            use_tanh=use_tanh_phi
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_e = self.ne_net(x)
        phi = self.phi_net(x)
        return n_e, phi
