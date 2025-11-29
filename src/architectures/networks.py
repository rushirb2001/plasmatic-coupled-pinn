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

from .fourier import FourierFeatureMapping, FourierFeatureMapping2D
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


class GatedSequentialModel(nn.Module):
    """FFM + Gated MLP with modulation encoders."""

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
