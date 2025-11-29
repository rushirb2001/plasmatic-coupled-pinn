"""
Fourier feature mappings for coordinate encoding.
"""

import math
import torch
import torch.nn as nn


class FourierFeatureMapping(nn.Module):
    """Random Fourier features with fixed B matrix."""

    def __init__(self, in_features: int = 2, mapping_size: int = 256, sigma: float = 10.0):
        super().__init__()
        B = torch.randn((mapping_size, in_features)) * sigma
        self.register_buffer('B', B)
        self.out_dim = mapping_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatureMapping2D(nn.Module):
    """Dyadic scale Fourier features for 2D coordinates (x, t)."""

    def __init__(self, in_features: int = 2, num_frequencies: int = 2):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        freq_bands = [2 ** i * math.pi for i in range(num_frequencies)]
        self.register_buffer('frequencies', torch.tensor(freq_bands).float().view(1, 1, num_frequencies))
        self.out_dim = in_features + 2 * in_features * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        t = coords[..., 1:2]
        freqs = self.frequencies.to(coords.device)

        x_proj = x * freqs
        t_proj = t * freqs

        encoded = [
            coords,
            torch.sin(x_proj).squeeze(0),
            torch.cos(x_proj).squeeze(0),
            torch.sin(t_proj).squeeze(0),
            torch.cos(t_proj).squeeze(0),
        ]
        return torch.cat(encoded, dim=-1)
