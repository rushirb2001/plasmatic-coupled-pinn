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


class FourierFeatureMapping1D(nn.Module):
    """Dyadic scale Fourier features for 1D coordinate (spatial x only)."""

    def __init__(self, num_frequencies: int = 2, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        if num_frequencies > 0:
            freq_bands = [2 ** i * math.pi for i in range(num_frequencies)]
            self.register_buffer('frequencies', torch.tensor(freq_bands).float().view(1, num_frequencies))
        self.out_dim = (1 if include_input else 0) + (2 * num_frequencies if num_frequencies > 0 else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, 1]

        Returns:
            Encoded features of shape [N, out_dim]
        """
        feats = []
        if self.include_input:
            feats.append(x)
        if self.num_frequencies > 0:
            angles = x @ self.frequencies  # [N, 1] @ [1, F] -> [N, F]
            feats.extend([torch.sin(angles), torch.cos(angles)])
        return torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]


class PeriodicTimeEmbedding(nn.Module):
    """
    Encode time using only sin/cos harmonics for exact 1-periodicity.

    This ensures the network output is exactly periodic in time,
    which is essential for RF-driven plasma simulations where
    the solution must repeat every RF cycle.

    Args:
        max_harmonic: Number of harmonics (k = 1, 2, ..., max_harmonic)

    Output dimension: 2 * max_harmonic (sin and cos for each harmonic)
    """

    def __init__(self, max_harmonic: int = 4):
        super().__init__()
        self.max_harmonic = max_harmonic
        self.register_buffer('ks', torch.arange(1, max_harmonic + 1, dtype=torch.float32))
        self.out_dim = 2 * max_harmonic

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape [N, 1], normalized to [0, 1]

        Returns:
            Periodic features of shape [N, 2*max_harmonic]
        """
        # angles = 2π * t * k for k = 1..K
        # shapes: t -> [N, 1], ks -> [1, K], angles -> [N, K]
        angles = 2 * math.pi * t @ self.ks.view(1, -1)
        sins = torch.sin(angles)  # [N, K]
        coss = torch.cos(angles)  # [N, K]
        return torch.cat([sins, coss], dim=-1)  # [N, 2K]


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
