"""
Custom activation functions for PINN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PTanh(nn.Module):
    """Parametrized tanh: a * tanh(b * x + c) + d"""

    def __init__(self, channels: int = 1, per_channel: bool = False, monotone: bool = True):
        super().__init__()
        dim = channels if per_channel else 1
        self._a = nn.Parameter(torch.zeros(dim))
        self._b = nn.Parameter(torch.zeros(dim))
        self.c = nn.Parameter(torch.zeros(dim))
        self.d = nn.Parameter(torch.zeros(dim))
        self.monotone = monotone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self._a) if self.monotone else self._a
        b = F.softplus(self._b) if self.monotone else self._b
        return a * torch.tanh(b * x + self.c) + self.d


class PExp(nn.Module):
    """Parametrized exp: exp(alpha * x + beta) - gamma"""

    def __init__(self, channels: int = 1, per_channel: bool = False, safe: bool = True, monotone: bool = True):
        super().__init__()
        dim = channels if per_channel else 1
        self._alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.safe = safe
        self.monotone = monotone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self._alpha) if self.monotone else self._alpha
        z = alpha * x + self.beta
        if self.safe:
            z = torch.clamp(z, -20, 20)
            y = torch.expm1(z) + 1
        else:
            y = torch.exp(z)
        return y - self.gamma
