
import torch
import numpy as np
from typing import Tuple

class CollocationSampler:
    """Base class for collocation point sampling."""
    def sample(self, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class UniformSampler(CollocationSampler):
    def __init__(self, x_range: Tuple[float, float], t_range: Tuple[float, float]):
        self.x_range = x_range
        self.t_range = t_range

    def sample(self, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(num_points, 1).uniform_(*self.x_range)
        t = torch.FloatTensor(num_points, 1).uniform_(*self.t_range)
        return x, t

class BetaSampler(CollocationSampler):
    def __init__(self, x_range: Tuple[float, float], t_range: Tuple[float, float], alpha: float = 1.0, beta: float = 1.0):
        self.x_range = x_range
        self.t_range = t_range
        self.alpha = alpha
        self.beta = beta

    def update_beta(self, new_beta: float):
        self.beta = new_beta

    def sample(self, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m = torch.distributions.Beta(self.alpha, self.beta)
        x_normalized = m.sample((num_points, 1))
        x = x_normalized * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t = torch.FloatTensor(num_points, 1).uniform_(*self.t_range)
        return x, t

class GridSampler(CollocationSampler):
    def __init__(self, x_range: Tuple[float, float], t_range: Tuple[float, float], nx: int, nt: int):
        self.x_range = x_range
        self.t_range = t_range
        self.nx = nx
        self.nt = nt

    def sample(self, num_points: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(self.x_range[0], self.x_range[1], self.nx)
        t = torch.linspace(self.t_range[0], self.t_range[1], self.nt)
        grid_x, grid_t = torch.meshgrid(x, t, indexing='ij')
        return grid_x.reshape(-1, 1), grid_t.reshape(-1, 1)
