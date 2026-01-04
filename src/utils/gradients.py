"""
Gradient utilities for PINN training.

Provides:
- Gradient extraction and analysis
- Adaptive loss balancing (GradNorm-style)
- Hessian eigenvalue computation
- Gradient flow diagnostics
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def extract_gradients(
    model: nn.Module,
    losses: List[torch.Tensor],
    names: List[str],
    retain_graph: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract gradients for each loss component.

    Args:
        model: Neural network model
        losses: List of loss tensors
        names: Names for each loss component
        retain_graph: Whether to retain computation graph

    Returns:
        Dictionary mapping layer names to gradient arrays
    """
    grad_dict = {}

    for loss, name in zip(losses, names):
        for i, (layer_name, layer) in enumerate(model.named_modules()):
            if isinstance(layer, nn.Linear):
                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=layer.weight,
                    retain_graph=retain_graph or (i < len(list(model.modules())) - 1),
                    create_graph=False,
                    allow_unused=True
                )[0]

                if grad is not None:
                    key = f"{name}/{layer_name}" if layer_name else f"{name}/layer_{i}"
                    grad_dict[key] = grad.detach().cpu().flatten().numpy()

    return grad_dict


def compute_gradient_norm(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    norm_type: float = 2.0,
) -> float:
    """
    Compute gradient norm for a loss.

    Args:
        loss: Loss tensor
        params: List of parameters to compute gradients for
        norm_type: Type of norm (default: L2)

    Returns:
        Gradient norm value
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )

    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += g.norm(norm_type).item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


def hessian_vector_product(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hessian-vector product Hv.

    Args:
        loss: Scalar loss tensor
        params: List of parameters
        v: Vector to multiply with Hessian

    Returns:
        Hessian-vector product
    """
    grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grad])
    Hv = torch.autograd.grad(grad_flat, params, grad_outputs=v, retain_graph=True)
    Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])
    return Hv_flat


def compute_top_hessian_eigenvalue(
    loss: torch.Tensor,
    model: nn.Module,
    max_iters: int = 20,
    tol: float = 1e-6,
) -> float:
    """
    Compute top Hessian eigenvalue using power iteration.

    Args:
        loss: Loss tensor
        model: Neural network model
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Top eigenvalue estimate
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    v = torch.randn(n_params, device=loss.device)
    v = v / v.norm()

    eigenvalue = None
    for _ in range(max_iters):
        Hv = hessian_vector_product(loss, params, v)
        new_eigenvalue = torch.dot(v, Hv)
        Hv_norm = Hv.norm()
        if Hv_norm == 0:
            break
        v = Hv / Hv_norm

        if eigenvalue is not None and torch.abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = new_eigenvalue

    return eigenvalue.item() if eigenvalue is not None else 0.0


class AdaptiveLossBalancer:
    """
    Adaptive loss balancing using mean absolute gradient and max-based normalization.

    Balances gradients from different loss components by computing the mean
    absolute gradient (L1-style) for each loss and scaling weights so that
    all losses contribute equally to parameter updates.

    Uses fast EMA adaptation (alpha=0.9) for responsive weight updates during
    early training when gradient magnitudes change rapidly.

    Args:
        alpha: Smoothing coefficient for EMA (default: 0.9, meaning 10% old + 90% new)
        loss_names: Names of loss components
        initial_weights: Initial weights for each loss
    """

    def __init__(
        self,
        alpha: float = 0.9,
        loss_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        **kwargs,  # Accept but ignore legacy parameters for compatibility
    ):
        self.alpha = alpha

        if loss_names is None:
            loss_names = ["continuity", "poisson"]
        self.loss_names = loss_names
        self.n_losses = len(loss_names)

        if initial_weights is None:
            self.weights = {name: 1.0 for name in loss_names}
        else:
            self.weights = initial_weights.copy()

        # Track gradient norms for logging
        self.last_grad_norms: Dict[str, float] = {}

    def _flatten_grads(self, grads: Tuple) -> torch.Tensor:
        """Flatten gradient tensors into a single 1D tensor."""
        flat_grads = []
        for g in grads:
            if g is not None:
                flat_grads.append(g.flatten())
        if flat_grads:
            return torch.cat(flat_grads)
        return torch.tensor([0.0])

    def _mean_abs_norm(
        self,
        loss: torch.Tensor,
        params: List[torch.Tensor],
    ) -> float:
        """Compute mean absolute gradient (L1-style norm) for a loss."""
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, create_graph=False, allow_unused=True
        )

        flat = self._flatten_grads(grads)
        return torch.mean(torch.abs(flat)).item()

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        params: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Update loss weights based on gradient magnitudes.

        Uses max-based normalization: each weight is set to max(all_norms)/this_norm
        so that the loss with the largest gradient gets weight 1.0 and others
        are scaled up proportionally.

        Args:
            losses: Dictionary of loss tensors
            params: List of parameters to compute gradients for

        Returns:
            Updated weight dictionary
        """
        grad_norms = {}

        # Compute mean absolute gradient for each loss
        for name, loss in losses.items():
            if name not in self.loss_names:
                continue
            grad_norms[name] = self._mean_abs_norm(loss, params)

        self.last_grad_norms = grad_norms.copy()

        if not grad_norms:
            return self.weights

        # Find maximum gradient norm
        max_norm = max(grad_norms.values())

        # Avoid division by zero
        if max_norm < 1e-10:
            return self.weights

        # Update weights with EMA using max-based normalization
        for name in self.loss_names:
            if name not in grad_norms:
                continue

            norm = grad_norms[name]

            # Target weight: max_norm / norm
            # Loss with largest gradient gets weight ~1
            # Losses with smaller gradients get larger weights
            if norm > 1e-10:
                target_weight = max_norm / norm
            else:
                target_weight = self.weights[name]

            # Fast EMA update (alpha=0.9 means 10% old + 90% new)
            self.weights[name] = (
                (1 - self.alpha) * self.weights[name] +
                self.alpha * target_weight
            )

        return self.weights

    def get_grad_norms(self) -> Dict[str, float]:
        """Get last computed gradient norms (for logging/debugging)."""
        return self.last_grad_norms.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()

    def compute_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute total weighted loss.

        Args:
            losses: Dictionary of loss tensors

        Returns:
            Weighted sum of losses
        """
        total = 0.0
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            total = total + weight * loss
        return total


class GradientMonitor:
    """
    Monitor gradient statistics during training.

    Tracks gradient norms, flow, and potential issues
    like vanishing/exploding gradients.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}

    def record(
        self,
        model: nn.Module,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Record gradient statistics for all layers.

        Args:
            model: Neural network model
            prefix: Prefix for metric names

        Returns:
            Dictionary of gradient statistics
        """
        stats = {}

        total_norm = 0.0
        num_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                key = f"{prefix}{name}" if prefix else name
                stats[f"{key}/norm"] = grad_norm
                stats[f"{key}/mean"] = grad_mean
                stats[f"{key}/std"] = grad_std

                total_norm += grad_norm ** 2
                num_params += 1

                # Track history
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(grad_norm)
                if len(self.history[key]) > self.window_size:
                    self.history[key].pop(0)

        if num_params > 0:
            stats["total_norm"] = total_norm ** 0.5

        return stats

    def check_health(self) -> Dict[str, bool]:
        """
        Check gradient health indicators.

        Returns:
            Dictionary of health check results
        """
        health = {
            "vanishing": False,
            "exploding": False,
            "unstable": False,
        }

        for key, values in self.history.items():
            if len(values) < 10:
                continue

            recent = values[-10:]
            mean = np.mean(recent)
            std = np.std(recent)

            # Check for vanishing gradients
            if mean < 1e-7:
                health["vanishing"] = True

            # Check for exploding gradients
            if mean > 1e3:
                health["exploding"] = True

            # Check for instability
            if std / (mean + 1e-8) > 2.0:
                health["unstable"] = True

        return health

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f"{key}/mean"] = np.mean(values)
                summary[f"{key}/std"] = np.std(values)
                summary[f"{key}/min"] = np.min(values)
                summary[f"{key}/max"] = np.max(values)
        return summary
