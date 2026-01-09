"""
Gradient utilities for PINN training.

Provides:
- Gradient extraction and analysis
- Adaptive loss balancing (GradNorm-style) with multiple strategies
- Residual normalization (EMA, characteristic scale, running stats)
- Gradient surgery (PCGrad) for conflicting gradients
- Hessian eigenvalue computation
- Gradient flow diagnostics
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class BalancingStrategy(Enum):
    """Loss balancing strategy for adaptive weights."""
    MAX_BASED = "max"       # max(norms) / this_norm (original, can oscillate)
    MEAN_BASED = "mean"     # mean(norms) / this_norm (more stable)
    RMS_BASED = "rms"       # rms(norms) / this_norm (compromise)
    SOFTMAX = "softmax"     # softmax(-log(norms)) (bounded weights)


class ResidualNormStrategy(Enum):
    """Residual normalization strategy."""
    NONE = "none"                       # No normalization
    EMA = "ema"                         # EMA of RMS (current approach)
    CHARACTERISTIC_SCALE = "char_scale" # Physics-based scaling
    RUNNING_STATS = "running_stats"     # Track mean/std over training
    BATCH_NORM = "batch_norm"           # Normalize per batch


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
    Adaptive loss balancing with multiple strategies.

    Balances gradients from different loss components by computing the mean
    absolute gradient (L1-style) for each loss and scaling weights so that
    all losses contribute equally to parameter updates.

    Strategies:
    - MAX_BASED: weight = max(norms) / this_norm (original, can oscillate)
    - MEAN_BASED: weight = mean(norms) / this_norm (more stable)
    - RMS_BASED: weight = rms(norms) / this_norm (compromise)
    - SOFTMAX: weight = softmax(-log(norms)) (bounded weights, most stable)

    Args:
        alpha: Smoothing coefficient for EMA (default: 0.1, meaning 90% old + 10% new)
               Lower alpha = more stable but slower adaptation
        loss_names: Names of loss components
        initial_weights: Initial weights for each loss
        strategy: Balancing strategy (default: "mean" for stability)
        max_weight: Maximum allowed weight to prevent explosion (default: 100.0)
        min_weight: Minimum allowed weight (default: 0.01)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        loss_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        strategy: Union[str, BalancingStrategy] = "mean",
        max_weight: float = 100.0,
        min_weight: float = 0.01,
        **kwargs,  # Accept but ignore legacy parameters for compatibility
    ):
        self.alpha = alpha
        self.max_weight = max_weight
        self.min_weight = min_weight

        # Parse strategy
        if isinstance(strategy, str):
            strategy_map = {
                "max": BalancingStrategy.MAX_BASED,
                "mean": BalancingStrategy.MEAN_BASED,
                "rms": BalancingStrategy.RMS_BASED,
                "softmax": BalancingStrategy.SOFTMAX,
            }
            self.strategy = strategy_map.get(strategy.lower(), BalancingStrategy.MEAN_BASED)
        else:
            self.strategy = strategy

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

        # Track weight history for debugging
        self.weight_history: Dict[str, List[float]] = {name: [] for name in loss_names}

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
    ) -> torch.Tensor:
        """Compute mean absolute gradient (L1-style norm) for a loss. Returns tensor to avoid sync."""
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, create_graph=False, allow_unused=True
        )

        flat = self._flatten_grads(grads)
        return torch.mean(torch.abs(flat))  # Keep as tensor, no .item()

    def _compute_target_weights(
        self,
        norms_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target weights based on strategy."""
        eps = 1e-10

        if self.strategy == BalancingStrategy.MAX_BASED:
            # Original: max(norms) / this_norm
            max_norm = norms_tensor.max()
            target = max_norm / (norms_tensor + eps)

        elif self.strategy == BalancingStrategy.MEAN_BASED:
            # More stable: mean(norms) / this_norm
            mean_norm = norms_tensor.mean()
            target = mean_norm / (norms_tensor + eps)

        elif self.strategy == BalancingStrategy.RMS_BASED:
            # Compromise: rms(norms) / this_norm
            rms_norm = torch.sqrt((norms_tensor ** 2).mean())
            target = rms_norm / (norms_tensor + eps)

        elif self.strategy == BalancingStrategy.SOFTMAX:
            # Most stable: bounded weights via softmax
            # Use -log(norms) so smaller norms get higher weights
            log_norms = torch.log(norms_tensor + eps)
            target = torch.softmax(-log_norms, dim=0) * len(norms_tensor)

        else:
            # Default to mean-based
            mean_norm = norms_tensor.mean()
            target = mean_norm / (norms_tensor + eps)

        return target

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        params: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Update loss weights based on gradient magnitudes.

        Uses the configured strategy to compute target weights, then applies
        EMA smoothing with weight capping for stability.

        Optimized: All computations stay on GPU, no CPU sync.

        Args:
            losses: Dictionary of loss tensors
            params: List of parameters to compute gradients for

        Returns:
            Updated weight dictionary (tensors on GPU)
        """
        # Compute all gradient norms as tensors on GPU
        grad_norm_tensors = {}
        for name, loss in losses.items():
            if name not in self.loss_names:
                continue
            grad_norm_tensors[name] = self._mean_abs_norm(loss, params)

        if not grad_norm_tensors:
            return self.weights

        # Stack norms
        names = list(grad_norm_tensors.keys())
        norms_tensor = torch.stack([grad_norm_tensors[n] for n in names])

        # Store for logging (keep as tensors)
        self.last_grad_norms = grad_norm_tensors

        # Compute target weights using configured strategy
        target_weights = self._compute_target_weights(norms_tensor)

        # Update weights with EMA and capping (all on GPU)
        for i, name in enumerate(names):
            if not isinstance(self.weights[name], torch.Tensor):
                # First call: convert float to tensor
                device = target_weights.device
                self.weights[name] = torch.tensor(self.weights[name], device=device)

            # EMA update
            new_weight = (
                (1 - self.alpha) * self.weights[name] +
                self.alpha * target_weights[i]
            )

            # Clamp to prevent explosion/collapse
            new_weight = torch.clamp(new_weight, self.min_weight, self.max_weight)
            self.weights[name] = new_weight

            # Track history (only keep last 100 for memory)
            if name in self.weight_history:
                self.weight_history[name].append(new_weight.detach().cpu().item())
                if len(self.weight_history[name]) > 100:
                    self.weight_history[name].pop(0)

        return self.weights

    def get_grad_norms(self) -> Dict[str, float]:
        """Get last computed gradient norms (for logging/debugging). Syncs to CPU here."""
        result = {}
        for name, val in self.last_grad_norms.items():
            if isinstance(val, torch.Tensor):
                result[name] = val.detach().cpu().item()
            else:
                result[name] = val
        return result

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights. Syncs to CPU if needed."""
        result = {}
        for name, val in self.weights.items():
            if isinstance(val, torch.Tensor):
                result[name] = val.detach().cpu().item()
            else:
                result[name] = val
        return result

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

    def get_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about weight history for debugging."""
        stats = {}
        for name, history in self.weight_history.items():
            if history:
                stats[name] = {
                    "current": history[-1],
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                }
        return stats


class ResidualNormalizer:
    """
    Modular residual normalization for PINN training.

    Normalizes PDE residuals to balance their magnitudes before computing loss.
    This helps when different terms have vastly different scales (e.g., δ ~ 100 vs α ~ 1e-7).

    Strategies:
    - NONE: No normalization (raw residuals)
    - EMA: Running EMA of RMS (adapts during training)
    - CHARACTERISTIC_SCALE: Divide by physics coefficient magnitudes
    - RUNNING_STATS: Track mean/std over training
    - BATCH_NORM: Normalize per batch (like BatchNorm but for residuals)

    Args:
        strategy: Normalization strategy
        ema_decay: Decay for EMA (default 0.99 = slow adaptation)
        char_scales: Dict of characteristic scales for each residual type
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        strategy: Union[str, ResidualNormStrategy] = "ema",
        ema_decay: float = 0.99,
        char_scales: Optional[Dict[str, float]] = None,
        eps: float = 1e-8,
    ):
        # Parse strategy
        if isinstance(strategy, str):
            strategy_map = {
                "none": ResidualNormStrategy.NONE,
                "ema": ResidualNormStrategy.EMA,
                "char_scale": ResidualNormStrategy.CHARACTERISTIC_SCALE,
                "characteristic_scale": ResidualNormStrategy.CHARACTERISTIC_SCALE,
                "running_stats": ResidualNormStrategy.RUNNING_STATS,
                "batch_norm": ResidualNormStrategy.BATCH_NORM,
            }
            self.strategy = strategy_map.get(strategy.lower(), ResidualNormStrategy.EMA)
        else:
            self.strategy = strategy

        self.ema_decay = ema_decay
        self.eps = eps

        # Characteristic scales (physics-based)
        self.char_scales = char_scales or {}

        # EMA buffers (will be initialized on first call)
        self.ema_rms: Dict[str, Optional[torch.Tensor]] = {}

        # Running stats (for running_stats strategy)
        self.running_mean: Dict[str, Optional[torch.Tensor]] = {}
        self.running_var: Dict[str, Optional[torch.Tensor]] = {}
        self.num_batches: int = 0

    def set_characteristic_scales(self, scales: Dict[str, float]):
        """Set characteristic scales for physics-based normalization."""
        self.char_scales = scales

    def normalize(
        self,
        residual: torch.Tensor,
        name: str,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Normalize a residual tensor.

        Args:
            residual: Raw residual tensor
            name: Name of the residual (e.g., "continuity", "poisson")
            training: Whether in training mode (affects running stats)

        Returns:
            Normalized residual tensor
        """
        if self.strategy == ResidualNormStrategy.NONE:
            return residual

        elif self.strategy == ResidualNormStrategy.EMA:
            return self._normalize_ema(residual, name, training)

        elif self.strategy == ResidualNormStrategy.CHARACTERISTIC_SCALE:
            return self._normalize_char_scale(residual, name)

        elif self.strategy == ResidualNormStrategy.RUNNING_STATS:
            return self._normalize_running_stats(residual, name, training)

        elif self.strategy == ResidualNormStrategy.BATCH_NORM:
            return self._normalize_batch(residual)

        return residual

    def _normalize_ema(
        self,
        residual: torch.Tensor,
        name: str,
        training: bool,
    ) -> torch.Tensor:
        """EMA-based normalization."""
        # Compute current RMS
        current_rms = torch.sqrt(torch.mean(residual ** 2) + self.eps)

        # Initialize or update EMA
        if name not in self.ema_rms or self.ema_rms[name] is None:
            self.ema_rms[name] = current_rms.detach()
        elif training:
            self.ema_rms[name] = (
                self.ema_decay * self.ema_rms[name] +
                (1 - self.ema_decay) * current_rms.detach()
            )

        # Normalize by EMA RMS
        return residual / (self.ema_rms[name] + self.eps)

    def _normalize_char_scale(
        self,
        residual: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Physics-based characteristic scale normalization."""
        scale = self.char_scales.get(name, 1.0)
        if scale == 0:
            scale = 1.0
        return residual / (abs(scale) + self.eps)

    def _normalize_running_stats(
        self,
        residual: torch.Tensor,
        name: str,
        training: bool,
    ) -> torch.Tensor:
        """Running statistics normalization (like BatchNorm)."""
        if training:
            # Compute batch statistics
            batch_mean = residual.mean()
            batch_var = residual.var()

            # Update running stats
            if name not in self.running_mean or self.running_mean[name] is None:
                self.running_mean[name] = batch_mean.detach()
                self.running_var[name] = batch_var.detach()
            else:
                momentum = 0.1
                self.running_mean[name] = (
                    (1 - momentum) * self.running_mean[name] +
                    momentum * batch_mean.detach()
                )
                self.running_var[name] = (
                    (1 - momentum) * self.running_var[name] +
                    momentum * batch_var.detach()
                )

            self.num_batches += 1

            # Use batch stats during training
            return (residual - batch_mean) / (torch.sqrt(batch_var) + self.eps)
        else:
            # Use running stats during eval
            mean = self.running_mean.get(name)
            var = self.running_var.get(name)
            if mean is None or var is None:
                return residual
            return (residual - mean) / (torch.sqrt(var) + self.eps)

    def _normalize_batch(self, residual: torch.Tensor) -> torch.Tensor:
        """Simple batch normalization (zero-mean, unit-variance)."""
        mean = residual.mean()
        std = residual.std()
        return (residual - mean) / (std + self.eps)

    def get_scales(self) -> Dict[str, float]:
        """Get current normalization scales for logging."""
        scales = {}
        for name, ema in self.ema_rms.items():
            if ema is not None:
                if isinstance(ema, torch.Tensor):
                    scales[f"ema_rms/{name}"] = ema.detach().cpu().item()
                else:
                    scales[f"ema_rms/{name}"] = ema

        for name, scale in self.char_scales.items():
            scales[f"char_scale/{name}"] = scale

        return scales


class PCGrad:
    """
    Projected Conflicting Gradients (PCGrad) for multi-task learning.

    When gradients from different losses conflict (negative dot product),
    project one onto the normal plane of the other to remove the conflicting
    component.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)

    Args:
        reduction: How to combine gradients after surgery ('sum' or 'mean')
    """

    def __init__(self, reduction: str = "sum"):
        self.reduction = reduction

    def _flatten_grad(self, grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Flatten gradients into a single vector."""
        flat = []
        for g in grads:
            if g is not None:
                flat.append(g.flatten())
            else:
                flat.append(torch.zeros(1, device=grads[0].device if grads[0] is not None else "cpu"))
        return torch.cat(flat) if flat else torch.tensor([0.0])

    def _unflatten_grad(
        self,
        flat_grad: torch.Tensor,
        shapes: List[torch.Size],
        params: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Unflatten gradient vector back to parameter shapes."""
        grads = []
        offset = 0
        for shape, param in zip(shapes, params):
            numel = param.numel()
            grads.append(flat_grad[offset:offset + numel].view(shape))
            offset += numel
        return grads

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        params: List[torch.Tensor],
        retain_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients with PCGrad surgery.

        Args:
            losses: Dictionary of loss tensors
            params: List of parameters
            retain_graph: Whether to retain computation graph

        Returns:
            Dictionary with 'combined_grad' key containing the surgically combined gradient
        """
        # Compute gradients for each loss
        grads = {}
        shapes = [p.shape for p in params]

        for name, loss in losses.items():
            grad_tuple = torch.autograd.grad(
                loss, params, retain_graph=retain_graph, create_graph=False, allow_unused=True
            )
            # Replace None grads with zeros
            grad_tuple = tuple(
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grad_tuple, params)
            )
            grads[name] = self._flatten_grad(grad_tuple)

        # Apply PCGrad surgery
        names = list(grads.keys())
        n_tasks = len(names)

        if n_tasks < 2:
            # No surgery needed
            combined = list(grads.values())[0] if grads else torch.zeros(1)
        else:
            # Surgery: project conflicting gradients
            grad_list = [grads[n].clone() for n in names]

            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i == j:
                        continue

                    # Check for conflict
                    dot = torch.dot(grad_list[i], grads[names[j]])
                    if dot < 0:
                        # Project gradient i onto normal plane of gradient j
                        norm_sq = torch.dot(grads[names[j]], grads[names[j]])
                        if norm_sq > 1e-10:
                            grad_list[i] = grad_list[i] - (dot / norm_sq) * grads[names[j]]

            # Combine gradients
            if self.reduction == "sum":
                combined = sum(grad_list)
            else:  # mean
                combined = sum(grad_list) / n_tasks

        return {"combined_grad": combined, "shapes": shapes}

    def apply_gradients(
        self,
        combined_grad: torch.Tensor,
        shapes: List[torch.Size],
        params: List[torch.Tensor],
    ):
        """
        Apply the combined gradient to parameters.

        Args:
            combined_grad: Flattened combined gradient
            shapes: Original parameter shapes
            params: List of parameters
        """
        unflat_grads = self._unflatten_grad(combined_grad, shapes, params)
        for param, grad in zip(params, unflat_grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.copy_(grad)


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
