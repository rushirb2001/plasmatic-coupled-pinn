"""
PINN Models for CCP-II Plasma Simulation.

This module provides:
- BasePINN: PyTorch Lightning base class with MLP that works out of the box
- Specialized PINN variants that inherit and add specific features
- MODEL_REGISTRY: Dictionary mapping hyphenated names to classes for YAML lookup

Usage with Lightning CLI:
    model:
      class_path: src.model.SequentialPINN  # or use registry lookup
      init_args:
        hidden_layers: [64, 64, 64]
"""

import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from src.architectures import (
    MLP,
    SequentialModel,
    SequentialModelPeriodic,
    GatedSequentialModel,
    ModulatedSequentialModel,
    ModulatedPINN,
    FourierMLP,
    TwoNetworkModel,
    DensityNetwork,
    PotentialNetwork,
    PoissonSolverCPU,
    PoissonSolverGPU,
    FieldCache,
)
from src.utils.physics import ParameterSpace, PhysicalConstants
from src.utils.nondim import NonDimensionalizer
from src.utils.gradients import AdaptiveLossBalancer, GradientMonitor
from src.data.fdm_solver import get_or_generate_fdm
from src.visualization.plotting import visualize_model
from src.visualization.gif_generator import generate_model_animation


class BasePINN(pl.LightningModule):
    """
    Base PINN model for CCP-II plasma simulation.

    Uses a simple MLP architecture by default and works out of the box.
    Subclasses can override `build_network()` to use different architectures.

    Features:
    - Complete training/validation/test loops for physics-informed learning
    - PDE residual computation for continuity and Poisson equations
    - Boundary condition enforcement
    - Configurable optimizer and scheduler
    - Automatic metric tracking

    Args:
        hidden_layers: List of hidden layer dimensions
        activation: Activation function ('tanh', 'relu', 'gelu', 'silu')
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Scheduler type ('constant', 'cosine', 'step')
        loss_weights: Dictionary of loss component weights
        params_path: Path to physics parameters YAML (optional)
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        loss_weights: Dict[str, float] = None,
        params_path: Optional[str] = None,
        use_adaptive_weights: bool = False,
        adaptive_alpha: float = 0.1,
        exact_bc: bool = True,
        ic_num_points: int = 10000,
        visualize_on_train_end: bool = True,
        output_dir: str = "./experiments",
        fdm_dir: str = "data/fdm",
        **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters()

        # Defaults
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]

        # Ensure loss_weights has all required keys (merge with defaults)
        default_loss_weights = {"continuity": 1.0, "poisson": 1.0, "bc": 10.0, "ic": 1.0}
        if loss_weights is None:
            loss_weights = default_loss_weights
        else:
            # Merge user-provided weights with defaults (user values take precedence)
            loss_weights = {**default_loss_weights, **loss_weights}

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.use_adaptive_weights = use_adaptive_weights
        self.exact_bc = exact_bc
        self.ic_num_points = ic_num_points
        self.visualize_on_train_end = visualize_on_train_end
        self.output_dir = output_dir
        self.fdm_dir = fdm_dir

        # Load physics parameters
        if params_path:
            self.params = ParameterSpace.from_yaml(params_path)
        else:
            self.params = ParameterSpace()

        # Generate or load FDM reference data for this parameter configuration
        get_or_generate_fdm(self.params, fdm_dir=fdm_dir)

        # Non-dimensionalization
        self.nondim = NonDimensionalizer(self.params)

        # Build the network
        self.net = self.build_network()

        # Compile network for faster execution (PyTorch 2.0+)
        self._compile_network()

        # Adaptive loss balancing
        loss_names = ["continuity", "poisson", "ic"]
        if not exact_bc:
            loss_names.append("bc")

        if use_adaptive_weights:
            self.loss_balancer = AdaptiveLossBalancer(
                alpha=adaptive_alpha,
                loss_names=loss_names,
                initial_weights=loss_weights,
            )
        else:
            self.loss_balancer = None

        # Gradient monitoring
        self.grad_monitor = GradientMonitor()

        # Metrics
        self.train_metrics = torchmetrics.MetricCollection({
            "loss_cont": torchmetrics.MeanMetric(),
            "loss_pois": torchmetrics.MeanMetric(),
            "loss_bc": torchmetrics.MeanMetric(),
            "loss_ic": torchmetrics.MeanMetric(),
        })
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def build_network(self) -> nn.Module:
        """
        Build the neural network architecture.

        Override this method in subclasses to use different architectures.
        Default: Simple MLP with 2 inputs (x, t) and 2 outputs (n_e, phi).
        """
        return MLP(
            in_dim=2,
            hidden_dims=self.hidden_layers,
            out_dim=2,
            activation=self.activation
        )

    def _compile_network(self):
        """
        Compile network with torch.compile for faster execution.

        Note: Disabled for PINNs because torch.compile doesn't support
        double backward (needed for computing PDE residuals with create_graph=True).
        The fused optimizer and pre-loaded data still provide speedups.
        """
        # torch.compile currently incompatible with autograd.grad(create_graph=True)
        # which is required for PINN PDE residual computation
        pass

    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (n_e, phi).

        Args:
            x_t: Input tensor of shape [B, 2] with normalized coordinates

        Returns:
            n_e: Electron density [B, 1]
            phi: Electric potential [B, 1]
        """
        return self.net(x_t)

    def compute_pde_residuals(
        self, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PDE residuals for continuity and Poisson equations.

        Uses automatic differentiation to compute spatial and temporal derivatives.
        Returns scaled residuals for numerical stability.
        """
        x_t = x_t.clone().requires_grad_(True)

        n_e, phi = self(x_t)

        # Compute gradients
        ones = torch.ones_like(n_e)

        # dn_e/dx, dn_e/dt
        grad_ne = torch.autograd.grad(n_e, x_t, ones, create_graph=True)[0]
        dn_e_dx = grad_ne[:, 0:1]
        dn_e_dt = grad_ne[:, 1:2]

        # dphi/dx
        grad_phi = torch.autograd.grad(phi, x_t, ones, create_graph=True)[0]
        dphi_dx = grad_phi[:, 0:1]

        # d2phi/dx2
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx, x_t, ones, create_graph=True
        )[0][:, 0:1]

        # Physics coefficients (from non-dimensionalization)
        coeff = self.nondim.coeffs

        # Electron flux: Gamma_e = -D * dn_e/dx - mu * n_e * dphi/dx
        Gamma_e = -coeff.alpha * dn_e_dx - coeff.beta * n_e * dphi_dx

        # d(Gamma_e)/dx
        dGamma_e_dx = torch.autograd.grad(
            Gamma_e, x_t, ones, create_graph=True
        )[0][:, 0:1]

        # Reaction rate R(x) - symmetric reaction zones
        x_norm = x_t[:, 0:1]
        R_val = torch.zeros_like(x_norm)

        # Reaction zone parameters (normalized)
        d = self.params.domain
        x1_norm = d.x1 / d.L
        x2_norm = d.x2 / d.L

        mask1 = (x_norm >= x1_norm) & (x_norm <= x2_norm)
        mask2 = (x_norm >= 1.0 - x2_norm) & (x_norm <= 1.0 - x1_norm)
        R_val = torch.where(mask1 | mask2, torch.ones_like(R_val), R_val)

        # Ion density (normalized) - computed from Boltzmann relation
        n_io = self.params.compute_n_io()

        # Continuity residual: dn_e/dt + d(Gamma_e)/dx - gamma*R = 0
        # gamma scales the reaction rate term
        res_cont = dn_e_dt + dGamma_e_dx - coeff.gamma * R_val

        # Poisson residual: d2phi/dx2 + delta*(n_e - n_io) = 0
        res_pois = d2phi_dx2 + coeff.delta * (n_e - n_io)

        return res_cont, res_pois

    def compute_boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition losses.

        BCs (normalized coordinates):
        - x=0: phi = sin(2*pi*t), n_e = 0
        - x=1: phi = 0, n_e = 0
        """
        B = t.shape[0]
        device = t.device

        # Left boundary (x=0)
        x0 = torch.zeros(B, 1, device=device)
        x_t_left = torch.cat([x0, t], dim=1)
        n_e_0, phi_0 = self(x_t_left)

        # Driving voltage (normalized)
        V_t = torch.sin(2 * math.pi * t)

        loss_left = torch.mean(n_e_0**2) + torch.mean((phi_0 - V_t)**2)

        # Right boundary (x=1)
        x1 = torch.ones(B, 1, device=device)
        x_t_right = torch.cat([x1, t], dim=1)
        n_e_L, phi_L = self(x_t_right)

        loss_right = torch.mean(n_e_L**2) + torch.mean(phi_L**2)

        return loss_left + loss_right

    def compute_ic_loss(self) -> torch.Tensor:
        """
        Compute initial condition loss.

        IC: n_e(x, t=0) = n_io (quasi-neutrality at t=0)

        Uses a fixed set of spatial points at t=0 to enforce the initial condition.
        """
        device = next(self.parameters()).device

        # Sample spatial points at t=0
        x = torch.linspace(0, 1, self.ic_num_points, device=device, dtype=torch.float32)
        t0 = torch.zeros_like(x)
        x_t0 = torch.stack([x, t0], dim=1)

        # Forward pass at t=0
        n_e_0, _ = self(x_t0)

        # Ion density (normalized) - computed from Boltzmann relation
        n_io = self.params.compute_n_io()

        # IC residual: n_e(x, t=0) - n_io = 0
        res_ic = n_e_0 - n_io
        loss_ic = torch.mean(res_ic**2)

        return loss_ic

    def pretrain_ic(
        self,
        num_steps: int = 500,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Pretrain the network to satisfy the initial condition.

        This optional pretraining phase helps the network start with
        a physically consistent initial state (n_e = n_io everywhere).

        Args:
            num_steps: Number of pretraining optimization steps
            learning_rate: Learning rate for pretraining optimizer
        """
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Generate pretraining points covering full domain at all times
        # (we want the network to predict n_io everywhere initially)
        n_points = self.ic_num_points
        x = torch.rand(n_points, device=device)
        t = torch.rand(n_points, device=device)
        x_t = torch.stack([x, t], dim=1)

        n_io = self.params.compute_n_io()

        self.train()
        for step in range(num_steps):
            optimizer.zero_grad()
            n_e, _ = self(x_t)
            loss = torch.mean((n_e - n_io)**2)
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"IC pretrain step {step + 1}/{num_steps}, loss: {loss.item():.6f}")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with PDE, IC, and optional BC losses."""
        x_t = batch[0] if isinstance(batch, (list, tuple)) else batch

        # PDE residuals (autocast disabled - incompatible with higher-order grads)
        res_cont, res_pois = self.compute_pde_residuals(x_t)
        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)

        # Initial condition loss (n_e = n_io at t=0)
        loss_ic = self.compute_ic_loss()

        # Boundary condition loss (only when exact_bc=False)
        if not self.exact_bc:
            t = x_t[:, 1:2]
            loss_bc = self.compute_boundary_loss(t)
        else:
            loss_bc = torch.tensor(0.0, device=x_t.device)

        # Get loss weights (static or adaptive)
        if self.loss_balancer is not None:
            # Update adaptive weights
            losses = {"continuity": loss_cont, "poisson": loss_pois, "ic": loss_ic}
            if not self.exact_bc:
                losses["bc"] = loss_bc

            params = list(self.parameters())
            weights = self.loss_balancer.update(losses, params)

            # Log adaptive weights
            self.log("weight/continuity", weights["continuity"], on_step=False, on_epoch=True)
            self.log("weight/poisson", weights["poisson"], on_step=False, on_epoch=True)
            self.log("weight/ic", weights["ic"], on_step=False, on_epoch=True)
            if not self.exact_bc:
                self.log("weight/bc", weights["bc"], on_step=False, on_epoch=True)

            # Log gradient norms for debugging
            grad_norms = self.loss_balancer.get_grad_norms()
            for name, norm in grad_norms.items():
                self.log(f"grad_norm/{name}", norm, on_step=False, on_epoch=True)
        else:
            weights = self.loss_weights

        # Total loss
        loss = (
            weights["continuity"] * loss_cont +
            weights["poisson"] * loss_pois +
            weights["ic"] * loss_ic
        )
        if not self.exact_bc:
            loss = loss + weights["bc"] * loss_bc

        # Metrics
        self.train_metrics["loss_cont"].update(loss_cont)
        self.train_metrics["loss_pois"].update(loss_pois)
        self.train_metrics["loss_ic"].update(loss_ic)
        self.train_metrics["loss_bc"].update(loss_bc)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with PDE, IC, and optional BC losses."""
        x_t = batch[0] if isinstance(batch, (list, tuple)) else batch

        with torch.enable_grad():
            res_cont, res_pois = self.compute_pde_residuals(x_t)

        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)

        # Initial condition loss
        loss_ic = self.compute_ic_loss()

        # Boundary condition loss (only when exact_bc=False)
        if not self.exact_bc:
            t = x_t[:, 1:2]
            loss_bc = self.compute_boundary_loss(t)
        else:
            loss_bc = torch.tensor(0.0, device=x_t.device)

        loss = (
            self.loss_weights["continuity"] * loss_cont +
            self.loss_weights["poisson"] * loss_pois +
            self.loss_weights["ic"] * loss_ic
        )
        if not self.exact_bc:
            loss = loss + self.loss_weights["bc"] * loss_bc

        self.val_metrics["loss_cont"].update(loss_cont)
        self.val_metrics["loss_pois"].update(loss_pois)
        self.val_metrics["loss_ic"].update(loss_ic)
        self.val_metrics["loss_bc"].update(loss_bc)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step with PDE and IC losses."""
        x_t = batch[0] if isinstance(batch, (list, tuple)) else batch

        with torch.enable_grad():
            res_cont, res_pois = self.compute_pde_residuals(x_t)

        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)

        # Initial condition loss
        loss_ic = self.compute_ic_loss()

        loss = loss_cont + loss_pois + loss_ic

        self.test_metrics["loss_cont"].update(loss_cont)
        self.test_metrics["loss_pois"].update(loss_pois)
        self.test_metrics["loss_ic"].update(loss_ic)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Use fused kernels on CUDA for faster optimization
        use_fused = torch.cuda.is_available()

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                fused=use_fused
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                fused=use_fused
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                fused=use_fused
            )

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        elif self.scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1000,
                gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }

        return optimizer

    def on_train_epoch_start(self):
        """Disable GC during training epoch for consistent performance."""
        gc.disable()

    def on_train_epoch_end(self):
        """Re-enable GC and collect at epoch boundary."""
        gc.enable()
        gc.collect()

    def on_train_end(self):
        """Generate visualizations at end of training."""
        if not self.visualize_on_train_end:
            return

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            visualize_model(
                model=self,
                nx=100,
                nt=100,
                save_dir=str(output_path / "visualizations"),
                device=str(self.device),
                fdm_dir=self.fdm_dir,
            )

            generate_model_animation(
                model=self,
                save_path=str(output_path / "visualizations" / "solution.gif"),
                nx=100,
                nt=100,
                fps=30,
                device=str(self.device),
                fdm_dir=self.fdm_dir,
            )
        except Exception as e:
            print(f"Visualization failed: {e}")


class SequentialPINN(BasePINN):
    """
    PINN with FFM + MLP architecture and exact BC enforcement.

    Uses Fourier Feature Mapping for better high-frequency learning.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True,
        **kwargs: Any
    ):
        self.num_ffm_frequencies = num_ffm_frequencies
        # Pass exact_bc to parent class
        super().__init__(hidden_layers=hidden_layers, exact_bc=exact_bc, **kwargs)

    def build_network(self) -> nn.Module:
        layers = self.hidden_layers + [2]
        return SequentialModel(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            exact_bc=self.exact_bc
        )


class SequentialPeriodicPINN(BasePINN):
    """
    PINN with FFM (spatial) + Periodic Time Embedding and exact BC enforcement.

    This model guarantees exact 1-periodicity in time by encoding time
    using only sin/cos harmonics. This matches the architecture from
    the well-performing archive script (SequentialmodelPeriodicFFM).

    Key features:
    - Spatial x uses FourierFeatureMapping1D (dyadic frequencies)
    - Time t uses PeriodicTimeEmbedding (pure harmonics, no raw t)
    - Applies exp() to n_e for positivity before BC enforcement
    - Exact boundary condition enforcement

    This is ideal for RF-driven plasma simulations where the solution
    must repeat every RF cycle.

    Args:
        hidden_layers: List of hidden layer dimensions
        num_ffm_frequencies: Number of dyadic frequencies for spatial FFM
        max_t_harmonic: Number of time harmonics (k=1..max_t_harmonic)
        exact_bc: Whether to enforce exact boundary conditions
        use_exp_ne: Whether to apply exp() to n_e for positivity
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        max_t_harmonic: int = 4,
        exact_bc: bool = True,
        use_exp_ne: bool = True,
        **kwargs: Any
    ):
        # Store these before calling super().__init__()
        self.num_ffm_frequencies = num_ffm_frequencies
        self.max_t_harmonic = max_t_harmonic
        self.use_exp_ne = use_exp_ne
        # Pass exact_bc to parent class
        super().__init__(hidden_layers=hidden_layers, exact_bc=exact_bc, **kwargs)

    def build_network(self) -> nn.Module:
        layers = self.hidden_layers + [2]
        return SequentialModelPeriodic(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            max_t_harmonic=self.max_t_harmonic,
            exact_bc=self.exact_bc,
            use_exp_ne=self.use_exp_ne,
        )


class GatedPINN(BasePINN):
    """
    PINN with FFM + true Gated MLP (GLU-style).

    Uses Gated Linear Units with sigmoid gates:
        output = tanh(W_v @ x) * sigmoid(W_g @ x)

    This architecture provides:
    - Proper information gating for multi-scale physics
    - Better gradient flow through deep networks
    - Selective activation for coupled PDE dynamics
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True,
        **kwargs: Any
    ):
        self.num_ffm_frequencies = num_ffm_frequencies
        # Pass exact_bc to parent class
        super().__init__(hidden_layers=hidden_layers, exact_bc=exact_bc, **kwargs)

    def build_network(self) -> nn.Module:
        layers = self.hidden_layers + [2]
        return GatedSequentialModel(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            exact_bc=self.exact_bc
        )


class ModulatedMLPPINN(BasePINN):
    """
    PINN with FFM + Modulated MLP (Modified MLP).

    Uses two encoder pathways with interpolation:
        output = h * enc1 + (1 - h) * enc2

    Based on Wang et al. (2021) Modified MLP architecture.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        exact_bc: bool = True,
        **kwargs: Any
    ):
        self.num_ffm_frequencies = num_ffm_frequencies
        super().__init__(hidden_layers=hidden_layers, exact_bc=exact_bc, **kwargs)

    def build_network(self) -> nn.Module:
        layers = self.hidden_layers + [2]
        return ModulatedSequentialModel(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            exact_bc=self.exact_bc
        )


class ModulatedPINNModel(BasePINN):
    """
    PINN with lightweight modulated architecture.

    Simple modulation without FFM. Uses raw (x, t) input.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        **kwargs: Any
    ):
        super().__init__(hidden_layers=hidden_layers, **kwargs)

    def build_network(self) -> nn.Module:
        layers = [2] + self.hidden_layers + [2]
        return ModulatedPINN(layers=layers)


class FourierPINN(BasePINN):
    """
    PINN with random Fourier features.

    Uses fixed random B matrix for frequency encoding.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        mapping_size: int = 256,
        sigma: float = 10.0,
        **kwargs: Any
    ):
        self.mapping_size = mapping_size
        self.sigma = sigma
        super().__init__(hidden_layers=hidden_layers, **kwargs)

    def build_network(self) -> nn.Module:
        return FourierMLP(
            hidden_dims=self.hidden_layers,
            mapping_size=self.mapping_size,
            sigma=self.sigma,
            activation=self.activation
        )


class TwoNetworkPINN(BasePINN):
    """
    PINN with separate networks for n_e and phi.

    Allows different architectures/capacities for each output.
    """

    def __init__(
        self,
        ne_layers: List[int] = None,
        phi_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        use_exp_ne: bool = True,
        use_tanh_phi: bool = False,
        **kwargs: Any
    ):
        self.ne_layers = ne_layers
        self.phi_layers = phi_layers
        self.num_ffm_frequencies = num_ffm_frequencies
        self.use_exp_ne = use_exp_ne
        self.use_tanh_phi = use_tanh_phi
        super().__init__(**kwargs)

    def build_network(self) -> nn.Module:
        ne_layers = self.ne_layers if self.ne_layers else [64, 64, 64, 1]
        phi_layers = self.phi_layers if self.phi_layers else [64, 64, 64, 1]
        return TwoNetworkModel(
            ne_layers=ne_layers,
            phi_layers=phi_layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            use_exp_ne=self.use_exp_ne,
            use_tanh_phi=self.use_tanh_phi
        )


class HybridPINN(BasePINN):
    """
    Hybrid PINN: Neural network for n_e, numerical solver for Poisson.

    Uses NN to predict electron density, then solves Poisson equation
    numerically for more accurate electric potential.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        solver_nx: int = 50,
        solver_nt: int = 1000,
        use_gpu_solver: bool = False,
        **kwargs: Any
    ):
        self.num_ffm_frequencies = num_ffm_frequencies
        self.solver_nx = solver_nx
        self.solver_nt = solver_nt
        self.use_gpu_solver = use_gpu_solver
        super().__init__(hidden_layers=hidden_layers, **kwargs)

        # Build Poisson solver
        self._build_solver()

        # Field cache for interpolation
        self.field_cache = None

    def _build_solver(self):
        """Initialize the Poisson solver."""
        t_max = 1.0 / self.params.plasma.f  # One RF period

        if self.use_gpu_solver and torch.cuda.is_available():
            self.solver = PoissonSolverGPU(
                nx=self.solver_nx,
                nt=self.solver_nt,
                t_max=t_max,
                device="cuda"
            )
        else:
            self.solver = PoissonSolverCPU(
                nx=self.solver_nx,
                nt=self.solver_nt,
                t_max=t_max,
                device="cpu"
            )

    def build_network(self) -> nn.Module:
        """Build density-only network."""
        layers = self.hidden_layers + [1]
        return DensityNetwork(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            use_exp=True
        )

    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hybrid NN + solver approach.

        1. Predict n_e using neural network
        2. Solve Poisson for phi using numerical solver
        3. Interpolate phi to query points
        """
        n_e = self.net(x_t)

        # Get phi from cache if available, otherwise return zeros
        if self.field_cache is not None:
            phi, _, _ = self.field_cache.get_phi(x_t)
        else:
            phi = torch.zeros_like(n_e)

        return n_e, phi

    def update_poisson_cache(self):
        """
        Solve Poisson equation on uniform grid and update cache.

        Should be called periodically during training.
        """
        with torch.no_grad():
            # Evaluate n_e on uniform grid
            x_t_uniform = self.solver.x_t_uniform.to(self.device)
            n_e_flat = self.net(x_t_uniform)
            n_e_grid = n_e_flat.view(self.solver.nx, self.solver.nt)

            # Solve Poisson
            if isinstance(self.solver, PoissonSolverGPU):
                phi, dphi_dx, dphi_dxx = self.solver.solve(n_e_grid)
            else:
                n_e_np = n_e_grid.cpu().numpy()
                phi, dphi_dx, dphi_dxx = self.solver.solve(n_e_np)
                phi = torch.from_numpy(phi).float()
                dphi_dx = torch.from_numpy(dphi_dx).float()
                dphi_dxx = torch.from_numpy(dphi_dxx).float()

            # Update cache
            self.field_cache = FieldCache(
                phi=phi,
                dphi_dx=dphi_dx,
                dphi_dxx=dphi_dxx,
                device=self.device
            )

    def on_train_epoch_start(self):
        """Update Poisson cache at start of each epoch."""
        super().on_train_epoch_start()  # GC disable from BasePINN
        self.update_poisson_cache()

    def compute_pde_residuals(
        self, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute only continuity residual (Poisson is hard-constrained).

        Uses cached phi and its derivatives from numerical solver.
        """
        x_t = x_t.clone().requires_grad_(True)

        n_e = self.net(x_t)

        # Get cached phi derivatives
        if self.field_cache is not None:
            _, dphi_dx, dphi_dxx = self.field_cache.get_phi(x_t)
        else:
            dphi_dx = torch.zeros_like(n_e)
            dphi_dxx = torch.zeros_like(n_e)

        # Compute n_e gradients
        ones = torch.ones_like(n_e)
        grad_ne = torch.autograd.grad(n_e, x_t, ones, create_graph=True)[0]
        dn_e_dx = grad_ne[:, 0:1]
        dn_e_dt = grad_ne[:, 1:2]

        # Physics coefficients
        coeff = self.nondim.coeffs

        # Electron flux using cached dphi_dx
        Gamma_e = -coeff.alpha * dn_e_dx - coeff.beta * n_e * dphi_dx

        # d(Gamma_e)/dx
        dGamma_e_dx = torch.autograd.grad(
            Gamma_e, x_t, ones, create_graph=True
        )[0][:, 0:1]

        # Reaction rate R(x)
        x_norm = x_t[:, 0:1]
        R_val = torch.zeros_like(x_norm)
        d = self.params.domain
        x1_norm = d.x1 / d.L
        x2_norm = d.x2 / d.L
        mask1 = (x_norm >= x1_norm) & (x_norm <= x2_norm)
        mask2 = (x_norm >= 1.0 - x2_norm) & (x_norm <= 1.0 - x1_norm)
        R_val = torch.where(mask1 | mask2, torch.ones_like(R_val), R_val)

        # Continuity residual only (Poisson is exact from solver)
        res_cont = dn_e_dt + dGamma_e_dx - R_val

        # Poisson residual is zero by construction (hard-constrained)
        res_pois = torch.zeros_like(res_cont)

        return res_cont, res_pois


class NonDimPINN(BasePINN):
    """
    PINN with explicit non-dimensionalization.

    All inputs/outputs are in dimensionless form.
    Useful for problems with widely varying scales.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_ffm_frequencies: int = 2,
        **kwargs: Any
    ):
        self.num_ffm_frequencies = num_ffm_frequencies
        super().__init__(hidden_layers=hidden_layers, **kwargs)

    def build_network(self) -> nn.Module:
        layers = self.hidden_layers + [2]
        return SequentialModel(
            layers=layers,
            num_ffm_frequencies=self.num_ffm_frequencies,
            exact_bc=True
        )

    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with automatic non-dimensionalization."""
        # Input is already normalized to [0,1] x [0,1]
        n_e, phi = self.net(x_t)
        return n_e, phi

    def to_physical(
        self, n_e: torch.Tensor, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert dimensionless outputs to physical units."""
        n_e_phys = self.nondim.unscale_ne(n_e)
        phi_phys = self.nondim.unscale_phi(phi)
        return n_e_phys, phi_phys


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, type] = {
    # Hyphenated names for YAML
    "base-pinn": BasePINN,
    "sequential-pinn": SequentialPINN,
    "sequential-periodic-pinn": SequentialPeriodicPINN,
    "gated-pinn": GatedPINN,
    "modulated-mlp-pinn": ModulatedMLPPINN,
    "modulated-pinn": ModulatedPINNModel,
    "fourier-pinn": FourierPINN,
    "two-network-pinn": TwoNetworkPINN,
    "hybrid-pinn": HybridPINN,
    "nondim-pinn": NonDimPINN,

    # Aliases
    "mlp": BasePINN,
    "ffm": SequentialPINN,
    "periodic": SequentialPeriodicPINN,
    "periodic-ffm": SequentialPeriodicPINN,
    "gated": GatedPINN,
    "modulated-mlp": ModulatedMLPPINN,
    "modulated": ModulatedPINNModel,
    "fourier": FourierPINN,
    "hybrid": HybridPINN,
}


def get_model_class(name: str) -> type:
    """
    Get model class by name from registry.

    Args:
        name: Model name (hyphenated form, e.g., 'sequential-pinn')

    Returns:
        Model class

    Raises:
        ValueError: If model name not found in registry
    """
    name_lower = name.lower().replace("_", "-")
    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_REGISTRY[name_lower]


def create_model(name: str, **kwargs: Any) -> BasePINN:
    """
    Create model instance by name.

    Args:
        name: Model name from registry
        **kwargs: Model initialization arguments

    Returns:
        Initialized model instance
    """
    model_class = get_model_class(name)
    return model_class(**kwargs)


def list_models() -> List[str]:
    """List all available model names."""
    return sorted(set(MODEL_REGISTRY.keys()))
