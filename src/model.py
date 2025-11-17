# ============================================================================
# src/model.py
# ============================================================================
"""
Module: model
Description:
    PyTorch Lightning model library for CCP-II PINN.
    Adapted from original model.py with proper BaseModel pattern.
    
    BaseModel provides the common PINN training loop, handling:
    - PDE residual computation
    - Boundary condition enforcement
    - Metric tracking
    - Optimizer/scheduler configuration
    
    Inherited models (CCPPinn, etc.) define:
    - Custom forward() with specific architecture
    - Architecture-specific hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Any, Tuple, List

import torchmetrics

from src.utils.richco import console
from src.utils.physics import PhysicalConstants, DefaultParameters, ScalingParameters


class BaseModel(pl.LightningModule):
    """
    Base model for PINN implementations.
    
    Provides complete training/validation/test loop for physics-informed neural networks.
    All PINN models should inherit from this class and implement forward().
    """
    
    def __init__(
            self,
            learning_rate: float = 1e-3,
            optimizer: str = 'adamw',
            scheduler: str = 'constant',
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            loss_weights: dict = None,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.enable_benchmarking = enable_benchmarking
        self.experiment_name = experiment_name
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {'continuity': 1.0, 'poisson': 1.0, 'bc': 10.0}
        self.loss_weights = loss_weights

        # Initialize physics parameters
        self.params = DefaultParameters()
        self.scales = ScalingParameters()
        
        # Metrics for tracking physics losses
        self.train_metrics = torchmetrics.MetricCollection({
            'loss_cont': torchmetrics.MeanMetric(),
            'loss_pois': torchmetrics.MeanMetric(),
            'loss_bc': torchmetrics.MeanMetric(),
        })
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - to be implemented by subclass.
        
        Args:
            x: Spatial coordinate tensor (B, 1)
            t: Temporal coordinate tensor (B, 1)
            
        Returns:
            n_e: Electron density (B, 1) in physical units
            phi: Electric potential (B, 1) in physical units
        """
        raise NotImplementedError("Subclass must implement forward()")
    
    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PDE residuals for the CCP-II model.
        
        This method is shared across all PINN models as the physics is the same.
        Only the forward() architecture differs between models.
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Forward returns physical quantities
        n_e, phi = self(x, t)
        
        # Compute derivatives
        dn_e_dt = torch.autograd.grad(n_e, t, grad_outputs=torch.ones_like(n_e), create_graph=True)[0]
        dn_e_dx = torch.autograd.grad(n_e, x, grad_outputs=torch.ones_like(n_e), create_graph=True)[0]
        dphi_dx = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        d2phi_dx2 = torch.autograd.grad(dphi_dx, x, grad_outputs=torch.ones_like(dphi_dx), create_graph=True)[0]
        
        # R(x) - Reaction rate
        R_val = torch.zeros_like(x)
        mask1 = (x >= self.params.x1) & (x <= self.params.x2)
        mask2 = (x >= self.params.L - self.params.x2) & (x <= self.params.L - self.params.x1)
        R_val[mask1 | mask2] = self.params.R0
        
        # n_i0 - Ion density
        term_sqrt = (self.params.m_i / (PhysicalConstants.e * self.params.T_e))**0.5
        n_i0_val = self.params.R0 * (self.params.x2 - self.params.x1) * term_sqrt
        n_i0 = torch.full_like(x, n_i0_val)
        
        # Gamma_e - Electron flux
        Gamma_e = -self.params.D * dn_e_dx - self.params.mu * n_e * dphi_dx
        dGamma_e_dx = torch.autograd.grad(Gamma_e, x, grad_outputs=torch.ones_like(Gamma_e), create_graph=True)[0]
        
        # Physical Residuals
        res_cont = dn_e_dt + dGamma_e_dx - R_val
        term_poisson = (PhysicalConstants.e / PhysicalConstants.epsilon_0) * (n_e - n_i0)
        res_pois = d2phi_dx2 + term_poisson
        
        # Scale Residuals to O(1) for numerical stability
        scale_cont = 1.0 / (self.scales.n_ref * self.params.f)
        scale_pois = 1.0 / (self.scales.phi_ref / (self.scales.x_ref**2))
        
        return res_cont * scale_cont, res_pois * scale_pois
    
    def compute_boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition losses.
        
        BCs:
        - x=0: phi = V(t), n_e = 0
        - x=L: phi = 0, n_e = 0
        """
        # Left boundary (x=0)
        x0 = torch.zeros_like(t)
        n_e_0, phi_0 = self(x0, t)
        V_t = self.params.V0 * torch.sin(2 * torch.pi * self.params.f * t)
        
        loss_bc_left = (torch.mean((n_e_0 - 0)**2) / self.scales.n_ref**2 + 
                        torch.mean((phi_0 - V_t)**2) / self.scales.phi_ref**2)
        
        # Right boundary (x=L)
        xL = torch.full_like(t, self.params.L)
        n_e_L, phi_L = self(xL, t)
        loss_bc_right = (torch.mean((n_e_L - 0)**2) / self.scales.n_ref**2 + 
                         torch.mean((phi_L - 0)**2) / self.scales.phi_ref**2)
        
        return loss_bc_left + loss_bc_right

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - compute losses and log metrics."""
        x, t = batch
        
        # Compute PDE residuals
        res_cont, res_pois = self.compute_pde_residuals(x, t)
        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)
        
        # Compute boundary condition loss
        loss_bc = self.compute_boundary_loss(t)
        
        # Total loss
        loss = (self.loss_weights['continuity'] * loss_cont + 
                self.loss_weights['poisson'] * loss_pois + 
                self.loss_weights['bc'] * loss_bc)
        
        # Update metrics
        self.train_metrics['loss_cont'].update(loss_cont)
        self.train_metrics['loss_pois'].update(loss_pois)
        self.train_metrics['loss_bc'].update(loss_bc)
        
        # Logging - only log total loss on_step, use metrics for components
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step - evaluate on validation set."""
        x, t = batch
        
        # Need gradients for PDE residuals even in validation
        with torch.enable_grad():
            res_cont, res_pois = self.compute_pde_residuals(x, t)
        
        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)
        
        # Also compute BC loss for validation
        loss_bc = self.compute_boundary_loss(t)
        
        # Total validation loss (same as training)
        loss = (self.loss_weights['continuity'] * loss_cont + 
                self.loss_weights['poisson'] * loss_pois + 
                self.loss_weights['bc'] * loss_bc)
        
        # Update metrics
        self.val_metrics['loss_cont'].update(loss_cont)
        self.val_metrics['loss_pois'].update(loss_pois)
        self.val_metrics['loss_bc'].update(loss_bc)
        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step - evaluate on test set."""
        x, t = batch
        
        with torch.enable_grad():
            res_cont, res_pois = self.compute_pde_residuals(x, t)
        
        loss_cont = torch.mean(res_cont**2)
        loss_pois = torch.mean(res_pois**2)
        loss = loss_cont + loss_pois
        
        # Update metrics
        self.test_metrics['loss_cont'].update(loss_cont)
        self.test_metrics['loss_pois'].update(loss_pois)
        
        # Logging
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        
        return loss
    
    def visualize_solution(self, save_path: str = None):
        """
        Visualize the trained PINN solution.
        
        Creates plots of electron density and electric potential over space and time.
        """
        import matplotlib.pyplot as plt
        
        if save_path is None:
            save_path = f"experiments/{self.experiment_name}/solution_visualization.png"
        
        # Create output directory
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate evaluation grid
        x = torch.linspace(0, self.params.L, 100, device=self.device)
        t = torch.linspace(0, 4e-7, 100, device=self.device)  # 5 RF cycles
        
        grid_x, grid_t = torch.meshgrid(x, t, indexing='ij')
        flat_x = grid_x.reshape(-1, 1)
        flat_t = grid_t.reshape(-1, 1)
        
        # Evaluate model
        self.eval()
        with torch.no_grad():
            n_e, phi = self(flat_x, flat_t)
        
        n_e = n_e.reshape(100, 100).cpu().numpy()
        phi = phi.reshape(100, 100).cpu().numpy()
        x_np = x.cpu().numpy()
        t_np = t.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Electron density heatmap
        im1 = axes[0, 0].imshow(n_e, extent=[t_np[0]*1e6, t_np[-1]*1e6, x_np[0]*1e3, x_np[-1]*1e3], 
                                aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title("Electron Density ($n_e$)")
        axes[0, 0].set_xlabel("Time (μs)")
        axes[0, 0].set_ylabel("Position (mm)")
        plt.colorbar(im1, ax=axes[0, 0], label='Density (m⁻³)')
        
        # Plot 2: Electric potential heatmap
        im2 = axes[0, 1].imshow(phi, extent=[t_np[0]*1e6, t_np[-1]*1e6, x_np[0]*1e3, x_np[-1]*1e3],
                                aspect='auto', origin='lower', cmap='plasma')
        axes[0, 1].set_title("Electric Potential ($\\phi$)")
        axes[0, 1].set_xlabel("Time (μs)")
        axes[0, 1].set_ylabel("Position (mm)")
        plt.colorbar(im2, ax=axes[0, 1], label='Potential (V)')
        
        # Plot 3: Spatial profiles at different times
        time_indices = [0, 25, 50, 75, 99]
        for i, idx in enumerate(time_indices):
            axes[1, 0].plot(x_np*1e3, n_e[:, idx], label=f't={t_np[idx]*1e6:.2f}μs', alpha=0.7)
        axes[1, 0].set_xlabel("Position (mm)")
        axes[1, 0].set_ylabel("Electron Density (m⁻³)")
        axes[1, 0].set_title("Spatial Profiles of $n_e$")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Temporal evolution at different positions
        pos_indices = [0, 25, 50, 75, 99]
        for i, idx in enumerate(pos_indices):
            axes[1, 1].plot(t_np*1e6, phi[idx, :], label=f'x={x_np[idx]*1e3:.1f}mm', alpha=0.7)
        axes[1, 1].set_xlabel("Time (μs)")
        axes[1, 1].set_ylabel("Electric Potential (V)")
        axes[1, 1].set_title("Temporal Evolution of $\\phi$")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.log(f"[green]✓[/green] Saved visualization to: {save_path}")
        plt.close()
        
        return save_path
    
    def on_train_end(self):
        """
        Called automatically at the end of training.
        Generates visualization of the solution.
        """
        console.log("\n[cyan]Training complete! Generating visualization...[/cyan]")
        try:
            save_path = self.visualize_solution()
            console.log(f"[green]✓[/green] Visualization complete!")
        except Exception as e:
            console.log(f"[red]✗[/red] Visualization failed: {e}")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.scheduler_name == 'linear' or self.scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.estimated_stepping_batches
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer


# ============================================================================
# Model Components
# ============================================================================

class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping for input coordinates."""
    
    def __init__(self, input_dim: int, num_features: int, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron backbone."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = 'tanh'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.Tanh()

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# PINN Models
# ============================================================================

class CCPPinn(BaseModel):
    """
    Capacitively Coupled Plasma PINN model.
    
    Default architecture: MLP with optional Fourier features.
    All hyperparameters can be overridden via YAML config.
    """
    
    def __init__(
            self,
            hidden_dims: List[int] = None,
            use_fourier: bool = True,
            fourier_scale: float = 10.0,
            num_fourier_features: int = 32,
            log_space_ne: bool = True,
            activation: str = 'tanh',
            **kwargs: Any
    ):
        # Set defaults
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
            
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.hidden_dims = hidden_dims
        self.use_fourier = use_fourier
        self.fourier_scale = fourier_scale
        self.num_fourier_features = num_fourier_features
        self.log_space_ne = log_space_ne
        self.activation = activation

        # Build network
        input_dim = 2  # (x, t)
        if use_fourier:
            self.feature_mapping = FourierFeatureMapping(input_dim, num_fourier_features, fourier_scale)
            input_dim = num_fourier_features * 2

        self.net = MLP(input_dim, hidden_dims, output_dim=2, activation=activation)
        
        console.log(f"[green]✓[/green] Initialized CCPPinn: {self.experiment_name}")
        console.log(f"  Hidden dims: {hidden_dims}")
        console.log(f"  Fourier features: {use_fourier} (scale={fourier_scale}, n={num_fourier_features})")
        console.log(f"  Log-space n_e: {log_space_ne}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CCPPinn.
        
        Returns physical quantities (n_e, phi) by:
        1. Normalizing inputs
        2. Applying optional Fourier features
        3. MLP forward pass
        4. Scaling outputs to physical units
        """
        # Normalize inputs to dimensionless form
        inputs = torch.cat([x / self.scales.x_ref, t * self.params.f], dim=1)
        
        if hasattr(self, 'feature_mapping'):
            inputs = self.feature_mapping(inputs)
        outputs = self.net(inputs)
        
        out1 = outputs[:, 0:1]
        out2 = outputs[:, 1:2]
        
        # Dimensionless predictions
        if self.log_space_ne:
            n_e_hat = torch.exp(out1)
        else:
            n_e_hat = out1
        phi_hat = out2
        
        # Scale to physical units
        n_e = n_e_hat * self.scales.n_ref
        phi = phi_hat * self.scales.phi_ref
        
        return n_e, phi


class SimplePinn(BaseModel):
    """
    Simple PINN baseline without Fourier features.
    
    Demonstrates how to create alternative architectures.
    """
    
    def __init__(
            self,
            hidden_dims: List[int] = None,
            activation: str = 'relu',
            **kwargs: Any
    ):
        if hidden_dims is None:
            hidden_dims = [32, 32]
            
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # Simple MLP
        self.net = MLP(input_dim=2, hidden_dims=hidden_dims, output_dim=2, activation=activation)
        
        console.log(f"[green]✓[/green] Initialized SimplePinn: {self.experiment_name}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple forward without normalization or Fourier features."""
        inputs = torch.cat([x, t], dim=1)
        outputs = self.net(inputs)
        
        n_e = outputs[:, 0:1] * self.scales.n_ref
        phi = outputs[:, 1:2] * self.scales.phi_ref
        
        return n_e, phi
