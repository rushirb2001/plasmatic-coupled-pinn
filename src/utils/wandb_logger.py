"""
Weights & Biases integration for PINN training.

Provides utilities for:
- Experiment tracking and logging
- Hyperparameter sweeps
- Model artifact management
- Custom PINN metrics logging
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    Weights & Biases logger for PINN experiments.

    Provides easy integration with PyTorch Lightning and custom logging
    for physics-informed metrics.

    Args:
        project: W&B project name
        entity: W&B entity (username or team)
        name: Run name
        config: Configuration dictionary
        tags: List of tags for the run
        notes: Run notes
        mode: 'online', 'offline', or 'disabled'
    """

    def __init__(
        self,
        project: str = "pinn-ccp2",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
            reinit=True,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        wandb.log(metrics, step=step)

    def log_losses(
        self,
        loss_total: float,
        loss_cont: float,
        loss_pois: float,
        loss_bc: float,
        step: Optional[int] = None,
        prefix: str = "train",
    ):
        """Log physics loss components."""
        self.log({
            f"{prefix}/loss_total": loss_total,
            f"{prefix}/loss_continuity": loss_cont,
            f"{prefix}/loss_poisson": loss_pois,
            f"{prefix}/loss_bc": loss_bc,
        }, step=step)

    def log_gradients(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
    ):
        """Log gradient statistics."""
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[f"gradients/{name}"] = grad_norm

        if grad_norms:
            grad_norms["gradients/total_norm"] = sum(grad_norms.values())
            self.log(grad_norms, step=step)

    def log_weights(
        self,
        lambda_cont: float,
        lambda_pois: float,
        step: Optional[int] = None,
    ):
        """Log adaptive loss weights."""
        self.log({
            "loss_weights/continuity": lambda_cont,
            "loss_weights/poisson": lambda_pois,
        }, step=step)

    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """Log learning rate."""
        self.log({"learning_rate": lr}, step=step)

    def log_solution_image(
        self,
        n_e: np.ndarray,
        phi: np.ndarray,
        x: np.ndarray,
        t: np.ndarray,
        step: Optional[int] = None,
        caption: str = "Solution",
    ):
        """Log solution heatmaps as images."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(n_e, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Electron Density")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(phi, aspect="auto", origin="lower", cmap="plasma")
        axes[1].set_title("Electric Potential")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()

        self.log({
            "solution": wandb.Image(fig, caption=caption)
        }, step=step)

        plt.close(fig)

    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        name: str = "model",
        aliases: Optional[List[str]] = None,
    ):
        """Log model checkpoint as artifact."""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact, aliases=aliases)

    def log_config(self, config: Dict[str, Any]):
        """Update run configuration."""
        wandb.config.update(config)

    def watch_model(
        self,
        model: torch.nn.Module,
        log: str = "all",
        log_freq: int = 100,
    ):
        """Watch model for gradient and parameter logging."""
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish the W&B run."""
        wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_sweep_config(
    method: str = "bayes",
    metric_name: str = "val_loss",
    metric_goal: str = "minimize",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a W&B sweep configuration.

    Args:
        method: Sweep method ('bayes', 'random', 'grid')
        metric_name: Metric to optimize
        metric_goal: 'minimize' or 'maximize'
        parameters: Parameter search space

    Returns:
        Sweep configuration dictionary
    """
    if parameters is None:
        parameters = {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            "hidden_layers": {
                "values": [[32, 32], [64, 64], [64, 64, 64], [128, 128]],
            },
            "num_ffm_frequencies": {
                "values": [2, 4, 8],
            },
            "batch_size": {
                "values": [1024, 2048, 4096],
            },
            "loss_weight_bc": {
                "distribution": "log_uniform_values",
                "min": 1.0,
                "max": 100.0,
            },
        }

    return {
        "method": method,
        "metric": {
            "name": metric_name,
            "goal": metric_goal,
        },
        "parameters": parameters,
    }


class WandBCallback:
    """
    PyTorch Lightning callback for W&B logging.

    Use this as a Lightning callback for automatic logging.
    """

    def __init__(
        self,
        project: str = "pinn-ccp2",
        entity: Optional[str] = None,
        log_gradients: bool = True,
        log_freq: int = 100,
    ):
        self.project = project
        self.entity = entity
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self.logger = None

    def on_train_start(self, trainer, pl_module):
        """Initialize W&B at training start."""
        if WANDB_AVAILABLE:
            config = dict(pl_module.hparams) if hasattr(pl_module, "hparams") else {}
            self.logger = WandBLogger(
                project=self.project,
                entity=self.entity,
                config=config,
            )
            if self.log_gradients:
                self.logger.watch_model(pl_module, log_freq=self.log_freq)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log batch metrics."""
        if self.logger and batch_idx % self.log_freq == 0:
            step = trainer.global_step
            self.logger.log({"train/loss": outputs["loss"].item()}, step=step)

    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics."""
        if self.logger:
            metrics = trainer.callback_metrics
            val_metrics = {
                f"val/{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()
                if "val" in k
            }
            self.logger.log(val_metrics, step=trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        """Finish W&B run."""
        if self.logger:
            self.logger.finish()
