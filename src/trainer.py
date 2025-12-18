"""
Training script for CCP-II PINN models.

This module provides:
- CollocationDataModule: PyTorch Lightning DataModule with configurable samplers
- PINNLightningCLI: Custom Lightning CLI with model registry support

Usage:
    # Train with YAML config
    python -m src.trainer fit --config configs/default.yaml

    # Train with command line args
    python -m src.trainer fit --model.class_path=src.model.SequentialPINN \
        --model.hidden_layers=[64,64,64] --trainer.max_epochs=1000

    # Test a checkpoint
    python -m src.trainer test --ckpt_path=path/to/checkpoint.ckpt
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Use TensorCores on A100

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from src.model import BasePINN, MODEL_REGISTRY
from src.data.collocation import (
    SamplingConfig,
    create_sampler,
    UniformSampler,
    GridSampler,
    BetaSampler,
    LatinHypercubeSampler,
)


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """
    Auto-generate experiment name from config.

    Pattern: {model_class}_{sampler_type}_lr{learning_rate}_{timestamp}

    Examples:
    - SequentialPINN_uniform_lr1e-03_20241231_143052
    - GatedPINN_beta_lr5e-04_20241231_150000
    """
    # Extract model class name
    model_config = config.get("model", {})
    if isinstance(model_config, dict):
        class_path = model_config.get("class_path", "BasePINN")
        # Extract class name from path like "src.model.SequentialPINN"
        model_name = class_path.split(".")[-1] if "." in class_path else class_path
    else:
        model_name = "PINN"

    # Extract sampler type
    data_config = config.get("data", {})
    if isinstance(data_config, dict):
        sampler_type = data_config.get("sampler_type", "uniform")
    else:
        sampler_type = "uniform"

    # Extract learning rate
    if isinstance(model_config, dict):
        init_args = model_config.get("init_args", {})
        if isinstance(init_args, dict):
            lr = init_args.get("learning_rate", 1e-3)
        else:
            lr = 1e-3
    else:
        lr = 1e-3

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{model_name}_{sampler_type}_lr{lr:.0e}_{timestamp}"


def get_wandb_run_name(experiment_name: str, config: Dict[str, Any]) -> str:
    """
    Generate WandB run name with additional context.

    Includes hidden layer info and key hyperparameters.
    """
    model_config = config.get("model", {})
    if isinstance(model_config, dict):
        init_args = model_config.get("init_args", {})
        if isinstance(init_args, dict):
            hidden = init_args.get("hidden_layers", [64, 64, 64])
            hidden_str = "x".join(str(h) for h in hidden) if hidden else "default"
        else:
            hidden_str = "default"
    else:
        hidden_str = "default"

    return f"{experiment_name}_h{hidden_str}"


class CollocationDataModule(pl.LightningDataModule):
    """
    DataModule for collocation point sampling.

    Supports multiple sampling strategies configurable via YAML:
    - uniform: Random uniform sampling
    - beta: Beta distribution sampling (concentrated at boundaries)
    - grid: Regular grid sampling
    - latin-hypercube: Latin hypercube sampling for better coverage

    Args:
        batch_size: Batch size for training
        num_points: Number of collocation points
        sampler_type: Type of sampler ('uniform', 'beta', 'grid', 'latin-hypercube')
        x_range: Spatial domain range (normalized to [0, 1])
        t_range: Temporal domain range (normalized to [0, 1])
        beta_param: Beta parameter for beta sampling
        val_grid_size: Grid size for validation (nx=nt)
        num_workers: Number of data loading workers
    """

    def __init__(
        self,
        batch_size: int = 4096,
        num_points: int = 20000,
        sampler_type: str = "uniform",
        x_range: tuple = (0.0, 1.0),
        t_range: tuple = (0.0, 1.0),
        beta_param: float = 1.0,
        val_grid_size: int = 100,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_points = num_points
        self.sampler_type = sampler_type
        self.x_range = x_range
        self.t_range = t_range
        self.beta_param = beta_param
        self.val_grid_size = val_grid_size
        self.num_workers = num_workers

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        # Create sampler based on type
        config = SamplingConfig(
            x_range=self.x_range,
            t_range=self.t_range,
            num_samples=self.num_points,
        )

        if self.sampler_type == "uniform":
            sampler = UniformSampler(config)
        elif self.sampler_type == "beta":
            sampler = BetaSampler(config, beta_x=self.beta_param, beta_t=self.beta_param)
        elif self.sampler_type == "grid":
            nx = int(self.num_points ** 0.5)
            nt = int(self.num_points ** 0.5)
            sampler = GridSampler(config, nx=nx, nt=nt)
        elif self.sampler_type == "latin-hypercube":
            sampler = LatinHypercubeSampler(config)
        else:
            sampler = create_sampler(self.sampler_type, config)

        # Generate training points and pre-load to device for faster access
        x_t = sampler.samples.to(self.device)
        self.train_dataset = TensorDataset(x_t)

        # Validation: always use grid for consistent evaluation
        val_config = SamplingConfig(
            x_range=self.x_range,
            t_range=self.t_range,
            num_samples=self.val_grid_size ** 2,
        )
        val_sampler = GridSampler(val_config, nx=self.val_grid_size, nt=self.val_grid_size)
        val_x_t = val_sampler.samples.to(self.device)
        self.val_dataset = TensorDataset(val_x_t)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        # Data is pre-loaded to device, no need for pin_memory or workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,  # Avoid small last batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


class PINNLightningCLI(LightningCLI):
    """
    Custom Lightning CLI with PINN-specific features.

    Features:
    - Model registry support: use model name instead of full class path
    - Automatic experiment directory setup
    - Default callbacks for checkpointing and progress
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments."""
        super().add_arguments_to_parser(parser)

        parser.add_argument(
            "--experiment_name",
            type=str,
            default=None,
            help="Name of the experiment (auto-generated if not provided)",
        )
        parser.add_argument(
            "--auto_name",
            type=bool,
            default=True,
            help="Auto-generate experiment name from config",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./experiments",
            help="Output directory for checkpoints and logs",
        )
        parser.add_argument(
            "--use_wandb",
            type=bool,
            default=False,
            help="Enable Weights & Biases logging",
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="pinn-ccp2",
            help="W&B project name",
        )
        parser.add_argument(
            "--wandb_entity",
            type=str,
            default=None,
            help="W&B entity (username or team)",
        )

    def before_instantiate_classes(self):
        """Setup callbacks and loggers before class instantiation."""
        super().before_instantiate_classes()

        subcommand = self.config.get("subcommand")
        if not subcommand:
            return

        config = self.config[subcommand]
        output_dir = config.get("output_dir", "./experiments")

        # Auto-generate experiment name if not provided
        auto_name = config.get("auto_name", True)
        experiment_name = config.get("experiment_name")

        if experiment_name is None or (auto_name and experiment_name == "pinn_experiment"):
            experiment_name = generate_experiment_name(config)
            config["experiment_name"] = experiment_name
            print(f"Auto-generated experiment name: {experiment_name}")

        # Generate WandB run name (store as instance variable, not in config)
        self._wandb_run_name = get_wandb_run_name(experiment_name, config)

        # Create experiment directory
        exp_dir = Path(output_dir) / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Update model output_dir to match experiment directory
        if "model" in config and isinstance(config["model"], dict):
            if "init_args" not in config["model"]:
                config["model"]["init_args"] = {}
            config["model"]["init_args"]["output_dir"] = str(exp_dir)

        if subcommand == "fit":
            self._setup_fit_callbacks(config, experiment_name, output_dir)

    def _setup_fit_callbacks(self, config, experiment_name, output_dir):
        """Setup training callbacks."""
        callbacks = [
            {
                "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": os.path.join(output_dir, experiment_name, "checkpoints"),
                    "filename": "{epoch:03d}-{val_loss:.2e}",
                    "monitor": "val_loss",
                    "save_top_k": 3,
                    "mode": "min",
                    "save_last": True,
                },
            },
            {
                "class_path": "pytorch_lightning.callbacks.RichProgressBar",
            },
            {
                "class_path": "pytorch_lightning.callbacks.RichModelSummary",
                "init_args": {"max_depth": 2},
            },
            {
                "class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "step"},
            },
        ]

        loggers = [
            {
                "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": output_dir,
                    "name": experiment_name,
                    "version": "tensorboard",
                },
            },
            {
                "class_path": "pytorch_lightning.loggers.CSVLogger",
                "init_args": {
                    "save_dir": output_dir,
                    "name": experiment_name,
                    "version": "csv",
                },
            },
        ]

        # Add WandB logger if enabled
        use_wandb = config.get("use_wandb", False)
        if use_wandb:
            wandb_project = config.get("wandb_project", "pinn-ccp2")
            wandb_entity = config.get("wandb_entity", None)
            wandb_run_name = getattr(self, "_wandb_run_name", experiment_name)

            # Build tags from config
            wandb_tags = [
                experiment_name.split("_")[0],  # Model name
                config.get("data", {}).get("sampler_type", "uniform"),
            ]

            wandb_config = {
                "class_path": "pytorch_lightning.loggers.WandbLogger",
                "init_args": {
                    "project": wandb_project,
                    "name": wandb_run_name,
                    "save_dir": output_dir,
                    "tags": wandb_tags,
                    "group": experiment_name.split("_")[0],  # Group by model
                },
            }
            if wandb_entity:
                wandb_config["init_args"]["entity"] = wandb_entity
            loggers.append(wandb_config)
            print(f"WandB run name: {wandb_run_name}")

        if "trainer" not in config:
            config["trainer"] = {}

        config["trainer"]["callbacks"] = callbacks
        config["trainer"]["logger"] = loggers


def main():
    """Main entry point for training."""
    cli = PINNLightningCLI(
        model_class=BasePINN,
        datamodule_class=CollocationDataModule,
        subclass_mode_model=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={
            "fit": {"default_config_files": ["configs/default.yaml"]},
        },
    )


if __name__ == "__main__":
    main()
