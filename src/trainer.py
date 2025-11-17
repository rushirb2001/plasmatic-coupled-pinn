
# ============================================================================
# src/trainer.py
# ============================================================================
from src.utils.richco import console
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
import time

# Initial loading spinner (kept from original)
spinner = Spinner("dots", text=f"Loading libraries...")
loading_panel = Panel(spinner, title="Initializing", border_style="cyan", width=40)
console.log()

# We can't easily do the "with Live" block if we want to be importable, but let's keep it for the script execution feel
# or just standard imports. The original had it at top level.

import os, warnings, sys
from pathlib import Path
from typing import Dict, List, Optional, Union

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.enable_flash_sdp(True) # Might fail on Mac
# torch.set_float32_matmul_precision('highest') # Might fail on Mac

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from src.model import BaseModel, CCPPinn
from src.data.sampling import UniformSampler, BetaSampler, GridSampler
from src.utils.physics import DefaultParameters

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, 
        batch_size: int = 10000,
        num_collocation_points: int = 20000,
        domain_length: float = 0.025,
        time_duration: float = 1.0,
        sampler_type: str = "uniform",
        beta_start: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_collocation_points = num_collocation_points
        self.domain_length = domain_length
        self.time_duration = time_duration
        self.sampler_type = sampler_type
        self.beta_param = beta_start
        self.num_workers = num_workers
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        console.log(f"[red] Using device: {self.device}")

    def setup(self, stage=None) -> None:
        x_range = (0, self.domain_length)
        t_range = (0, self.time_duration)

        if self.sampler_type == "uniform":
            sampler = UniformSampler(x_range, t_range)
        elif self.sampler_type == "beta":
            sampler = BetaSampler(x_range, t_range, beta=self.beta_param)
        elif self.sampler_type == "grid":
            nx = int(self.num_collocation_points**0.5)
            nt = int(self.num_collocation_points**0.5)
            sampler = GridSampler(x_range, t_range, nx, nt)
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

        console.log(f"[yellow] Generating {self.num_collocation_points} points using {self.sampler_type} sampler")
        x, t = sampler.sample(self.num_collocation_points)
        
        self.train_dataset = TensorDataset(x, t)
        
        # Validation (Grid)
        val_sampler = GridSampler(x_range, t_range, nx=100, nt=100)
        vx, vt = val_sampler.sample()
        self.val_dataset = TensorDataset(vx, vt)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument('--experiment_name', type=str, default='experiment')
        parser.add_argument('--output_dir', type=str, default='./experiments')
        parser.add_argument('--early_stopping_patience', type=int, default=10)
        parser.add_argument('--monitor_metric', type=str, default='val_loss')

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        # Setup callbacks similar to original trainer.py
        subcommand = self.config.get('subcommand')
        if not subcommand: return
        
        config = self.config[subcommand]
        experiment_name = config.get('experiment_name', 'experiment')
        output_dir = config.get('output_dir', './experiments')

        if subcommand == 'fit':
            self._setup_fit_callbacks(config, experiment_name, output_dir)

    def _setup_fit_callbacks(self, config, experiment_name, output_dir):
        callbacks = [
            {
                "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": os.path.join(output_dir, experiment_name),
                    "filename": "{epoch:02d}-{val_loss:.2e}",
                    "monitor": config.get('monitor_metric', 'val_loss'),
                    "save_top_k": 3,
                    "mode": "min",
                },
            },
            {
                "class_path": "pytorch_lightning.callbacks.RichProgressBar",
            },
            {
                "class_path": "pytorch_lightning.callbacks.RichModelSummary",
            },
        ]
        
        loggers = [
            {
                "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": output_dir,
                    "name": experiment_name,
                },
            }
        ]
        
        if 'trainer' not in config: config['trainer'] = {}
        config['trainer']['callbacks'] = callbacks
        config['trainer']['logger'] = loggers
        
        console.log(f"✓ Experiment: {experiment_name}")

def main():
    CustomLightningCLI(
        model_class=BaseModel,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == "__main__":
    main()
