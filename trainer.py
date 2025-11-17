from utils.richco import console

from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
import time

spinner = Spinner("dots", text=f"Loading libraries...")
loading_panel = Panel(spinner, title="Initializing", border_style="cyan", width=40)
console.log()
with Live(loading_panel, console=console, refresh_per_second=30) as live:
    import os, warnings, sys
    from pathlib import Path
    from typing import Dict, List, Optional, Union

    warnings.filterwarnings("ignore", category=UserWarning)
    spinner.text="Loaded standard libraries"
    live.update(loading_panel)

    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision('highest')
    spinner.text="Loaded torch libraries"
    live.update(loading_panel)

    import pytorch_lightning as pl
    from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
    spinner.text="Loaded lightning libraries"
    live.update(loading_panel)

    import numpy as np
    import pandas as pd
    spinner.text="Loaded sklearn libraries"
    live.update(loading_panel)

    from model import BaseModel, DyAbModel, CustomModel, CustomPooledModel, CustomReArchModel, LinearPredictorModel
    spinner.text="Loaded standard models"
    live.update(loading_panel)

    from utils.loader import Loader
    from utils.resource import ResourceManager
    from utils.embeddings import EmbeddingEngine
    spinner.text="Loaded custom utilities and libraries"
    live.update(loading_panel)

    time.sleep(2)
    spinner.text="✅ All libraries loaded successfully!"
    loading_panel.style = "green"
    live.update(loading_panel)

class FlexibleDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, torch.HalfTensor],
        features: Union[str, List[str]] = None,
        features_2: Union[str, List[str]] = None,
        target: Optional[str] = None,
        target_2: Optional[str] = None,
        device = 'cpu | cuda | mps',
        embedding_paths=None,
    ) -> None:
        
        self.data = data
        self.features = None if features is None else (features if isinstance(features, list) else [features])
        self.features_2 = None if features_2 is None else (features_2 if isinstance(features_2, list) else [features_2])
        self.target = target
        self.target_2 = target_2

        self._embedding_paths = [embedding_paths] if isinstance(embedding_paths, str) else embedding_paths
        self._embeddings = {}
        self.device = device
        self._embeddings = self._get_embedding_files()
        
        self._target_tensor = None
        self._target_2_tensor = None
        if self.target:
            self._target_tensor = torch.tensor(self.data[self.target], dtype=torch.float32, device=self.device)
        if self.target_2:
            self._target_2_tensor = torch.tensor(self.data[self.target_2], dtype=torch.float32, device=self.device)
    
    def _get_embedding_files(self):
        if not self._embeddings and self._embedding_paths:
            for idx, file_path in enumerate(self._embedding_paths):
                if file_path and os.path.isfile(file_path):
                    emb_tensor = torch.load(file_path, weights_only=False, map_location=self.device)['embeddings']
                    if not emb_tensor.is_contiguous():
                        emb_tensor = emb_tensor.contiguous()
                    self._embeddings[f'embedding_{idx}'] = emb_tensor
        return self._embeddings
            
    def __getitem__(self, index):

        idx1 = torch.tensor(self.data[self.features[0]][index]).to(dtype=torch.int32)
        if self.features_2 is not None:
            idx2 = torch.tensor(self.data[self.features_2[0]][index]).to(dtype=torch.int32)
        emb1 = torch.cat([embedding[idx1] for _, embedding in self._embeddings.items()]).to(device=self.device, dtype=torch.float32)
        if self.features_2 is not None:
            emb2 = torch.cat([embedding[idx2] for _, embedding in self._embeddings.items()]).to(device=self.device, dtype=torch.float32)
            outs = [emb1, emb2]
        else:
            outs = [emb1]
        if self.target:
            outs.append(self._target_tensor[index])
        if self.target_2:
            outs.append(self._target_2_tensor[index])
        return tuple(outs)
    
    def __len__(self):
        return len(self.data[self.features[0]])

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, 
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        features: Union[str, List[str]] = None,
        target: Optional[str] = None,
        features_2: Union[str, List[str]] = None,
        target_2: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.features = features
        self.target = target
        self.features_2 = features_2
        self.target_2 = target_2
        self.batch_size = batch_size
    
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        console.log(f"[red] Using device: {self.device}")

        self.dl = Loader()
        self.rm = ResourceManager()

    def setup(self, stage=None) -> None:
        if self.dataset_name:
            console.log(f"[yellow] Loading dataset '{self.dataset_name}'")
            try:
                self.dataset = self.rm.get_resource(self.dataset_name)
            except Exception as e:
                raise ValueError(f"Failed to load dataset '{self.dataset_name}': {e}")
        elif self.dataset_path:
            self.dataset = self.dl.load(self.dataset_path, map_location=self.device, weights_only=False)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided.")
        
        paths = self.dataset.get('path', None)
        console.log(f" Path from dataset: {paths}")
        if paths is not None:
            if isinstance(paths, str):
                paths = [paths]
                
            for idx, file_path in enumerate(paths or []):
                if '/Users/rbhavsar/Code' in file_path and '/Users/rbhavsar/Code' not in str(Path(__file__).parent.parent):
                    paths[idx] = file_path.replace('/Users/rbhavsar/Code', '/home/rbhavsar/antibody-property-prediction-ml')
            
            if len(self.dataset['train']) == 4:
                self.train_dataset = FlexibleDataset(
                    data=self.dataset['train'],
                    features=self.features,
                    target=self.target,
                    features_2=self.features_2,
                    target_2=self.target_2,
                    embedding_paths=paths,
                    device=self.device
                )

                self.val_dataset = FlexibleDataset(
                    data=self.dataset['val'],
                    features=self.features,
                    target=self.target,
                    features_2=self.features_2,
                    target_2=self.target_2,
                    embedding_paths=paths,
                    device=self.device
                )

                self.test_dataset = FlexibleDataset(
                    data=self.dataset['test'],
                    features=self.features,
                    target=self.target,
                    features_2=self.features_2,
                    target_2=self.target_2,
                    embedding_paths=paths,
                    device=self.device
                )
            elif len(self.dataset['train']) == 2:
                self.train_dataset = FlexibleDataset(
                    data=self.dataset['train'],
                    features=self.features,
                    target=self.target,
                    embedding_paths=paths,
                    device=self.device
                )

                self.val_dataset = FlexibleDataset(
                    data=self.dataset['val'],
                    features=self.features,
                    target=self.target,
                    embedding_paths=paths,
                    device=self.device
                )

                self.test_dataset = FlexibleDataset(
                    data=self.dataset['test'],
                    features=self.features,
                    target=self.target,
                    embedding_paths=paths,
                    device=self.device
                )

        elif paths is None:
            # console.log(self.dataset['train']['embedding'].shape, self.dataset['train']['embedding_2'].shape)
            if len(self.dataset['train']) == 4:
                self.train_dataset = TensorDataset(self.dataset['train']['embedding'].float(),self.dataset['train']['embedding_2'].float(),self.dataset['train']['target'].float(),self.dataset['train']['target_2'].float()) 
                self.val_dataset = TensorDataset(self.dataset['val']['embedding'].float(),self.dataset['val']['embedding_2'].float(),self.dataset['val']['target'].float(),self.dataset['val']['target_2'].float())
                self.test_dataset = TensorDataset(self.dataset['test']['embedding'].float(),self.dataset['test']['embedding_2'].float(),self.dataset['test']['target'].float(),self.dataset['test']['target_2'].float())
            elif len(self.dataset['train']) == 2:
                self.train_dataset = TensorDataset(self.dataset['train']['embedding'].float(),self.dataset['train']['target'].float())
                self.val_dataset = TensorDataset(self.dataset['val']['embedding'].float(),self.dataset['val']['target'].float())
                self.test_dataset = TensorDataset(self.dataset['test']['embedding'].float(),self.dataset['test']['target'].float())


    def _create_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_loader(self.test_dataset, shuffle=False)

class CustomLightningCLI(LightningCLI):
    
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to parser."""
        super().add_arguments_to_parser(parser)
        
        parser.add_argument('--experiment_name', type=str, default='experiment')
        parser.add_argument('--output_dir', type=str, default='./src/experiments')
        parser.add_argument('--early_stopping_patience', type=int, default=5)
        parser.add_argument('--monitor_metric', type=str, default='val_loss')
        parser.add_argument('--monitor_mode', type=str, default='min', choices=['min', 'max'])
        parser.add_argument('--save_top_k', type=int, default=3)
        parser.add_argument('--use_wandb', type=bool, default=False)
        parser.add_argument('--wandb_project', type=str, default='antibody-ml')
    
    def before_instantiate_classes(self):
        """Setup callbacks and loggers before instantiation."""
        super().before_instantiate_classes()
        
        subcommand = self.config.get('subcommand')
        if not subcommand:
            return
            
        config = self.config[subcommand]
        
        experiment_name = config.get('experiment_name', 'experiment')
        output_dir = config.get('output_dir', './experiments')
        
        # Sync experiment_name to model if needed
        if 'model' in config and hasattr(config['model'], 'init_args'):
            if 'experiment_name' not in config['model']['init_args']:
                config['model']['init_args']['experiment_name'] = experiment_name
        
        if subcommand == 'fit':
            self._setup_fit_callbacks_and_loggers(config, experiment_name, output_dir)
        elif subcommand == 'test':
            self._setup_test_callbacks_and_loggers(config, experiment_name, output_dir)
        elif subcommand == 'predict':
            self._setup_predict_callbacks_and_loggers(config, experiment_name, output_dir)
    
    def _setup_fit_callbacks_and_loggers(self, config, experiment_name, output_dir):
        """Setup callbacks and loggers for fit command."""
        
        callbacks = [
            {
                "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": os.path.join(output_dir, experiment_name),
                    "filename": "{epoch:02d}-{val_loss:.2f}",
                    "monitor": config.get('monitor_metric', 'val_loss'),
                    "mode": config.get('monitor_mode', 'min'),
                    "save_top_k": config.get('save_top_k', 3),
                    "save_weights_only": False,
                    "verbose": True,
                },
            },
            {
                "class_path": "pytorch_lightning.callbacks.EarlyStopping",
                "init_args": {
                    "monitor": config.get('monitor_metric', 'val_loss'),
                    "patience": config.get('early_stopping_patience', 5),
                    "mode": config.get('monitor_mode', 'min'),
                    "verbose": True,
                },
            },
            {
                "class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "step"},
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
                    "version": None,
                    "default_hp_metric": False,
                },
            }
        ]
        
        if config.get('use_wandb', False):
            loggers.append({
                "class_path": "pytorch_lightning.loggers.WandbLogger",
                "init_args": {
                    "project": config.get('wandb_project', 'antibody-ml'),
                    "name": experiment_name,
                    "save_dir": os.path.join(output_dir, experiment_name),
                },
            })
        
        if 'trainer' not in config:
            config['trainer'] = {}
        
        config['trainer']['callbacks'] = callbacks
        config['trainer']['logger'] = loggers
        
        console.log(f"✓ Experiment: {experiment_name}")
        console.log(f"✓ Output: {os.path.join(output_dir, experiment_name)}")
    
    def _setup_test_callbacks_and_loggers(self, config, experiment_name, output_dir):
        """Setup minimal callbacks and loggers for test command."""
        
        callbacks = [{"class_path": "pytorch_lightning.callbacks.RichProgressBar"}]
        
        loggers = [
            {
                "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": output_dir,
                    "name": f"{experiment_name}_test",
                    "version": None,
                    "default_hp_metric": False,
                },
            }
        ]
        
        if 'trainer' not in config:
            config['trainer'] = {}
        
        config['trainer']['callbacks'] = callbacks
        config['trainer']['logger'] = loggers
    
    def _setup_predict_callbacks_and_loggers(self, config, experiment_name, output_dir):
        """Setup minimal callbacks and loggers for predict command."""
        
        callbacks = [{"class_path": "pytorch_lightning.callbacks.RichProgressBar"}]
        
        if 'trainer' not in config:
            config['trainer'] = {}
        
        config['trainer']['callbacks'] = callbacks
        config['trainer']['logger'] = False  # No logging for predictions


def fix_checkpoint_before_cli():
    """Fix checkpoint file before CLI parses it, if needed."""
    
    # Check if we're running test or predict with a ckpt_path
    if len(sys.argv) < 2:
        return
    
    subcommand = sys.argv[1]
    if subcommand not in ['test', 'predict']:
        return
    
    # Find ckpt_path in arguments
    ckpt_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--ckpt_path' and i + 1 < len(sys.argv):
            ckpt_path = sys.argv[i + 1]
            break
    
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        needs_fix = False
        if 'hyper_parameters' in checkpoint:
            hyper_params = checkpoint['hyper_parameters']
            
            # Check for _class_path or _instantiator at any level
            if isinstance(hyper_params, dict):
                if '_class_path' in hyper_params or '_instantiator' in hyper_params:
                    needs_fix = True
                    console.log("[yellow]Found old format keys in hyperparameters")
        
        if needs_fix:
            # Create fixed checkpoint
            fixed_checkpoint = checkpoint.copy()
            hyper_params = fixed_checkpoint['hyper_parameters']
            
            # Fix top-level _class_path and remove _instantiator
            if '_class_path' in hyper_params:
                class_path = hyper_params.pop('_class_path')
                hyper_params.pop('_instantiator', None)
                fixed_hyper_params = {
                    'class_path': class_path,
                    'init_args': {k: v for k, v in hyper_params.items() if k not in ['class_path', '_instantiator', 'model']}
                }
                # Preserve model if it exists
                if 'model' in hyper_params:
                    fixed_hyper_params['init_args']['model'] = hyper_params['model']
                fixed_checkpoint['hyper_parameters'] = fixed_hyper_params
                hyper_params = fixed_hyper_params.get('init_args', fixed_hyper_params)
            
            # Save fixed checkpoint
            torch.save(fixed_checkpoint, ckpt_path)
            
    except Exception as e:
        import traceback
        console.log(f"[red]{traceback.format_exc()}")
        return None


def main():
    
    fix_checkpoint_before_cli()
    
    CustomLightningCLI(
        model_class=BaseModel,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        seed_everything_default=42,
        save_config_callback=None,
    )
if __name__ == "__main__":
    main()