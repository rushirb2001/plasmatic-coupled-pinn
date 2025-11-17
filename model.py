# ============================================================================
# models.py
# ============================================================================
"""
Module: models
Part of: Antibody Thermostability Prediction Framework
Author: [Author Name]
Version: 2.0.0

Description:
    Comprehensive PyTorch Lightning model library for predicting antibody
    thermostability changes from protein sequence embeddings. Implements
    multiple deep learning architectures optimized for regression and
    classification tasks on protein data, with extensive support for
    training, validation, benchmarking, and visualization.
    
    This module provides a flexible base class hierarchy enabling rapid
    experimentation with different model architectures, loss functions,
    optimizers, and learning rate schedules. All models inherit from
    BaseModel, which provides unified training loops, metric tracking,
    and experiment management through PyTorch Lightning.

Core Components:
    Base Classes:
        - BaseModel: Foundation class with training/validation loops,
          optimizer/scheduler configuration, metric tracking, and
          benchmarking capabilities. Supports both regression and
          classification tasks.
    
    Regression Models (Thermostability Prediction):
        - DyAbModel: ResNet18-based architecture that converts sequence
          embedding differences into 2D images for CNN processing.
          Supports multiple channel operations (diff, add, mul, div).
          
        - CustomModel: Transformer-based architecture with positional
          encoding, cross-attention between sequence pairs, multi-scale
          pooling, and stability metric extraction. Designed for learning
          complex sequence interactions.
          
        - CustomReArchModel: Refined variant of CustomModel with deeper
          input projection layers and optimized regularization for
          improved generalization.
          
        - CustomPooledModel: Simplified architecture using mean pooling
          instead of transformers, designed for faster training and
          inference on large datasets.
          
        - CNN1DModel: 1D convolutional network for efficient processing
          of sequential embedding data with batch normalization and
          global average pooling.
          
        - LinearPredictorModel: Multi-layer perceptron baseline for
          direct prediction from single sequence embeddings.
    
    Placeholder Models (Future Development):
        - Transformers: Reserved for pure transformer implementations
        - VAEModel: Reserved for variational autoencoder approaches
        - TLearningModel: Reserved for transfer learning pipelines
        - CNN2DModel: Reserved for 2D convolutional variants

Key Features:
    Training & Optimization:
        - Multiple optimizers: AdamW, Adam, RAdam, NAdam, SGD with Nesterov
        - 10+ learning rate schedulers: linear warmup, cosine annealing,
          ReduceLROnPlateau, OneCycleLR, CyclicLR, exponential decay
        - Configurable warmup steps and weight decay
        - Mixed precision training support via PyTorch Lightning
    
    Loss Functions:
        - Regression: MSE, MAE (L1), Huber loss
        - Classification: Cross-entropy, BCE with logits, Hinge loss
        - Task-appropriate loss selection with smooth configuration
    
    Metrics & Evaluation:
        - Regression: MAE, MSE, RMSE, Pearson correlation, R²
        - Classification: Accuracy, F1, AUROC, Precision, Recall
        - Automatic metric computation via torchmetrics
        - Per-epoch and per-step logging to TensorBoard/WandB
    
    Benchmarking & Analysis:
        - Automated generation of prediction vs. target scatter plots
        - Residual analysis with distribution histograms
        - Signal-to-noise ratio (SNR) stratified performance metrics
        - Correlation analysis between embedding magnitude and errors
        - High-DPI publication-ready plots (1200 DPI)
        - Automatic versioning of output plots
    
    Architecture Design Patterns:
        - Sequence pair processing: Models accept two embeddings and
          predict their difference (ΔTm prediction)
        - Cross-attention mechanisms for learning sequence interactions
        - Multi-scale pooling combining attention and average pooling
        - Stability metrics: cosine similarity, Euclidean distance,
          L1/L2 norms, dot products, norm ratios
        - Positional encoding with learnable components
        - Adaptive input projections for variable embedding dimensions

Dependencies:
    Core Framework:
        - pytorch_lightning (>=2.0.0): Training orchestration
        - torch (>=2.0.0): Deep learning framework
        - torch.nn: Neural network modules
        - torch.nn.functional: Functional API
        - torchmetrics (>=1.0.0): Metric computation
    
    Computer Vision:
        - torchvision.models: Pre-trained models (ResNet18)
        - torchvision.transforms: Image preprocessing
    
    Data & Computation:
        - numpy: Numerical operations
        - sklearn.metrics: Additional evaluation metrics
        - transformers: Learning rate schedulers
    
    Visualization:
        - matplotlib.pyplot: Plot generation
        - Custom font configurations for publication quality
    
    Utilities:
        - utils.richco: Rich console logging
        - glob: File pattern matching
        - os, pathlib: Path management
        - typing: Type annotations

Configuration:
    Models are instantiated with configuration dictionaries containing:
        - embedding_dim: Input embedding dimension (e.g., 640, 1280)
        - seq_len: Sequence length for positional encoding
        - hidden_dim: Hidden layer dimensionality
        - learning_rate: Initial learning rate (typically 1e-4 to 1e-6)
        - optimizer: Choice of optimizer algorithm
        - scheduler: Learning rate schedule strategy
        - task_type: 'regression', 'binary_classification', or
          'multiclass_classification'
        - loss_type: Loss function identifier
        - enable_benchmarking: Enable detailed evaluation plots
        - experiment_name: Name for organizing outputs
        - warmup_steps: Number of warmup steps for schedulers
        - weight_decay: L2 regularization strength

Model Architecture Details:
    BaseModel provides:
        - Configurable training/validation/test loops
        - Automatic metric tracking and logging
        - Optimizer and scheduler configuration
        - Loss computation with task-specific functions
        - Prediction step for inference
        - Epoch-end callbacks for metric aggregation
    
    DyAbModel architecture:
        Input: Two embeddings (B, L, D) or (B, L*D)
        1. Reshape to square matrices if needed
        2. Apply channel operations (diff/add/mul/div)
        3. Resize to fixed image size (e.g., 192x192)
        4. Normalize to [0, 1] range
        5. Process through ResNet18
        6. Output: Single value prediction
    
    CustomModel architecture:
        Input: Two embeddings (B, L, D)
        1. Linear projection to hidden_dim
        2. Add positional encoding (sinusoidal + learnable)
        3. Transformer encoder layers (2-6 layers adaptive)
        4. Cross-attention between sequences
        5. Multi-scale pooling (70% attention + 30% mean)
        6. Feature extraction with LayerNorm and dropout
        7. Stability metric computation (6 features)
        8. Dual regression heads for tm1, tm2
        9. Output: tm1 - tm2 (ΔTm)
    
    CustomPooledModel architecture:
        Input: Two embeddings (B, L, D)
        1. Linear projection to hidden_dim
        2. Embedding encoder (2-layer MLP)
        3. Mean pooling to (B, D)
        4. Pairwise operations (diff, product)
        5. Feature extraction
        6. Dual regression heads
        7. Output: tm1 - tm2
    
    CNN1DModel architecture:
        Input: Two embeddings (B, L, D)
        1. Transpose to (B, D, L)
        2. Conv1D layers with batch normalization
        3. Global average pooling
        4. Concatenate processed embeddings
        5. MLP regressor
        6. Output: ΔTm

Benchmarking System:
    When enable_benchmarking=True, models automatically generate:
    
    1. Prediction vs. Target Scatter Plots:
       - Color-coded by dataset (Specifica/Ginkgo/NbThermo)
       - R² correlation coefficient annotation
       - Pearson r² annotation
       - Ideal prediction line (y=x)
       - 1200 DPI resolution
    
    2. SNR (Signal-to-Noise Ratio) Analysis:
       - Stratifies samples by embedding magnitude
       - Computes MAE for low/medium/high SNR bins
       - Correlation between SNR and prediction errors
       - SNR distribution histogram
    
    3. Residual Analysis:
       - Residual distribution histogram
       - Residuals vs. targets scatter
       - Error distribution histogram
    
    All plots are versioned and saved to:
        experiments/{experiment_name}/graphs/

Training Best Practices:
    1. Start with linear warmup scheduler for stable training
    2. Use AdamW with weight_decay=0.01 to 0.1
    3. Learning rates: 1e-4 to 1e-6 depending on model size
    4. Enable mixed precision for faster training
    5. Monitor validation Pearson correlation
    6. Use gradient clipping for stability
    7. Implement early stopping on val_loss
    8. Save checkpoints based on val_pearson

Performance Optimization:
    - Models use torch.compile() compatible operations
    - Efficient attention mechanisms with batch_first=True
    - Memory-efficient gradient checkpointing available
    - Supports distributed training via PyTorch Lightning
    - Automatic mixed precision (AMP) support
    - DataLoader with num_workers for parallel data loading

Dataset Compatibility:
    Models are designed for:
        - Specifica dataset (seq_len=138)
        - Ginkgo GDPa1 dataset (seq_len=149)
        - NbThermo VHH dataset (seq_len=155)
    
    Batch formats supported:
        - (embedding1, embedding2, target): Direct ΔTm
        - (embedding1, embedding2, y1, y2): Compute ΔTm = y1 - y2
        - (embedding, target): Single sequence prediction

Output Files:
    Checkpoints:
        - experiments/{name}/checkpoints/epoch={N}-step={S}.ckpt
    
    Logs:
        - experiments/{name}/logs/version_{V}/
        - TensorBoard logs
        - CSV metrics
    
    Plots:
        - experiments/{name}/graphs/evaluation_predictions_vs_targets_version_{V}.png
        - experiments/{name}/graphs/snr_analysis_version_{V}.png
        - experiments/{name}/graphs/evaluation_benchmarking_version_{V}.png

Notes:
    - All models support both single GPU and distributed training
    - Automatic handling of variable sequence lengths via padding
    - Models log comprehensive metrics to TensorBoard/WandB
    - Thread-safe file versioning prevents overwriting
    - Rich console output with progress bars via utils.richco
    - Models checkpoint best validation performance automatically
    - Supports resuming training from checkpoints
    - Compatible with PyTorch Lightning Callbacks ecosystem
    
    Known Limitations:
        - DyAbModel requires square-able embedding dimensions
        - Transformer models require significant GPU memory
        - Benchmarking requires test data to fit in memory
        - SNR analysis assumes flattened embedding inputs

See Also:
    - pipelinerunner.py: Data preprocessing and pipeline orchestration
    - utils/loader.py: Data loading utilities
    - utils/resource.py: Resource management
    - configs/: YAML configuration examples
"""
# ============================================================================

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['savefig.bbox'] = 'tight'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import pytorch_lightning as pl
import torchvision.models as models
from torchvision.transforms import Resize

import numpy as np
import glob
from utils.richco import console
from typing import Optional, Any, Tuple

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

import torchmetrics


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-3,
            optimizer: str = 'adamw',
            scheduler: str = 'constant',
            warmup_steps: int = 0,
            weight_decay: float = 0.9,
            task_type: str = 'regression',
            num_classes: Optional[int] = None,
            loss_type: str = 'mse',
            enable_benchmarking: bool = False,
            **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.output_hidden = kwargs.get('output_hidden', False) 
        self.enable_benchmarking = enable_benchmarking

        if self.task_type == 'regression':
            self.train_metrics = torchmetrics.MetricCollection({
                'mae': torchmetrics.MeanAbsoluteError(),
                'mse': torchmetrics.MeanSquaredError(),
                'pearson': torchmetrics.PearsonCorrCoef(),
                'rmse': torchmetrics.MeanSquaredError(squared=False)
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')
        elif self.task_type == 'binary_classification':
            self.train_metrics = torchmetrics.MetricCollection({
                'accuracy': torchmetrics.Accuracy(),
                'f1_score': torchmetrics.F1Score(num_classes=self.num_classes, average='macro'),
                'auc': torchmetrics.AUROC(num_classes=self.num_classes, average='macro'),
                'precision': torchmetrics.Precision(num_classes=self.num_classes, average='macro'),
                'recall': torchmetrics.Recall(num_classes=self.num_classes, average='macro')
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')
        elif self.task_type == 'multiclass_classification':
            self.train_metrics = torchmetrics.MetricCollection({
                'accuracy': torchmetrics.Accuracy(num_classes=self.num_classes),
                'f1_score': torchmetrics.F1Score(num_classes=self.num_classes, average='macro'),
                'auc': torchmetrics.AUROC(num_classes=self.num_classes, average='macro'),
                'precision': torchmetrics.Precision(num_classes=self.num_classes, average='macro'),
                'recall': torchmetrics.Recall(num_classes=self.num_classes, average='macro')
            })
            self.val_metrics = self.train_metrics.clone(prefix='val_')
            self.test_metrics = self.train_metrics.clone(prefix='test_')
        
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(outputs, targets)
        elif self.loss_type == 'mae':
            return F.l1_loss(outputs, targets)
        elif self.loss_type == 'cross_entropy':
            return F.cross_entropy(outputs, targets)
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(outputs, targets.float())
        elif self.loss_type == 'hinge':
            return F.hinge_embedding_loss(outputs, targets.float())
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        predictions = self(inputs)

        if hasattr(self, 'output_hidden') and self.output_hidden:
            return predictions, targets, inputs

        return predictions, targets
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()
    
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")

    def configure_optimizers(self):
        if self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'radam':
            optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        if self.scheduler_name == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches)
        elif self.scheduler_name == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches)
        elif self.scheduler_name == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches)
        elif self.scheduler_name == 'lronplat':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, threshold=1e-3)
        elif self.scheduler_name == 'cosann':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
        elif self.scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_name == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.learning_rate * 10,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif self.scheduler_name == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.scheduler_name == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate / 10,
                max_lr=self.learning_rate * 5,
                step_size_up=2000,
                mode='triangular2'
            )
        elif self.scheduler_name == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

        if self.scheduler_name == 'lronplat':
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val_loss',
                    'frequency': 1
                }
            }
        elif self.scheduler_name != 'constant':
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return {
            'optimizer': optimizer,}
    
class DyAbModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 1280,
            embedding_img_size: int = 192,
            diff_channel0: str = "diff",
            diff_channel1: Optional[str] = None,
            diff_channel2: Optional[str] = None,
            output_hidden: bool = False,
            enable_batchnorm: bool = False,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            seq_len: int = 155,
            optimizer: str = "adamw",
            scheduler: str = "linear",
            learning_rate: float = 1e-6,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.embedding_img_size = embedding_img_size
        self.diff_channel0 = diff_channel0
        self.diff_channel1 = diff_channel1
        self.diff_channel2 = diff_channel2
        self.output_hidden = output_hidden
        self.enable_benchmarking = enable_benchmarking
        self.enable_batchnorm = enable_batchnorm
        self.experiment_name = experiment_name
        self.seq_len = seq_len
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        self.output_directory = None

        console.log(experiment_name)
        self.resize = Resize((self.embedding_img_size, self.embedding_img_size))
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

        console.log(self.learning_rate)
        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        if param_count == 0:
            print("ResNet parameters:", sum(p.numel() for p in self.resnet.parameters()))
            for name, module in self.named_modules():
                print(f"Module {name}: {sum(p.numel() for p in module.parameters())}")
            raise RuntimeError("Model has no parameters.")
        print("Model parameters initialised with", param_count)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs.squeeze(), targets.squeeze())
    
    def resize_embeddings(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor):
        img_size = int(self.embedding_img_size)
        B = embeddings1.shape[0]
        input_image = torch.zeros((B, 3, img_size, img_size), dtype=torch.float32, device=self.device)

        for channel, op in enumerate([self.diff_channel0, self.diff_channel1, self.diff_channel2]):
            if op is not None:
                if op == "diff":
                    embeddings = embeddings1 - embeddings2 
                elif op == "add":
                    embeddings = embeddings1 + embeddings2
                elif op == "mul":
                    embeddings = embeddings1 * embeddings2
                elif op == "div":
                    embeddings = embeddings1 / (embeddings2 + 1e-8)
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            else:
                embeddings = torch.zeros_like(embeddings1)

            pooled_embeddings = self.resize(embeddings)
            
            min_val = torch.amin(pooled_embeddings, dim=1, keepdim=True)
            max_val = torch.amax(pooled_embeddings, dim=1, keepdim=True)
            range_val = max_val - min_val
            pooled_embeddings = (pooled_embeddings - min_val) / (range_val + 1e-8)

            input_image[:, channel, :, :] = pooled_embeddings

        return input_image

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, return_img: bool = False) -> torch.Tensor:

        if len(embeddings1.shape) == 2:
            B, L = embeddings1.shape[0], int(np.sqrt(self.embedding_dim))
            H = self.embedding_dim // L
            
            if L * H != self.embedding_dim:
                L = int(np.ceil(np.sqrt(self.embedding_dim)))
                H = L

                padded_dim = L * H
                padding1 = torch.zeros(B, padded_dim - self.embedding_dim, device=self.device)
                padding2 = torch.zeros(B, padded_dim - self.embedding_dim, device=self.device)
                embeddings1 = torch.cat([embeddings1, padding1], dim=1)
                embeddings2 = torch.cat([embeddings2, padding2], dim=1)
            
            embeddings1 = embeddings1.view(B, L, H)
            embeddings2 = embeddings2.view(B, L, H)

        embedding_img = self.resize_embeddings(embeddings1, embeddings2)

        if return_img:
            return embedding_img

        if self.enable_batchnorm:
            embedding_img = self.batch_norm(embedding_img) # * Batch-Norm for gradient stabilisation
            
        predictions = self.resnet(embedding_img).squeeze().float()

        return predictions
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)


        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            
            input_embeddings = torch.cat([embeddings1.flatten(1), embeddings2.flatten(1)], dim=1)
            
            self._test_outputs.append({
                'predictions': outputs.detach(),
                'targets': targets.detach(),
                'loss': loss,
                'inputs': input_embeddings.detach()
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True)

        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            if isinstance(self._test_outputs[0], dict) and 'predictions' in self._test_outputs[0]:
                all_preds = torch.cat([x['predictions'] for x in self._test_outputs]).cpu().numpy()
                all_targets = torch.cat([x['targets'] for x in self._test_outputs]).cpu().numpy()
                all_inputs = torch.cat([x['inputs'] for x in self._test_outputs]).cpu().numpy()

                self.benchmark(all_preds, all_targets, all_inputs, test_metrics['test_pearson'].cpu().numpy())

                delattr(self, '_test_outputs')
        self.test_metrics.reset()

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> None:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[-1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 0

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs, pearson_r) -> None:
        
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        self.evaluation(preds, targets, pearson_r)
        self.snr(preds, targets, inputs)
      
    def evaluation(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        if self.seq_len == 138 or self.seq_len == 150:
            dataset_name = 'Specifica' 
            color = 'orange'
        elif self.seq_len == 149 or self.seq_len == 139:
            dataset_name = 'Ginkgo GDPa1'
            color = 'red'
        elif self.seq_len == 155 or self.seq_len == 153:
            dataset_name = 'NbThermo VHH'
            color = 'green'

        if r2 < 0.2:
            color = 'blue'
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(targets, predictions, alpha=0.6, color=color)
        ax.set_title(f'Predictions vs Targets :: {dataset_name}')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', alpha=0.8)
        
        # Add R² annotations
        ax.text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_predictions_vs_targets', 'png', self.output_directory), 
                    dpi=900, bbox_inches='tight')
        plt.close()
        
        return r2
        
    def evaluation_snr(self, predictions, targets) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].scatter(targets, predictions)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_xlabel('Targets')
        axes[0].set_ylabel('Predictions')
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        axes[0].legend(['Predictions', 'Targets'])
        axes[0].text(0.95, 0.05, f'R² = {r2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        residuals = predictions - targets
        axes[1].hist(residuals, bins=50)
        axes[1].set_title('Residuals Histogram')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(['Residuals'])

        axes[2].scatter(targets, residuals)
        axes[2].set_title('Residuals vs Targets')
        axes[2].set_xlabel('Targets')
        axes[2].set_ylabel('Residuals')
        axes[2].plot([targets.min(), targets.max()], [0, 0], 'k--')
        axes[2].legend(['Residuals', 'Zero Line'])

        errors = np.abs(predictions - targets)
        axes[3].hist(errors, bins=50)
        axes[3].set_title('Errors Histogram')
        axes[3].set_xlabel('Errors')
        axes[3].set_ylabel('Frequency')
        axes[3].legend(['Errors'])
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_benchmarking', 'png', self.output_directory), dpi=900, bbox_inches='tight')

    def snr(self, preds, targets, inputs) -> None:

        predictions = preds.flatten()
        targets = targets.flatten()

        if inputs.ndim > 1:
            inputs = inputs.reshape(inputs.shape[0], -1)
        else:
            inputs = inputs.reshape(1, -1)
            
        snr_magnitude = np.linalg.norm(inputs, axis=1)

        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=900, bbox_inches='tight')

class Transformers(BaseModel):
    pass

class VAEModel(BaseModel):
    pass

class CustomModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 640,
            seq_len: int = 247,
            hidden_dim: int = 512,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            optimizer: str = "adamw",
            scheduler: str = "linear",
            learning_rate: float = 1e-6,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.enable_benchmarking = enable_benchmarking
        console.log(enable_benchmarking)
        self.experiment_name = experiment_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        
        """ Custom model architecture """
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        console.log(f"Embedding Dim: {self.embedding_dim}, Sequence Length: {self.seq_len}, Hidden Dim: {self.hidden_dim}")

        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_dim) # * Adaptive Input Project to handle multiple experiment types across AbS Sequences
        
        num_layers = max(2, min(6, self.seq_len // 32)) # * Dynamic scaling to encode long sequences better
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, 
                nhead=8, 
                dim_feedforward=self.hidden_dim*2, # * Pre-Interaction, it is important to learn rich representations in the embedding space itself to augment the limited data scenarios
                dropout=0.1, 
                activation='gelu',
                norm_first=True,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.GELU(),                                      # * Non-Linearity for complex attention mapping, helps in learning non-linear relationships between residues
            nn.Linear(self.hidden_dim//4, 1),
            nn.Softmax(dim=1)
        )
        
        self.pos_encoding = self._create_positional_encoding()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim = self.hidden_dim,                # * Cross-Attention to learn interactions between two sequences, this would help in learning specific regions of interest
            num_heads = 8, 
            dropout = 0.1,
            batch_first = True
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),                                    # * Simple Feed-Forward Network for feature extraction for the attention head of the two sequences
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),                                # * Final Regressor to predict the thermostability change for the sequence pair
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//4, 1),
        )
        
        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        print("Model parameters initialised with", param_count)

    def _create_positional_encoding(self) -> nn.Parameter:
        position = torch.arange(self.seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           -(math.log(10000.0) / self.hidden_dim))
        
        pos_encoding = torch.zeros(1, self.seq_len, self.hidden_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        learnable_component = torch.randn(1, self.seq_len, self.hidden_dim) * 0.02

        combined_encoding = pos_encoding + learnable_component

        return nn.Parameter(combined_encoding)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(outputs.squeeze(), targets.squeeze(), reduction='mean')
    
    def multi_scale_pooling(self, embedding: torch.Tensor) -> torch.Tensor:
        return (
            0.7 * torch.sum(embedding * self.attention_pool(embedding), dim=1) +
            0.3 * torch.mean(embedding, dim=1)
        )
    
    def stability_metric(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8).unsqueeze(1)
        euclidean_dist = F.pairwise_distance(embedding1, embedding2, p=2, eps=1e-8, keepdim=True)

        l2_dist = torch.norm(embedding1 - embedding2, p=2, dim=1, keepdim=True)
        l1_dist = torch.norm(embedding1 - embedding2, p=1, dim=1, keepdim=True)
        
        dot = torch.sum(embedding1 * embedding2, dim=1, keepdim=True)
        norm_ratio = torch.norm(embedding1, p=2, dim=1, keepdim=True) / (torch.norm(embedding2, p=2, dim=1, keepdim=True) + 1e-8)

        return torch.cat([cosine_sim, euclidean_dist, l2_dist, l1_dist, dot, norm_ratio], dim=1)

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        
        emb1_proj = self.input_projection(embeddings1) + self.pos_encoding
        emb2_proj = self.input_projection(embeddings2) + self.pos_encoding
        
        enc1 = self.sequence_encoder(emb1_proj)
        enc2 = self.sequence_encoder(emb2_proj)
        
        attn_output1, _ = self.cross_attention(enc1, enc2, enc2)
        attn_output2, _ = self.cross_attention(enc2, enc1, enc1)
        
        enc1_seq = self.multi_scale_pooling(attn_output1)
        enc2_seq = self.multi_scale_pooling(attn_output2)
        
        diff1, diff2 = enc1_seq - enc2_seq, enc2_seq - enc1_seq
        
        input_tm1 = torch.cat([
            enc1_seq,
            diff1,
        ], dim=1)
        features1 = self.feature_extractor(input_tm1)
        
        input_tm2 = torch.cat([
            diff2,
            enc2_seq,
        ], dim=1)
        features2 = self.feature_extractor(input_tm2)
            
        stability1 = self.stability_metric(enc1_seq, enc2_seq)
        stability2 = self.stability_metric(enc2_seq, enc1_seq)
        input_reg1 = torch.cat([features1, stability1], dim=1)
        input_reg2 = torch.cat([features2, stability2], dim=1)
        tm1 = self.regressor(input_reg1).squeeze(-1).float()
        tm2 = self.regressor(input_reg2).squeeze(-1).float()

        return tm1 - tm2
        
        # diff = enc1_seq - enc2_seq  # Only one difference
        # # prod = enc1_seq * enc2_seq
        # stability = self.stability_metric(enc1_seq, enc2_seq)
        
        # # Concatenate all interaction features
        # combined = torch.cat([
        #     enc1_seq,
        #     enc2_seq,
        #     diff,
        # ], dim=1)
        
        # features = self.feature_extractor(combined)
        # features = torch.cat([features, stability], dim=1)
        # delta_tm = self.regressor(features).squeeze(-1).float()
        
        # return delta_tm

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)


        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            input_embeddings = torch.cat([embeddings1.flatten(1), embeddings2.flatten(1)], dim=1)
        
            self._test_outputs.append({
                'predictions': outputs.detach(),
                'targets': targets.detach(),
                'inputs': input_embeddings.detach().cpu(),
                'loss': loss
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True)
        console.log(test_metrics)
        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            if isinstance(self._test_outputs[0], dict) and 'predictions' in self._test_outputs[0]:
                
                all_preds = torch.cat([x['predictions'] for x in self._test_outputs]).cpu().numpy()
                all_targets = torch.cat([x['targets'] for x in self._test_outputs]).cpu().numpy()
                # all_inputs = torch.cat([x['inputs'] for x in self._test_outputs]).cpu().numpy()

                # self.benchmark(all_preds, all_targets, all_inputs)
                self.benchmark(all_preds, all_targets, None, test_metrics['test_pearson'].cpu().numpy())
                

                delattr(self, '_test_outputs')
        self.test_metrics.reset()

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> None:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[-1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 0

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs = None, pearson_r = None):
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        self.evaluation(preds, targets, pearson_r)
    
    def evaluation(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')
        console.log(f'  {self.seq_len}')
        
        if self.seq_len == 138 or self.seq_len == 150:
            dataset_name = 'Specifica' 
            color = 'orange'
        elif self.seq_len == 149 or self.seq_len == 139:
            dataset_name = 'Ginkgo GDPa1'
            color = 'red'
        elif self.seq_len == 155 or self.seq_len == 153:
            dataset_name = 'NbThermo VHH'
            color = 'green'

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(targets, predictions, alpha=0.6, color=color)
        ax.set_title(f'Predictions vs Targets :: {dataset_name}')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', alpha=0.8)
        
        # Add R² annotations
        ax.text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_predictions_vs_targets', 'png', self.output_directory), 
                    dpi=900, bbox_inches='tight')
        plt.close()
        
        return r2
        
    def evaluation_snr(self, predictions, targets) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].scatter(targets, predictions)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_xlabel('Targets')
        axes[0].set_ylabel('Predictions')
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        axes[0].legend(['Predictions', 'Targets'])
        axes[0].text(0.95, 0.05, f'R² = {r2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        residuals = predictions - targets
        axes[1].hist(residuals, bins=50)
        axes[1].set_title('Residuals Histogram')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(['Residuals'])

        axes[2].scatter(targets, residuals)
        axes[2].set_title('Residuals vs Targets')
        axes[2].set_xlabel('Targets')
        axes[2].set_ylabel('Residuals')
        axes[2].plot([targets.min(), targets.max()], [0, 0], 'k--')
        axes[2].legend(['Residuals', 'Zero Line'])

        errors = np.abs(predictions - targets)
        axes[3].hist(errors, bins=50)
        axes[3].set_title('Errors Histogram')
        axes[3].set_xlabel('Errors')
        axes[3].set_ylabel('Frequency')
        axes[3].legend(['Errors'])
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_benchmarking', 'png', self.output_directory), dpi=900, bbox_inches='tight')

    def snr(self, preds, targets,) -> None:

        predictions = preds.flatten()
        targets = targets.flatten()
        
        # min_length = min(len(predictions), len(targets), len(inputs))
        min_length = min(len(predictions), len(targets))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        # inputs = inputs[:min_length]    
        
        # console.log(f'Inputs shape for SNR analysis: {inputs.shape}')
        console.log(f'Predictions shape for SNR analysis: {predictions.shape}')
        console.log(f'Targets shape for SNR analysis: {targets.shape}')

        # Ensure snr_magnitude has the same length as predictions/targets
        if len(snr_magnitude) != min_length:
            snr_magnitude = snr_magnitude[:min_length]

        console.log(f'SNR magnitude shape: {snr_magnitude.shape}')
        
        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=900, bbox_inches='tight')

class CustomReArchModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 640,
            seq_len: int = 247,
            hidden_dim: int = 512,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            optimizer: str = "adamw",
            scheduler: str = "linear",
            learning_rate: float = 1e-6,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.enable_benchmarking = enable_benchmarking
        console.log(enable_benchmarking)
        self.experiment_name = experiment_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        
        """ Custom model architecture """
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        console.log(f"Embedding Dim: {self.embedding_dim}, Sequence Length: {self.seq_len}, Hidden Dim: {self.hidden_dim}")

        self.input_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim * 8),
            nn.LayerNorm(self.hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        ) # * Adaptive Input Project to handle multiple experiment types across AbS Sequences 
        
        num_layers = max(2, min(6, self.seq_len // 32)) # * Dynamic scaling to encode long sequences better
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, 
                nhead=8, 
                dim_feedforward=self.hidden_dim*2, # * Pre-Interaction, it is important to learn rich representations in the embedding space itself to augment the limited data scenarios
                dropout=0.1, 
                activation='gelu',
                norm_first=True,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.GELU(),                                      # * Non-Linearity for complex attention mapping, helps in learning non-linear relationships between residues
            nn.Linear(self.hidden_dim//4, 1),
            nn.Softmax(dim=1)
        )
        
        self.pos_encoding = self._create_positional_encoding()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim = self.hidden_dim,                # * Cross-Attention to learn interactions between two sequences, this would help in learning specific regions of interest
            num_heads = 8, 
            dropout = 0.1,
            batch_first = True
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),                                    # * Simple Feed-Forward Network for feature extraction for the attention head of the two sequences
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Tanh(),                               # * Final Regressor to predict the thermostability change for the sequence pair
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//4, 1),
            nn.ReLU()
        )
        
        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        print("Model parameters initialised with", param_count)

    def _create_positional_encoding(self) -> nn.Parameter:
        position = torch.arange(self.seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           -(math.log(10000.0) / self.hidden_dim))
        
        pos_encoding = torch.zeros(1, self.seq_len, self.hidden_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        learnable_component = torch.randn(1, self.seq_len, self.hidden_dim) * 0.02

        combined_encoding = pos_encoding + learnable_component

        return nn.Parameter(combined_encoding)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(outputs.squeeze(), targets.squeeze(), reduction='mean')
    
    def multi_scale_pooling(self, embedding: torch.Tensor) -> torch.Tensor:
        return (
            0.7 * torch.sum(embedding * self.attention_pool(embedding), dim=1) +
            0.3 * torch.mean(embedding, dim=1)
        )
    
    def stability_metric(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8).unsqueeze(1)
        euclidean_dist = F.pairwise_distance(embedding1, embedding2, p=2, eps=1e-8, keepdim=True)

        l2_dist = torch.norm(embedding1 - embedding2, p=2, dim=1, keepdim=True)
        l1_dist = torch.norm(embedding1 - embedding2, p=1, dim=1, keepdim=True)
        
        dot = torch.sum(embedding1 * embedding2, dim=1, keepdim=True)
        norm_ratio = torch.norm(embedding1, p=2, dim=1, keepdim=True) / (torch.norm(embedding2, p=2, dim=1, keepdim=True) + 1e-8)

        return torch.cat([cosine_sim, euclidean_dist, l2_dist, l1_dist, dot, norm_ratio], dim=1)

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        
        emb1_proj = self.input_projection(embeddings1) + self.pos_encoding
        emb2_proj = self.input_projection(embeddings2) + self.pos_encoding
        
        enc1 = self.sequence_encoder(emb1_proj)
        enc2 = self.sequence_encoder(emb2_proj)
        
        attn_output1, _ = self.cross_attention(enc1, enc2, enc2)
        attn_output2, _ = self.cross_attention(enc2, enc1, enc1)
        
        enc1_seq = self.multi_scale_pooling(attn_output1)
        enc2_seq = self.multi_scale_pooling(attn_output2)
        
        features1 = self.feature_extractor(enc1_seq)
        features2 = self.feature_extractor(enc2_seq)

        tm1 = self.regressor(features1).squeeze(-1).float()
        tm2 = self.regressor(features2).squeeze(-1).float()

        return tm1 - tm2


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)


        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            input_embeddings = torch.cat([embeddings1.flatten(1), embeddings2.flatten(1)], dim=1)
        
            self._test_outputs.append({
                'predictions': outputs.detach(),
                'targets': targets.detach(),
                'inputs': input_embeddings.detach().cpu(),
                'loss': loss
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True)
        console.log(test_metrics)
        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            if isinstance(self._test_outputs[0], dict) and 'predictions' in self._test_outputs[0]:
                
                all_preds = torch.cat([x['predictions'] for x in self._test_outputs]).cpu().numpy()
                all_targets = torch.cat([x['targets'] for x in self._test_outputs]).cpu().numpy()
                # all_inputs = torch.cat([x['inputs'] for x in self._test_outputs]).cpu().numpy()

                # self.benchmark(all_preds, all_targets, all_inputs)
                self.benchmark(all_preds, all_targets, None, test_metrics['test_pearson'].cpu().numpy())
                

                delattr(self, '_test_outputs')
        console.log(f" Value Range of Test Predictors: {all_preds.min()} to {all_preds.max()}")
        self.test_metrics.reset()

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> None:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[-1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 0

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs = None, pearson_r = None):
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        self.evaluation(preds, targets, pearson_r)
    
    def evaluation(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')
        console.log(f'  {self.seq_len}')
        
        if self.seq_len == 138 or self.seq_len == 150:
            dataset_name = 'Specifica' 
            color = 'orange'
        elif self.seq_len == 149 or self.seq_len == 139:
            dataset_name = 'Ginkgo GDPa1'
            color = 'red'
        elif self.seq_len == 155 or self.seq_len == 153:
            dataset_name = 'NbThermo VHH'
            color = 'green'

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(targets, predictions, alpha=0.6, color=color)
        ax.set_title(f'Predictions vs Targets :: {dataset_name}')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', alpha=0.8)
        
        # Add R² annotations
        ax.text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_predictions_vs_targets', 'png', self.output_directory), 
                    dpi=900, bbox_inches='tight')
        plt.close()
        
        return r2

    def snr(self, preds, targets,) -> None:

        predictions = preds.flatten()
        targets = targets.flatten()
        
        min_length = min(len(predictions), len(targets))
        predictions = predictions[:min_length]
        targets = targets[:min_length]

        console.log(f'Predictions shape for SNR analysis: {predictions.shape}')
        console.log(f'Targets shape for SNR analysis: {targets.shape}')

        if len(snr_magnitude) != min_length:
            snr_magnitude = snr_magnitude[:min_length]

        console.log(f'SNR magnitude shape: {snr_magnitude.shape}')
        
        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=900, bbox_inches='tight')

class CustomPooledModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 1280,
            seq_len: int = 247,
            hidden_dim: int = 512,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            optimizer: str = "adamw",
            scheduler: str = "linear",
            learning_rate: float = 1e-6,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.enable_benchmarking = enable_benchmarking
        console.log(enable_benchmarking)
        self.experiment_name = experiment_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        
        """ Custom model architecture """
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_dim) # * Adaptive Input Project to handle multiple experiment types across AbS Sequences
        
        self.embedding_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),                                      # * Embedding Encoder to refine the input embeddings before any interaction
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),                                    # * Simple Feed-Forward Network for feature extraction for the attention head of the two sequences
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),                                # * Final Regressor to predict the thermostability change for the sequence pair
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//4, 1),
        )
        
        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        print("Model parameters initialised with", param_count)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(outputs.squeeze(), targets.squeeze(), reduction='mean')
    
    def stability_metric(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8).unsqueeze(1)
        euclidean_dist = F.pairwise_distance(embedding1, embedding2, p=2, eps=1e-8, keepdim=True)

        l2_dist = torch.norm(embedding1 - embedding2, p=2, dim=1, keepdim=True)
        l1_dist = torch.norm(embedding1 - embedding2, p=1, dim=1, keepdim=True)
        
        dot = torch.sum(embedding1 * embedding2, dim=1, keepdim=True)
        norm_ratio = torch.norm(embedding1, p=2, dim=1, keepdim=True) / (torch.norm(embedding2, p=2, dim=1, keepdim=True) + 1e-8)

        return torch.cat([cosine_sim, euclidean_dist, l2_dist, l1_dist, dot, norm_ratio], dim=1)

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        
        emb1_proj = self.input_projection(embeddings1)
        emb2_proj = self.input_projection(embeddings2)
        
        enc1_seq = self.embedding_encoder(emb1_proj).squeeze(1)
        enc2_seq = self.embedding_encoder(emb2_proj).squeeze(1)
        
        diff1, diff2 = enc1_seq - enc2_seq, enc2_seq - enc1_seq
        prod = enc1_seq * enc2_seq
        
        input_tm1 = torch.cat([
            enc1_seq,
            diff1,
        ], dim=1)
        features1 = self.feature_extractor(input_tm1)
        
        input_tm2 = torch.cat([
            enc2_seq,
            diff2,
        ], dim=1)
        features2 = self.feature_extractor(input_tm2)
            
        # stability = self.stability_metric(enc1_seq, enc2_seq)
        # input_reg1 = torch.cat([features1, stability], dim=1)
        # input_reg2 = torch.cat([features2, stability], dim=1)
        tm1 = self.regressor(features1).squeeze(-1).float()
        tm2 = self.regressor(features2).squeeze(-1).float()

        return tm1 - tm2

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)


        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            input_embeddings = torch.cat([embeddings1.flatten(1), embeddings2.flatten(1)], dim=1)
        
            self._test_outputs.append({
                'predictions': outputs.detach(),
                'targets': targets.detach(),
                'inputs': input_embeddings.detach().cpu(),
                'loss': loss
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True)

        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            if isinstance(self._test_outputs[0], dict) and 'predictions' in self._test_outputs[0]:
                all_preds = torch.cat([x['predictions'] for x in self._test_outputs]).cpu().numpy()
                all_targets = torch.cat([x['targets'] for x in self._test_outputs]).cpu().numpy()
                all_inputs = torch.cat([x['inputs'] for x in self._test_outputs]).cpu().numpy()

                self.benchmark(all_preds, all_targets, all_inputs, test_metrics['test_pearson'].cpu().numpy())

                delattr(self, '_test_outputs')

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> None:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[-1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 0

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs = None, pearson_r = None):
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        self.evaluation(preds, targets, pearson_r)
    
    def evaluation(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')
        console.log(f'  {self.seq_len}')
        
        if self.seq_len == 138 or self.seq_len == 150:
            dataset_name = 'Specifica' 
            color = 'orange'
        elif self.seq_len == 149 or self.seq_len == 139:
            dataset_name = 'Ginkgo GDPa1'
            color = 'red'
        elif self.seq_len == 155 or self.seq_len == 153:
            dataset_name = 'NbThermo VHH'
            color = 'green'

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(targets, predictions, alpha=0.6, color=color)
        ax.set_title(f'Predictions vs Targets :: {dataset_name}')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', alpha=0.8)
        
        # Add R² annotations
        ax.text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_predictions_vs_targets', 'png', self.output_directory), 
                    dpi=900, bbox_inches='tight')
        plt.close()
        
        return r2  
      
    def evaluation_snr(self, predictions, targets) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].scatter(targets, predictions)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_xlabel('Targets')
        axes[0].set_ylabel('Predictions')
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        axes[0].legend(['Predictions', 'Targets'])
        axes[0].text(0.95, 0.05, f'R² = {r2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        residuals = predictions - targets
        axes[1].hist(residuals, bins=50)
        axes[1].set_title('Residuals Histogram')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(['Residuals'])

        axes[2].scatter(targets, residuals)
        axes[2].set_title('Residuals vs Targets')
        axes[2].set_xlabel('Targets')
        axes[2].set_ylabel('Residuals')
        axes[2].plot([targets.min(), targets.max()], [0, 0], 'k--')
        axes[2].legend(['Residuals', 'Zero Line'])

        errors = np.abs(predictions - targets)
        axes[3].hist(errors, bins=50)
        axes[3].set_title('Errors Histogram')
        axes[3].set_xlabel('Errors')
        axes[3].set_ylabel('Frequency')
        axes[3].legend(['Errors'])
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_benchmarking', 'png', self.output_directory), dpi=900, bbox_inches='tight')

    def snr(self, preds, targets, inputs) -> None:

        predictions = preds.flatten()
        targets = targets.flatten()
        
        min_length = min(len(predictions), len(targets), len(inputs))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        inputs = inputs[:min_length]    
        
        console.log(f'Inputs shape for SNR analysis: {inputs.shape}')
        console.log(f'Predictions shape for SNR analysis: {predictions.shape}')
        console.log(f'Targets shape for SNR analysis: {targets.shape}')

        min_length = min(len(predictions), len(targets), len(inputs))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        inputs = inputs[:min_length]

        # For 1D inputs, treat each element as a separate sample
        if inputs.ndim == 1:
            snr_magnitude = np.abs(inputs)  # Use absolute value as magnitude for 1D
        else:
            # For multi-dimensional inputs, compute L2 norm across features
            if inputs.ndim > 2:
                inputs = inputs.reshape(inputs.shape[0], -1)
            snr_magnitude = np.linalg.norm(inputs, axis=1)

        # Ensure snr_magnitude has the same length as predictions/targets
        if len(snr_magnitude) != min_length:
            snr_magnitude = snr_magnitude[:min_length]

        console.log(f'SNR magnitude shape: {snr_magnitude.shape}')
        
        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=900, bbox_inches='tight')

class CNN1DModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 1280,
            seq_len: int = 247,
            hidden_dim: int = 512,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            optimizer: str = "adamw",
            scheduler: str = "linear",
            learning_rate: float = 1e-6,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.enable_benchmarking = enable_benchmarking
        console.log(enable_benchmarking)
        self.experiment_name = experiment_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        
        """ Custom model architecture """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        console.log(f"Embedding Dim: {self.embedding_dim}, Hidden Dim: {self.hidden_dim}")

        self.conv1 = nn.Conv1d(self.embedding_dim, self.hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=5, padding=1)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.deltaregr = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        print("Model parameters initialised with", param_count)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs.squeeze(), targets.squeeze(), reduction='mean')
    
    def _process_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        
        input = embedding.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(input)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.global_pool(x).squeeze(-1)

        return x

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        
        embedding1 = self._process_embedding(embeddings1)
        embedding2 = self._process_embedding(embeddings2)

        tm_diff = self.deltaregr(torch.cat([embedding1, embedding2], dim=1)).squeeze(-1).float()

        return tm_diff
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)


        if self.task_type == 'regression':
            self.train_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.train_metrics.update(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.val_metrics.update(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            embeddings1, embeddings2, y1, y2 = batch
            targets = (y1 - y2).float()
        elif len(batch) == 3:
            embeddings1, embeddings2, targets = batch
            targets = targets.float()
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        
        inputs = self.forward(embeddings1, embeddings2)
        outputs = self.forward(embeddings1, embeddings2)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics.update(outputs.squeeze(), targets.squeeze())
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            if self.task_type == 'binary_classification':
                predictions = torch.sigmoid(outputs).squeeze()
            else:
                predictions = torch.softmax(outputs, dim=1)
            self.test_metrics.update(predictions, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            self._test_outputs.append({
                'predictions': outputs.detach(),
                'targets': targets.detach(),
                'inputs': inputs.detach(),
                'loss': loss
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            if isinstance(self._test_outputs[0], dict) and 'predictions' in self._test_outputs[0]:
                all_preds = torch.cat([x['predictions'] for x in self._test_outputs]).cpu().numpy()
                all_targets = torch.cat([x['targets'] for x in self._test_outputs]).cpu().numpy()
                all_inputs = torch.cat([x['inputs'] for x in self._test_outputs]).cpu().numpy()

                self.benchmark(all_preds, all_targets, all_inputs)

                delattr(self, '_test_outputs')

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> None:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[-1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 0

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs):
        
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        self.evaluation(preds, targets)
        self.snr(preds, targets, inputs)
        
    def evaluation(self, predictions, targets) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].scatter(targets, predictions)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_xlabel('Targets')
        axes[0].set_ylabel('Predictions')
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        axes[0].legend(['Predictions', 'Targets'])
        axes[0].text(0.95, 0.05, f'R² = {r2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        residuals = predictions - targets
        axes[1].hist(residuals, bins=50)
        axes[1].set_title('Residuals Histogram')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(['Residuals'])

        axes[2].scatter(targets, residuals)
        axes[2].set_title('Residuals vs Targets')
        axes[2].set_xlabel('Targets')
        axes[2].set_ylabel('Residuals')
        axes[2].plot([targets.min(), targets.max()], [0, 0], 'k--')
        axes[2].legend(['Residuals', 'Zero Line'])

        errors = np.abs(predictions - targets)
        axes[3].hist(errors, bins=50)
        axes[3].set_title('Errors Histogram')
        axes[3].set_xlabel('Errors')
        axes[3].set_ylabel('Frequency')
        axes[3].legend(['Errors'])
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_benchmarking', 'png', self.output_directory), dpi=900, bbox_inches='tight')

    def snr(self, preds, targets, inputs) -> None:

        predictions = preds.flatten()
        targets = targets.flatten()
        
        min_length = min(len(predictions), len(targets), len(inputs))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        inputs = inputs[:min_length]    
        
        console.log(f'Inputs shape for SNR analysis: {inputs.shape}')
        console.log(f'Predictions shape for SNR analysis: {predictions.shape}')
        console.log(f'Targets shape for SNR analysis: {targets.shape}')

        min_length = min(len(predictions), len(targets), len(inputs))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        inputs = inputs[:min_length]

        # For 1D inputs, treat each element as a separate sample
        if inputs.ndim == 1:
            snr_magnitude = np.abs(inputs)  # Use absolute value as magnitude for 1D
        else:
            # For multi-dimensional inputs, compute L2 norm across features
            if inputs.ndim > 2:
                inputs = inputs.reshape(inputs.shape[0], -1)
            snr_magnitude = np.linalg.norm(inputs, axis=1)

        # Ensure snr_magnitude has the same length as predictions/targets
        if len(snr_magnitude) != min_length:
            snr_magnitude = snr_magnitude[:min_length]

        console.log(f'SNR magnitude shape: {snr_magnitude.shape}')
        
        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=900, bbox_inches='tight')

class CNN2DModel(BaseModel):
    pass

class TLearningModel(BaseModel):
    pass

class LinearPredictorModel(BaseModel):
    def __init__(
            self,
            embedding_dim: int = 1280,
            hidden_dim: int = 512,
            num_layers: int = 4,
            seq_len: int = 247,
            enable_benchmarking: bool = False,
            experiment_name: str = "default",
            optimizer: str = "adamw",
            scheduler: str = "constant",
            learning_rate: float = 1e-4,
            dropout_rate: float = 0.2,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.enable_benchmarking = enable_benchmarking
        self.experiment_name = experiment_name
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate

        console.log(f"Linear Predictor - Embedding Dim: {self.embedding_dim}, Hidden Dim: {self.hidden_dim}, Layers: {self.num_layers}")

        layers = []

        layers.append(nn.Linear(self.embedding_dim, 128))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(128, 256))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 1))

        self.predictor = nn.Sequential(*layers)
        
        self._verify_parameters()

    def _verify_parameters(self):
        param_count = sum(p.numel() for p in self.parameters())
        console.log(f"Linear Predictor Model parameters initialized with {param_count}")

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(outputs.squeeze(), targets.squeeze())
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        predictions = self.predictor(embeddings).squeeze(-1).float()
        return predictions

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        embeddings, targets = batch
        outputs = self.forward(embeddings)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.train_metrics(outputs, targets)
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            preds = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else torch.softmax(outputs, dim=1)
            self.train_metrics(preds, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        embeddings, targets = batch
        outputs = self.forward(embeddings)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.val_metrics(outputs, targets)
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            preds = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else torch.softmax(outputs, dim=1)
            self.val_metrics(preds, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        embeddings, targets = batch
        outputs = self.forward(embeddings)
        loss = self.compute_loss(outputs, targets)

        if self.task_type == 'regression':
            self.test_metrics(outputs, targets)
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            preds = torch.sigmoid(outputs) if self.task_type == 'binary_classification' else torch.softmax(outputs, dim=1)
            self.test_metrics(preds, targets)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression':
            if not hasattr(self, '_test_outputs'):
                self._test_outputs = []
            self._test_outputs.append({
                'predictions': outputs.detach().cpu(),
                'targets': targets.detach().cpu(),
                'inputs': embeddings.detach().cpu()
            })

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True)
        
        if self.enable_benchmarking and self.task_type == 'regression' and hasattr(self, '_test_outputs') and self._test_outputs:
            all_preds = torch.cat([batch['predictions'] for batch in self._test_outputs])
            all_targets = torch.cat([batch['targets'] for batch in self._test_outputs])
            all_inputs = torch.cat([batch['inputs'] for batch in self._test_outputs])

            self.benchmark(all_preds.numpy(), all_targets.numpy(), all_inputs.numpy(), test_metrics['test_pearson'].cpu().numpy())
            self._test_outputs = []
        self.test_metrics.reset()

    def _version_file(self, base_name: str, file_ext: str, output_dir: str) -> str:
        versioned_files = glob.glob(f"{output_dir}/{base_name}_version_*.{file_ext}")
        if versioned_files:
            versions = [int(f.split('_version_')[1].split('.')[0]) for f in versioned_files]
            new_version = max(versions) + 1
        else:
            new_version = 1

        return f"{output_dir}/{base_name}_version_{new_version}.{file_ext}"

    def benchmark(self, preds, targets, inputs, pearson_r):
        self.output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", self.experiment_name, "graphs")
        os.makedirs(self.output_directory, exist_ok=True)
        r2 = self.evaluation(preds, targets, pearson_r)
        self.snr(preds, targets, inputs, r2)
    
    def evaluation(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        if self.seq_len == 138 or self.seq_len == 150:
            dataset_name = 'Specifica' 
            color = 'orange'
        elif self.seq_len == 149 or self.seq_len == 139:
            dataset_name = 'Ginkgo GDPa1'
            color = 'red'
        elif self.seq_len == 155 or self.seq_len == 153:
            dataset_name = 'NbThermo VHH'
            color = 'green'

        if r2 < 0.2:
            color = 'blue'
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(targets, predictions, alpha=0.6, color=color)
        ax.set_title(f'Predictions vs Targets :: {dataset_name}')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', alpha=0.8)
        
        # Add R² annotations
        ax.text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_predictions_vs_targets', 'png', self.output_directory), 
                    dpi=900, bbox_inches='tight')
        plt.close()
        
        return r2

    def evaluation_snr(self, predictions, targets, pearson_r) -> None:
        predictions = predictions.flatten()
        targets = targets.flatten()

        rmse = root_mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        console.log('Test Evaluation Metrics:')
        console.log(f'  RMSE: {rmse}')
        console.log(f'  R2: {r2}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].scatter(targets, predictions)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_xlabel('Targets')
        axes[0].set_ylabel('Predictions')
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        axes[0].legend(['Predictions', 'Targets'])
        axes[0].text(0.95, 0.15, f'Corr R² = {r2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0].text(0.95, 0.05, f'Pearson r² = {(pearson_r)**2:.4f}', 
             transform=axes[0].transAxes,
             fontsize=12,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

        residuals = predictions - targets
        axes[1].hist(residuals, bins=50)
        axes[1].set_title('Residuals Histogram')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(['Residuals'])

        axes[2].scatter(targets, residuals)
        axes[2].set_title('Residuals vs Targets')
        axes[2].set_xlabel('Targets')
        axes[2].set_ylabel('Residuals')
        axes[2].plot([targets.min(), targets.max()], [0, 0], 'k--')
        axes[2].legend(['Residuals', 'Zero Line'])

        errors = np.abs(predictions - targets)
        axes[3].hist(errors, bins=50)
        axes[3].set_title('Errors Histogram')
        axes[3].set_xlabel('Errors')
        axes[3].set_ylabel('Frequency')
        axes[3].legend(['Errors'])
        
        plt.tight_layout()
        plt.savefig(self._version_file('evaluation_benchmarking', 'png', self.output_directory), dpi=900, bbox_inches='tight')
        
        return r2

    def snr(self, preds, targets, inputs, r2) -> None:
        predictions = preds.flatten()
        targets = targets.flatten()
        
        min_length = min(len(predictions), len(targets), len(inputs))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        inputs = inputs[:min_length]    
        
        console.log(f'Inputs shape for SNR analysis: {inputs.shape}')
        console.log(f'Predictions shape for SNR analysis: {predictions.shape}')
        console.log(f'Targets shape for SNR analysis: {targets.shape}')

        # Calculate SNR magnitude from input embeddings
        if inputs.ndim == 1:
            snr_magnitude = np.abs(inputs)
        else:
            # For multi-dimensional inputs, use L2 norm across features
            snr_magnitude = np.linalg.norm(inputs, axis=1) if inputs.ndim == 2 else np.linalg.norm(inputs.reshape(len(inputs), -1), axis=1)

        # Ensure snr_magnitude has the same length as predictions/targets
        if len(snr_magnitude) != min_length:
            snr_magnitude = snr_magnitude[:min_length]

        console.log(f'SNR magnitude shape: {snr_magnitude.shape}')
        
        low_snr = snr_magnitude <= np.percentile(snr_magnitude, 25)
        med_snr = (snr_magnitude > np.percentile(snr_magnitude, 25)) & (snr_magnitude <= np.percentile(snr_magnitude, 75))
        high_snr = snr_magnitude > np.percentile(snr_magnitude, 75)

        mae_low = mean_absolute_error(targets[low_snr], predictions[low_snr])        
        mae_med = mean_absolute_error(targets[med_snr], predictions[med_snr])
        mae_high = mean_absolute_error(targets[high_snr], predictions[high_snr])

        mse_low = mean_squared_error(targets[low_snr], predictions[low_snr])
        mse_med = mean_squared_error(targets[med_snr], predictions[med_snr])
        mse_high = mean_squared_error(targets[high_snr], predictions[high_snr])

        console.log(f'MAE (Low SNR): {mae_low}')
        console.log(f'MAE (Med SNR): {mae_med}')
        console.log(f'MAE (High SNR): {mae_high}')
        console.log(f'MSE (Low SNR): {mse_low}')
        console.log(f'MSE (Med SNR): {mse_med}')
        console.log(f'MSE (High SNR): {mse_high}')
        
        correlation = np.corrcoef(snr_magnitude, np.abs(predictions - targets))[0,1]
        console.log(f'Correlation (SNR vs Errors): {correlation}')
        
        plt.figure(figsize=(15,5))

        plt.subplot(1, 3, 1)
        plt.scatter(snr_magnitude, np.abs(predictions - targets))
        plt.title(f'SNR vs Errors (r: {correlation:.2f})')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Absolute Error')

        plt.subplot(1, 3, 2)
        snr_bins = ['Low-SNR', 'Medium-SNR', 'High-SNR']
        mae_val = [mae_low, mae_med, mae_high]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(snr_bins, mae_val, color=colors)
        plt.title('MAE by SNR')
        plt.ylabel('Mean Absolute Error')

        for bar, val in zip(bars, mae_val):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.3f}', ha='center', va='bottom')

        plt.subplot(1, 3, 3)
        plt.hist(snr_magnitude, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(snr_magnitude, 25), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(np.percentile(snr_magnitude, 75), color='green', linestyle='dashed', linewidth=1)
        plt.title('SNR Distribution')
        plt.xlabel('SNR (Embedding Magnitude)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self._version_file('snr_analysis', 'png', self.output_directory), dpi=1200, bbox_inches='tight')