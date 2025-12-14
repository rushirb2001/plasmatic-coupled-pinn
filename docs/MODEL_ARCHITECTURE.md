# PINN Model Architecture

This document explains the refactored PINN model architecture following the patterns from the original `model.py`.

## Architecture Overview

```
src/model.py
├── BaseModel (abstract PINN base class)
│   ├── __init__() - Common hyperparameters
│   ├── forward() - Abstract method (implemented by subclasses)
│   ├── compute_pde_residuals() - Shared physics computation
│   ├── compute_boundary_loss() - Shared BC computation
│   ├── training_step() - Complete training logic
│   ├── validation_step() - Complete validation logic
│   ├── test_step() - Complete test logic
│   └── configure_optimizers() - Optimizer/scheduler setup
│
├── CCPPinn (default PINN model)
│   └── forward() - MLP with optional Fourier features
│
└── SimplePinn (baseline model)
    └── forward() - Simple MLP without features
```

## Key Design Principles

### 1. BaseModel Contains All Common Logic

Unlike the previous version, `BaseModel` now has **complete implementations** of all methods:
- ✅ `training_step()` - Fully implemented
- ✅ `validation_step()` - Fully implemented  
- ✅ `test_step()` - Fully implemented
- ✅ `compute_pde_residuals()` - Shared physics equations
- ✅ `compute_boundary_loss()` - Shared boundary conditions

**No `NotImplementedError` placeholders!**

### 2. Subclasses Only Implement `forward()`

Each PINN model (e.g., `CCPPinn`, `SimplePinn`) only needs to implement:
- `forward(x, t) -> (n_e, phi)` - The neural network architecture

All training logic, PDE residuals, and boundary conditions are handled by `BaseModel`.

### 3. Default Hyperparameters with YAML Override

Each model has sensible defaults that can be overridden via YAML config:

```python
class CCPPinn(BaseModel):
    def __init__(
        self,
        hidden_dims: List[int] = None,  # Default: [64, 64, 64]
        use_fourier: bool = True,
        fourier_scale: float = 10.0,
        num_fourier_features: int = 32,
        **kwargs
    ):
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        # ...
```

## How to Create a New Model

### Step 1: Inherit from BaseModel

```python
class MyCustomPinn(BaseModel):
    def __init__(
        self,
        # Your architecture hyperparameters with defaults
        num_layers: int = 5,
        hidden_size: int = 128,
        dropout: float = 0.1,
        **kwargs  # Always include **kwargs for BaseModel params
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        # Build your architecture
        self.layers = nn.ModuleList([...])
        
    def forward(self, x, t):
        """
        Must return physical quantities (n_e, phi).
        Inputs x, t are in physical units.
        """
        # Your custom architecture
        inputs = torch.cat([x, t], dim=1)
        # ... neural network forward pass ...
        return n_e, phi  # Physical units
```

### Step 2: Create a YAML Config

```yaml
# configs/my_custom_pinn.yaml
model:
  class_path: src.model.MyCustomPinn
  init_args:
    # Architecture params
    num_layers: 5
    hidden_size: 128
    dropout: 0.1
    
    # Training params (from BaseModel)
    learning_rate: 1.0e-3
    optimizer: adamw
    loss_weights:
      continuity: 1.0
      poisson: 1.0
      bc: 10.0
    experiment_name: my_custom_pinn

data:
  batch_size: 10000
  num_collocation_points: 20000
  time_duration: 4.0e-7
```

### Step 3: Train

```bash
python -m src.trainer fit --config configs/my_custom_pinn.yaml
```

## Available Models

### CCPPinn (Default)

**Architecture:** MLP with optional Fourier feature mapping

**Key Parameters:**
- `hidden_dims: List[int]` - Layer dimensions (default: `[64, 64, 64]`)
- `use_fourier: bool` - Enable Fourier features (default: `True`)
- `fourier_scale: float` - Fourier frequency scale (default: `10.0`)
- `num_fourier_features: int` - Number of Fourier features (default: `32`)
- `log_space_ne: bool` - Predict log(n_e) (default: `True`)
- `activation: str` - Activation function: `tanh`, `relu`, `gelu` (default: `tanh`)

**Example:**
```bash
python -m src.trainer fit --config configs/ccppinn_default.yaml
```

### SimplePinn (Baseline)

**Architecture:** Simple MLP without feature engineering

**Key Parameters:**
- `hidden_dims: List[int]` - Layer dimensions (default: `[32, 32]`)
- `activation: str` - Activation function (default: `relu`)

**Example:**
```bash
python -m src.trainer fit --config configs/simplepinn.yaml
```

## BaseModel Parameters

All models inherit these parameters from `BaseModel`:

### Training
- `learning_rate: float` - Learning rate (default: `1e-3`)
- `optimizer: str` - Optimizer: `adamw`, `adam` (default: `adamw`)
- `scheduler: str` - LR scheduler: `constant`, `cosine`, `linear` (default: `constant`)
- `weight_decay: float` - L2 regularization (default: `0.0`)
- `warmup_steps: int` - Warmup steps (default: `0`)

### Loss Configuration
- `loss_weights: dict` - Loss component weights
  ```python
  {
      'continuity': 1.0,  # Continuity equation weight
      'poisson': 1.0,     # Poisson equation weight
      'bc': 10.0          # Boundary condition weight
  }
  ```

### Experiment
- `experiment_name: str` - Name for checkpoints/logs (default: `"default"`)
- `enable_benchmarking: bool` - Enable detailed eval plots (default: `False`)

## YAML Configuration Examples

### Override Individual Parameters

```yaml
# configs/custom_experiment.yaml
model:
  class_path: src.model.CCPPinn
  init_args:
    hidden_dims: [128, 128, 128, 128]  # Deeper network
    learning_rate: 5.0e-4              # Lower LR
    loss_weights:
      continuity: 2.0                  # Higher continuity weight
      poisson: 1.0
      bc: 20.0                         # Higher BC weight
```

### Experiment with Different Architectures

```bash
# Try CCPPinn
python -m src.trainer fit --config configs/ccppinn_default.yaml

# Try SimplePinn baseline
python -m src.trainer fit --config configs/simplepinn.yaml

# Override parameters via CLI
python -m src.trainer fit \
  --config configs/ccppinn_default.yaml \
  --model.hidden_dims=[256,256] \
  --model.learning_rate=1e-4
```

## Extending the Framework

### Adding New Architectures

You can add any architecture as long as `forward()` returns `(n_e, phi)` in physical units:

```python
class ResNetPinn(BaseModel):
    """PINN using ResNet backbone."""
    
    def forward(self, x, t):
        # Input normalization
        x_norm = x / self.scales.x_ref
        t_norm = t * self.params.f
        
        # ResNet forward pass
        features = self.resnet(torch.cat([x_norm, t_norm], dim=1))
        
        # Output scaling
        n_e = features[:, 0:1] * self.scales.n_ref
        phi = features[:, 1:2] * self.scales.phi_ref
        
        return n_e, phi
```

### Using Different Physics

To solve a different PDE, override `compute_pde_residuals()`:

```python
class NavierStokesPinn(BaseModel):
    def compute_pde_residuals(self, x, t):
        # Your custom PDE residuals
        u, v, p = self(x, t)  # Velocity and pressure
        
        # Compute NS residuals
        res_momentum_x = ...
        res_momentum_y = ...
        res_continuity = ...
        
        return res_momentum_x, res_momentum_y, res_continuity
```

## Best Practices

1. **Always call `super().__init__(**kwargs)`** - Passes BaseModel params
2. **Always call `self.save_hyperparameters()`** - Enables checkpoint loading
3. **Return physical units from `forward()`** - BaseModel expects this
4. **Use default parameters** - Makes models work out-of-the-box
5. **Document your architecture** - Help others understand your model

## Troubleshooting

### "Module has no forward()"
- Make sure your model class implements `forward(x, t)`

### "Missing hyperparameter"
- Check that `**kwargs` is in `__init__()` signature
- Check that `super().__init__(**kwargs)` is called

### "Checkpoint loading fails"
- Make sure `self.save_hyperparameters()` is called after `super().__init__()`

### "Loss is NaN/Inf"
- Check your `forward()` returns physical units
- Try adjusting `loss_weights` in config
- Reduce learning rate
