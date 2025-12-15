# CCP-II PINN: Physics-Informed Neural Networks for Plasma Simulation

A Physics-Informed Neural Network (PINN) implementation for solving the one-dimensional Capacitively Coupled Plasma (CCP-II) model. The model couples electron transport with electrostatic potential variations through the continuity and Poisson equations.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plasmatic-coupled-pinn.git
cd plasmatic-coupled-pinn

# Install with Poetry
poetry install

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# Train with default configuration
poetry run python -m src.trainer fit --config configs/default.yaml

# Train with GatedPINN (recommended for CCP-II)
poetry run python -m src.trainer fit --config configs/gated.yaml

# Train with WandB logging
poetry run python -m src.trainer fit --config configs/gated.yaml --use_wandb=true

# Test a checkpoint
poetry run python -m src.trainer test --ckpt_path=path/to/checkpoint.ckpt
```

## Physical Model

### Governing Equations

The CCP-II model is described by a coupled system of PDEs:

**Continuity Equation (Electron Transport):**

$$\frac{\partial n_e}{\partial t} + \frac{\partial \Gamma_e}{\partial x} = R$$

Electron density evolves due to transport and local generation/loss through reactions.

**Poisson Equation:**

$$\frac{\partial^2 \phi}{\partial x^2} = -\frac{e}{\varepsilon_0}(n_e - n_{i0})$$

Electrostatic potential varies in response to charge distribution.

### Electron Flux

The electron flux combines diffusion and drift:

$$\Gamma_e = -D\frac{\partial n_e}{\partial x} - \mu n_e \frac{\partial \phi}{\partial x}$$

Where:
- $D$ = Electron diffusion coefficient = $\frac{eT_e}{m_e \nu_m}$
- $\mu$ = Electron mobility = $\frac{e}{m_e \nu_m}$

### Ion Density (Boltzmann Relation)

$$n_{i0} = R_0(x_2 - x_1)\sqrt{\frac{m_i}{eT_e}}$$

### Boundary Conditions

| Boundary | Potential $\phi$ | Electron Density $n_e$ |
|----------|------------------|------------------------|
| Left ($x=0$) | $V(t) = V_0\sin(2\pi f t)$ | 0 |
| Right ($x=L$) | 0 | 0 |

**Reaction Zones:** $R(x) = R_0$ for $x \in [x_1, x_2] \cup [L-x_2, L-x_1]$, zero elsewhere.

### Default Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Domain Length | $L$ | 0.025 | m |
| Driving Frequency | $f$ | 13.56 | MHz |
| Voltage Amplitude | $V_0$ | 100 | V |
| Reaction Rate | $R_0$ | $2.7 \times 10^{20}$ | $\text{m}^{-3}\text{s}^{-1}$ |
| Reaction Zone Start | $x_1$ | 0.005 | m |
| Reaction Zone End | $x_2$ | 0.01 | m |
| Electron Temperature | $T_e$ | 3 | eV |
| Ion Mass | $m_i$ | 40 | amu |
| Collision Frequency | $\nu_m$ | $1 \times 10^8$ | $\text{s}^{-1}$ |

### Physical Constants

| Constant | Symbol | Value |
|----------|--------|-------|
| Elementary Charge | $e$ | $1.6 \times 10^{-19}$ C |
| Electron Mass | $m_e$ | $9.109 \times 10^{-31}$ kg |
| Vacuum Permittivity | $\varepsilon_0$ | $8.854 \times 10^{-12}$ F/m |

## Model Architectures

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `BasePINN` | Simple MLP baseline | Debugging, baselines |
| `SequentialPINN` | FFM + MLP with exact BC | General use |
| `GatedPINN` | FFM + Gated MLP | Complex dynamics (recommended) |
| `FourierPINN` | Random Fourier features | High-frequency signals |
| `ModulatedPINN` | Lightweight gating | Quick experiments |
| `NonDimPINN` | Explicit non-dimensionalization | Multi-scale physics |
| `HybridPINN` | Exact Poisson solver | When Poisson is difficult |

### Fourier Feature Mapping (FFM)

Transforms input $(x,t)$ into higher-dimensional space to capture high-frequency oscillations:

$$\text{Features} = [\sin(\omega_1 x), \cos(\omega_1 x), \sin(\omega_2 x), \cos(\omega_2 x), \ldots]$$

Mitigates spectral bias inherent in standard neural networks.

### Exact Boundary Conditions

Solutions satisfy boundary conditions exactly through network construction:

$$n_e(x,t) = x \cdot (L-x) \cdot \text{network}(x,t)$$

$$\phi(x,t) = \frac{L-x}{L} \cdot V(t) + x \cdot (L-x) \cdot \text{network}(x,t)$$

## Collocation Sampling

### Implemented Samplers

| Sampler | Description |
|---------|-------------|
| `uniform` | Random uniform sampling |
| `beta` | Beta distribution (boundary-focused) |
| `grid` | Regular grid sampling |
| `latin-hypercube` | LHS for better coverage |

### Beta Sampling

Concentrates samples near boundaries (sheath regions):

$$x \sim \text{Beta}(\alpha, \beta)$$

Useful for capturing sharp gradients near electrodes.

## Training

### Configuration

Training is configured via YAML files in `configs/`:

```yaml
model:
  class_path: src.model.GatedPINN
  init_args:
    hidden_layers: [128, 128, 128]
    num_ffm_frequencies: 4
    exact_bc: true
    learning_rate: 1e-3

data:
  batch_size: 4096
  num_points: 20000
  sampler_type: uniform

trainer:
  max_epochs: 5000
  accelerator: auto
```

### Outputs

Training produces:
- Checkpoints in `experiments/<name>/checkpoints/`
- TensorBoard logs in `experiments/<name>/tensorboard/`
- CSV metrics in `experiments/<name>/csv/`
- Visualizations in `experiments/<name>/visualizations/`

## Project Structure

```
plasmatic-coupled-pinn/
├── configs/           # YAML training configurations
├── docs/              # Documentation and references
├── scripts/           # Utility scripts
├── src/
│   ├── architectures/ # Neural network components
│   ├── data/          # Data loading and sampling
│   ├── utils/         # Physics, gradients, logging
│   ├── visualization/ # Plotting and animation
│   ├── model.py       # PINN model definitions
│   └── trainer.py     # PyTorch Lightning training
└── experiments/       # Training outputs (gitignored)
```

## Validation

Run model validation:

```bash
poetry run python scripts/validate_experiments.py
```

## Documentation

- [Theoretical Methods](docs/THEORETICAL_METHODS.md) - Advanced techniques and future work
- [Model Architecture](docs/MODEL_ARCHITECTURE.md) - Neural network details
- [Visualization](docs/VISUALIZATION.md) - Plotting and animation guide
- `docs/ASU_PIML_1D.pdf` - Primary physics documentation
- `docs/CCP-II Model_10_31_update.docx` - Model update notes

## References

### Core Methods
- Raissi et al. (2019): Physics-Informed Neural Networks
- Tancik et al. (2020): Fourier Features Let Networks Learn High Frequency Functions

### Adaptive Techniques
- Kendall & Gal (2018): Multi-Task Learning Using Uncertainty to Weigh Losses
- Wang et al. (2021): Understanding and Mitigating Gradient Flow Pathologies in PINNs
- Wang et al. (2022): When and Why PINNs Fail to Train (NTK Analysis)

### Advanced Architectures
- Wang et al. (2024): PirateNets - Physics-informed Residual Adaptive Networks
- Cho et al. (NeurIPS 2023): Hypernetwork-based Meta-Learning for Low-Rank PINNs

### Sampling & Optimization
- Wu et al. (2022): Residual-based Adaptive Sampling for PINNs
- Liu et al. (2025): ConFIG - Conflict-free Training of PINNs

## License

MIT License
