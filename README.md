# CCP-II Model: Physics-Informed Neural Networks for Plasma Simulation

## Overview

This project implements a Physics-Informed Neural Network (PINN) approach to solve a one-dimensional Capacitively Coupled Plasma (CCP) model. The model is governed by two fundamental equations: the electron continuity equation and the Poisson equation, which together describe electron transport and electrostatic potential variations in plasma systems.

## Physical Model

### Governing Equations

The CCP-II model is described by a coupled system of partial differential equations:

**Continuity Equation (Electron Transport):**
$$\frac{\partial n_e}{\partial t} + \frac{\partial \Gamma_e}{\partial x} = R$$

Expresses particle conservation, indicating that electron density evolves over time due to electron transport toward surfaces and local generation or loss through chemical reactions.

**Poisson Equation:**
$$\frac{\partial^2 \phi}{\partial x^2} = -\frac{e}{\varepsilon_0}(n_e - n_{i0})$$

Describes how electrostatic potential spatially varies in response to charge distribution of electrons and ions within the plasma.

### Key Variables

- $n_e$: Electron density
- $\phi$: Electric potential
- $\Gamma_e$: Electron flux (combination of diffusion and drift components)
- $R(x)$: Spatially dependent reaction rate

### Electron Flux Components

The electron flux consists of two components:
$$\Gamma_e = -D\frac{\partial n_e}{\partial x} - \mu n_e \frac{\partial \phi}{\partial x}$$

Where:
- **Diffusion flux**: $-D\frac{\partial n_e}{\partial x}$
- **Drift flux**: $-\mu n_e \frac{\partial \phi}{\partial x}$

Transport coefficients:
- $D$ = Electron diffusion coefficient = $\frac{eT_e}{m_e \nu_m}$
- $\mu$ = Electron mobility = $\frac{e}{m_e \nu_m}$

### Ion Density

The ion density is expressed through the Boltzmann relation:
$$n_{i0} = R_0(x_2 - x_1)\sqrt{\frac{m_i}{eT_e}}$$

Where $R_0$ is the spatially varying reaction rate coefficient.

## Boundary Conditions

- **Left boundary** ($x=0$): $\phi = V(t)$, $n_e = 0$
- **Right boundary** ($x=L$): $\phi = 0$, $n_e = 0$
- **Driving voltage**: $V(t) = V_0 \sin(2\pi f t)$ (sinusoidal AC)
- **Reaction zones**: $R(x) = R_0$ for $x \in [x_1, x_2] \cup [L-x_2, L-x_1]$, $R = 0$ elsewhere

## Default Parameters

| Parameter | Symbol | Default Value |
|-----------|--------|----------------|
| Domain Length | $L$ | 0.025 m |
| Driving Frequency | $f$ | 13.56 MHz |
| Driving Voltage Amplitude | $V_0$ | 100 V |
| Reaction Rate Coefficient | $R_0$ | $2.7 \times 10^{20}$ |
| Reaction zone start | $x_1$ | 0.005 m |
| Reaction zone width | $x_2$ | 0.01 m |
| Electron Temperature | $T_e$ | 3 eV |
| Ion Mass | $m_i$ | 40 amu |
| Collision Frequency | $\nu_m$ | $1 \times 10^8$ s$^{-1}$ |

### Physical Constants

- Elementary charge ($e$): $1.6 \times 10^{-19}$ C
- Electron mass ($m_e$): $9.109 \times 10^{-31}$ kg
- Vacuum permittivity ($\varepsilon_0$): $8.854 \times 10^{-12}$ C$^2$kg$^{-1}$m$^{-3}$s$^{-2}$

## PINN Architecture and Techniques

### Standard PINN Approach

Physics-Informed Neural Networks solve PDEs by embedding governing equations, initial conditions, and boundary conditions directly into the loss function during training. Rather than relying solely on labeled data, PINNs enforce physics constraints during training.

### Network Architecture Components

#### 1. Poisson Solver
- Numerically hard constraining the Poisson equation on uniform grids
- Random collocation point approaches
- Centered, forward, and backward finite difference approximations

#### 2. PirateNets
Implements Physics-informed Deep Learning with Residual Adaptive Networks (Wang et al., 2024)

#### 3. Alternating PINN Architecture
Sequential training of different components

#### 4. Hypernetwork Architecture
Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks (Cho et al., NeurIPS 2023)

### Adaptive Weighting Techniques

#### Loss Variance-Based Adaptive Weights

Derived from Kendall and Gal's (2018) Multi-Task Learning framework. Each task's loss is interpreted as arising from a Gaussian likelihood with learnable observation noise. The composite loss function:

$$L = \sum_i \frac{L_i}{2\sigma_i^2} + \log(\sigma_i)$$

The exponential term $\frac{1}{2\sigma_i^2}$ acts as an adaptive scaling factor that down-weights noisy or unstable losses (large $\sigma_i$) and up-weights reliable ones (small $\sigma_i$).

#### Gradient Pathology-Based Weights

Addresses imbalances between multiple objectives by monitoring gradient norms rather than loss magnitudes. This approach mitigates stiffness and vanishing gradient issues.

Algorithm:
1. Compute gradient magnitudes $g_i = \left\|\nabla_\theta L_i\right\|$ for each loss term
2. Calculate maximum gradients across tasks: $g_{\max} = \max_i g_i$
3. Normalize gradients: $\bar{g}_i = \frac{g_i}{g_{\max}}$
4. Apply exponential moving average
5. Update task-specific smoothing parameters

#### Neural Tangent Kernel (NTK) Perspective

Balances effective learning dynamics across loss components by monitoring kernel spectra and coupling strength. Under gradient flow:
$$\frac{dL_i}{dt} = -\frac{\partial L_i^T}{\partial \theta} K(t) \frac{\partial L_i}{\partial \theta}$$

Adjusts weights to equalize convergence rates of different loss terms. In the infinite-width limit, the Neural Tangent Kernel converges to a deterministic kernel, and training is equivalent to kernel regression.

### Optimizer Strategies

#### ConFIG Optimizer

Conflict-Free Gradient optimizer addresses gradient conflicts in multi-loss PINN training by projecting each loss gradient onto a common descent direction that minimizes interference between objectives.

#### Adam + L-BFGS Hybrid Strategy

- **Early phase**: Adam optimizer for stable first-order updates
- **Later phase**: L-BFGS optimizer for refined convergence with higher precision

### Collocation Point Strategies

#### Beta Sampling

Progressively shifts focus from bulk plasma to sheath regions during training:

**Spatial sampling**: $x \sim \text{Beta}(\alpha, \beta)$

**Temporal sampling**: $t \sim \text{Uniform}(0, T)$

During training, the parameter $\beta$ is annealed according to:
$$\beta_{\text{epoch}} = \beta_0 \cdot \frac{\text{epoch}+1}{\text{TOTAL\_EPOCHS}+1}$$

When $\beta$ is large, samples concentrate near the sheath region; when $\beta$ is small, emphasis shifts to the bulk plasma region.

#### RAD Sampling (Residual-based Adaptive Distribution)

Adaptively redistributes training points based on PDE residual magnitudes:

1. Generate large candidate pool of random $(x,t)$ points over spatiotemporal domain
2. Evaluate continuity and Poisson residuals using automatic differentiation
3. Compute scalar residual magnitude: $\mathcal{R} = \left|\frac{\partial n_e}{\partial t} + \frac{\partial \Gamma_e}{\partial x} - R\right| + \left|\frac{\partial^2 \phi}{\partial x^2} + \frac{e}{\varepsilon_0}(n_e - n_{i0})\right|$
4. Convert residuals to probability density function: $p(x,t) = \frac{\mathcal{R}^k}{\sum \mathcal{R}^k} + c_{\text{cc}}$

   where $k$ controls focus on high-residual regions and $c_{\text{cc}}$ ensures a uniform sampling component for exploration

5. Sample new collocation points from this weighted distribution

## Training Strategies

### Soft vs. Hard Constraining of Boundary Conditions

**Hard Constraining (Exact Boundary Conditions):**

Solutions satisfy boundary conditions exactly through network construction:

$$n_e(x) = (x - 0) \cdot d(x) = x \cdot d(x)$$
$$\phi(x) = (x - L) \cdot (V_0\sin(2\pi f t) - 0) \cdot p(x)$$

where $d(x)$ and $p(x)$ are trainable functions learned by the neural network.

**Soft Constraining:**

Boundary conditions are enforced through loss terms with trainable network components.

### Time Marching Approach

- Temporal domain divided into 10 sub-windows of equal length ($\Delta t = 0.1$ time units)
- Separate PirateNet model trained for each window using Random Fourier Feature embeddings with hidden dimension and three residual blocks
- Each window trained with:
  - Boundary MSE: $\text{MSE}_{\text{BC}} = \frac{1}{N_{\text{BC}}}\sum (n_e^{\text{pred}} - 0)^2 + (\phi^{\text{pred}} - V(t))^2$
  - PDE residuals (continuity and Poisson terms)
  - Initial condition consistency term enforcing continuity with previous window
- First window includes full-grid FDM supervision
- Sequential initialization: each window's final prediction becomes the next window's initial condition
- Training: 20,000 collocation points per epoch, evenly split between boundary and interior points
- Optimization: Fixed learning rate, weight decay of 0.01

### Curriculum Training Strategy

Model is progressively trained across increasing parameter values rather than from scratch for each configuration:

1. Identify configuration with lowest L2 error (baseline, e.g., $R_0 = 2.5 \times 10^{20}$)
2. Use that trained network as initialization for next parameter value
3. Gradually increase reaction coefficient: $R_0: 2.5 \times 10^{20} \to 2.7 \times 10^{20} \to 2.9 \times 10^{20} \to 3.5 \times 10^{20}$

Or gradually increase voltage: $V_0: 40 \to 60 \to 80 \to 125 \to 150$ V

#### Output-Layer-Only Fine-Tuning

Variant of curriculum training where all network parameters except the final linear output layer are frozen during subsequent fine-tuning stages. Only weights and biases of the output layer remain trainable.

### Bounding Potential

Learnable gating mechanism on potential output to prevent overly rapid evolution and unstable solutions:

$$\phi_{\text{final}} = \alpha \cdot \phi_{\text{predicted}}$$

where $\alpha$ is a trainable scalar parameter enforced to remain positive using exponential parametrization: $\alpha = \exp(\log\_\text{scale})$

## Feature Engineering

### Fourier Feature Mapping

Transforms input $(x,t)$ into higher-dimensional space using sinusoidal functions to capture oscillatory behavior:

$$\text{Features} = [\sin(\omega_1 x), \cos(\omega_1 x), \sin(\omega_2 x), \cos(\omega_2 x), \ldots]$$

where $\omega_i$ are frequency components.

### Periodic Time Embedding

Enforces quasi-steady state behavior using periodic time embeddings with maximum of 2 harmonics:

$$\text{Features} = [\sin(t), \cos(t), \sin(2t), \cos(2t)]$$

### Log-Space Prediction

Predicts $\log(n_e)$ instead of $n_e$ directly to improve numerical stability:

$$\hat{n}_e = \exp(\text{network\_output})$$

with exponential transformation on the output to ensure positive electron density.

## Evaluation Criteria

### L2 Error

Standard L2 norm of prediction errors:
$$L_2 = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_{\text{pred},i} - y_{\text{true},i})^2}$$

### FRMSE (Fourier Root Mean Square Error)

Quantifies differences in frequency domain rather than physical domain. Measures how well the model preserves spectral content (oscillatory modes and harmonics).

**Spatial FRMSE:**
Apply FFT along spatial dimension for each fixed time slice. Calculate RMSE deviation between predicted and reference spectra. Average over all time slices:
$$\text{FRMSE}_x = \frac{1}{N_t}\sum_{j=1}^{N_t} \sqrt{\frac{1}{N_x}\sum_{i=1}^{N_x} |\hat{Y}(k_i, t_j) - Y(k_i, t_j)|^2}$$

**Temporal FRMSE:**
Apply FFT along temporal dimension for each spatial location. Measure frequency-domain discrepancies:
$$\text{FRMSE}_t = \frac{1}{N_x}\sum_{j=1}^{N_x} \sqrt{\frac{1}{N_t}\sum_{i=1}^{N_t} |\hat{Y}(x_j, \omega_i) - Y(x_j, \omega_i)|^2}$$

Normalized by spectral energy of true field for scale-invariance.

<!-- ## Experimental Results

### Best Results (Without FDM Data)

Configuration: Adaptive weights (gradient pathology - Algorithm 1), exact boundary conditions, Fourier Feature Mapping, periodic time embedding, and log-space prediction.

**Varying Reaction Coefficient ($R_0$):**

| $R_0$ | $L_2(n_e)$ | $L_2(\phi)$ |
|-------|-----------|-----------|
| $2.3 \times 10^{20}$ | $2.73 \times 10^{-1}$ | $5.60 \times 10^{-1}$ |
| $2.5 \times 10^{20}$ | $9.72 \times 10^{-2}$ | $1.49 \times 10^{-1}$ |
| $2.7 \times 10^{20}$ | $3.55 \times 10^{-2}$ | $3.89 \times 10^{-2}$ |
| $2.9 \times 10^{20}$ | $3.46 \times 10^{-1}$ | $9.75 \times 10^{-1}$ |
| $3.1 \times 10^{20}$ | $3.38 \times 10^{-1}$ | $9.59 \times 10^{-1}$ |
| $3.9 \times 10^{20}$ | $4.07 \times 10^{-1}$ | $1.55 \times 10^{0}$ |

**Varying Voltage ($V_0$):**

| $V_0$ (V) | $L_2(n_e)$ | $L_2(\phi)$ |
|-----------|-----------|-----------|
| 40 | $4.59 \times 10^{-2}$ | $1.11 \times 10^{-1}$ |
| 60 | $4.03 \times 10^{-2}$ | $7.00 \times 10^{-2}$ |
| 80 | $4.30 \times 10^{-2}$ | $6.29 \times 10^{-2}$ |
| 125 | $4.34 \times 10^{-2}$ | $4.37 \times 10^{-2}$ |
| 150 | $2.59 \times 10^{-1}$ | $4.77 \times 10^{-1}$ |

### Results With FDM Data [0, 0.1]

Training with FDM reference data from time window $[0, 0.1]$ improves results:

| $R_0$ | $L_2(n_e)$ | $L_2(\phi)$ |
|-------|-----------|-----------|
| $2.3 \times 10^{20}$ | $1.25 \times 10^{-1}$ | $2.21 \times 10^{-1}$ |
| $2.5 \times 10^{20}$ | $5.32 \times 10^{-2}$ | $7.36 \times 10^{-2}$ |
| $2.7 \times 10^{20}$ | $4.61 \times 10^{-2}$ | $6.19 \times 10^{-2}$ |
| $3.9 \times 10^{20}$ | $6.04 \times 10^{-2}$ | $9.64 \times 10^{-2}$ |

### Curriculum Learning Results

Training progression from lower to higher reaction coefficients improves stability.

**Starting from $R_0 = 2.5 \times 10^{20}$ (without FDM data, 10k epochs Phase 2):**

| $R_0$ | $L_2(n_e)$ | $L_2(\phi)$ |
|-------|-----------|-----------|
| $2.7 \times 10^{20}$ | $4.28 \times 10^{-2}$ | $5.26 \times 10^{-2}$ |
| $2.9 \times 10^{20}$ | $5.41 \times 10^{-2}$ | $7.49 \times 10^{-2}$ |
| $3.9 \times 10^{20}$ | $1.02 \times 10^{-1}$ | $1.91 \times 10^{-1}$ |

### RAD Sampling Strategy Results

Residual-based adaptive sampling shows mixed performance compared to baseline:

| Metric | RAD Sampling | Baseline |
|--------|-------------|----------|
| $L_2(n_e)$ | $5.30 \times 10^{-2}$ | $3.86 \times 10^{-2}$ |
| $L_2(\phi)$ | $6.98 \times 10^{-2}$ | $4.72 \times 10^{-2}$ |
| MSE (Continuity) | $9.13$ | $14.55$ |
| MSE (Poisson) | $0.011$ | $0.118$ |
| Spectral FRMSE($n_e$) | $0.0069$ | - |
| Spectral FRMSE($\phi$) | $0.0149$ | - |

### Detailed Performance Example

Configuration: $R_0 = 2.7 \times 10^{20}$, $V_0 = 100$ V, trained with FDM data $[0, 0.1]$, 50k epochs

**Error Metrics:**
- $L_2$ error (electron density): $4.61 \times 10^{-2}$
- $L_2$ error (electric potential): $6.19 \times 10^{-2}$

**Loss Components:**
- Continuity equation loss: $14.545$
- Poisson equation loss: $0.1183$

**Fourier Error Metrics:**
- Spectral FRMSE($n_e$) along $x$: $0.004917$
- Spectral FRMSE($\phi$) along $x$: $0.017183$
- Temporal FRMSE($n_e$) averaged over $x$: $0.000273$
- Temporal FRMSE($\phi$) averaged over $x$: $0.000247$ -->

## Non-Dimensionalization

The governing equations are non-dimensionalized to improve numerical stability using reference scales. The continuity equation after non-dimensionalization:

$$\frac{\partial \tilde{n}_e}{\partial \tilde{t}} + \frac{\partial \tilde{\Gamma}_e}{\partial \tilde{x}} = \tilde{R}$$

And the Poisson equation becomes dimensionless through appropriate scaling of all variables and parameters.

## References

### Key Papers Referenced

- Kendall & Gal (2018): Multi-Task Learning Using Uncertainty to Weigh Losses
- Wang et al. (2021): Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks
- Wang et al. (2022): Neural Tangent Kernel (NTK) Perspective on PINN convergence
- Wang et al. (2024): PirateNets - Physics-informed Deep Learning with Residual Adaptive Networks
- Wu et al. (2022): A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks
- Cho et al. (NeurIPS 2023): Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks
- Liu et al. (2025): ConFIG - Towards Conflict-free Training of Physics Informed Neural Networks
- Kiyani et al. (2025): Optimizing the optimizer for physics-informed neural networks and Kolmogorov-Arnold networks
- Ryck et al.: An operator preconditioning perspective on training in physics-informed machine learning

## Documentation

Primary model documentation and detailed information are available in:
- `docs/ASU_PIML_1D.pdf`
- `docs/CCP-II Model_10_31_update.docx`
