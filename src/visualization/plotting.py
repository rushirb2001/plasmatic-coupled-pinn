"""
Static plotting utilities for PINN visualization.

Optimized for speed while maintaining visual quality.

Provides functions for visualizing:
- Solution fields (n_e, phi) as heatmaps
- Spatial/temporal profiles
- Loss curves and training metrics
- Comparison plots with reference data
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from src.data.fdm_solver import get_fdm_for_visualization

# Module-level initialization (called once at import)
_MATPLOTLIB_CONFIGURED = False

def _ensure_matplotlib_configured():
    """Configure matplotlib once at module level."""
    global _MATPLOTLIB_CONFIGURED
    if not _MATPLOTLIB_CONFIGURED:
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "agg.path.chunksize": 10000,
        })
        _MATPLOTLIB_CONFIGURED = True

# Initialize on import
_ensure_matplotlib_configured()


def _log(msg: str) -> None:
    """Print a log message with prefix."""
    print(f"[Plotting] {msg}")


def setup_matplotlib():
    """Configure matplotlib for publication-quality plots. (No-op, configured at import)"""
    pass  # Already configured at module level


def _save_and_close(fig: plt.Figure, save_path: Optional[str], msg: str) -> None:
    """Save figure and close to free memory."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor='white', edgecolor='none')
        _log(msg)
    plt.close(fig)


def _convert_units(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    """Convert to display units and compute extent. Returns x_mm, t_us, extent."""
    x_mm = x * 1e3
    t_us = t * 1e6
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]
    return x_mm, t_us, extent


def compute_errors(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute error maps and L2 errors in a single pass.

    Returns:
        err_n_e: Relative error map for n_e
        err_phi: Relative error map for phi
        l2_n_e: Relative L2 error for n_e
        l2_phi: Relative L2 error for phi
    """
    eps = 1e-10
    ref_n_e_max = np.abs(ref_n_e).max() + eps
    ref_phi_max = np.abs(ref_phi).max() + eps

    err_n_e = np.abs(pred_n_e - ref_n_e) / ref_n_e_max
    err_phi = np.abs(pred_phi - ref_phi) / ref_phi_max

    l2_n_e = np.sqrt(np.mean((pred_n_e - ref_n_e)**2)) / ref_n_e_max
    l2_phi = np.sqrt(np.mean((pred_phi - ref_phi)**2)) / ref_phi_max

    return err_n_e, err_phi, l2_n_e, l2_phi


def plot_solution_heatmaps(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot electron density and potential as heatmaps.

    Args:
        n_e: Electron density array [Nx, Nt]
        phi: Electric potential array [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save figure (optional)
        title: Figure title (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    x_mm, t_us, extent = _convert_units(x, t)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Electron density - use rasterized for large arrays
    im1 = axes[0].imshow(
        n_e, extent=extent, aspect="auto", origin="lower",
        cmap="viridis", rasterized=True
    )
    axes[0].set_title(r"Electron Density $n_e$ (m$^{-3}$)")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Position (mm)")
    fig.colorbar(im1, ax=axes[0], format="%.2e")

    # Electric potential
    im2 = axes[1].imshow(
        phi, extent=extent, aspect="auto", origin="lower",
        cmap="plasma", rasterized=True
    )
    axes[1].set_title(r"Electric Potential $\phi$ (V)")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Position (mm)")
    fig.colorbar(im2, ax=axes[1])

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved solution heatmaps to {save_path}")

    return fig


def plot_spatial_profiles(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    time_indices: Optional[List[int]] = None,
    ref_n_e: Optional[np.ndarray] = None,
    ref_phi: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot spatial profiles at selected time instances.
    """
    if time_indices is None:
        nt = len(t)
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    x_mm = x * 1e3
    t_us = t * 1e6
    n_times = len(time_indices)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pre-compute colors once
    colors = plt.cm.viridis(np.linspace(0, 1, n_times))
    has_ref = ref_n_e is not None and ref_phi is not None

    # n_e profiles - batch label creation
    for i, idx in enumerate(time_indices):
        label = f"PINN t={t_us[idx]:.2f}μs" if has_ref else f"t = {t_us[idx]:.2f} μs"
        axes[0].plot(x_mm, n_e[:, idx], color=colors[i], label=label, linewidth=1.5)
        if has_ref:
            axes[0].plot(x_mm, ref_n_e[:, idx], color=colors[i], linewidth=1.5,
                        linestyle="--", alpha=0.7)

    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Profiles" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # phi profiles
    for i, idx in enumerate(time_indices):
        label = f"PINN t={t_us[idx]:.2f}μs" if has_ref else f"t = {t_us[idx]:.2f} μs"
        axes[1].plot(x_mm, phi[:, idx], color=colors[i], label=label, linewidth=1.5)
        if has_ref:
            axes[1].plot(x_mm, ref_phi[:, idx], color=colors[i], linewidth=1.5,
                        linestyle="--", alpha=0.7)

    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Profiles" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved spatial profiles to {save_path}")

    return fig


def plot_temporal_evolution(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    position_indices: Optional[List[int]] = None,
    ref_n_e: Optional[np.ndarray] = None,
    ref_phi: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot temporal evolution at selected positions.
    """
    if position_indices is None:
        nx = len(x)
        position_indices = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]

    x_mm = x * 1e3
    t_us = t * 1e6
    n_pos = len(position_indices)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0, 1, n_pos))
    has_ref = ref_n_e is not None and ref_phi is not None

    # n_e evolution
    for i, idx in enumerate(position_indices):
        label = f"PINN x={x_mm[idx]:.1f}mm" if has_ref else f"x = {x_mm[idx]:.1f} mm"
        axes[0].plot(t_us, n_e[idx, :], color=colors[i], label=label, linewidth=1.5)
        if has_ref:
            axes[0].plot(t_us, ref_n_e[idx, :], color=colors[i], linewidth=1.5,
                        linestyle="--", alpha=0.7)

    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Evolution" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # phi evolution
    for i, idx in enumerate(position_indices):
        label = f"PINN x={x_mm[idx]:.1f}mm" if has_ref else f"x = {x_mm[idx]:.1f} mm"
        axes[1].plot(t_us, phi[idx, :], color=colors[i], label=label, linewidth=1.5)
        if has_ref:
            axes[1].plot(t_us, ref_phi[idx, :], color=colors[i], linewidth=1.5,
                        linestyle="--", alpha=0.7)

    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Evolution" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved temporal evolution to {save_path}")

    return fig


def plot_loss_curves(
    losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot training loss curves.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in losses.items():
        ax.plot(values, label=name, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved loss curves to {save_path}")

    return fig


def plot_comparison_heatmaps(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12),
    err_n_e: Optional[np.ndarray] = None,
    err_phi: Optional[np.ndarray] = None,
    l2_n_e: Optional[float] = None,
    l2_phi: Optional[float] = None,
) -> plt.Figure:
    """
    Plot 3x2 grid comparing PINN predictions with FDM reference.
    Optimized: accepts pre-computed errors to avoid redundant computation.
    """
    x_mm, t_us, extent = _convert_units(x, t)

    # Use pre-computed errors if provided, otherwise compute
    if err_n_e is None or err_phi is None or l2_n_e is None or l2_phi is None:
        err_n_e, err_phi, l2_n_e, l2_phi = compute_errors(
            pred_n_e, pred_phi, ref_n_e, ref_phi
        )

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Common imshow kwargs for speed
    imshow_kwargs = dict(extent=extent, aspect="auto", origin="lower", rasterized=True)

    # Row 1: Reconstructed (PINN)
    im1 = axes[0, 0].imshow(pred_n_e, cmap="rainbow", **imshow_kwargs)
    fig.colorbar(im1, ax=axes[0, 0], format="%.2e")
    axes[0, 0].set_title(r"Reconstructed $n_e$ (m$^{-3}$)")
    axes[0, 0].set_ylabel("Position (mm)")

    im2 = axes[0, 1].imshow(pred_phi, cmap="rainbow", **imshow_kwargs)
    fig.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].set_title(r"Reconstructed $\phi$ (V)")

    # Row 2: Original (FDM)
    im3 = axes[1, 0].imshow(ref_n_e, cmap="rainbow", **imshow_kwargs)
    fig.colorbar(im3, ax=axes[1, 0], format="%.2e")
    axes[1, 0].set_title(r"FDM $n_e$ (m$^{-3}$)")
    axes[1, 0].set_ylabel("Position (mm)")

    im4 = axes[1, 1].imshow(ref_phi, cmap="rainbow", **imshow_kwargs)
    fig.colorbar(im4, ax=axes[1, 1])
    axes[1, 1].set_title(r"FDM $\phi$ (V)")

    # Row 3: Error
    im5 = axes[2, 0].imshow(err_n_e, cmap="hot", **imshow_kwargs)
    fig.colorbar(im5, ax=axes[2, 0])
    axes[2, 0].set_title(r"Relative Error $n_e$")
    axes[2, 0].set_xlabel("Time (μs)")
    axes[2, 0].set_ylabel("Position (mm)")

    im6 = axes[2, 1].imshow(err_phi, cmap="hot", **imshow_kwargs)
    fig.colorbar(im6, ax=axes[2, 1])
    axes[2, 1].set_title(r"Relative Error $\phi$")
    axes[2, 1].set_xlabel("Time (μs)")

    # Add L2 error text
    fig.text(0.5, 0.01, f"Relative L2 Error: $n_e$ = {l2_n_e:.4f}, $\\phi$ = {l2_phi:.4f}",
             ha="center", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save_and_close(fig, save_path, f"Saved comparison heatmaps to {save_path}")

    return fig


def plot_comparison(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    x: np.ndarray,
    t_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot comparison between predicted and reference solutions.
    """
    x_mm = x * 1e3

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # n_e comparison
    axes[0].plot(x_mm, ref_n_e[:, t_idx], "k-", label="Reference", linewidth=2)
    axes[0].plot(x_mm, pred_n_e[:, t_idx], "r--", label="PINN", linewidth=2)
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # phi comparison
    axes[1].plot(x_mm, ref_phi[:, t_idx], "k-", label="Reference", linewidth=2)
    axes[1].plot(x_mm, pred_phi[:, t_idx], "r--", label="PINN", linewidth=2)
    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved comparison plot to {save_path}")

    return fig


def plot_error_maps(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    err_n_e: Optional[np.ndarray] = None,
    err_phi: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot error maps between predicted and reference solutions.
    Accepts pre-computed errors to avoid redundant computation.
    """
    x_mm, t_us, extent = _convert_units(x, t)

    # Use pre-computed errors if provided, otherwise compute
    if err_n_e is None or err_phi is None:
        err_n_e, err_phi, _, _ = compute_errors(
            pred_n_e, pred_phi, ref_n_e, ref_phi
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Common kwargs
    imshow_kwargs = dict(extent=extent, aspect="auto", origin="lower",
                         cmap="Reds", rasterized=True)
    norm = mcolors.LogNorm(vmin=1e-4, vmax=1)

    # n_e error
    im1 = axes[0].imshow(err_n_e, norm=norm, **imshow_kwargs)
    axes[0].set_title(r"Relative Error in $n_e$")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Position (mm)")
    fig.colorbar(im1, ax=axes[0])

    # phi error
    im2 = axes[1].imshow(err_phi, norm=norm, **imshow_kwargs)
    axes[1].set_title(r"Relative Error in $\phi$")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Position (mm)")
    fig.colorbar(im2, ax=axes[1])

    fig.tight_layout()
    _save_and_close(fig, save_path, f"Saved error maps to {save_path}")

    return fig


def visualize_model(
    model: torch.nn.Module,
    nx: int = 100,
    nt: int = 100,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    save_dir: Optional[str] = None,
    device: str = "cpu",
    fdm_dir: str = "data/fdm",
    ref_n_e: Optional[np.ndarray] = None,
    ref_phi: Optional[np.ndarray] = None,
    ref_x: Optional[np.ndarray] = None,
    ref_t: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Visualize a trained PINN model with optional FDM comparison.
    Optimized: efficient tensor operations, batch model evaluation.
    """
    _log(f"Starting model visualization (nx={nx}, nt={nt}, device={device})")
    model.eval()
    model.to(device)

    # Auto-load FDM reference data if model has physics params and no manual ref provided
    if ref_n_e is None and hasattr(model, 'params'):
        _log("Attempting to load FDM reference data...")
        fdm_data = get_fdm_for_visualization(model.params, fdm_dir=fdm_dir)
        if fdm_data is not None:
            ref_n_e, ref_phi, ref_x, ref_t = fdm_data
            _log(f"Loaded FDM reference data: {model.params.get_fdm_filename()}")

    has_ref = ref_n_e is not None and ref_phi is not None
    _log(f"Reference data available: {has_ref}")

    # Use reference grid if available, otherwise create evaluation grid
    if has_ref and ref_x is not None and ref_t is not None:
        if hasattr(model, 'params'):
            L = model.params.domain.L
            T_period = model.params.scales.t_ref
        else:
            L = ref_x.max()
            T_period = ref_t.max()

        x_norm = ref_x / L
        t_norm = ref_t / T_period
        # Create tensors directly on device
        x = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_norm, dtype=torch.float32, device=device)
        nx, nt = len(ref_x), len(ref_t)
    else:
        x = torch.linspace(x_range[0], x_range[1], nx, device=device)
        t = torch.linspace(t_range[0], t_range[1], nt, device=device)

    # Create meshgrid efficiently
    X, T = torch.meshgrid(x, t, indexing="ij")
    x_t = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    _log(f"Evaluating model on grid of {x_t.shape[0]} points...")

    # Evaluate model with torch.inference_mode for speed
    with torch.inference_mode():
        n_e, phi = model(x_t)
    _log("Model evaluation complete")

    # Reshape and convert to numpy efficiently
    n_e = n_e.view(nx, nt).cpu().numpy()
    phi = phi.view(nx, nt).cpu().numpy()
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()

    # Scale predictions to physical units using physics parameters
    if has_ref and hasattr(model, 'params'):
        n_e *= model.params.scales.n_ref
        phi *= model.params.scales.phi_ref
        x_np = ref_x
        t_np = ref_t
    elif has_ref:
        n_e *= ref_n_e.max()
        phi *= ref_phi.max()
        x_np = ref_x
        t_np = ref_t

    # Create plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Generating plots in {save_dir}")

        if has_ref:
            _log("Generating comparison plots with FDM reference...")
            # Compute errors once, pass to both functions
            err_n_e, err_phi, l2_n_e, l2_phi = compute_errors(
                n_e, phi, ref_n_e, ref_phi
            )
            plot_comparison_heatmaps(
                n_e, phi, ref_n_e, ref_phi, x_np, t_np,
                save_path=str(save_dir / "comparison_heatmaps.png"),
                err_n_e=err_n_e, err_phi=err_phi, l2_n_e=l2_n_e, l2_phi=l2_phi
            )
            plot_spatial_profiles(
                n_e, phi, x_np, t_np,
                ref_n_e=ref_n_e, ref_phi=ref_phi,
                save_path=str(save_dir / "spatial_profiles.png")
            )
            plot_temporal_evolution(
                n_e, phi, x_np, t_np,
                ref_n_e=ref_n_e, ref_phi=ref_phi,
                save_path=str(save_dir / "temporal_evolution.png")
            )
            plot_error_maps(
                n_e, phi, ref_n_e, ref_phi, x_np, t_np,
                save_path=str(save_dir / "error_maps.png"),
                err_n_e=err_n_e, err_phi=err_phi
            )
        else:
            _log("Generating plots without FDM reference...")
            plot_solution_heatmaps(
                n_e, phi, x_np, t_np,
                save_path=str(save_dir / "solution_heatmaps.png")
            )
            plot_spatial_profiles(
                n_e, phi, x_np, t_np,
                save_path=str(save_dir / "spatial_profiles.png")
            )
            plot_temporal_evolution(
                n_e, phi, x_np, t_np,
                save_path=str(save_dir / "temporal_evolution.png")
            )

        _log("Plot generation complete")

    return {"n_e": n_e, "phi": phi, "x": x_np, "t": t_np}
