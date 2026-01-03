"""
Static plotting utilities for PINN visualization.

Provides functions for visualizing:
- Solution fields (n_e, phi) as heatmaps
- Spatial/temporal profiles
- Loss curves and training metrics
- Comparison plots with reference data
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from src.data.fdm_solver import get_fdm_for_visualization


def _log(msg: str) -> None:
    """Print a log message with prefix."""
    print(f"[Plotting] {msg}")


def setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


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
    setup_matplotlib()

    # Convert to display units
    x_mm = x * 1e3  # m to mm
    t_us = t * 1e6  # s to us

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Electron density
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]

    im1 = axes[0].imshow(
        n_e, extent=extent, aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0].set_title(r"Electron Density $n_e$ (m$^{-3}$)")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Position (mm)")
    plt.colorbar(im1, ax=axes[0], format="%.2e")

    # Electric potential
    im2 = axes[1].imshow(
        phi, extent=extent, aspect="auto", origin="lower", cmap="plasma"
    )
    axes[1].set_title(r"Electric Potential $\phi$ (V)")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Position (mm)")
    plt.colorbar(im2, ax=axes[1])

    if title:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved solution heatmaps to {save_path}")

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

    Args:
        n_e: Electron density array [Nx, Nt]
        phi: Electric potential array [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        time_indices: Indices of times to plot
        ref_n_e: Reference (FDM) electron density [Nx, Nt] (optional)
        ref_phi: Reference (FDM) potential [Nx, Nt] (optional)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

    if time_indices is None:
        nt = len(t)
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    x_mm = x * 1e3
    t_us = t * 1e6

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    has_ref = ref_n_e is not None and ref_phi is not None

    # n_e profiles
    for i, idx in enumerate(time_indices):
        axes[0].plot(
            x_mm, n_e[:, idx],
            color=colors[i],
            label=f"PINN t={t_us[idx]:.2f}μs" if has_ref else f"t = {t_us[idx]:.2f} μs",
            linewidth=1.5,
            linestyle="-"
        )
        if has_ref:
            axes[0].plot(
                x_mm, ref_n_e[:, idx],
                color=colors[i],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7
            )
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Profiles" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # phi profiles
    for i, idx in enumerate(time_indices):
        axes[1].plot(
            x_mm, phi[:, idx],
            color=colors[i],
            label=f"PINN t={t_us[idx]:.2f}μs" if has_ref else f"t = {t_us[idx]:.2f} μs",
            linewidth=1.5,
            linestyle="-"
        )
        if has_ref:
            axes[1].plot(
                x_mm, ref_phi[:, idx],
                color=colors[i],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7
            )
    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Profiles" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved spatial profiles to {save_path}")

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

    Args:
        n_e: Electron density array [Nx, Nt]
        phi: Electric potential array [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        position_indices: Indices of positions to plot
        ref_n_e: Reference (FDM) electron density [Nx, Nt] (optional)
        ref_phi: Reference (FDM) potential [Nx, Nt] (optional)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

    if position_indices is None:
        nx = len(x)
        position_indices = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]

    x_mm = x * 1e3
    t_us = t * 1e6

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0, 1, len(position_indices)))

    has_ref = ref_n_e is not None and ref_phi is not None

    # n_e evolution
    for i, idx in enumerate(position_indices):
        axes[0].plot(
            t_us, n_e[idx, :],
            color=colors[i],
            label=f"PINN x={x_mm[idx]:.1f}mm" if has_ref else f"x = {x_mm[idx]:.1f} mm",
            linewidth=1.5,
            linestyle="-"
        )
        if has_ref:
            axes[0].plot(
                t_us, ref_n_e[idx, :],
                color=colors[i],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7
            )
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Evolution" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # phi evolution
    for i, idx in enumerate(position_indices):
        axes[1].plot(
            t_us, phi[idx, :],
            color=colors[i],
            label=f"PINN x={x_mm[idx]:.1f}mm" if has_ref else f"x = {x_mm[idx]:.1f} mm",
            linewidth=1.5,
            linestyle="-"
        )
        if has_ref:
            axes[1].plot(
                t_us, ref_phi[idx, :],
                color=colors[i],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7
            )
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Evolution" + (" (solid=PINN, dashed=FDM)" if has_ref else ""))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved temporal evolution to {save_path}")

    return fig


def plot_loss_curves(
    losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot training loss curves.

    Args:
        losses: Dictionary of loss names to values
        save_path: Path to save figure
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

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

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved loss curves to {save_path}")

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
) -> plt.Figure:
    """
    Plot 3x2 grid comparing PINN predictions with FDM reference.

    Layout:
        Row 1: Reconstructed n_e | Reconstructed phi
        Row 2: FDM n_e          | FDM phi
        Row 3: Error n_e        | Error phi

    Args:
        pred_n_e: Predicted electron density [Nx, Nt]
        pred_phi: Predicted potential [Nx, Nt]
        ref_n_e: Reference electron density [Nx, Nt]
        ref_phi: Reference potential [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

    x_mm = x * 1e3
    t_us = t * 1e6
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]

    # Compute error maps
    eps = 1e-10
    err_n_e = np.abs(pred_n_e - ref_n_e) / (np.abs(ref_n_e).max() + eps)
    err_phi = np.abs(pred_phi - ref_phi) / (np.abs(ref_phi).max() + eps)

    # Get colormap limits for consistent coloring
    n_e_vmin = min(pred_n_e.min(), ref_n_e.min())
    n_e_vmax = max(pred_n_e.max(), ref_n_e.max())
    phi_vmin = min(pred_phi.min(), ref_phi.min())
    phi_vmax = max(pred_phi.max(), ref_phi.max())

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Row 1: Reconstructed (PINN)
    im1 = axes[0, 0].imshow(
        pred_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=n_e_vmin, vmax=n_e_vmax
    )
    plt.colorbar(im1, ax=axes[0, 0], format="%.2e")
    axes[0, 0].set_title(r"Reconstructed $n_e$ (m$^{-3}$)")
    axes[0, 0].set_ylabel("Position (mm)")

    im2 = axes[0, 1].imshow(
        pred_phi, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=phi_vmin, vmax=phi_vmax
    )
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].set_title(r"Reconstructed $\phi$ (V)")

    # Row 2: Original (FDM)
    im3 = axes[1, 0].imshow(
        ref_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=n_e_vmin, vmax=n_e_vmax
    )
    plt.colorbar(im3, ax=axes[1, 0], format="%.2e")
    axes[1, 0].set_title(r"FDM $n_e$ (m$^{-3}$)")
    axes[1, 0].set_ylabel("Position (mm)")

    im4 = axes[1, 1].imshow(
        ref_phi, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=phi_vmin, vmax=phi_vmax
    )
    plt.colorbar(im4, ax=axes[1, 1])
    axes[1, 1].set_title(r"FDM $\phi$ (V)")

    # Row 3: Error
    im5 = axes[2, 0].imshow(
        err_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="hot"
    )
    plt.colorbar(im5, ax=axes[2, 0])
    axes[2, 0].set_title(r"Relative Error $n_e$")
    axes[2, 0].set_xlabel("Time (μs)")
    axes[2, 0].set_ylabel("Position (mm)")

    im6 = axes[2, 1].imshow(
        err_phi, extent=extent, aspect="auto", origin="lower",
        cmap="hot"
    )
    plt.colorbar(im6, ax=axes[2, 1])
    axes[2, 1].set_title(r"Relative Error $\phi$")
    axes[2, 1].set_xlabel("Time (μs)")

    # Add L2 error text
    l2_n_e = np.sqrt(np.mean((pred_n_e - ref_n_e)**2)) / (np.abs(ref_n_e).max() + eps)
    l2_phi = np.sqrt(np.mean((pred_phi - ref_phi)**2)) / (np.abs(ref_phi).max() + eps)
    fig.text(0.5, 0.01, f"Relative L2 Error: $n_e$ = {l2_n_e:.4f}, $\\phi$ = {l2_phi:.4f}",
             ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved comparison heatmaps to {save_path}")

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

    Args:
        pred_n_e: Predicted electron density [Nx, Nt]
        pred_phi: Predicted potential [Nx, Nt]
        ref_n_e: Reference electron density [Nx, Nt]
        ref_phi: Reference potential [Nx, Nt]
        x: Spatial coordinates [Nx]
        t_idx: Time index for comparison
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

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

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved comparison plot to {save_path}")

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
) -> plt.Figure:
    """
    Plot error maps between predicted and reference solutions.

    Args:
        pred_n_e: Predicted electron density [Nx, Nt]
        pred_phi: Predicted potential [Nx, Nt]
        ref_n_e: Reference electron density [Nx, Nt]
        ref_phi: Reference potential [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_matplotlib()

    # Compute relative errors
    eps = 1e-10
    err_n_e = np.abs(pred_n_e - ref_n_e) / (np.abs(ref_n_e) + eps)
    err_phi = np.abs(pred_phi - ref_phi) / (np.abs(ref_phi) + eps)

    x_mm = x * 1e3
    t_us = t * 1e6
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # n_e error
    im1 = axes[0].imshow(
        err_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="Reds", norm=mcolors.LogNorm(vmin=1e-4, vmax=1)
    )
    axes[0].set_title(r"Relative Error in $n_e$")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Position (mm)")
    plt.colorbar(im1, ax=axes[0])

    # phi error
    im2 = axes[1].imshow(
        err_phi, extent=extent, aspect="auto", origin="lower",
        cmap="Reds", norm=mcolors.LogNorm(vmin=1e-4, vmax=1)
    )
    axes[1].set_title(r"Relative Error in $\phi$")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Position (mm)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        _log(f"Saved error maps to {save_path}")

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

    Automatically loads FDM reference data based on model's physics parameters
    if available. Manual reference data can still be provided.

    Args:
        model: Trained PINN model (should have .params attribute)
        nx: Number of spatial points
        nt: Number of temporal points
        x_range: Spatial domain range
        t_range: Temporal domain range
        save_dir: Directory to save plots
        device: Device for model evaluation
        fdm_dir: Directory containing FDM reference data files
        ref_n_e: Reference (FDM) electron density [Nx, Nt] (optional, overrides auto-load)
        ref_phi: Reference (FDM) potential [Nx, Nt] (optional, overrides auto-load)
        ref_x: Reference spatial coordinates [Nx] (optional, overrides auto-load)
        ref_t: Reference time coordinates [Nt] (optional, overrides auto-load)

    Returns:
        Dictionary with n_e, phi, x, t arrays
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
        # Normalize reference coordinates for model input
        x_norm = ref_x / ref_x.max() if ref_x.max() > 0 else ref_x
        t_norm = ref_t / ref_t.max() if ref_t.max() > 0 else ref_t
        x = torch.tensor(x_norm, dtype=torch.float32, device=device)
        t = torch.tensor(t_norm, dtype=torch.float32, device=device)
        nx, nt = len(ref_x), len(ref_t)
    else:
        x = torch.linspace(x_range[0], x_range[1], nx, device=device)
        t = torch.linspace(t_range[0], t_range[1], nt, device=device)

    X, T = torch.meshgrid(x, t, indexing="ij")
    x_t = torch.stack([X.flatten(), T.flatten()], dim=1)
    _log(f"Evaluating model on grid of {x_t.shape[0]} points...")

    # Evaluate model
    with torch.no_grad():
        n_e, phi = model(x_t)
    _log("Model evaluation complete")

    # Reshape to grid
    n_e = n_e.view(nx, nt).cpu().numpy()
    phi = phi.view(nx, nt).cpu().numpy()
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()

    # Scale predictions to physical units if reference is available
    if has_ref:
        n_e = n_e * ref_n_e.max()
        phi = phi * ref_phi.max()
        x_np = ref_x
        t_np = ref_t

    # Create plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Generating plots in {save_dir}")

        if has_ref:
            _log("Generating comparison plots with FDM reference...")
            # 3x2 comparison heatmaps
            plot_comparison_heatmaps(
                n_e, phi, ref_n_e, ref_phi, x_np, t_np,
                save_path=str(save_dir / "comparison_heatmaps.png")
            )
            # Spatial profiles with FDM comparison
            plot_spatial_profiles(
                n_e, phi, x_np, t_np,
                ref_n_e=ref_n_e, ref_phi=ref_phi,
                save_path=str(save_dir / "spatial_profiles.png")
            )
            # Temporal evolution with FDM comparison
            plot_temporal_evolution(
                n_e, phi, x_np, t_np,
                ref_n_e=ref_n_e, ref_phi=ref_phi,
                save_path=str(save_dir / "temporal_evolution.png")
            )
            # Error maps
            plot_error_maps(
                n_e, phi, ref_n_e, ref_phi, x_np, t_np,
                save_path=str(save_dir / "error_maps.png")
            )
        else:
            _log("Generating plots without FDM reference...")
            # Original plots without comparison
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
