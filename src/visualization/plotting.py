"""
Static plotting utilities for PINN visualization.

Provides functions for visualizing:
- Solution fields (n_e, phi) as heatmaps
- Spatial/temporal profiles
- Loss curves and training metrics
- Comparison plots with reference data
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


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

    return fig


def plot_spatial_profiles(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    time_indices: Optional[List[int]] = None,
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

    # n_e profiles
    for i, idx in enumerate(time_indices):
        axes[0].plot(
            x_mm, n_e[:, idx],
            color=colors[i],
            label=f"t = {t_us[idx]:.2f} μs",
            linewidth=1.5
        )
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Profiles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # phi profiles
    for i, idx in enumerate(time_indices):
        axes[1].plot(
            x_mm, phi[:, idx],
            color=colors[i],
            label=f"t = {t_us[idx]:.2f} μs",
            linewidth=1.5
        )
    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Profiles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return fig


def plot_temporal_evolution(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    position_indices: Optional[List[int]] = None,
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

    # n_e evolution
    for i, idx in enumerate(position_indices):
        axes[0].plot(
            t_us, n_e[idx, :],
            color=colors[i],
            label=f"x = {x_mm[idx]:.1f} mm",
            linewidth=1.5
        )
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density Evolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # phi evolution
    for i, idx in enumerate(position_indices):
        axes[1].plot(
            t_us, phi[idx, :],
            color=colors[i],
            label=f"x = {x_mm[idx]:.1f} mm",
            linewidth=1.5
        )
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential Evolution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

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

    return fig


def visualize_model(
    model: torch.nn.Module,
    nx: int = 100,
    nt: int = 100,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    save_dir: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Visualize a trained PINN model.

    Args:
        model: Trained PINN model
        nx: Number of spatial points
        nt: Number of temporal points
        x_range: Spatial domain range
        t_range: Temporal domain range
        save_dir: Directory to save plots
        device: Device for model evaluation

    Returns:
        Dictionary with n_e, phi, x, t arrays
    """
    model.eval()
    model.to(device)

    # Create evaluation grid
    x = torch.linspace(x_range[0], x_range[1], nx, device=device)
    t = torch.linspace(t_range[0], t_range[1], nt, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x_t = torch.stack([X.flatten(), T.flatten()], dim=1)

    # Evaluate model
    with torch.no_grad():
        n_e, phi = model(x_t)

    # Reshape to grid
    n_e = n_e.view(nx, nt).cpu().numpy()
    phi = phi.view(nx, nt).cpu().numpy()
    x = x.cpu().numpy()
    t = t.cpu().numpy()

    # Create plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_solution_heatmaps(
            n_e, phi, x, t,
            save_path=str(save_dir / "solution_heatmaps.png")
        )
        plot_spatial_profiles(
            n_e, phi, x, t,
            save_path=str(save_dir / "spatial_profiles.png")
        )
        plot_temporal_evolution(
            n_e, phi, x, t,
            save_path=str(save_dir / "temporal_evolution.png")
        )

    return {"n_e": n_e, "phi": phi, "x": x, "t": t}
