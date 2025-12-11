"""
Animation generation utilities for PINN visualization.

Creates GIF animations of solution evolution over time.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch


def create_solution_gif(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (12, 5),
) -> str:
    """
    Create animated GIF of solution evolution.

    Args:
        n_e: Electron density array [Nx, Nt]
        phi: Electric potential array [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save GIF
        fps: Frames per second
        dpi: Resolution
        figsize: Figure size

    Returns:
        Path to saved GIF
    """
    x_mm = x * 1e3
    t_us = t * 1e6

    # Determine y-axis limits
    n_e_min, n_e_max = np.min(n_e), np.max(n_e)
    phi_min, phi_max = np.min(phi), np.max(phi)

    # Add some padding
    n_e_pad = (n_e_max - n_e_min) * 0.1
    phi_pad = (phi_max - phi_min) * 0.1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Initialize plots
    line_ne, = axes[0].plot(x_mm, n_e[:, 0], "b-", linewidth=2)
    axes[0].set_xlim(x_mm[0], x_mm[-1])
    axes[0].set_ylim(n_e_min - n_e_pad, n_e_max + n_e_pad)
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0].set_title("Electron Density")
    axes[0].grid(True, alpha=0.3)

    line_phi, = axes[1].plot(x_mm, phi[:, 0], "r-", linewidth=2)
    axes[1].set_xlim(x_mm[0], x_mm[-1])
    axes[1].set_ylim(phi_min - phi_pad, phi_max + phi_pad)
    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel(r"$\phi$ (V)")
    axes[1].set_title("Electric Potential")
    axes[1].grid(True, alpha=0.3)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(frame):
        line_ne.set_ydata(n_e[:, frame])
        line_phi.set_ydata(phi[:, frame])
        time_text.set_text(f"t = {t_us[frame]:.3f} μs")
        return line_ne, line_phi, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t),
        interval=1000 // fps, blit=True
    )

    # Save GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)

    return str(save_path)


def create_heatmap_gif(
    data: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    title: str = "Solution",
    cmap: str = "viridis",
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (8, 6),
) -> str:
    """
    Create animated GIF of heatmap evolution.

    Shows vertical slice moving through time.

    Args:
        data: Data array [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save GIF
        title: Plot title
        cmap: Colormap
        fps: Frames per second
        dpi: Resolution
        figsize: Figure size

    Returns:
        Path to saved GIF
    """
    x_mm = x * 1e3
    t_us = t * 1e6

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]
    im = ax.imshow(
        data, extent=extent, aspect="auto",
        origin="lower", cmap=cmap
    )
    plt.colorbar(im, ax=ax)

    # Create vertical line indicator
    vline = ax.axvline(x=t_us[0], color="white", linewidth=2, linestyle="--")

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Position (mm)")
    ax.set_title(title)

    def update(frame):
        vline.set_xdata([t_us[frame], t_us[frame]])
        return vline,

    ani = animation.FuncAnimation(
        fig, update, frames=len(t),
        interval=1000 // fps, blit=True
    )

    # Save GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)

    return str(save_path)


def create_dual_heatmap_gif(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (14, 5),
) -> str:
    """
    Create animated GIF showing both n_e and phi heatmaps.

    Args:
        n_e: Electron density [Nx, Nt]
        phi: Electric potential [Nx, Nt]
        x: Spatial coordinates [Nx]
        t: Time coordinates [Nt]
        save_path: Path to save GIF
        fps: Frames per second
        dpi: Resolution
        figsize: Figure size

    Returns:
        Path to saved GIF
    """
    x_mm = x * 1e3
    t_us = t * 1e6
    extent = [t_us[0], t_us[-1], x_mm[0], x_mm[-1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # n_e heatmap
    im1 = axes[0].imshow(
        n_e, extent=extent, aspect="auto",
        origin="lower", cmap="viridis"
    )
    plt.colorbar(im1, ax=axes[0], format="%.2e")
    vline1 = axes[0].axvline(x=t_us[0], color="white", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_title(r"Electron Density $n_e$ (m$^{-3}$)")

    # phi heatmap
    im2 = axes[1].imshow(
        phi, extent=extent, aspect="auto",
        origin="lower", cmap="plasma"
    )
    plt.colorbar(im2, ax=axes[1])
    vline2 = axes[1].axvline(x=t_us[0], color="white", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Position (mm)")
    axes[1].set_title(r"Electric Potential $\phi$ (V)")

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(frame):
        vline1.set_xdata([t_us[frame], t_us[frame]])
        vline2.set_xdata([t_us[frame], t_us[frame]])
        time_text.set_text(f"t = {t_us[frame]:.3f} μs")
        return vline1, vline2, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t),
        interval=1000 // fps, blit=True
    )

    # Save GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)

    return str(save_path)


def generate_model_animation(
    model: torch.nn.Module,
    save_path: str,
    nx: int = 100,
    nt: int = 100,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    fps: int = 30,
    device: str = "cpu",
) -> str:
    """
    Generate animation from a trained PINN model.

    Args:
        model: Trained PINN model
        save_path: Path to save GIF
        nx: Number of spatial points
        nt: Number of temporal points
        x_range: Spatial domain range
        t_range: Temporal domain range
        fps: Frames per second
        device: Device for model evaluation

    Returns:
        Path to saved GIF
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

    # Reshape to grid and convert to numpy
    n_e = n_e.view(nx, nt).cpu().numpy()
    phi = phi.view(nx, nt).cpu().numpy()
    x = x.cpu().numpy()
    t = t.cpu().numpy()

    # Create animation
    return create_solution_gif(n_e, phi, x, t, save_path, fps=fps)
