"""
Animation generation utilities for PINN visualization.

Creates GIF animations of solution evolution over time.
Supports comparison with FDM reference data.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def create_comparison_heatmap_gif(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (14, 12),
) -> str:
    """
    Create animated GIF with 3x2 grid comparing PINN vs FDM.

    Layout:
        Row 1: Reconstructed n_e | Reconstructed phi
        Row 2: FDM n_e          | FDM phi
        Row 3: Error n_e        | Error phi

    Args:
        pred_n_e: Predicted electron density [Nx, Nt]
        pred_phi: Predicted electric potential [Nx, Nt]
        ref_n_e: Reference (FDM) electron density [Nx, Nt]
        ref_phi: Reference (FDM) electric potential [Nx, Nt]
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
    im_pred_ne = axes[0, 0].imshow(
        pred_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=n_e_vmin, vmax=n_e_vmax
    )
    plt.colorbar(im_pred_ne, ax=axes[0, 0], format="%.2e")
    axes[0, 0].set_title(r"Reconstructed $n_e$ (m$^{-3}$)")
    axes[0, 0].set_ylabel("Position (mm)")

    im_pred_phi = axes[0, 1].imshow(
        pred_phi, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=phi_vmin, vmax=phi_vmax
    )
    plt.colorbar(im_pred_phi, ax=axes[0, 1])
    axes[0, 1].set_title(r"Reconstructed $\phi$ (V)")

    # Row 2: Original (FDM)
    im_ref_ne = axes[1, 0].imshow(
        ref_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=n_e_vmin, vmax=n_e_vmax
    )
    plt.colorbar(im_ref_ne, ax=axes[1, 0], format="%.2e")
    axes[1, 0].set_title(r"FDM $n_e$ (m$^{-3}$)")
    axes[1, 0].set_ylabel("Position (mm)")

    im_ref_phi = axes[1, 1].imshow(
        ref_phi, extent=extent, aspect="auto", origin="lower",
        cmap="rainbow", vmin=phi_vmin, vmax=phi_vmax
    )
    plt.colorbar(im_ref_phi, ax=axes[1, 1])
    axes[1, 1].set_title(r"FDM $\phi$ (V)")

    # Row 3: Error
    im_err_ne = axes[2, 0].imshow(
        err_n_e, extent=extent, aspect="auto", origin="lower",
        cmap="hot", vmin=0, vmax=err_n_e.max()
    )
    plt.colorbar(im_err_ne, ax=axes[2, 0])
    axes[2, 0].set_title(r"Relative Error $n_e$")
    axes[2, 0].set_xlabel("Time (μs)")
    axes[2, 0].set_ylabel("Position (mm)")

    im_err_phi = axes[2, 1].imshow(
        err_phi, extent=extent, aspect="auto", origin="lower",
        cmap="hot", vmin=0, vmax=err_phi.max()
    )
    plt.colorbar(im_err_phi, ax=axes[2, 1])
    axes[2, 1].set_title(r"Relative Error $\phi$")
    axes[2, 1].set_xlabel("Time (μs)")

    # Create vertical time indicator lines
    vlines = []
    for ax in axes.flatten():
        vline = ax.axvline(x=t_us[0], color="white", linewidth=2, linestyle="--")
        vlines.append(vline)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(frame):
        for vline in vlines:
            vline.set_xdata([t_us[frame], t_us[frame]])
        time_text.set_text(f"t = {t_us[frame]:.3f} μs")
        return vlines + [time_text]

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


def create_comparison_profile_gif(
    pred_n_e: np.ndarray,
    pred_phi: np.ndarray,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """
    Create animated GIF showing spatial profiles with PINN vs FDM comparison.

    Layout:
        Row 1: n_e profiles (PINN vs FDM) | phi profiles (PINN vs FDM)
        Row 2: n_e error profile          | phi error profile

    Args:
        pred_n_e: Predicted electron density [Nx, Nt]
        pred_phi: Predicted electric potential [Nx, Nt]
        ref_n_e: Reference (FDM) electron density [Nx, Nt]
        ref_phi: Reference (FDM) electric potential [Nx, Nt]
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

    # Compute error
    eps = 1e-10
    err_n_e = np.abs(pred_n_e - ref_n_e) / (np.abs(ref_n_e).max() + eps)
    err_phi = np.abs(pred_phi - ref_phi) / (np.abs(ref_phi).max() + eps)

    # Get y-axis limits
    n_e_min = min(pred_n_e.min(), ref_n_e.min())
    n_e_max = max(pred_n_e.max(), ref_n_e.max())
    phi_min = min(pred_phi.min(), ref_phi.min())
    phi_max = max(pred_phi.max(), ref_phi.max())
    err_n_e_max = err_n_e.max()
    err_phi_max = err_phi.max()

    n_e_pad = (n_e_max - n_e_min) * 0.1
    phi_pad = (phi_max - phi_min) * 0.1

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Row 1: n_e comparison
    line_pred_ne, = axes[0, 0].plot(x_mm, pred_n_e[:, 0], "r-", linewidth=2, label="PINN")
    line_ref_ne, = axes[0, 0].plot(x_mm, ref_n_e[:, 0], "k--", linewidth=2, label="FDM")
    axes[0, 0].set_xlim(x_mm[0], x_mm[-1])
    axes[0, 0].set_ylim(n_e_min - n_e_pad, n_e_max + n_e_pad)
    axes[0, 0].set_xlabel("Position (mm)")
    axes[0, 0].set_ylabel(r"$n_e$ (m$^{-3}$)")
    axes[0, 0].set_title("Electron Density")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1: phi comparison
    line_pred_phi, = axes[0, 1].plot(x_mm, pred_phi[:, 0], "r-", linewidth=2, label="PINN")
    line_ref_phi, = axes[0, 1].plot(x_mm, ref_phi[:, 0], "k--", linewidth=2, label="FDM")
    axes[0, 1].set_xlim(x_mm[0], x_mm[-1])
    axes[0, 1].set_ylim(phi_min - phi_pad, phi_max + phi_pad)
    axes[0, 1].set_xlabel("Position (mm)")
    axes[0, 1].set_ylabel(r"$\phi$ (V)")
    axes[0, 1].set_title("Electric Potential")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: n_e error profile
    line_err_ne, = axes[1, 0].plot(x_mm, err_n_e[:, 0], "b-", linewidth=2)
    axes[1, 0].set_xlim(x_mm[0], x_mm[-1])
    axes[1, 0].set_ylim(0, err_n_e_max * 1.1)
    axes[1, 0].set_xlabel("Position (mm)")
    axes[1, 0].set_ylabel("Relative Error")
    axes[1, 0].set_title(r"Error in $n_e$")
    axes[1, 0].grid(True, alpha=0.3)

    # Row 2: phi error profile
    line_err_phi, = axes[1, 1].plot(x_mm, err_phi[:, 0], "b-", linewidth=2)
    axes[1, 1].set_xlim(x_mm[0], x_mm[-1])
    axes[1, 1].set_ylim(0, err_phi_max * 1.1)
    axes[1, 1].set_xlabel("Position (mm)")
    axes[1, 1].set_ylabel("Relative Error")
    axes[1, 1].set_title(r"Error in $\phi$")
    axes[1, 1].grid(True, alpha=0.3)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(frame):
        line_pred_ne.set_ydata(pred_n_e[:, frame])
        line_ref_ne.set_ydata(ref_n_e[:, frame])
        line_pred_phi.set_ydata(pred_phi[:, frame])
        line_ref_phi.set_ydata(ref_phi[:, frame])
        line_err_ne.set_ydata(err_n_e[:, frame])
        line_err_phi.set_ydata(err_phi[:, frame])
        time_text.set_text(f"t = {t_us[frame]:.3f} μs")
        return line_pred_ne, line_ref_ne, line_pred_phi, line_ref_phi, line_err_ne, line_err_phi, time_text

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


def generate_comparison_animation(
    model: torch.nn.Module,
    ref_n_e: np.ndarray,
    ref_phi: np.ndarray,
    ref_x: np.ndarray,
    ref_t: np.ndarray,
    save_path: str,
    fps: int = 30,
    device: str = "cpu",
    animation_type: str = "heatmap",
) -> str:
    """
    Generate comparison animation from a trained PINN model vs FDM reference.

    Args:
        model: Trained PINN model
        ref_n_e: Reference electron density [Nx, Nt] (physical units)
        ref_phi: Reference potential [Nx, Nt] (physical units)
        ref_x: Reference spatial coordinates [Nx] (physical units, m)
        ref_t: Reference time coordinates [Nt] (physical units, s)
        save_path: Path to save GIF
        fps: Frames per second
        device: Device for model evaluation
        animation_type: "heatmap" for 3x2 grid, "profile" for spatial profiles

    Returns:
        Path to saved GIF
    """
    model.eval()
    model.to(device)

    nx, nt = len(ref_x), len(ref_t)

    # Normalize coordinates for model input
    x_norm = ref_x / ref_x.max() if ref_x.max() > 0 else ref_x
    t_norm = ref_t / ref_t.max() if ref_t.max() > 0 else ref_t

    # Create evaluation grid
    x = torch.tensor(x_norm, dtype=torch.float32, device=device)
    t = torch.tensor(t_norm, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x_t = torch.stack([X.flatten(), T.flatten()], dim=1)

    # Evaluate model (outputs are normalized)
    with torch.no_grad():
        pred_n_e, pred_phi = model(x_t)

    # Reshape to grid
    pred_n_e = pred_n_e.view(nx, nt).cpu().numpy()
    pred_phi = pred_phi.view(nx, nt).cpu().numpy()

    # Scale to physical units (approximate - should match training scaling)
    # For now, use reference data range to scale predictions
    pred_n_e = pred_n_e * ref_n_e.max()
    pred_phi = pred_phi * ref_phi.max()

    # Create animation
    if animation_type == "heatmap":
        return create_comparison_heatmap_gif(
            pred_n_e, pred_phi, ref_n_e, ref_phi,
            ref_x, ref_t, save_path, fps=fps
        )
    else:
        return create_comparison_profile_gif(
            pred_n_e, pred_phi, ref_n_e, ref_phi,
            ref_x, ref_t, save_path, fps=fps
        )
