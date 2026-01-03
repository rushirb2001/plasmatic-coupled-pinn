"""
Animation generation utilities for PINN visualization.

Creates GIF/MP4 animations of solution evolution over time.
Supports comparison with FDM reference data.

Optimizations:
- Direct frame rendering with imageio (bypasses slow matplotlib animation)
- Automatic frame limiting (max 300 frames)
- tqdm progress bars for real-time feedback
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.animation as animation
mplstyle.use('fast')  # Disable anti-aliasing for speed
import numpy as np
import torch
from tqdm import tqdm

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from src.data.fdm_solver import get_fdm_for_visualization


def _fig_to_array(fig) -> np.ndarray:
    """Convert matplotlib figure to numpy array (fast)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)


def _render_frames_fast(
    fig,
    update_func,
    frame_indices: np.ndarray,
    desc: str = "Rendering",
) -> List[np.ndarray]:
    """Render frames by updating figure and capturing to array.

    This bypasses matplotlib's slow animation.save() entirely.
    ~10-50x faster than FuncAnimation.save().
    """
    frames = []
    for i, frame_idx in enumerate(tqdm(frame_indices, desc=desc, unit="frame")):
        update_func(frame_idx)
        frame = _fig_to_array(fig).copy()
        frames.append(frame)
    return frames


def _save_frames_as_video(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 30,
) -> str:
    """Save frames as MP4 or GIF using imageio/ffmpeg."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_IMAGEIO:
        # Use imageio for fast saving
        if save_path.suffix.lower() == ".gif":
            # For GIF, reduce fps for smaller file
            imageio.mimwrite(str(save_path), frames, fps=min(fps, 15), loop=0)
        else:
            # For MP4, use ffmpeg backend
            mp4_path = save_path.with_suffix(".mp4")
            imageio.mimwrite(str(mp4_path), frames, fps=fps, codec="libx264",
                           output_params=["-crf", "23", "-pix_fmt", "yuv420p"])
            if save_path.suffix.lower() == ".gif":
                _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
                mp4_path.unlink()
            else:
                return str(mp4_path)
    else:
        # Fallback: save as individual PNGs, then use ffmpeg
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(tqdm(frames, desc="Saving PNGs")):
                plt.imsave(f"{tmpdir}/frame_{i:05d}.png", frame)
            # Use ffmpeg to create video
            mp4_path = save_path.with_suffix(".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                str(mp4_path)
            ], capture_output=True)
            if save_path.suffix.lower() == ".gif":
                _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
                mp4_path.unlink()
                return str(save_path)
            return str(mp4_path)

    return str(save_path)


class TqdmProgressCallback:
    """tqdm-based progress callback for animation saving."""

    def __init__(self, total_frames: int, desc: str = "Saving GIF"):
        self.pbar = tqdm(total=total_frames, desc=desc, unit="frame",
                         ncols=80, leave=True)
        self._last_frame = 0

    def __call__(self, current_frame: int, total_frames: int):
        """Called by animation writer for each frame."""
        # Update by the difference since last call
        delta = current_frame - self._last_frame
        if delta > 0:
            self.pbar.update(delta)
        self._last_frame = current_frame

        # Close on completion
        if current_frame >= total_frames - 1:
            self.pbar.close()


def _get_writer(fps: int, output_format: str = "mp4"):
    """Get the best available animation writer.

    Args:
        fps: Frames per second
        output_format: 'mp4' (fast, recommended) or 'gif' (slow)

    MP4 with H.264 is ~50-100x faster than GIF encoding.
    """
    if shutil.which("ffmpeg"):
        if output_format == "gif":
            return animation.FFMpegWriter(fps=fps, codec="gif")
        else:
            # H.264 is extremely fast and produces small files
            return animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "23"]
            )
    return animation.PillowWriter(fps=fps)


def _convert_to_gif(mp4_path: str, gif_path: str, fps: int = 15) -> str:
    """Convert MP4 to GIF using ffmpeg (much faster than direct GIF encoding).

    Uses palette generation for better quality.
    """
    import subprocess

    # Generate palette for better GIF quality
    palette_cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", f"fps={fps},scale=640:-1:flags=lanczos,palettegen",
        "-t", "10",  # Sample first 10s for palette
        "/tmp/palette.png"
    ]

    # Create GIF using palette
    gif_cmd = [
        "ffmpeg", "-y", "-i", mp4_path, "-i", "/tmp/palette.png",
        "-lavfi", f"fps={fps},scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
        gif_path
    ]

    subprocess.run(palette_cmd, capture_output=True)
    subprocess.run(gif_cmd, capture_output=True)

    return gif_path


def _compute_frame_indices(
    total_frames: int,
    skip_frames: int = 1,
    max_frames: int = 300,
) -> np.ndarray:
    """Compute frame indices with automatic skipping for reasonable GIF size.

    Args:
        total_frames: Total number of frames in data
        skip_frames: Manual skip interval (1=no skip)
        max_frames: Maximum frames in output GIF (default 300 = 10s @ 30fps)

    Returns:
        Array of frame indices to use
    """
    # Auto-calculate skip to stay under max_frames
    auto_skip = max(1, total_frames // max_frames)
    effective_skip = max(skip_frames, auto_skip)

    indices = np.arange(0, total_frames, effective_skip)

    if len(indices) > max_frames:
        indices = indices[:max_frames]

    return indices


def create_solution_gif(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (12, 5),
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    # Compute frame indices with automatic limiting
    total_frames = len(t)
    frame_indices = _compute_frame_indices(total_frames, skip_frames)
    n_frames = len(frame_indices)

    if n_frames < total_frames:
        print(f"Creating solution GIF: {n_frames} frames (reduced from {total_frames}) @ {fps} fps")
    else:
        print(f"Creating solution GIF: {n_frames} frames @ {fps} fps")

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

    def update(frame_idx):
        actual_frame = frame_indices[frame_idx]
        line_ne.set_ydata(n_e[:, actual_frame])
        line_phi.set_ydata(phi[:, actual_frame])
        time_text.set_text(f"t = {t_us[actual_frame]:.3f} μs")
        return line_ne, line_phi, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    # Save as MP4 first (fast), then convert to GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = save_path.with_suffix(".mp4")
    writer = _get_writer(fps, output_format="mp4")
    progress = TqdmProgressCallback(n_frames, "Rendering MP4")
    ani.save(str(mp4_path), writer=writer, dpi=dpi, progress_callback=progress)
    plt.close(fig)

    if save_path.suffix.lower() == ".gif":
        print("Converting MP4 to GIF...")
        _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
        mp4_path.unlink()
        return str(save_path)
    else:
        return str(mp4_path)


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
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    total_frames = len(t)
    frame_indices = _compute_frame_indices(total_frames, skip_frames)
    n_frames = len(frame_indices)

    if n_frames < total_frames:
        print(f"Creating heatmap GIF: {n_frames} frames (reduced from {total_frames}) @ {fps} fps")
    else:
        print(f"Creating heatmap GIF: {n_frames} frames @ {fps} fps")

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

    def update(frame_idx):
        actual_frame = frame_indices[frame_idx]
        vline.set_xdata([t_us[actual_frame], t_us[actual_frame]])
        return vline,

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    # Save as MP4 first (fast), then convert to GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = save_path.with_suffix(".mp4")
    writer = _get_writer(fps, output_format="mp4")
    progress = TqdmProgressCallback(n_frames, "Rendering MP4")
    ani.save(str(mp4_path), writer=writer, dpi=dpi, progress_callback=progress)
    plt.close(fig)

    if save_path.suffix.lower() == ".gif":
        print("Converting MP4 to GIF...")
        _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
        mp4_path.unlink()
        return str(save_path)
    else:
        return str(mp4_path)


def create_dual_heatmap_gif(
    n_e: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_path: str,
    fps: int = 30,
    dpi: int = 100,
    figsize: Tuple[int, int] = (14, 5),
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    total_frames = len(t)
    frame_indices = _compute_frame_indices(total_frames, skip_frames)
    n_frames = len(frame_indices)

    if n_frames < total_frames:
        print(f"Creating dual heatmap GIF: {n_frames} frames (reduced from {total_frames}) @ {fps} fps")
    else:
        print(f"Creating dual heatmap GIF: {n_frames} frames @ {fps} fps")

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

    def update(frame_idx):
        actual_frame = frame_indices[frame_idx]
        vline1.set_xdata([t_us[actual_frame], t_us[actual_frame]])
        vline2.set_xdata([t_us[actual_frame], t_us[actual_frame]])
        time_text.set_text(f"t = {t_us[actual_frame]:.3f} μs")
        return vline1, vline2, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    # Save as MP4 first (fast), then convert to GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = save_path.with_suffix(".mp4")
    writer = _get_writer(fps, output_format="mp4")
    progress = TqdmProgressCallback(n_frames, "Rendering MP4")
    ani.save(str(mp4_path), writer=writer, dpi=dpi, progress_callback=progress)
    plt.close(fig)

    if save_path.suffix.lower() == ".gif":
        print("Converting MP4 to GIF...")
        _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
        mp4_path.unlink()
        return str(save_path)
    else:
        return str(mp4_path)


def generate_model_animation(
    model: torch.nn.Module,
    save_path: str,
    nx: int = 100,
    nt: int = 100,
    x_range: Tuple[float, float] = (0.0, 1.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    fps: int = 30,
    device: str = "cpu",
    fdm_dir: str = "data/fdm",
    ref_n_e: Optional[np.ndarray] = None,
    ref_phi: Optional[np.ndarray] = None,
    ref_x: Optional[np.ndarray] = None,
    ref_t: Optional[np.ndarray] = None,
) -> str:
    """
    Generate animation from a trained PINN model.

    Automatically loads FDM reference data based on model's physics parameters
    if available. If FDM data exists, generates a 3x2 comparison GIF.
    Otherwise, generates a simple solution evolution GIF.

    Args:
        model: Trained PINN model (should have .params attribute)
        save_path: Path to save GIF
        nx: Number of spatial points
        nt: Number of temporal points
        x_range: Spatial domain range
        t_range: Temporal domain range
        fps: Frames per second
        device: Device for model evaluation
        fdm_dir: Directory containing FDM reference data files
        ref_n_e: Optional reference electron density [Nx, Nt] (overrides auto-load)
        ref_phi: Optional reference electric potential [Nx, Nt] (overrides auto-load)
        ref_x: Optional reference spatial coordinates [Nx] (overrides auto-load)
        ref_t: Optional reference time coordinates [Nt] (overrides auto-load)

    Returns:
        Path to saved GIF
    """
    model.eval()
    model.to(device)

    # Auto-load FDM reference data if model has physics params and no manual ref provided
    if ref_n_e is None and hasattr(model, 'params'):
        fdm_data = get_fdm_for_visualization(model.params, fdm_dir=fdm_dir)
        if fdm_data is not None:
            ref_n_e, ref_phi, ref_x, ref_t = fdm_data
            print(f"Loaded FDM reference data for animation: {model.params.get_fdm_filename()}")

    # Check if FDM reference data is available
    has_reference = all(v is not None for v in [ref_n_e, ref_phi, ref_x, ref_t])

    if has_reference:
        # Use generate_comparison_animation for FDM comparison
        return generate_comparison_animation(
            model=model,
            ref_n_e=ref_n_e,
            ref_phi=ref_phi,
            ref_x=ref_x,
            ref_t=ref_t,
            save_path=save_path,
            fps=fps,
            device=device,
            animation_type="heatmap",
        )
    else:
        # Original behavior without reference data
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
    dpi: int = 150,
    figsize: Tuple[float, float] = (10, 8),
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    total_frames = len(t)
    frame_indices = _compute_frame_indices(total_frames, skip_frames)
    n_frames = len(frame_indices)

    if n_frames < total_frames:
        print(f"Creating comparison heatmap GIF (3x2 grid): {n_frames} frames (reduced from {total_frames}) @ {fps} fps")
    else:
        print(f"Creating comparison heatmap GIF (3x2 grid): {n_frames} frames @ {fps} fps")

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

    def update(frame_idx):
        actual_frame = frame_indices[frame_idx]
        for vline in vlines:
            vline.set_xdata([t_us[actual_frame], t_us[actual_frame]])
        time_text.set_text(f"t = {t_us[actual_frame]:.3f} μs")
        return vlines + [time_text]

    # Use tqdm on frames for progress, disable internal callback
    frames_iter = tqdm(range(n_frames), desc="Rendering", unit="frame", ncols=80)
    ani = animation.FuncAnimation(
        fig, update, frames=frames_iter,
        interval=1000 // fps, blit=True
    )

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Use pillow for GIF (simpler), reduce DPI for speed
    effective_dpi = min(dpi, 100)
    ani.save(str(save_path), writer="pillow", fps=fps, dpi=effective_dpi,
             progress_callback=lambda i, n: None)
    plt.close(fig)

    print(f"  Saved: {save_path}")
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
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    total_frames = len(t)
    frame_indices = _compute_frame_indices(total_frames, skip_frames)
    n_frames = len(frame_indices)

    if n_frames < total_frames:
        print(f"Creating comparison profile GIF (2x2 grid): {n_frames} frames (reduced from {total_frames}) @ {fps} fps")
    else:
        print(f"Creating comparison profile GIF (2x2 grid): {n_frames} frames @ {fps} fps")

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

    def update(frame_idx):
        actual_frame = frame_indices[frame_idx]
        line_pred_ne.set_ydata(pred_n_e[:, actual_frame])
        line_ref_ne.set_ydata(ref_n_e[:, actual_frame])
        line_pred_phi.set_ydata(pred_phi[:, actual_frame])
        line_ref_phi.set_ydata(ref_phi[:, actual_frame])
        line_err_ne.set_ydata(err_n_e[:, actual_frame])
        line_err_phi.set_ydata(err_phi[:, actual_frame])
        time_text.set_text(f"t = {t_us[actual_frame]:.3f} μs")
        return line_pred_ne, line_ref_ne, line_pred_phi, line_ref_phi, line_err_ne, line_err_phi, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True
    )

    # Save as MP4 first (fast), then convert to GIF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = save_path.with_suffix(".mp4")
    writer = _get_writer(fps, output_format="mp4")
    progress = TqdmProgressCallback(n_frames, "Rendering MP4")
    ani.save(str(mp4_path), writer=writer, dpi=dpi, progress_callback=progress)
    plt.close(fig)

    if save_path.suffix.lower() == ".gif":
        print("Converting MP4 to GIF...")
        _convert_to_gif(str(mp4_path), str(save_path), fps=min(fps, 15))
        mp4_path.unlink()
        return str(save_path)
    else:
        return str(mp4_path)


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
    skip_frames: int = 1,
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
        skip_frames: Skip every N frames for faster generation (1=no skip)

    Returns:
        Path to saved GIF
    """
    print(f"Generating comparison animation ({animation_type})...")

    model.eval()
    model.to(device)

    nx, nt = len(ref_x), len(ref_t)

    print(f"  Evaluating model on {nx}x{nt} grid...")

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

    print("  Model evaluation complete. Creating animation...")

    # Create animation
    if animation_type == "heatmap":
        return create_comparison_heatmap_gif(
            pred_n_e, pred_phi, ref_n_e, ref_phi,
            ref_x, ref_t, save_path, fps=fps, skip_frames=skip_frames
        )
    else:
        return create_comparison_profile_gif(
            pred_n_e, pred_phi, ref_n_e, ref_phi,
            ref_x, ref_t, save_path, fps=fps, skip_frames=skip_frames
        )
