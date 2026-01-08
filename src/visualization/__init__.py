"""
Visualization module for PINN models.

Provides utilities for:
- Static plots: heatmaps, profiles, loss curves
- FDM comparison: 3x2 comparison grids, error plots
"""

from .plotting import (
    setup_matplotlib,
    compute_errors,
    plot_solution_heatmaps,
    plot_spatial_profiles,
    plot_temporal_evolution,
    plot_loss_curves,
    plot_comparison,
    plot_comparison_heatmaps,
    plot_error_maps,
    visualize_model,
)

__all__ = [
    # Plotting
    "setup_matplotlib",
    "compute_errors",
    "plot_solution_heatmaps",
    "plot_spatial_profiles",
    "plot_temporal_evolution",
    "plot_loss_curves",
    "plot_comparison",
    "plot_comparison_heatmaps",
    "plot_error_maps",
    "visualize_model",
]
