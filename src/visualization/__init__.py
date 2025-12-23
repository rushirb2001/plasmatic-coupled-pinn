"""
Visualization module for PINN models.

Provides utilities for:
- Static plots: heatmaps, profiles, loss curves
- Animations: solution evolution GIFs
- FDM comparison: 3x2 comparison grids, error plots
"""

from .plotting import (
    setup_matplotlib,
    plot_solution_heatmaps,
    plot_spatial_profiles,
    plot_temporal_evolution,
    plot_loss_curves,
    plot_comparison,
    plot_comparison_heatmaps,
    plot_error_maps,
    visualize_model,
)

from .gif_generator import (
    create_solution_gif,
    create_heatmap_gif,
    create_dual_heatmap_gif,
    generate_model_animation,
    create_comparison_heatmap_gif,
    create_comparison_profile_gif,
    generate_comparison_animation,
)

__all__ = [
    # Plotting
    "setup_matplotlib",
    "plot_solution_heatmaps",
    "plot_spatial_profiles",
    "plot_temporal_evolution",
    "plot_loss_curves",
    "plot_comparison",
    "plot_comparison_heatmaps",
    "plot_error_maps",
    "visualize_model",
    # Animation
    "create_solution_gif",
    "create_heatmap_gif",
    "create_dual_heatmap_gif",
    "generate_model_animation",
    "create_comparison_heatmap_gif",
    "create_comparison_profile_gif",
    "generate_comparison_animation",
]
