# Automatic Visualization

The PINN models now automatically generate visualizations when training completes!

## How It Works

When training finishes, the `on_train_end()` hook is triggered, which:
1. Generates a 100x100 grid over the full domain (space × time)
2. Evaluates the trained model
3. Creates a 4-panel visualization showing:
   - **Top Left**: Electron density heatmap
   - **Top Right**: Electric potential heatmap
   - **Bottom Left**: Spatial profiles at different times
   - **Bottom Right**: Temporal evolution at different positions

## Output Location

Visualizations are saved to:
```
experiments/{experiment_name}/solution_visualization.png
```

For example, if `experiment_name: ccppinn_default`, the plot will be saved to:
```
experiments/ccppinn_default/solution_visualization.png
```

## Manual Visualization

You can also manually generate visualizations at any time:

### From Python

```python
from src.model import CCPPinn

# Load trained model
model = CCPPinn.load_from_checkpoint("path/to/checkpoint.ckpt")

# Generate visualization
model.visualize_solution()

# Or specify custom path
model.visualize_solution(save_path="my_results/custom_viz.png")
```

### From Command Line (Legacy Script)

The standalone `src/visualize.py` script still works:

```bash
python -m src.visualize experiments/experiment/epoch=49-val_loss=5.81e+05.ckpt
```

## Customizing Visualization

To customize the visualization, edit the `visualize_solution()` method in `src/model.py`:

```python
def visualize_solution(self, save_path: str = None):
    # Change grid resolution
    x = torch.linspace(0, self.params.L, 200, device=self.device)  # Higher res
    t = torch.linspace(0, 4e-7, 200, device=self.device)
    
    # Change time range
    t = torch.linspace(0, 1e-6, 100, device=self.device)  # More cycles
    
    # Change colormap
    im1 = axes[0, 0].imshow(..., cmap='hot')  # Different colormap
    
    # Add more plots, etc.
```

## Disabling Auto-Visualization

If you want to disable automatic visualization, override `on_train_end()` in your model:

```python
class MyCustomPinn(BaseModel):
    def on_train_end(self):
        # Don't visualize
        pass
```

Or just remove the `on_train_end()` method from `BaseModel`.

## Example Output

The visualization includes:

1. **Heatmaps**: Show how electron density and potential vary over space and time
2. **Spatial Profiles**: Show snapshots at 5 different time points
3. **Temporal Evolution**: Show time-series at 5 different spatial locations

All axes are labeled with proper units (mm for position, μs for time).
