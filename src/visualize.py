
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import CCPPinn
from src.utils.physics import DefaultParameters

def visualize_checkpoint(checkpoint_path: str, output_path: str = "results.png"):
    # Load model
    model = CCPPinn.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Generate grid
    params = DefaultParameters()
    x = torch.linspace(0, params.L, 100)
    t = torch.linspace(0, 4e-7, 100) # 5 cycles
    
    grid_x, grid_t = torch.meshgrid(x, t, indexing='ij')
    flat_x = grid_x.reshape(-1, 1)
    flat_t = grid_t.reshape(-1, 1)
    
    with torch.no_grad():
        n_e, phi = model(flat_x, flat_t)
        
    n_e = n_e.reshape(100, 100).numpy()
    phi = phi.reshape(100, 100).numpy()
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axs[0].imshow(n_e, extent=[0, 4e-7, 0, params.L], aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title("Electron Density ($n_e$)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (m)")
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(phi, extent=[0, 4e-7, 0, params.L], aspect='auto', origin='lower', cmap='plasma')
    axs[1].set_title("Electric Potential ($\phi$)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position (m)")
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
        visualize_checkpoint(ckpt)
    else:
        print("Usage: python src/visualize.py <checkpoint_path>")
