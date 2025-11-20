"""
Reproducibility utilities for deterministic training.

Provides seed_everything() to ensure reproducible experiments across runs.
"""

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
    set_single_thread_blas: bool = True
) -> int:
    """
    Set random seeds for reproducibility across all libraries.

    Call this ONCE, as early as possible (before any CUDA call or model creation).

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (may reduce performance)
        set_single_thread_blas: If True, force single-threaded BLAS operations

    Returns:
        The seed value used
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Torch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic kernels (turn off perf heuristics that change algorithms)
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass  # older PyTorch versions

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Disable TF32 so matmuls are bitwise stable across runs
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    # cuBLAS determinism (must be set before the first CUDA context is created)
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Keep SciPy/BLAS from reordering work across threads (helps determinism)
    if set_single_thread_blas:
        for var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS"
        ]:
            os.environ.setdefault(var, "1")

    return seed


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device for training.

    Args:
        prefer_mps: If True, prefer MPS (Apple Silicon) over CPU when CUDA unavailable

    Returns:
        torch.device for training
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_accelerator() -> str:
    """
    Get accelerator string for PyTorch Lightning.

    Returns:
        "gpu", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
