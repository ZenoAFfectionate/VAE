"""General-purpose helper utilities.

Includes reproducibility helpers (seed fixing), device detection, a running
average meter, parameter counting, the model factory, and checkpoint
save / load utilities.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for full reproducibility.

    Seeds Python's ``random``, NumPy and PyTorch (CPU & CUDA) and enables
    deterministic cuDNN behaviour.

    Args:
        seed: The integer seed to apply across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favour determinism over raw throughput for reproducible experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        A ``torch.device`` instance (``cuda`` or ``cpu``).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Tracks and computes the running average of a streamed scalar value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Incorporate ``n`` observations of value ``val``."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate a VAE / SparseVAE from a configuration dictionary.

    Using a single factory keeps the training and validation scripts in sync
    and guarantees that a checkpoint can always be reconstructed from its
    stored config.

    Args:
        config: Dict produced from parsed CLI arguments plus dataset metadata.
            Must contain ``model``, ``image_channels``, ``image_size`` etc.

    Returns:
        An (uninitialised-on-device) model instance.
    """
    # Imported lazily inside the function to avoid a circular import:
    # `model` depends on `utils.losses`, which would otherwise pull in this
    # module (and `model`) again at import time.
    from model import SparseVAE, VAE

    common = dict(
        image_channels=config["image_channels"],
        image_size=config["image_size"],
        backbone=config["backbone"],
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        activation=config["activation"],
        recon_loss_type=config["recon_loss_type"],
        beta=config["beta"],
    )

    model_type = config["model"].lower()
    if model_type == "vae":
        return VAE(**common)
    if model_type == "sparsevae":
        return SparseVAE(
            target_sparsity=config["target_sparsity"],
            sparse_weight=config["sparse_weight"],
            **common,
        )
    raise ValueError(f"Unsupported model type {config['model']!r}. "
                     f"Choose from {{'VAE', 'SparseVAE'}}.")


def save_checkpoint(state: Dict[str, Any], save_path: str) -> None:
    """Save a checkpoint dictionary to disk, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(ckpt_path: str,
                    map_location: torch.device | str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk.

    Args:
        ckpt_path: Path to the checkpoint file.
        map_location: Device onto which tensors are mapped.

    Returns:
        The deserialized checkpoint dictionary.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)
