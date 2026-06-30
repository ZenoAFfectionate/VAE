"""Visualization utilities for reconstructions and generated samples."""

from __future__ import annotations

import os

import torch
from torchvision.utils import save_image


@torch.no_grad()
def save_reconstruction(model: torch.nn.Module,
                        data: torch.Tensor,
                        save_path: str,
                        n_samples: int = 8) -> None:
    """Save a side-by-side comparison of inputs and their reconstructions.

    The top row contains the original images and the bottom row the
    corresponding reconstructions.

    Args:
        model: A trained VAE / SparseVAE in eval mode.
        data: A batch of input images of shape ``(B, C, H, W)``.
        save_path: Destination PNG path (parent dirs created automatically).
        n_samples: Number of examples to display.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(n_samples, data.size(0))
    inputs = data[:n]
    output = model(inputs)
    recon = output["recon"][:n]

    # Stack originals on top, reconstructions on the bottom.
    comparison = torch.cat([inputs.cpu(), recon.cpu()], dim=0)
    save_image(comparison, save_path, nrow=n)


@torch.no_grad()
def save_generated_samples(model: torch.nn.Module,
                           device: torch.device,
                           save_path: str,
                           n_samples: int = 64) -> None:
    """Decode random latent vectors and save the generated image grid.

    Args:
        model: A trained VAE / SparseVAE in eval mode.
        device: Device on which to draw the latent samples.
        save_path: Destination PNG path (parent dirs created automatically).
        n_samples: Number of samples to generate.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    samples = model.sample(n_samples, device).cpu()
    nrow = int(n_samples ** 0.5) or 1
    save_image(samples, save_path, nrow=nrow)
