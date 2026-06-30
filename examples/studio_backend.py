"""Backend inference logic for the VAE Studio web app.

All functions here are framework-agnostic (no Gradio dependency) and operate on
NumPy arrays / tensors, so they can be unit tested directly. The Gradio UI in
``examples/app.py`` is a thin presentation layer on top of these functions.

Image convention:
    * UI images are NumPy arrays in ``HWC`` (or ``HW``) with values in
      ``[0, 255]`` (uint8) or ``[0, 1]`` (float).
    * Model tensors are ``(1, C, H, W)`` float in ``[0, 1]`` matching the
      checkpoint's channels / resolution.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from examples.common import (
    add_gaussian_noise, add_salt_pepper, build_interpolation_grid, encode_mean,
    make_center_mask, psnr,
)
from examples.inpainting import inpaint as _inpaint


# ===================================================================== #
# Image <-> tensor conversion (pure, unit tested)
# ===================================================================== #
def preprocess_image(arr: np.ndarray, image_channels: int,
                     image_size: int) -> torch.Tensor:
    """Convert an uploaded image array into a model-ready tensor.

    Handles arbitrary input resolution, channel count (grayscale / RGB / RGBA)
    and value range, producing a ``(1, C, H, W)`` float tensor in ``[0, 1]``
    that matches the model's expected channels and size.

    Args:
        arr: Uploaded image as a NumPy array (``HW``, ``HWC``).
        image_channels: Channels the model expects (1 or 3).
        image_size: Target square resolution.

    Returns:
        A ``(1, image_channels, image_size, image_size)`` tensor in ``[0, 1]``.
    """
    t = torch.from_numpy(np.asarray(arr)).float()
    if t.max() > 1.5:            # uint8 [0, 255] -> [0, 1]
        t = t / 255.0
    if t.dim() == 2:             # HW -> HW1
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1)       # HWC -> CHW
    cur_c = t.size(0)

    # Reconcile channel count with the model's expectation.
    if image_channels == 1:
        t = t[:3].mean(dim=0, keepdim=True) if cur_c >= 3 else t[:1]
    else:  # image_channels == 3
        t = t.repeat(3, 1, 1) if cur_c == 1 else t[:3]

    t = t.unsqueeze(0)           # (1, C, H, W)
    t = F.interpolate(t, size=(image_size, image_size),
                      mode="bilinear", align_corners=False)
    return t.clamp(0.0, 1.0)


def to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a model tensor to a displayable NumPy image in ``[0, 1]``."""
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    img = t.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = img[:, :, 0]       # grayscale -> HW
    return np.clip(img, 0.0, 1.0)


def tile_images(imgs: torch.Tensor, nrow: int, padding: int = 2,
                pad_value: float = 0.04) -> np.ndarray:
    """Tile a batch of images into a single grid NumPy image.

    Args:
        imgs: Tensor ``(N, C, H, W)`` in ``[0, 1]``.
        nrow: Number of images per row.
        padding: Pixels of padding between tiles.
        pad_value: Fill value for padding (matches the dark UI background).

    Returns:
        A NumPy grid image (``HWC`` or ``HW``) in ``[0, 1]``.
    """
    imgs = imgs.detach().cpu()
    n, c, h, w = imgs.shape
    ncol = max(1, min(nrow, n))
    nrows = math.ceil(n / ncol)
    grid_h = nrows * h + (nrows + 1) * padding
    grid_w = ncol * w + (ncol + 1) * padding
    grid = np.full((grid_h, grid_w, c), pad_value, dtype=np.float32)
    for idx in range(n):
        r, col = divmod(idx, ncol)
        y = padding + r * (h + padding)
        x = padding + col * (w + padding)
        grid[y:y + h, x:x + w, :] = imgs[idx].permute(1, 2, 0).numpy()
    if c == 1:
        grid = grid[:, :, 0]
    return np.clip(grid, 0.0, 1.0)


# ===================================================================== #
# Model-driven operations (return display-ready NumPy images)
# ===================================================================== #
@torch.no_grad()
def reconstruct(model, arr: np.ndarray, image_channels: int,
                image_size: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Encode then decode an uploaded image; return input, recon and stats."""
    x = preprocess_image(arr, image_channels, image_size).to(_device(model))
    output = model(x)
    mu = output["mu"]
    err = F.mse_loss(output["recon"], x).item()
    info = (f"Reconstruction MSE: {err:.4f}\n"
            f"Latent dim: {mu.numel()}\n"
            f"Latent mean / std: {mu.mean().item():.3f} / {mu.std().item():.3f}")
    return to_display(x), to_display(output["recon"]), info


@torch.no_grad()
def generate_samples(model, n: int, seed: int) -> np.ndarray:
    """Sample ``n`` images from the prior and tile them into a grid."""
    torch.manual_seed(int(seed))
    samples = model.sample(int(n), _device(model))
    nrow = max(1, int(round(n ** 0.5)))
    return tile_images(samples, nrow=nrow)


@torch.no_grad()
def denoise(model, arr: np.ndarray, noise_type: str, level: float,
            image_channels: int, image_size: int
            ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Corrupt an uploaded image and reconstruct (denoise) it."""
    x = preprocess_image(arr, image_channels, image_size).to(_device(model))
    if noise_type == "salt_pepper":
        noisy = add_salt_pepper(x, amount=float(level))
    else:
        noisy = add_gaussian_noise(x, std=float(level))
    denoised = model(noisy)["recon"]
    gain = psnr(x, denoised) - psnr(x, noisy)
    info = f"PSNR gain after denoising: {gain:+.2f} dB"
    return to_display(noisy), to_display(denoised), info


def inpaint_image(model, arr: np.ndarray, mask_frac: float, steps: int,
                  lr: float, image_channels: int, image_size: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Mask the centre of an uploaded image and inpaint via latent optimization."""
    x = preprocess_image(arr, image_channels, image_size).to(_device(model))
    mask = make_center_mask(image_size, image_size, frac=float(mask_frac))
    masked = x * mask.to(x.device)
    filled = _inpaint(model, x, mask, steps=int(steps), lr=float(lr))
    return to_display(masked), to_display(filled)


@torch.no_grad()
def interpolate_images(model, arr_a: np.ndarray, arr_b: np.ndarray, steps: int,
                       use_slerp: bool, image_channels: int,
                       image_size: int) -> np.ndarray:
    """Morph between two uploaded images through the latent space."""
    device = _device(model)
    a = preprocess_image(arr_a, image_channels, image_size).to(device)
    b = preprocess_image(arr_b, image_channels, image_size).to(device)
    steps = max(2, int(steps))
    grid = build_interpolation_grid(model, a, b, steps, use_slerp=use_slerp)
    return tile_images(grid, nrow=steps)


@torch.no_grad()
def latent_generate(model, slider_values, latent_dim: int) -> np.ndarray:
    """Decode a latent vector assembled from explorer slider values.

    For standard VAEs the latent is a flat vector ``(1, latent_dim)``. For
    PretrainedVAE (HuggingFace AutoencoderKL) the latent is spatial
    ``(1, C, H/f, W/f)``; slider values are broadcast across the spatial
    dimensions so each slider controls one latent channel.
    """
    device = _device(model)

    # Detect spatial-latent models (PretrainedVAE) by checking for the
    # `downscale` attribute set in model/pretrained.py.
    if hasattr(model, "downscale") and hasattr(model, "latent_channels"):
        # Spatial latent: build (1, C, H/f, W/f) and broadcast slider values.
        spatial = model.image_size // model.downscale
        c = model.latent_channels
        z = torch.zeros(1, c, spatial, spatial, device=device)
        for i, val in enumerate(slider_values[:c]):
            z[0, i] = float(val)  # broadcast across H×W via broadcasting
        return to_display(model.decode(z))

    # Standard flat-latent VAE.
    z = torch.zeros(1, latent_dim, device=device)
    for i, val in enumerate(slider_values):
        if i < latent_dim:
            z[0, i] = float(val)
    return to_display(model.decode(z))


def _device(model) -> torch.device:
    """Infer the device a model's parameters live on (default CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have params
        return torch.device("cpu")
