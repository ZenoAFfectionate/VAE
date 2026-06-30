"""Tests for the VAE Studio backend (examples/studio_backend.py).

These validate the image preprocessing / display conversion and every
model-driven operation using a tiny in-memory VAE. The Gradio UI itself is a
thin layer over these functions and needs no separate test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from examples import studio_backend as be


@pytest.fixture
def tiny_vae():
    from model import VAE
    torch.manual_seed(0)
    return VAE(image_channels=1, image_size=28, backbone="mlp",
               hidden_dims=[32], latent_dim=8).eval()


@pytest.fixture
def tiny_rgb_vae():
    from model import VAE
    torch.manual_seed(0)
    return VAE(image_channels=3, image_size=16, backbone="cnn",
               hidden_dims=[16, 32], latent_dim=8, recon_loss_type="mse").eval()


# --------------------------------------------------------------------- #
# Preprocessing / display conversion (pure)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("arr,channels,size,expected_c", [
    (np.random.randint(0, 256, (20, 15, 3), dtype=np.uint8), 1, 28, 1),
    (np.random.randint(0, 256, (20, 15), dtype=np.uint8), 1, 28, 1),
    (np.random.randint(0, 256, (20, 15), dtype=np.uint8), 3, 28, 3),
    (np.random.randint(0, 256, (12, 12, 4), dtype=np.uint8), 3, 16, 3),
])
def test_preprocess_image_shapes(arr, channels, size, expected_c):
    t = be.preprocess_image(arr, channels, size)
    assert t.shape == (1, expected_c, size, size)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_preprocess_image_float_input_not_rescaled():
    arr = np.full((10, 10, 3), 0.5, dtype=np.float32)  # already in [0, 1]
    t = be.preprocess_image(arr, 1, 8)
    assert t.max() <= 1.0
    assert t.mean().item() == pytest.approx(0.5, abs=1e-3)


def test_to_display_grayscale_and_rgb():
    gray = be.to_display(torch.rand(1, 1, 8, 8))
    assert gray.shape == (8, 8)
    rgb = be.to_display(torch.rand(1, 3, 8, 8))
    assert rgb.shape == (8, 8, 3)
    assert gray.min() >= 0.0 and gray.max() <= 1.0


def test_tile_images_grid_shape():
    imgs = torch.rand(4, 1, 8, 8)
    grid = be.tile_images(imgs, nrow=2, padding=1)
    # 2x2 tiles of 8px + padding -> (2*8+3, 2*8+3)
    assert grid.shape == (8 * 2 + 3, 8 * 2 + 3)
    assert grid.min() >= 0.0 and grid.max() <= 1.0

    rgb = be.tile_images(torch.rand(3, 3, 8, 8), nrow=3, padding=0)
    assert rgb.shape == (8, 8 * 3, 3)


# --------------------------------------------------------------------- #
# Model-driven operations
# --------------------------------------------------------------------- #
def test_reconstruct(tiny_vae):
    arr = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    inp, recon, info = be.reconstruct(tiny_vae, arr, 1, 28)
    assert inp.shape == (28, 28) and recon.shape == (28, 28)
    assert "MSE" in info and "Latent" in info


def test_generate_samples(tiny_vae):
    grid = be.generate_samples(tiny_vae, n=9, seed=0)
    assert grid.ndim == 2  # grayscale grid
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_generate_samples_is_seeded(tiny_vae):
    g1 = be.generate_samples(tiny_vae, n=4, seed=123)
    g2 = be.generate_samples(tiny_vae, n=4, seed=123)
    assert np.allclose(g1, g2)


def test_denoise(tiny_vae):
    arr = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    noisy, denoised, info = be.denoise(tiny_vae, arr, "gaussian", 0.4, 1, 28)
    assert noisy.shape == (28, 28) and denoised.shape == (28, 28)
    assert "PSNR" in info


def test_denoise_salt_pepper(tiny_vae):
    arr = np.random.rand(28, 28).astype(np.float32)
    noisy, denoised, _ = be.denoise(tiny_vae, arr, "salt_pepper", 0.2, 1, 28)
    # Salt & pepper pushes some pixels to the extremes.
    assert ((noisy == 0.0) | (noisy == 1.0)).any()
    assert denoised.shape == (28, 28)


def test_inpaint_image(tiny_vae):
    arr = np.random.rand(28, 28).astype(np.float32)
    masked, filled = be.inpaint_image(tiny_vae, arr, mask_frac=0.5,
                                      steps=3, lr=0.05, image_channels=1,
                                      image_size=28)
    assert masked.shape == (28, 28) and filled.shape == (28, 28)
    assert filled.min() >= 0.0 and filled.max() <= 1.0


def test_interpolate_images(tiny_vae):
    a = np.random.rand(28, 28).astype(np.float32)
    b = np.random.rand(28, 28).astype(np.float32)
    grid = be.interpolate_images(tiny_vae, a, b, steps=5, use_slerp=False,
                                 image_channels=1, image_size=28)
    assert grid.ndim == 2
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_latent_generate(tiny_vae):
    img = be.latent_generate(tiny_vae, [0.5] * 8, latent_dim=8)
    assert img.shape == (28, 28)
    # Fewer slider values than latent_dim is handled gracefully.
    img2 = be.latent_generate(tiny_vae, [0.1, 0.2], latent_dim=8)
    assert img2.shape == (28, 28)


def test_rgb_pipeline(tiny_rgb_vae):
    """RGB / CNN model: reconstruct and generate produce 3-channel output."""
    arr = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
    inp, recon, _ = be.reconstruct(tiny_rgb_vae, arr, 3, 16)
    assert inp.shape == (16, 16, 3) and recon.shape == (16, 16, 3)
    grid = be.generate_samples(tiny_rgb_vae, n=4, seed=0)
    assert grid.ndim == 3 and grid.shape[2] == 3
