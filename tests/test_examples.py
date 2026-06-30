"""Tests for the reusable pure-logic helpers in examples/common.py.

These cover the math used across the demos (interpolation, noising, masking,
AUC, PSNR, PCA) without requiring a trained checkpoint, matplotlib or gradio.
"""

from __future__ import annotations

import math

import pytest
import torch

from examples.common import (
    add_gaussian_noise,
    add_salt_pepper,
    compute_auc,
    interpolation_steps,
    lerp,
    make_center_mask,
    pca_2d,
    psnr,
    slerp,
)


# --------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------- #
def test_lerp_endpoints_and_midpoint():
    z0 = torch.zeros(1, 8)
    z1 = torch.ones(1, 8)
    assert torch.allclose(lerp(z0, z1, 0.0), z0)
    assert torch.allclose(lerp(z0, z1, 1.0), z1)
    assert torch.allclose(lerp(z0, z1, 0.5), torch.full((1, 8), 0.5))


def test_slerp_endpoints():
    torch.manual_seed(0)
    z0 = torch.randn(1, 16)
    z1 = torch.randn(1, 16)
    assert torch.allclose(slerp(z0, z1, 0.0), z0, atol=1e-4)
    assert torch.allclose(slerp(z0, z1, 1.0), z1, atol=1e-4)


def test_slerp_collinear_falls_back_to_lerp():
    z0 = torch.randn(1, 8)
    z1 = 2.0 * z0  # collinear -> sin(omega) ~ 0
    out = slerp(z0, z1, 0.5)
    assert torch.isfinite(out).all()


def test_interpolation_steps():
    steps = interpolation_steps(5)
    assert steps.shape == (5,)
    assert steps[0].item() == pytest.approx(0.0)
    assert steps[-1].item() == pytest.approx(1.0)
    # Always at least two endpoints even if fewer requested.
    assert interpolation_steps(1).shape[0] == 2


# --------------------------------------------------------------------- #
# Noising
# --------------------------------------------------------------------- #
def test_add_gaussian_noise_range_and_shape():
    x = torch.rand(4, 1, 8, 8)
    noisy = add_gaussian_noise(x, std=0.5)
    assert noisy.shape == x.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not torch.allclose(noisy, x)  # noise actually applied


def test_add_salt_pepper_values_in_range():
    x = torch.rand(4, 1, 16, 16)
    noisy = add_salt_pepper(x, amount=0.5)
    assert noisy.shape == x.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    # Some pixels should be pushed to the extremes.
    assert ((noisy == 0.0) | (noisy == 1.0)).any()


# --------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------- #
def test_make_center_mask_structure():
    mask = make_center_mask(10, 10, frac=0.4)
    assert mask.shape == (1, 10, 10)
    # Borders kept (1), centre hole zeroed (0).
    assert mask[0, 0, 0].item() == 1.0
    assert mask[0, 5, 5].item() == 0.0
    # Hole area roughly matches the requested fraction.
    hole = (mask == 0).sum().item()
    assert hole == 4 * 4


# --------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------- #
def test_compute_auc_known_value():
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(0.75)


def test_compute_auc_perfect_and_reversed():
    scores = torch.tensor([0.0, 0.1, 0.9, 1.0])
    labels = torch.tensor([0, 0, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(1.0)
    # Flipping the labels gives the complementary AUC.
    assert compute_auc(scores, 1 - labels) == pytest.approx(0.0)


def test_compute_auc_single_class_is_half():
    scores = torch.rand(8)
    labels = torch.zeros(8, dtype=torch.long)
    assert compute_auc(scores, labels) == pytest.approx(0.5)


def test_compute_auc_handles_ties():
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    labels = torch.tensor([0, 1, 0, 1])
    # All scores equal -> no discrimination -> AUC 0.5.
    assert compute_auc(scores, labels) == pytest.approx(0.5)


# --------------------------------------------------------------------- #
# PSNR & PCA
# --------------------------------------------------------------------- #
def test_psnr_identical_is_infinite():
    x = torch.rand(2, 1, 8, 8)
    assert math.isinf(psnr(x, x))


def test_psnr_known_value():
    x = torch.zeros(1, 1, 4, 4)
    y = torch.full((1, 1, 4, 4), 0.1)  # MSE = 0.01 -> PSNR = 20 dB
    assert psnr(x, y) == pytest.approx(20.0, abs=1e-4)


def test_pca_2d_shape_and_variance_order():
    torch.manual_seed(0)
    # Data with most variance along the first axis.
    x = torch.randn(200, 5)
    x[:, 0] *= 10.0
    coords = pca_2d(x)
    assert coords.shape == (200, 2)
    # First principal component should capture the larger variance.
    assert coords[:, 0].var() >= coords[:, 1].var()


# ===================================================================== #
# Integration: run the actual demo logic with a tiny in-memory VAE
# ===================================================================== #
@pytest.fixture
def tiny_vae():
    from model import VAE
    torch.manual_seed(0)
    return VAE(image_channels=1, image_size=28, backbone="mlp",
               hidden_dims=[32], latent_dim=8).eval()


def test_build_interpolation_grid(tiny_vae):
    from examples.common import build_interpolation_grid
    starts = torch.rand(3, 1, 28, 28)
    ends = torch.rand(3, 1, 28, 28)
    grid = build_interpolation_grid(tiny_vae, starts, ends, steps=5)
    assert grid.shape == (3 * 5, 1, 28, 28)
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_build_interpolation_grid_length_mismatch(tiny_vae):
    from examples.common import build_interpolation_grid
    with pytest.raises(ValueError):
        build_interpolation_grid(tiny_vae, torch.rand(3, 1, 28, 28),
                                 torch.rand(2, 1, 28, 28), steps=4)


def test_build_traversal_grid(tiny_vae):
    from examples.common import build_traversal_grid
    image = torch.rand(1, 1, 28, 28)
    grid, n_dims = build_traversal_grid(tiny_vae, image, num_dims=4,
                                        steps=6, span=3.0)
    assert n_dims == 4
    assert grid.shape == (4 * 6, 1, 28, 28)
    # num_dims is clamped to the latent dimensionality.
    _, n_dims_clamped = build_traversal_grid(tiny_vae, image, num_dims=999,
                                             steps=3, span=2.0)
    assert n_dims_clamped == 8


def test_inpaint_preserves_observed_pixels(tiny_vae):
    from examples.common import make_center_mask
    from examples.inpainting import inpaint
    images = torch.rand(2, 1, 28, 28)
    mask = make_center_mask(28, 28, frac=0.5)
    filled = inpaint(tiny_vae, images, mask, steps=3, lr=0.05)
    assert filled.shape == images.shape
    assert filled.min() >= 0.0 and filled.max() <= 1.0
    # Observed (unmasked) pixels must be left untouched.
    m = mask.to(filled.device)
    assert torch.allclose(filled * m, images * m, atol=1e-6)


def test_reconstruction_scores(tiny_vae):
    from examples.anomaly_detection import reconstruction_scores
    images = torch.rand(5, 1, 28, 28)
    scores = reconstruction_scores(tiny_vae, images)
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all() and (scores >= 0).all()


@pytest.mark.parametrize("latent_dim,expected", [(8, 2), (2, 2), (1, 2)])
def test_reduce_to_2d_always_2d(latent_dim, expected):
    from examples.latent_map import reduce_to_2d
    mu = torch.randn(20, latent_dim)
    coords = reduce_to_2d(mu, method="pca", seed=0)
    assert coords.shape == (20, expected)
