"""Tests for the HuggingFace pretrained VAE adapter and valid.py source guard.

The full wrapper tests require the optional ``diffusers`` package and are
skipped automatically when it is not installed (a tiny in-memory AutoencoderKL
is built, so no network download occurs). The pixel-range helpers and the
``valid.py`` source-selection guard are tested unconditionally.
"""

from __future__ import annotations

import sys

import pytest
import torch

from model.pretrained import PretrainedVAE


# --------------------------------------------------------------------- #
# Pixel-range helpers (no diffusers required)
# --------------------------------------------------------------------- #
def test_pixel_range_roundtrip():
    """[0,1] -> [-1,1] -> [0,1] is an identity transform."""
    x = torch.rand(4, 3, 8, 8)
    model_range = PretrainedVAE._to_model_range(x)
    assert model_range.min() >= -1.0 and model_range.max() <= 1.0
    back = PretrainedVAE._to_pixel_range(model_range)
    assert torch.allclose(back, x, atol=1e-6)


def test_to_pixel_range_clamps():
    """Out-of-range VAE outputs are clamped into [0, 1]."""
    out = PretrainedVAE._to_pixel_range(torch.tensor([-3.0, 0.0, 3.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_requires_a_source():
    """Constructing without hf_model_id or vae raises a clear error."""
    with pytest.raises(ValueError):
        PretrainedVAE(hf_model_id=None, vae=None)


# --------------------------------------------------------------------- #
# valid.py runtime source guard (needs torchvision, not diffusers)
# --------------------------------------------------------------------- #
def test_valid_requires_exactly_one_source(monkeypatch):
    """valid.main must reject 'neither' and 'both' source specifications."""
    import valid as valid_mod

    monkeypatch.setattr(sys, "argv", ["valid.py"])  # neither
    with pytest.raises(SystemExit):
        valid_mod.main()

    monkeypatch.setattr(sys, "argv",
                        ["valid.py", "--checkpoint", "a.pth",
                         "--hf-model", "some/model"])  # both
    with pytest.raises(SystemExit):
        valid_mod.main()


# --------------------------------------------------------------------- #
# Full wrapper behaviour (requires diffusers; uses a tiny in-memory model)
# --------------------------------------------------------------------- #
@pytest.fixture
def tiny_pretrained():
    """Build a tiny AutoencoderKL-backed PretrainedVAE without downloading."""
    diffusers = pytest.importorskip("diffusers")
    AutoencoderKL = diffusers.AutoencoderKL
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        sample_size=16,
    )
    # block_out_channels of length 2 -> spatial down-sampling factor 2.
    return PretrainedVAE(vae=vae, image_channels=3, image_size=16,
                         recon_loss_type="mse", beta=1.0)


def test_pretrained_forward_shapes(tiny_pretrained):
    """forward returns recon matching the input and 4D spatial latents."""
    x = torch.rand(2, 3, 16, 16)
    out = tiny_pretrained(x)
    assert out["recon"].shape == x.shape
    assert out["mu"].shape == out["logvar"].shape
    assert out["mu"].dim() == 4  # (B, C_z, H/f, W/f)
    assert out["recon"].min() >= 0.0 and out["recon"].max() <= 1.0


def test_pretrained_loss_is_finite(tiny_pretrained):
    """Loss decomposition yields finite scalar terms."""
    x = torch.rand(2, 3, 16, 16)
    losses = tiny_pretrained.loss_function(x, tiny_pretrained(x))
    for key in ("total_loss", "recon_loss", "kl_loss"):
        assert losses[key].dim() == 0 and torch.isfinite(losses[key])


def test_pretrained_sample_shape(tiny_pretrained):
    """sample decodes random spatial latents into images."""
    samples = tiny_pretrained.sample(2, torch.device("cpu"))
    assert samples.shape == (2, 3, 16, 16)
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_pretrained_rejects_indivisible_size(tiny_pretrained):
    """Inputs not divisible by the down-sampling factor are rejected."""
    with pytest.raises(ValueError):
        tiny_pretrained(torch.rand(2, 3, 15, 15))
    with pytest.raises(ValueError):
        tiny_pretrained(torch.rand(2, 1, 16, 16))  # wrong channels
