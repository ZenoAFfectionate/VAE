"""Unit tests for the VAE and Sparse VAE model architectures."""

from __future__ import annotations

import copy

import pytest
import torch

from model import SparseVAE, VAE
from utils.helpers import build_model

# All four model x backbone combinations exercised by the parametrized tests.
_COMBOS = [
    ("VAE", "mlp", 1, 28, [128, 64]),
    ("VAE", "cnn", 3, 32, [32, 64, 128]),
    ("SparseVAE", "mlp", 1, 28, [128, 64]),
    ("SparseVAE", "cnn", 3, 32, [32, 64, 128]),
]


def _build(model_type, backbone, channels, size, hidden, latent_dim=16):
    cfg = dict(
        model=model_type, backbone=backbone, image_channels=channels,
        image_size=size, hidden_dims=hidden, latent_dim=latent_dim,
        activation="relu",
        recon_loss_type="bce" if channels == 1 else "mse",
        beta=1.0, target_sparsity=0.05, sparse_weight=1.0,
    )
    return build_model(cfg)


@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_forward_output_shapes(model_type, backbone, channels, size, hidden):
    """Forward pass returns structured outputs with correct shapes."""
    model = _build(model_type, backbone, channels, size, hidden)
    x = torch.rand(4, channels, size, size)
    out = model(x)

    assert set(out) >= {"recon", "mu", "logvar", "z"}
    # Reconstruction must match the input resolution exactly.
    assert out["recon"].shape == x.shape
    assert out["mu"].shape == (4, 16)
    assert out["logvar"].shape == (4, 16)
    assert out["z"].shape == (4, 16)


@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_loss_terms_are_valid(model_type, backbone, channels, size, hidden):
    """Loss decomposition yields finite scalars; KL is non-negative."""
    model = _build(model_type, backbone, channels, size, hidden)
    x = torch.rand(4, channels, size, size)
    out = model(x)
    losses = model.loss_function(x, out)

    for key in ("total_loss", "recon_loss", "kl_loss"):
        assert losses[key].dim() == 0, f"{key} should be a scalar"
        assert torch.isfinite(losses[key]), f"{key} is not finite"

    # KL divergence between a Gaussian and the standard normal is >= 0.
    assert losses["kl_loss"].item() >= -1e-5

    if model_type == "SparseVAE":
        assert "sparse_loss" in losses
        assert losses["sparse_loss"].item() >= -1e-5
        ratio = losses["sparsity_ratio"].item()
        assert 0.0 <= ratio <= 1.0


@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_gradient_flow(model_type, backbone, channels, size, hidden):
    """Backprop populates gradients in encoder and decoder parameters."""
    model = _build(model_type, backbone, channels, size, hidden)
    x = torch.rand(4, channels, size, size)
    out = model(x)
    loss = model.loss_function(x, out)["total_loss"]
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
    # At least one parameter must receive a non-zero gradient.
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_reparameterization_is_stochastic_and_differentiable():
    """Reparameterization injects noise yet keeps the graph differentiable."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8)
    mu = torch.zeros(4, 8, requires_grad=True)
    logvar = torch.zeros(4, 8, requires_grad=True)

    torch.manual_seed(0)
    z1 = model.reparameterize(mu, logvar)
    torch.manual_seed(1)
    z2 = model.reparameterize(mu, logvar)
    # Different noise draws should produce different samples.
    assert not torch.allclose(z1, z2)

    # Gradient must flow back to mu through the reparameterized sample.
    z1.sum().backward()
    assert mu.grad is not None and torch.allclose(mu.grad, torch.ones_like(mu))


def test_reparameterization_collapses_with_tiny_variance():
    """With very small variance, z should be (almost) equal to mu.

    Note: ``reparameterize`` clamps logvar to a minimum of -10 for numerical
    stability, so the residual std is ~exp(-5) ≈ 0.0067 rather than exactly 0.
    """
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8)
    mu = torch.randn(4, 8)
    logvar = torch.full((4, 8), -30.0)  # clamped internally to -10
    z = model.reparameterize(mu, logvar)
    assert torch.allclose(z, mu, atol=5e-2)


@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_batch_size_one(model_type, backbone, channels, size, hidden):
    """Edge case: a single-sample batch must work end-to-end."""
    model = _build(model_type, backbone, channels, size, hidden)
    x = torch.rand(1, channels, size, size)
    out = model(x)
    losses = model.loss_function(x, out)
    assert out["recon"].shape == x.shape
    assert torch.isfinite(losses["total_loss"])


def test_minimum_cnn_resolution():
    """Edge case: the smallest valid CNN input (size == 2^num_stages)."""
    # Two stages -> minimum size 4.
    model = VAE(image_channels=1, image_size=4, backbone="cnn",
                hidden_dims=[8, 16], latent_dim=4, recon_loss_type="mse")
    x = torch.rand(2, 1, 4, 4)
    out = model(x)
    assert out["recon"].shape == x.shape


def test_input_validation_rejects_wrong_shape():
    """Model raises a clear error on mismatched input shapes / rank."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8)
    with pytest.raises(ValueError):
        model(torch.rand(4, 3, 28, 28))      # wrong channels
    with pytest.raises(ValueError):
        model(torch.rand(4, 1, 16, 16))      # wrong spatial size
    with pytest.raises(ValueError):
        model(torch.rand(4, 784))            # wrong rank


def test_cnn_rejects_indivisible_resolution():
    """CNN backbone rejects sizes not divisible by 2**num_stages."""
    with pytest.raises(ValueError):
        # 28 is not divisible by 2**3 = 8.
        VAE(image_channels=1, image_size=28, backbone="cnn",
            hidden_dims=[16, 32, 64], latent_dim=8)


def test_sample_generation_shape():
    """`sample` decodes random latents into properly shaped images."""
    model = VAE(image_channels=3, image_size=32, backbone="cnn",
                hidden_dims=[32, 64, 128], latent_dim=16, recon_loss_type="mse")
    samples = model.sample(5, torch.device("cpu"))
    assert samples.shape == (5, 3, 32, 32)


def test_device_compatibility():
    """Model runs on CPU and (when present) on CUDA without code changes."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for dev in devices:
        model = SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                          hidden_dims=[64], latent_dim=8).to(dev)
        x = torch.rand(3, 1, 28, 28, device=dev)
        out = model(x)
        losses = model.loss_function(x, out)
        assert out["recon"].device.type == dev.type
        assert torch.isfinite(losses["total_loss"])


def test_vae_rejects_negative_beta():
    with pytest.raises(ValueError):
        VAE(image_channels=1, image_size=28, backbone="mlp",
            hidden_dims=[64], latent_dim=8, beta=-1.0)


@pytest.mark.parametrize("bad_rho", [0.0, 1.0, -0.1, 1.2])
def test_sparse_vae_rejects_invalid_target_sparsity(bad_rho):
    with pytest.raises(ValueError):
        SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                  hidden_dims=[64], latent_dim=8, target_sparsity=bad_rho)


def test_sparse_vae_is_drop_in_for_vae():
    """SparseVAE exposes the same public interface as VAE."""
    sparse = SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                       hidden_dims=[64], latent_dim=8)
    x = torch.rand(4, 1, 28, 28)
    out = sparse(x)
    losses = sparse.loss_function(x, out)
    # Superset of the standard VAE loss keys.
    assert {"total_loss", "recon_loss", "kl_loss"} <= set(losses)
    assert hasattr(sparse, "sample") and callable(sparse.sample)


# ===================================================================== #
# Mathematical correctness of the loss composition
# ===================================================================== #
@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_elbo_decomposition_is_exact(model_type, backbone, channels, size, hidden):
    """total_loss must equal recon_loss + beta * kl_loss (+ sparse term)."""
    model = _build(model_type, backbone, channels, size, hidden)
    x = torch.rand(4, channels, size, size)
    out = model(x)
    losses = model.loss_function(x, out)

    expected = losses["recon_loss"] + model.beta * losses["kl_loss"]
    if "sparse_loss" in losses:
        expected = expected + model.sparse_weight * losses["sparse_loss"]
    assert torch.allclose(losses["total_loss"], expected, atol=1e-5)


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0, 4.0])
def test_beta_scales_kl_contribution(beta):
    """The beta coefficient correctly scales the KL term in the total loss."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8, beta=beta)
    x = torch.rand(4, 1, 28, 28)
    out = model(x)
    losses = model.loss_function(x, out)
    expected = losses["recon_loss"] + beta * losses["kl_loss"]
    assert torch.allclose(losses["total_loss"], expected, atol=1e-5)


def test_sparse_weight_scales_penalty():
    """The sparse weight correctly scales the sparsity penalty contribution."""
    x = torch.rand(8, 1, 28, 28)
    base = SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                     hidden_dims=[64], latent_dim=8, sparse_weight=3.0)
    out = base(x)
    losses = base.loss_function(x, out)
    vae_part = losses["recon_loss"] + base.beta * losses["kl_loss"]
    assert torch.allclose(
        losses["total_loss"], vae_part + 3.0 * losses["sparse_loss"], atol=1e-5)


# ===================================================================== #
# Output ranges and the reparameterization distribution
# ===================================================================== #
@pytest.mark.parametrize("model_type,backbone,channels,size,hidden", _COMBOS)
def test_reconstruction_in_unit_range(model_type, backbone, channels, size, hidden):
    """Decoder output is bounded to [0, 1] by the final Sigmoid."""
    model = _build(model_type, backbone, channels, size, hidden).eval()
    x = torch.rand(4, channels, size, size)
    recon = model(x)["recon"]
    assert recon.min().item() >= 0.0 and recon.max().item() <= 1.0


def test_generated_samples_in_unit_range():
    """`sample` outputs are also bounded to [0, 1]."""
    model = VAE(image_channels=3, image_size=32, backbone="cnn",
                hidden_dims=[32, 64, 128], latent_dim=16,
                recon_loss_type="mse").eval()
    samples = model.sample(8, torch.device("cpu"))
    assert samples.min().item() >= 0.0 and samples.max().item() <= 1.0


def test_reparameterization_recovers_distribution():
    """Empirically, z ~ N(mu, sigma^2): sample mean/std match the parameters."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[32], latent_dim=4)
    torch.manual_seed(0)
    mu = torch.full((40000, 4), 2.0)
    logvar = torch.zeros(40000, 4)  # sigma = 1
    z = model.reparameterize(mu, logvar)
    assert torch.allclose(z.mean(0), torch.full((4,), 2.0), atol=0.05)
    assert torch.allclose(z.std(0), torch.ones(4), atol=0.05)


def test_eval_forward_is_deterministic_under_fixed_seed():
    """With the same RNG seed the stochastic forward pass is reproducible."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8).eval()
    x = torch.rand(4, 1, 28, 28)
    torch.manual_seed(123)
    out1 = model(x)["recon"]
    torch.manual_seed(123)
    out2 = model(x)["recon"]
    assert torch.allclose(out1, out2)


# ===================================================================== #
# Encode / decode shapes and gradient targeting
# ===================================================================== #
@pytest.mark.parametrize("backbone,channels,size,hidden", [
    ("mlp", 1, 28, [128, 64]),
    ("cnn", 3, 32, [32, 64, 128]),
])
def test_encode_decode_shapes(backbone, channels, size, hidden):
    """encode -> (mu, logvar); decode(z) -> image-shaped tensor."""
    model = VAE(image_channels=channels, image_size=size, backbone=backbone,
                hidden_dims=hidden, latent_dim=10,
                recon_loss_type="mse" if channels == 3 else "bce")
    x = torch.rand(5, channels, size, size)
    mu, logvar = model.encode(x)
    assert mu.shape == (5, 10) and logvar.shape == (5, 10)
    recon = model.decode(mu)
    assert recon.shape == (5, channels, size, size)


def test_latent_heads_receive_gradients():
    """Backprop reaches the latent mean/logvar projection heads."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8)
    x = torch.rand(4, 1, 28, 28)
    model.loss_function(x, model(x))["total_loss"].backward()
    assert model.fc_mu.weight.grad is not None
    assert model.fc_logvar.weight.grad is not None
    assert model.fc_mu.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("latent_dim", [2, 8, 32, 128])
def test_various_latent_dims(latent_dim):
    """Models build and run across a range of latent dimensionalities."""
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=latent_dim)
    out = model(torch.rand(3, 1, 28, 28))
    assert out["mu"].shape == (3, latent_dim)
    assert out["recon"].shape == (3, 1, 28, 28)


# ===================================================================== #
# Numerical stability under extreme inputs
# ===================================================================== #
@pytest.mark.parametrize("filler", [0.0, 1.0])
def test_extreme_constant_inputs_are_stable(filler):
    """All-zero / all-one inputs must not produce NaN / Inf losses."""
    model = SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                      hidden_dims=[64], latent_dim=8)
    x = torch.full((4, 1, 28, 28), filler)
    losses = model.loss_function(x, model(x))
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["recon_loss"])
    assert torch.isfinite(losses["kl_loss"])


def test_sparsity_ratio_reported_in_unit_interval():
    """SparseVAE's reported sparsity_ratio always lies within [0, 1]."""
    model = SparseVAE(image_channels=1, image_size=28, backbone="mlp",
                      hidden_dims=[64], latent_dim=16)
    losses = model.loss_function(torch.rand(8, 1, 28, 28),
                                 model(torch.rand(8, 1, 28, 28)))
    assert 0.0 <= losses["sparsity_ratio"].item() <= 1.0

