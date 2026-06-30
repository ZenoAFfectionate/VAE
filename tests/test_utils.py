"""Tests for utility modules: losses, metrics, helpers, options, logger."""

from __future__ import annotations

import logging
import math
import os

import pytest
import torch
import torch.nn.functional as F

from model import SparseVAE, VAE
from utils.helpers import (
    AverageMeter,
    build_model,
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from utils.logger import setup_logger
from utils.losses import kl_divergence, reconstruction_loss, sparsity_penalty
from utils.metrics import MetricTracker, latent_sparsity_ratio
from utils.options import get_train_args, get_valid_args
from utils.visualization import save_generated_samples, save_reconstruction


# --------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------- #
def test_reconstruction_loss_mse_zero_on_identity():
    """MSE reconstruction loss is ~0 when reconstruction equals the input."""
    x = torch.rand(4, 1, 8, 8)
    loss = reconstruction_loss(x, x, loss_type="mse")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_bce_is_finite_and_nonnegative():
    """BCE reconstruction loss is finite and non-negative."""
    x = torch.rand(4, 1, 8, 8)
    recon = torch.rand(4, 1, 8, 8)
    loss = reconstruction_loss(recon, x, loss_type="bce")
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_reconstruction_loss_invalid_type_raises():
    x = torch.rand(2, 1, 4, 4)
    with pytest.raises(ValueError):
        reconstruction_loss(x, x, loss_type="huber")


def test_kl_divergence_zero_for_standard_normal():
    """KL(N(0, I) || N(0, I)) == 0."""
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_positive_when_shifted():
    """A shifted / scaled Gaussian yields strictly positive KL."""
    mu = torch.full((4, 16), 2.0)
    logvar = torch.zeros(4, 16)
    assert kl_divergence(mu, logvar).item() > 0.0


def test_kl_divergence_stable_under_extreme_logvar():
    """Extreme log-variance values must not produce NaN / Inf (clamping)."""
    mu = torch.zeros(4, 8)
    logvar = torch.full((4, 8), 1e4)
    assert torch.isfinite(kl_divergence(mu, logvar))


def test_sparsity_penalty_zero_at_target():
    """The penalty vanishes when mean activation equals the target."""
    rho = 0.1
    activations = torch.full((32, 16), rho)
    assert sparsity_penalty(activations, rho).item() == pytest.approx(0.0, abs=1e-5)


def test_sparsity_penalty_positive_away_from_target():
    """The penalty is positive when activations deviate from the target."""
    activations = torch.full((32, 16), 0.8)
    assert sparsity_penalty(activations, target_sparsity=0.05).item() > 0.0


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def test_metric_tracker_weighted_average():
    """MetricTracker computes the correct sample-weighted average."""
    tracker = MetricTracker()
    tracker.update({"loss": 1.0}, n=2)
    tracker.update({"loss": 4.0}, n=2)
    assert tracker.averages()["loss"] == pytest.approx(2.5)


def test_metric_tracker_empty_is_safe():
    """Averaging before any update returns an empty dict without error."""
    assert MetricTracker().averages() == {}


def test_latent_sparsity_ratio_bounds():
    """Sparsity ratio always lies within [0, 1]."""
    mu = torch.randn(8, 16)
    ratio = latent_sparsity_ratio(mu)
    assert 0.0 <= ratio <= 1.0


def test_average_meter():
    """AverageMeter tracks running averages correctly."""
    meter = AverageMeter("x")
    meter.update(2.0, n=1)
    meter.update(4.0, n=3)
    assert meter.avg == pytest.approx((2.0 + 12.0) / 4)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def test_set_seed_reproducibility():
    """Identical seeds yield identical random draws."""
    set_seed(123)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_get_device_returns_torch_device():
    assert isinstance(get_device(prefer_cuda=False), torch.device)
    assert get_device(prefer_cuda=False).type == "cpu"


def test_count_parameters_positive():
    model = VAE(image_channels=1, image_size=28, backbone="mlp",
                hidden_dims=[64], latent_dim=8)
    assert count_parameters(model) > 0


@pytest.mark.parametrize("model_type,expected", [
    ("VAE", VAE), ("SparseVAE", SparseVAE),
])
def test_build_model_factory(model_type, expected, mlp_config):
    cfg = dict(mlp_config)
    cfg["model"] = model_type
    model = build_model(cfg)
    assert isinstance(model, expected)


def test_build_model_invalid_type_raises(mlp_config):
    cfg = dict(mlp_config)
    cfg["model"] = "GAN"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_checkpoint_save_load_roundtrip(tmp_path, mlp_config):
    """A saved checkpoint reloads with identical weights."""
    model = build_model(mlp_config)
    ckpt_path = os.path.join(tmp_path, "sub", "ckpt.pth")
    save_checkpoint({"model_state": model.state_dict(), "config": mlp_config},
                    ckpt_path)
    assert os.path.isfile(ckpt_path)

    loaded = load_checkpoint(ckpt_path, map_location="cpu")
    restored = build_model(loaded["config"])
    restored.load_state_dict(loaded["model_state"])
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(p1, p2)


def test_load_checkpoint_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("definitely/not/here.pth")


# --------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------- #
def test_train_args_parsing():
    args = get_train_args([
        "--model", "SparseVAE", "--dataset", "CIFAR10", "--backbone", "cnn",
        "--latent-dim", "64", "--epochs", "7", "--sparse-weight", "2.5",
    ])
    assert args.model == "SparseVAE"
    assert args.dataset == "CIFAR10"
    assert args.backbone == "cnn"
    assert args.latent_dim == 64
    assert args.epochs == 7
    assert args.sparse_weight == 2.5


def test_train_args_defaults():
    args = get_train_args([])
    assert args.model == "VAE"
    assert args.seed == 42


def test_valid_args_checkpoint_and_hf_model():
    """Validation parser accepts either a checkpoint or a HF model id."""
    a1 = get_valid_args(["--checkpoint", "outputs/run/best.pth"])
    assert a1.checkpoint == "outputs/run/best.pth"
    assert a1.hf_model is None

    a2 = get_valid_args(["--hf-model", "stabilityai/sd-vae-ft-mse"])
    assert a2.hf_model == "stabilityai/sd-vae-ft-mse"
    assert a2.checkpoint is None

    # Both optional at parse time (the runtime enforces exactly one).
    a3 = get_valid_args([])
    assert a3.checkpoint is None and a3.hf_model is None


# --------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------- #
def test_setup_logger_writes_to_file(tmp_path):
    """Logger writes messages to a file inside the given directory."""
    logger = setup_logger(str(tmp_path), log_name="test.log", name="test_logger")
    logger.info("hello-vae")

    for handler in logger.handlers:
        handler.flush()

    log_file = os.path.join(str(tmp_path), "test.log")
    assert os.path.isfile(log_file)
    with open(log_file, encoding="utf-8") as f:
        assert "hello-vae" in f.read()
    logging.shutdown()


# --------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------- #
def test_save_reconstruction_creates_file(tmp_path, mlp_config):
    """Reconstruction comparison images are written to disk."""
    model = build_model(mlp_config).eval()
    data = torch.rand(8, 1, 28, 28)
    path = os.path.join(str(tmp_path), "figs", "recon.png")
    save_reconstruction(model, data, path, n_samples=4)
    assert os.path.isfile(path)


def test_save_generated_samples_creates_file(tmp_path, mlp_config):
    """Generated sample grids are written to disk."""
    model = build_model(mlp_config).eval()
    path = os.path.join(str(tmp_path), "figs", "samples.png")
    save_generated_samples(model, torch.device("cpu"), path, n_samples=9)
    assert os.path.isfile(path)


# --------------------------------------------------------------------- #
# Closed-form / reference-value correctness of the loss functions
# --------------------------------------------------------------------- #
def test_kl_divergence_matches_closed_form_for_unit_variance():
    """For logvar=0, KL = 0.5 * sum(mu^2) averaged over the batch."""
    mu = torch.full((4, 16), 1.0)
    logvar = torch.zeros(4, 16)
    # Per sample: -0.5 * sum(1 + 0 - 1 - 1) = 0.5 * D ; here D = 16.
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.5 * 16, abs=1e-4)


def test_kl_divergence_matches_closed_form_with_variance():
    """KL matches the analytical Gaussian formula for arbitrary logvar."""
    d = 8
    v = math.log(2.0)  # variance = 2
    mu = torch.zeros(4, d)
    logvar = torch.full((4, d), v)
    # -0.5 * sum(1 + v - 0 - e^v) = -0.5 * d * (1 + v - 2) = 0.5 * d * (1 - v)
    expected = 0.5 * d * (1.0 - v)
    assert kl_divergence(mu, logvar).item() == pytest.approx(expected, abs=1e-4)


def test_reconstruction_bce_matches_reference():
    """BCE reconstruction equals the reference torch BCE (sum / batch)."""
    torch.manual_seed(0)
    x = torch.rand(6, 1, 8, 8)
    recon = 0.2 + 0.6 * torch.rand(6, 1, 8, 8)  # safely inside (0, 1)
    got = reconstruction_loss(recon, x, loss_type="bce")
    ref = F.binary_cross_entropy(
        recon.reshape(6, -1), x.reshape(6, -1), reduction="sum") / 6
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)


def test_reconstruction_mse_matches_reference():
    """MSE reconstruction equals the summed MSE divided by the batch size."""
    torch.manual_seed(0)
    x = torch.rand(6, 3, 8, 8)
    recon = torch.rand(6, 3, 8, 8)
    got = reconstruction_loss(recon, x, loss_type="mse")
    ref = F.mse_loss(recon, x, reduction="sum") / 6
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)


def test_sparsity_penalty_matches_bernoulli_kl():
    """The penalty equals the analytical Bernoulli KL summed over units."""
    rho, c, d = 0.1, 0.5, 8
    activations = torch.full((32, d), c)  # rho_hat == c for every unit
    expected = d * (rho * math.log(rho / c)
                    + (1 - rho) * math.log((1 - rho) / (1 - c)))
    got = sparsity_penalty(activations, target_sparsity=rho)
    assert got.item() == pytest.approx(expected, rel=1e-5)


def test_reconstruction_loss_independent_of_batch_size():
    """Per-sample normalization keeps loss scale stable across batch sizes."""
    torch.manual_seed(0)
    single = torch.rand(1, 1, 8, 8)
    recon = torch.rand(1, 1, 8, 8)
    big_x = single.repeat(16, 1, 1, 1)
    big_recon = recon.repeat(16, 1, 1, 1)
    loss1 = reconstruction_loss(recon, single, "mse")
    loss16 = reconstruction_loss(big_recon, big_x, "mse")
    assert loss1.item() == pytest.approx(loss16.item(), rel=1e-5)


# --------------------------------------------------------------------- #
# Metric edge cases & seeding completeness
# --------------------------------------------------------------------- #
def test_latent_sparsity_ratio_all_active():
    """Large positive means -> sigmoid ~1 -> no inactive units (ratio 0)."""
    mu = torch.full((8, 16), 20.0)
    assert latent_sparsity_ratio(mu, threshold=0.05) == pytest.approx(0.0)


def test_latent_sparsity_ratio_all_inactive():
    """Large negative means -> sigmoid ~0 -> all units inactive (ratio 1)."""
    mu = torch.full((8, 16), -20.0)
    assert latent_sparsity_ratio(mu, threshold=0.05) == pytest.approx(1.0)


def test_metric_tracker_tracks_multiple_keys():
    """The tracker maintains independent weighted averages per key."""
    tracker = MetricTracker()
    tracker.update({"a": 1.0, "b": 10.0}, n=1)
    tracker.update({"a": 3.0, "b": 30.0}, n=3)
    avgs = tracker.averages()
    assert avgs["a"] == pytest.approx((1.0 + 9.0) / 4)
    assert avgs["b"] == pytest.approx((10.0 + 90.0) / 4)


def test_set_seed_affects_numpy_and_random():
    """set_seed makes NumPy and Python's random reproducible too."""
    import random

    import numpy as np

    set_seed(99)
    a_np, a_py = np.random.rand(5), [random.random() for _ in range(5)]
    set_seed(99)
    b_np, b_py = np.random.rand(5), [random.random() for _ in range(5)]
    assert (a_np == b_np).all()
    assert a_py == b_py


def test_checkpoint_preserves_metadata(tmp_path, mlp_config):
    """Saved checkpoints round-trip their non-tensor metadata fields."""
    path = os.path.join(str(tmp_path), "ck.pth")
    save_checkpoint({"config": mlp_config, "epoch": 7,
                     "val_metrics": {"total_loss": 1.23}}, path)
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 7
    assert loaded["config"]["latent_dim"] == mlp_config["latent_dim"]
    assert loaded["val_metrics"]["total_loss"] == pytest.approx(1.23)


def test_options_reject_invalid_choices():
    """Invalid enum-like arguments cause argparse to exit."""
    with pytest.raises(SystemExit):
        get_train_args(["--model", "NotAModel"])
    with pytest.raises(SystemExit):
        get_train_args(["--backbone", "transformer"])
    with pytest.raises(SystemExit):
        get_train_args(["--dataset", "ImageNet"])


def test_options_hidden_dims_parse_as_list():
    """`--hidden-dims` accepts a variable-length integer list."""
    args = get_train_args(["--hidden-dims", "256", "128", "64"])
    assert args.hidden_dims == [256, 128, 64]
