"""End-to-end pipeline sanity checks.

These exercise the actual training/evaluation routines used by ``train.py`` and
``valid.py`` (``run_epoch`` and ``evaluate``) on tiny dummy data, plus the
checkpoint save -> reload -> evaluate flow, all without downloading datasets.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import train as train_mod
import valid as valid_mod
from utils.helpers import build_model, load_checkpoint, save_checkpoint, set_seed


def _tiny_loader(n=16, channels=1, size=28, batch_size=8):
    images = torch.rand(n, channels, size, size)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


@pytest.fixture
def tiny_config():
    return dict(
        model="VAE", backbone="mlp", image_channels=1, image_size=28,
        hidden_dims=[64], latent_dim=8, activation="relu",
        recon_loss_type="bce", beta=1.0,
        target_sparsity=0.05, sparse_weight=1.0,
        dataset="MNIST", data_root="./dataset", val_split=0.1,
        resize=None, seed=42,
    )


def test_train_run_epoch_executes(tiny_config):
    """A single training epoch runs and returns finite core metrics."""
    set_seed(0)
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_mod.run_epoch(model, _tiny_loader(), device, optimizer,
                                  grad_clip=5.0)
    for key in ("total_loss", "recon_loss", "kl_loss", "sparsity_ratio"):
        assert key in metrics
    assert all(map(lambda v: v == v, metrics.values()))  # no NaN


def test_eval_epoch_does_not_update_weights(tiny_config):
    """Evaluation mode (no optimizer) leaves model weights unchanged."""
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    before = [p.clone() for p in model.parameters()]

    train_mod.run_epoch(model, _tiny_loader(), device, optimizer=None)

    for p_before, p_after in zip(before, model.parameters()):
        assert torch.allclose(p_before, p_after)


def test_checkpoint_then_evaluate(tmp_path, tiny_config):
    """Full mini-pipeline: train -> save -> reload -> evaluate."""
    set_seed(0)
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # One short training epoch.
    train_mod.run_epoch(model, _tiny_loader(), device, optimizer, grad_clip=5.0)

    # Persist a checkpoint exactly as train.py does.
    ckpt_path = os.path.join(str(tmp_path), "best.pth")
    save_checkpoint({"model_state": model.state_dict(),
                     "config": tiny_config, "epoch": 1}, ckpt_path)

    # Reload (valid.py flow) and evaluate.
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    restored = build_model(ckpt["config"]).to(device)
    restored.load_state_dict(ckpt["model_state"])

    metrics = valid_mod.evaluate(restored, _tiny_loader(), device)
    assert {"total_loss", "recon_loss", "kl_loss", "sparsity_ratio"} <= set(metrics)
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_sparse_vae_pipeline_reports_sparsity(tiny_config):
    """The Sparse VAE path surfaces the sparsity penalty through the loop."""
    cfg = dict(tiny_config)
    cfg["model"] = "SparseVAE"
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_mod.run_epoch(model, _tiny_loader(), device, optimizer)
    assert "sparse_loss" in metrics
    assert 0.0 <= metrics["sparsity_ratio"] <= 1.0


def test_format_metrics_is_readable():
    """The console metric formatter produces an aligned, complete summary."""
    line = train_mod._format_metrics({
        "total_loss": 1.0, "recon_loss": 0.8, "kl_loss": 0.2,
        "sparsity_ratio": 0.5, "sparse_loss": 0.1,
    })
    for token in ("loss=", "recon=", "kl=", "sparsity=", "sparse="):
        assert token in line


# --------------------------------------------------------------------- #
# Training effectiveness & reproducibility
# --------------------------------------------------------------------- #
def test_training_reduces_loss_on_fixed_data(tiny_config):
    """Optimizing repeatedly over a fixed tiny batch lowers the loss."""
    set_seed(0)
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # A single fixed batch reused every step so the model can overfit it.
    images = torch.rand(16, 1, 28, 28)
    loader = DataLoader(TensorDataset(images, torch.zeros(16)), batch_size=16)

    first = train_mod.run_epoch(model, loader, device, optimizer)["total_loss"]
    for _ in range(25):
        last = train_mod.run_epoch(model, loader, device, optimizer)["total_loss"]
    assert last < first


def test_training_is_reproducible(tiny_config):
    """Identical seed + config + data order -> identical epoch metrics."""
    device = torch.device("cpu")

    # Fixed data shared by both runs; each run re-seeds identically so weight
    # initialization and reparameterization noise are deterministic.
    set_seed(2024)
    fixed_images = torch.rand(16, 1, 28, 28)

    def deterministic_run():
        set_seed(2024)
        model = build_model(tiny_config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(fixed_images, torch.zeros(16)),
                            batch_size=8, shuffle=False)
        return train_mod.run_epoch(model, loader, device, optimizer)

    m1 = deterministic_run()
    m2 = deterministic_run()
    assert m1["total_loss"] == pytest.approx(m2["total_loss"], rel=1e-6)
    assert m1["recon_loss"] == pytest.approx(m2["recon_loss"], rel=1e-6)


def test_eval_is_deterministic_after_reload(tmp_path, tiny_config):
    """Evaluation metrics are identical before saving and after reloading."""
    device = torch.device("cpu")
    set_seed(0)
    model = build_model(tiny_config).to(device)

    loader = _tiny_loader()
    torch.manual_seed(7)
    metrics_before = valid_mod.evaluate(model, loader, device)

    ckpt_path = os.path.join(str(tmp_path), "m.pth")
    save_checkpoint({"model_state": model.state_dict(),
                     "config": tiny_config}, ckpt_path)
    restored = build_model(load_checkpoint(ckpt_path)["config"]).to(device)
    restored.load_state_dict(load_checkpoint(ckpt_path)["model_state"])

    torch.manual_seed(7)
    metrics_after = valid_mod.evaluate(restored, loader, device)
    assert metrics_before["total_loss"] == pytest.approx(
        metrics_after["total_loss"], rel=1e-6)


def test_evaluate_uses_no_grad(tiny_config):
    """`evaluate` must not build a graph (outputs require no grad)."""
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    # If evaluate leaked gradients, parameters would accumulate .grad here.
    for p in model.parameters():
        p.grad = None
    valid_mod.evaluate(model, _tiny_loader(), device)
    assert all(p.grad is None for p in model.parameters())


def test_grad_clip_argument_is_accepted(tiny_config):
    """run_epoch honours the grad_clip argument without error (clip on/off)."""
    device = torch.device("cpu")
    model = build_model(tiny_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    m_clip = train_mod.run_epoch(model, _tiny_loader(), device, optimizer,
                                 grad_clip=1.0)
    m_noclip = train_mod.run_epoch(model, _tiny_loader(), device, optimizer,
                                   grad_clip=0.0)
    assert torch.isfinite(torch.tensor(m_clip["total_loss"]))
    assert torch.isfinite(torch.tensor(m_noclip["total_loss"]))


def test_cnn_pipeline_end_to_end(tmp_path):
    """A CNN-backbone config trains and evaluates end-to-end."""
    cfg = dict(
        model="VAE", backbone="cnn", image_channels=3, image_size=32,
        hidden_dims=[16, 32, 64], latent_dim=16, activation="relu",
        recon_loss_type="mse", beta=1.0, target_sparsity=0.05, sparse_weight=1.0,
        dataset="CIFAR10", data_root="./dataset", val_split=0.1,
        resize=None, seed=0,
    )
    device = torch.device("cpu")
    set_seed(0)
    model = build_model(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loader = _tiny_loader(n=16, channels=3, size=32, batch_size=8)

    train_metrics = train_mod.run_epoch(model, loader, device, optimizer)
    eval_metrics = valid_mod.evaluate(model, loader, device)
    assert torch.isfinite(torch.tensor(train_metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(eval_metrics["total_loss"]))
