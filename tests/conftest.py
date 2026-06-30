"""Shared pytest fixtures and path setup for the VAE test suite.

Ensures the project root is importable regardless of how pytest is invoked,
and provides reusable dummy configurations / tensors / dataloaders so that
individual test modules stay concise and fast.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Make the project root importable (so `import model`, `utils`, ... work)
# no matter the current working directory used to launch pytest.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture
def device() -> torch.device:
    """CPU device used by all tests for determinism and portability."""
    return torch.device("cpu")


@pytest.fixture
def mlp_config() -> dict:
    """A small MLP-backbone configuration (MNIST-like, 1x28x28)."""
    return dict(
        model="VAE", backbone="mlp", image_channels=1, image_size=28,
        hidden_dims=[128, 64], latent_dim=16, activation="relu",
        recon_loss_type="bce", beta=1.0,
        target_sparsity=0.05, sparse_weight=1.0,
    )


@pytest.fixture
def cnn_config() -> dict:
    """A small CNN-backbone configuration (CIFAR-like, 3x32x32)."""
    return dict(
        model="VAE", backbone="cnn", image_channels=3, image_size=32,
        hidden_dims=[32, 64, 128], latent_dim=16, activation="relu",
        recon_loss_type="mse", beta=1.0,
        target_sparsity=0.05, sparse_weight=1.0,
    )


@pytest.fixture
def make_batch():
    """Factory returning a random image batch in ``[0, 1]``."""
    def _make(n: int = 4, channels: int = 1, size: int = 28) -> torch.Tensor:
        return torch.rand(n, channels, size, size)
    return _make


@pytest.fixture
def dummy_loader():
    """Factory building a tiny in-memory dataloader of random images."""
    from torch.utils.data import DataLoader, TensorDataset

    def _make(n: int = 16, channels: int = 1, size: int = 28,
              batch_size: int = 8) -> DataLoader:
        images = torch.rand(n, channels, size, size)
        labels = torch.zeros(n, dtype=torch.long)
        return DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    return _make
