"""Tests for dataset loading, preprocessing and the train/val split logic.

A lightweight in-memory fake dataset is monkeypatched into the registry so the
tests run instantly without downloading any real benchmark data.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import Dataset

import data.dataset as ds
from data import DatasetInfo, get_dataloaders


class _FakeImageDataset(Dataset):
    """Minimal stand-in for a torchvision image dataset."""

    def __init__(self, root=None, train=True, download=False, transform=None,
                 length=40, channels=1, size=28):
        self.length = length
        self.channels = channels
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random image already in [0, 1] (mimics ToTensor output) + label.
        return torch.rand(self.channels, self.size, self.size), 0


@pytest.fixture
def patch_dataset(monkeypatch):
    """Patch a registry entry so get_dataloaders uses the fake dataset."""
    def _patch(key="mnist", channels=1, size=28, length=40):
        def factory(root, train, download, transform):
            return _FakeImageDataset(length=length, channels=channels, size=size)
        monkeypatch.setitem(ds._DATASET_REGISTRY[key], "cls", factory)
    return _patch


def test_dataloaders_batch_shape_and_range(patch_dataset, tmp_path):
    """Batches have the expected shape and pixel values lie in [0, 1]."""
    patch_dataset(key="mnist", channels=1, size=28, length=40)
    train_loader, val_loader, info = get_dataloaders(
        dataset_name="MNIST", data_root=str(tmp_path),
        batch_size=8, val_split=0.25, num_workers=0)

    images, labels = next(iter(train_loader))
    assert images.shape == (8, 1, 28, 28)
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert labels.shape[0] == 8


def test_train_val_split_sizes(patch_dataset, tmp_path):
    """The split partitions the data according to `val_split`."""
    patch_dataset(key="mnist", channels=1, size=28, length=40)
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name="MNIST", data_root=str(tmp_path),
        batch_size=4, val_split=0.25, num_workers=0)

    train_count = sum(b[0].size(0) for b in train_loader)
    val_count = sum(b[0].size(0) for b in val_loader)
    # 40 samples, 25% validation -> 30 train / 10 val (train drops last partial).
    assert val_count == 10
    assert train_count <= 30


def test_loader_is_iterable_multiple_times(patch_dataset, tmp_path):
    """The validation loader can be iterated more than once."""
    patch_dataset(key="mnist", length=24)
    _, val_loader, _ = get_dataloaders(
        dataset_name="MNIST", data_root=str(tmp_path),
        batch_size=8, val_split=0.5, num_workers=0)
    first = sum(1 for _ in val_loader)
    second = sum(1 for _ in val_loader)
    assert first == second and first > 0


def test_cifar_shape(patch_dataset, tmp_path):
    """CIFAR-10 style data yields 3x32x32 batches."""
    patch_dataset(key="cifar10", channels=3, size=32, length=40)
    train_loader, _, info = get_dataloaders(
        dataset_name="CIFAR10", data_root=str(tmp_path),
        batch_size=8, val_split=0.25, num_workers=0)
    images, _ = next(iter(train_loader))
    assert images.shape == (8, 3, 32, 32)
    assert info.image_channels == 3 and info.image_size == 32


def test_dataset_info_input_dim():
    """DatasetInfo computes the flattened input dimension correctly."""
    info = DatasetInfo(name="mnist", image_channels=1, image_size=28)
    assert info.input_dim == 784
    info_rgb = DatasetInfo(name="cifar10", image_channels=3, image_size=32)
    assert info_rgb.input_dim == 3 * 32 * 32


def test_unknown_dataset_raises():
    """Requesting an unsupported dataset raises a clear error."""
    with pytest.raises(ValueError):
        get_dataloaders(dataset_name="NotADataset", num_workers=0)


def test_build_transform_returns_callable_pipeline():
    """`build_transform` returns a usable preprocessing pipeline."""
    transform = ds.build_transform(image_size=28, resize=None)
    assert transform is not None
    # With a resize different from the native size, the pipeline still builds.
    transform_resized = ds.build_transform(image_size=28, resize=32)
    assert transform_resized is not None


# --------------------------------------------------------------------- #
# Dataset name normalization & metadata
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("MNIST", "mnist"),
    ("mnist", "mnist"),
    ("Fashion-MNIST", "fashionmnist"),
    ("fashion_mnist", "fashionmnist"),
    ("FashionMNIST", "fashionmnist"),
    ("CIFAR-10", "cifar10"),
    ("cifar10", "cifar10"),
])
def test_normalize_name_variants(name, expected):
    """Dataset names are normalized regardless of case / separators."""
    assert ds._normalize_name(name) == expected


def test_normalize_name_rejects_unknown():
    with pytest.raises(ValueError):
        ds._normalize_name("svhn")


def test_fashionmnist_shape(patch_dataset, tmp_path):
    """Fashion-MNIST style data yields 1x28x28 batches."""
    patch_dataset(key="fashionmnist", channels=1, size=28, length=32)
    train_loader, _, info = get_dataloaders(
        dataset_name="FashionMNIST", data_root=str(tmp_path),
        batch_size=8, val_split=0.25, num_workers=0)
    images, _ = next(iter(train_loader))
    assert images.shape == (8, 1, 28, 28)
    assert info.name == "fashionmnist"


@pytest.mark.parametrize("val_split,length,expected_val", [
    (0.1, 100, 10),
    (0.2, 50, 10),
    (0.5, 40, 20),
])
def test_val_split_ratio(patch_dataset, tmp_path, val_split, length, expected_val):
    """Validation subset size matches the requested split ratio."""
    patch_dataset(key="mnist", length=length)
    _, val_loader, _ = get_dataloaders(
        dataset_name="MNIST", data_root=str(tmp_path),
        batch_size=5, val_split=val_split, num_workers=0)
    val_count = sum(b[0].size(0) for b in val_loader)
    assert val_count == expected_val


def test_split_is_reproducible_with_same_seed(patch_dataset, tmp_path):
    """A fixed seed yields the same train/val partition across calls."""
    patch_dataset(key="mnist", length=40)
    common = dict(dataset_name="MNIST", data_root=str(tmp_path),
                  batch_size=40, val_split=0.25, num_workers=0, seed=7)
    _, val1, _ = get_dataloaders(**common)
    _, val2, _ = get_dataloaders(**common)
    # Same seed -> identical validation indices -> identical underlying data.
    idx1 = sorted(val1.dataset.indices)
    idx2 = sorted(val2.dataset.indices)
    assert idx1 == idx2


def test_train_loader_drops_last_partial_batch(patch_dataset, tmp_path):
    """The training loader drops the final partial batch for stable shapes."""
    patch_dataset(key="mnist", length=40)  # 30 train samples, batch 8
    train_loader, _, _ = get_dataloaders(
        dataset_name="MNIST", data_root=str(tmp_path),
        batch_size=8, val_split=0.25, num_workers=0)
    for images, _ in train_loader:
        assert images.size(0) == 8  # no smaller trailing batch


def test_creates_data_root_directory(patch_dataset, tmp_path):
    """The download directory is created automatically if missing."""
    target = os.path.join(str(tmp_path), "nested", "dataset")
    assert not os.path.exists(target)
    patch_dataset(key="mnist", length=16)
    get_dataloaders(dataset_name="MNIST", data_root=target,
                    batch_size=8, val_split=0.5, num_workers=0)
    assert os.path.isdir(target)
