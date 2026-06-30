"""Dataset loading and preprocessing for VAE experiments.

Provides a single unified entry point :func:`get_dataloaders` that supports
MNIST, Fashion-MNIST and CIFAR-10 with automatic download. Data loading is
fully decoupled from training logic and uses only relative paths so the
repository stays portable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Module logger; integrates with the experiment logger configured in
# train.py / valid.py (logger name "vae") so download status is visible.
logger = logging.getLogger("vae")

# Per-dataset metadata: torchvision class, channels and native image size.
_DATASET_REGISTRY = {
    "mnist": {"cls": datasets.MNIST, "channels": 1, "size": 28},
    "fashionmnist": {"cls": datasets.FashionMNIST, "channels": 1, "size": 28},
    "cifar10": {"cls": datasets.CIFAR10, "channels": 3, "size": 32},
    # Larger datasets for richer VAE training:
    #   CelebA  — ~200k celebrity face images (3×178×218), auto-downloads.
    #   STL10   — 100k unlabeled + 5k labelled images (3×96×96), designed for
    #             unsupervised / representation learning.
    "celeba": {"cls": datasets.CelebA, "channels": 3, "size": 178},
    "stl10": {"cls": datasets.STL10, "channels": 3, "size": 96},
}


@dataclass
class DatasetInfo:
    """Lightweight container describing dataset tensor shapes."""

    name: str
    image_channels: int
    image_size: int

    @property
    def input_dim(self) -> int:
        """Flattened input dimensionality (for the MLP backbone)."""
        return self.image_channels * self.image_size * self.image_size


def _normalize_name(dataset_name: str) -> str:
    """Normalize a user-provided dataset name to its registry key."""
    key = dataset_name.lower().replace("-", "").replace("_", "")
    if key not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            f"Choose from {{'MNIST', 'FashionMNIST', 'CIFAR10'}}."
        )
    return key


def build_transform(image_size: int, resize: int | None = None) -> transforms.Compose:
    """Build the preprocessing pipeline.

    Images are converted to tensors with values in ``[0, 1]`` (which matches
    the Sigmoid output of the decoders and the BCE reconstruction loss). An
    optional resize is supported for custom resolutions.

    Args:
        image_size: Native dataset image size.
        resize: Optional target size; if given and different from the native
            size, a resize transform is prepended.
    """
    ops = []
    if resize is not None and resize != image_size:
        ops.append(transforms.Resize((resize, resize)))
    ops.append(transforms.ToTensor())  # scales pixels to [0, 1]
    return transforms.Compose(ops)


def _load_split(dataset_cls, root: str, train: bool, transform,
                dataset_key: str = "") -> Dataset:
    """Load one dataset split, downloading it only if not already present.

    Different torchvision datasets use different constructor signatures:
    ``MNIST``/``CIFAR10`` use ``train=True/False``; ``CelebA`` uses
    ``split='train'/'valid'/'test'``; ``STL10`` uses ``split='train'/'test'/'unlabeled'``.
    This helper abstracts that away.

    Args:
        dataset_cls: A torchvision dataset class.
        root: Directory where the dataset is stored.
        train: Whether to load the training split.
        transform: Preprocessing pipeline applied to each sample.
        dataset_key: Registry key, used to pick the correct constructor args.

    Returns:
        The instantiated dataset split.
    """
    split_label = "train" if train else "test"

    # Build kwargs appropriate for each dataset's constructor.
    if dataset_key == "celeba":
        # CelebA: split='train'/'valid'/'test', no `train` param.
        kwargs = dict(split=split_label)
    elif dataset_key == "stl10":
        # STL10: split='train'/'test'/'unlabeled'/'train+unlabeled'
        kwargs = dict(split=split_label)
    else:
        # Standard MNIST / FashionMNIST / CIFAR10 style.
        kwargs = dict(train=train)

    try:
        dataset = dataset_cls(root=root, download=False, transform=transform,
                              **kwargs)
        logger.info(f"Found existing {split_label} dataset under '{root}'.")
        return dataset
    except (RuntimeError, FileNotFoundError):
        logger.info(f"{split_label.capitalize()} dataset not found under '{root}', "
                    f"downloading ...")
        return dataset_cls(root=root, download=True, transform=transform,
                           **kwargs)


def get_dataloaders(dataset_name: str = "MNIST",
                    data_root: str = "./dataset",
                    batch_size: int = 128,
                    val_split: float = 0.1,
                    num_workers: int = 4,
                    resize: int | None = None,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DatasetInfo]:
    """Create training and validation dataloaders for a benchmark dataset.

    The official training split is partitioned into train / validation subsets
    according to ``val_split``. The dataset is downloaded automatically to
    ``data_root`` on first use and reused on subsequent runs.

    Args:
        dataset_name: One of ``"MNIST"``, ``"FashionMNIST"``, ``"CIFAR10"``.
        data_root: Relative directory where raw datasets are stored / cached.
        batch_size: Mini-batch size for both loaders.
        val_split: Fraction of the training set reserved for validation.
        num_workers: Number of worker processes for data loading.
        resize: Optional image resize target.
        seed: Seed controlling the reproducible train / validation split.

    Returns:
        A tuple ``(train_loader, val_loader, info)``.
    """
    key = _normalize_name(dataset_name)
    meta = _DATASET_REGISTRY[key]
    dataset_cls = meta["cls"]
    image_size = resize if resize is not None else meta["size"]

    # Ensure the (relative) download directory exists before fetching data.
    os.makedirs(data_root, exist_ok=True)

    transform = build_transform(meta["size"], resize)
    full_train = _load_split(dataset_cls, data_root, train=True,
                             transform=transform, dataset_key=key)

    # Reproducible train / validation split driven by a fixed generator.
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # `pin_memory` accelerates host-to-device transfer when CUDA is available.
    pin_memory = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)

    info = DatasetInfo(name=key,
                       image_channels=meta["channels"],
                       image_size=image_size)
    return train_loader, val_loader, info
