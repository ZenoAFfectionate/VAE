"""Shared helpers for the VAE example demos.

Provides model / data loading utilities built on the project's own modules, plus
a set of small, dependency-free, unit-testable functions (interpolation,
noising, masking, AUC, PCA) reused across the demos.

Heavy / optional dependencies (matplotlib, gradio, scikit-learn) are imported
lazily inside the functions that need them, so importing this module stays cheap
and test-friendly.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

# Ensure the project root is importable when a demo is launched directly
# (e.g. `python examples/latent_interpolation.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402

from data import get_dataloaders  # noqa: E402
from utils.helpers import build_model, get_device, load_checkpoint, set_seed  # noqa: E402


# ===================================================================== #
# Model & data loading (reuse the project's own factory / pipeline)
# ===================================================================== #
def load_model(checkpoint: str, device: torch.device | None = None):
    """Load a trained model from a checkpoint produced by ``train.py``.

    Args:
        checkpoint: Path to the ``.pth`` checkpoint.
        device: Target device; auto-detected when ``None``.

    Returns:
        Tuple ``(model, config, device)`` with the model in eval mode.
    """
    device = device or get_device()
    ckpt = load_checkpoint(checkpoint, map_location=device)
    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config, device


def get_eval_loader(config: dict, batch_size: int = 128, num_workers: int = 0,
                    dataset_name: str | None = None):
    """Build a validation dataloader matching a checkpoint's data settings.

    Args:
        config: The checkpoint configuration dict.
        batch_size: Batch size for the loader.
        num_workers: Number of dataloader workers.
        dataset_name: Override the dataset (e.g. for cross-dataset anomaly
            detection); defaults to the training dataset.

    Returns:
        Tuple ``(val_loader, info)``.
    """
    _, val_loader, info = get_dataloaders(
        dataset_name=dataset_name or config["dataset"],
        data_root=config["data_root"],
        batch_size=batch_size,
        val_split=config["val_split"],
        num_workers=num_workers,
        resize=config.get("resize"),
        seed=config["seed"],
    )
    return val_loader, info


@torch.no_grad()
def collect_samples(loader, n: int, device: torch.device,
                    per_class: int | None = None):
    """Gather up to ``n`` images (optionally balanced per class) from a loader.

    Args:
        loader: A dataloader yielding ``(images, labels)``.
        n: Total number of images to collect (ignored if ``per_class`` set).
        device: Device to move the collected tensors to.
        per_class: If given, collect this many images for each distinct label.

    Returns:
        Tuple ``(images, labels)`` tensors.
    """
    images, labels = [], []
    counts: dict = {}
    stale_batches = 0               # prevent looping forever in per_class mode
    for batch_imgs, batch_lbls in loader:
        added = 0
        for img, lbl in zip(batch_imgs, batch_lbls):
            key = int(lbl)
            if per_class is not None:
                if counts.get(key, 0) >= per_class:
                    continue
                counts[key] = counts.get(key, 0) + 1
            images.append(img)
            labels.append(lbl)
            added += 1
            if per_class is None and len(images) >= n:
                break
        if per_class is None and len(images) >= n:
            break
        # When every batch contributes zero new images, the loader has no
        # more unseen classes to offer — quit early.
        if per_class is not None and added == 0:
            stale_batches += 1
            if stale_batches >= 3:
                break
        else:
            stale_batches = 0
    imgs = torch.stack(images).to(device)
    lbls = torch.tensor([int(x) for x in labels])
    return imgs, lbls


def encode_mean(model, x: torch.Tensor) -> torch.Tensor:
    """Return the latent mean ``mu`` of inputs (the deterministic code)."""
    mu, _ = model.encode(x)
    return mu


@torch.no_grad()
def build_interpolation_grid(model, start_imgs: torch.Tensor,
                             end_imgs: torch.Tensor, steps: int,
                             use_slerp: bool = False) -> torch.Tensor:
    """Build an interpolation grid morphing each start image into its end image.

    Args:
        model: A trained VAE exposing ``encode`` / ``decode``.
        start_imgs: Start images ``(P, C, H, W)``.
        end_imgs: End images ``(P, C, H, W)`` (same count as ``start_imgs``).
        steps: Number of interpolation steps (grid columns).
        use_slerp: Use spherical interpolation instead of linear.

    Returns:
        A tensor ``(P * steps, C, H, W)`` ordered row-major (one row per pair);
        render it with ``nrow=steps``.
    """
    if start_imgs.size(0) != end_imgs.size(0):
        raise ValueError("start_imgs and end_imgs must have the same length.")
    mu_start = encode_mean(model, start_imgs)
    mu_end = encode_mean(model, end_imgs)
    interp = slerp if use_slerp else lerp

    rows = [model.decode(interp(mu_start, mu_end, float(t))).cpu()
            for t in interpolation_steps(steps)]
    # (steps, P, C, H, W) -> (P, steps, C, H, W) -> (P*steps, C, H, W)
    return torch.stack(rows, dim=1).reshape(-1, *rows[0].shape[1:])


@torch.no_grad()
def build_traversal_grid(model, image: torch.Tensor, num_dims: int,
                         steps: int, span: float):
    """Build a latent-traversal grid for a single anchor image.

    Each row sweeps one latent dimension across ``[-span, span]`` while keeping
    the other dimensions fixed at the image's encoded value.

    Args:
        model: A trained VAE exposing ``encode`` / ``decode``.
        image: A single anchor image ``(1, C, H, W)``.
        num_dims: Number of latent dimensions to traverse (grid rows).
        steps: Number of values per dimension (grid columns).
        span: Sweep range; each dim is varied over ``[-span, span]``.

    Returns:
        Tuple ``(grid, n_dims)`` where grid is ``(n_dims * steps, C, H, W)``;
        render with ``nrow=steps``.
    """
    base_mu = encode_mean(model, image)  # (1, latent_dim)
    latent_dim = base_mu.size(1)
    n_dims = min(num_dims, latent_dim)
    values = torch.linspace(-span, span, max(steps, 2), device=base_mu.device)

    rows = []
    for dim in range(n_dims):
        for val in values:
            z = base_mu.clone()
            z[0, dim] = val  # override a single latent dimension
            rows.append(model.decode(z).cpu())
    return torch.cat(rows, dim=0), n_dims


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ===================================================================== #
# Pure, dependency-free helpers (unit tested in tests/test_examples.py)
# ===================================================================== #
def lerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two latent codes."""
    return (1.0 - t) * z0 + t * z1


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float,
          eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two latent codes.

    Falls back to :func:`lerp` when the two vectors are nearly collinear.
    """
    z0_flat, z1_flat = z0.reshape(-1), z1.reshape(-1)
    n0 = z0_flat.norm() + eps
    n1 = z1_flat.norm() + eps
    cos = torch.dot(z0_flat, z1_flat) / (n0 * n1)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos)
    sin = torch.sin(omega)
    if sin.abs() < eps:
        return lerp(z0, z1, t)
    a = torch.sin((1.0 - t) * omega) / sin
    b = torch.sin(t * omega) / sin
    return a * z0 + b * z1


def interpolation_steps(n: int) -> torch.Tensor:
    """Return ``n`` evenly spaced interpolation coefficients in ``[0, 1]``."""
    return torch.linspace(0.0, 1.0, max(n, 2))


def add_gaussian_noise(x: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    """Add Gaussian noise to images in ``[0, 1]`` and clamp back to range."""
    return torch.clamp(x + std * torch.randn_like(x), 0.0, 1.0)


def add_salt_pepper(x: torch.Tensor, amount: float = 0.1) -> torch.Tensor:
    """Apply salt-and-pepper corruption to a fraction ``amount`` of pixels."""
    noisy = x.clone()
    probs = torch.rand_like(x)
    noisy[probs < amount / 2] = 0.0          # pepper
    noisy[probs > 1.0 - amount / 2] = 1.0    # salt
    return noisy


def make_center_mask(height: int, width: int, frac: float = 0.5) -> torch.Tensor:
    """Build a binary mask (1=keep, 0=hole) with a centered square hole.

    Args:
        height: Image height.
        width: Image width.
        frac: Side length of the hole as a fraction of the image size.

    Returns:
        A ``(1, H, W)`` float mask with the central square zeroed out.
    """
    mask = torch.ones(1, height, width)
    h_hole = int(height * frac)
    w_hole = int(width * frac)
    top = (height - h_hole) // 2
    left = (width - w_hole) // 2
    mask[:, top:top + h_hole, left:left + w_hole] = 0.0
    return mask


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute ROC-AUC where higher ``score`` indicates the positive class.

    Uses the rank-sum (Mann-Whitney U) formulation, robust to ties.

    Args:
        scores: 1D tensor of anomaly scores.
        labels: 1D tensor of binary labels (1 = positive / anomaly, 0 = normal).

    Returns:
        The AUC as a float in ``[0, 1]`` (0.5 if a class is empty).
    """
    scores = scores.flatten().float()
    labels = labels.flatten().float()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks (1-indexed) handle ties correctly.
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    # Resolve ties by averaging ranks of equal scores.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    rank_sum_pos = ranks[labels == 1].sum().item()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def psnr(clean: torch.Tensor, recon: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB) between two ``[0, max_val]`` images."""
    mse = torch.mean((clean - recon) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    import math
    return 10.0 * math.log10((max_val ** 2) / mse)


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """Project feature vectors to 2D via PCA (top-2 principal components).

    Args:
        x: Tensor of shape ``(N, D)``.

    Returns:
        Tensor of shape ``(N, 2)`` (the 2D projection).
    """
    x = x.reshape(x.size(0), -1).float()
    x_centered = x - x.mean(dim=0, keepdim=True)
    # SVD-based PCA: columns of V are the principal directions.
    _, _, v = torch.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ v[:2].T
