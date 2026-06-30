"""Evaluation metric utilities for VAE models.

These helpers turn the raw per-batch loss tensors into aggregated, reportable
quantities and provide latent-space diagnostics such as the activation
sparsity ratio.
"""

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def latent_sparsity_ratio(mu: torch.Tensor, threshold: float = 0.05) -> float:
    """Measure the fraction of (near-)inactive latent units.

    Latent means are mapped through a sigmoid to bounded activations and the
    proportion of activations below ``threshold`` is returned. A higher value
    indicates a sparser latent representation.

    Args:
        mu: Latent mean tensor of shape ``(B, latent_dim)``.
        threshold: Activation level below which a unit is deemed inactive.

    Returns:
        The sparsity ratio as a Python float in ``[0, 1]``.
    """
    activations = torch.sigmoid(mu)
    return (activations < threshold).float().mean().item()


class MetricTracker:
    """Accumulates loss / metric values across an epoch and reports averages."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._count = 0

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        """Add a batch of scalar metrics weighted by sample count ``n``."""
        for key, value in metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value) * n
        self._count += n

    def averages(self) -> Dict[str, float]:
        """Return the per-sample average of every tracked metric."""
        if self._count == 0:
            return {key: 0.0 for key in self._sums}
        return {key: total / self._count for key, total in self._sums.items()}
