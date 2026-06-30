"""Sparse Variational Autoencoder.

Extends the standard :class:`~model.vae.VAE` by adding an explicit sparsity
constraint on the latent activations. The constraint pushes the average
activation of each latent unit towards a small target value ``rho`` using a
KL-divergence based penalty (as in classic sparse autoencoders).

The class keeps full interface compatibility with the training / validation
pipelines and can be used as a drop-in replacement for the standard VAE.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from model.vae import VAE
from utils.losses import sparsity_penalty


class SparseVAE(VAE):
    """Variational Autoencoder with a sparsity-regularized latent space.

    Args:
        target_sparsity: Desired average activation level ``rho`` in ``(0, 1)``.
        sparse_weight: Weight applied to the sparsity penalty term.
        (Remaining args are inherited from :class:`~model.vae.VAE`.)
    """

    def __init__(self,
                 image_channels: int = 1,
                 image_size: int = 28,
                 backbone: str = "mlp",
                 hidden_dims: Sequence[int] = (512, 256),
                 latent_dim: int = 32,
                 activation: str = "relu",
                 recon_loss_type: str = "bce",
                 beta: float = 1.0,
                 target_sparsity: float = 0.05,
                 sparse_weight: float = 1.0):
        super().__init__(
            image_channels=image_channels,
            image_size=image_size,
            backbone=backbone,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            recon_loss_type=recon_loss_type,
            beta=beta,
        )
        if not 0.0 < target_sparsity < 1.0:
            raise ValueError(
                f"target_sparsity must be in (0, 1), got {target_sparsity}.")
        self.target_sparsity = target_sparsity
        self.sparse_weight = sparse_weight

    @staticmethod
    def _latent_activation(mu: torch.Tensor) -> torch.Tensor:
        """Map the latent mean into bounded activations in ``(0, 1)``.

        A sigmoid is used so the resulting values can be interpreted as
        activation probabilities for the Bernoulli-based sparsity penalty.
        """
        return torch.sigmoid(mu)

    def loss_function(self, x: torch.Tensor,
                      output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss plus the sparsity penalty.

        Returns:
            Dict with ``total_loss``, ``recon_loss``, ``kl_loss``,
            ``sparse_loss`` and ``sparsity_ratio`` (the measured fraction of
            near-zero latent activations, for monitoring).
        """
        # Base reconstruction + KL terms from the parent VAE.
        losses = super().loss_function(x, output)

        # Sparsity penalty driving mean activations towards `target_sparsity`.
        activations = self._latent_activation(output["mu"])
        sparse_loss = sparsity_penalty(activations, self.target_sparsity)

        losses["total_loss"] = losses["total_loss"] + self.sparse_weight * sparse_loss
        losses["sparse_loss"] = sparse_loss

        # Monitoring metric: fraction of activations close to zero (inactive).
        with torch.no_grad():
            ratio = (activations < self.target_sparsity).float().mean()
        losses["sparsity_ratio"] = ratio
        return losses
