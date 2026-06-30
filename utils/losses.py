"""Centralized loss function implementations for VAE / Sparse VAE.

All functions are written to be device-agnostic and numerically stable.
Reconstruction and KL terms are normalized per sample (averaged over the
batch dimension) so that loss magnitudes stay comparable across different
batch sizes and image resolutions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Small constant used for numerical stability in log / division operations.
_EPS = 1e-8


def reconstruction_loss(recon_x: torch.Tensor,
                        x: torch.Tensor,
                        loss_type: str = "bce") -> torch.Tensor:
    """Compute the reconstruction loss between inputs and reconstructions.

    Args:
        recon_x: Reconstructed tensor with values in ``[0, 1]`` (after Sigmoid).
            Shape ``(B, ...)``.
        x: Ground-truth tensor with values in ``[0, 1]``. Shape ``(B, ...)``.
        loss_type: Either ``"bce"`` (binary cross entropy) or ``"mse"``
            (mean squared error).

    Returns:
        A scalar tensor: the reconstruction loss averaged over the batch
        (summed over all non-batch dimensions for each sample).
    """
    batch_size = x.size(0)
    # Flatten everything except the batch dimension so the sum is taken
    # over all pixel / feature dimensions of each individual sample.
    recon_flat = recon_x.reshape(batch_size, -1)
    x_flat = x.reshape(batch_size, -1)

    if loss_type == "bce":
        # Clamp to avoid log(0) numerical issues inside BCE.
        recon_flat = recon_flat.clamp(min=_EPS, max=1.0 - _EPS)
        loss = F.binary_cross_entropy(recon_flat, x_flat, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(recon_flat, x_flat, reduction="none")
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {loss_type!r}. "
                         f"Choose from {{'bce', 'mse'}}.")

    # Sum over feature dimensions, then average over the batch.
    return loss.sum(dim=1).mean(dim=0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between ``N(mu, sigma^2)`` and ``N(0, I)``.

    The closed-form expression is::

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: Latent mean tensor of shape ``(B, latent_dim)``.
        logvar: Latent log-variance tensor of shape ``(B, latent_dim)``.

    Returns:
        A scalar tensor: the KL divergence averaged over the batch.
    """
    # Clamp log-variance to a safe range to prevent exp() overflow / NaNs.
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kld_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    return kld_per_sample.mean(dim=0)


def sparsity_penalty(activations: torch.Tensor,
                     target_sparsity: float = 0.05) -> torch.Tensor:
    """KL-divergence based sparsity penalty for Sparse VAE latent activations.

    Encourages the average activation of each latent unit (over the batch) to
    approach a small target value ``rho``. The penalty for a single unit is the
    KL divergence between two Bernoulli distributions::

        KL(rho || rho_hat) = rho * log(rho / rho_hat)
                           + (1 - rho) * log((1 - rho) / (1 - rho_hat))

    Args:
        activations: Bounded activations in ``[0, 1]`` of shape
            ``(B, latent_dim)`` (e.g. ``sigmoid`` of the latent mean).
        target_sparsity: Desired average activation level ``rho`` in ``(0, 1)``.

    Returns:
        A scalar tensor: the summed sparsity penalty over all latent units.
    """
    rho = target_sparsity
    # Mean activation of each latent unit across the batch, clamped for stability.
    rho_hat = torch.clamp(activations.mean(dim=0), min=_EPS, max=1.0 - _EPS)

    kl = (rho * torch.log(rho / rho_hat)
          + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat)))
    return kl.sum()
