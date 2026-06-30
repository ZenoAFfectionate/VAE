"""HuggingFace pretrained VAE adapter.

Wraps a mature, ready-to-use ``diffusers.AutoencoderKL`` model (e.g. the
Stable Diffusion VAEs such as ``stabilityai/sd-vae-ft-mse``) behind the exact
same interface as the project's own :class:`~model.vae.VAE`. This lets a
pretrained model be plugged directly into the existing evaluation and
visualization pipelines without any changes to ``valid.py``'s core logic.

Key adaptations performed by the wrapper:
    * Pixel-range conversion: the project pipeline produces images in ``[0, 1]``
      whereas ``AutoencoderKL`` expects / returns ``[-1, 1]``.
    * Latent extraction: the diffusers ``DiagonalGaussianDistribution`` exposes
      ``mean`` and ``logvar``, which we surface as ``mu`` / ``logvar``.
    * Spatial latents: ``AutoencoderKL`` uses a 4D convolutional latent
      ``(B, C_z, H/f, W/f)`` rather than a flat vector, so the KL term is
      computed over the flattened latent dimensions.

``diffusers`` is an optional dependency and is imported lazily, so the rest of
the project (and its tests) work without it installed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.losses import kl_divergence, reconstruction_loss


class PretrainedVAE(nn.Module):
    """Adapter exposing a HuggingFace ``AutoencoderKL`` via the project API.

    Args:
        hf_model_id: HuggingFace repo id or local path of the pretrained VAE
            (e.g. ``"stabilityai/sd-vae-ft-mse"``). Ignored if ``vae`` is given.
        vae: An already-instantiated ``AutoencoderKL`` (mainly for testing /
            advanced use). When ``None``, the model is loaded from
            ``hf_model_id``.
        image_channels: Number of input image channels (AutoencoderKL is RGB=3).
        image_size: Spatial resolution of the evaluation images (square).
        recon_loss_type: Reconstruction loss type, ``"mse"`` (recommended) or
            ``"bce"``.
        beta: Weight on the KL divergence term.
        cache_dir: Optional directory to cache downloaded weights.
    """

    def __init__(self,
                 hf_model_id: Optional[str] = None,
                 vae: Optional[nn.Module] = None,
                 image_channels: int = 3,
                 image_size: int = 256,
                 recon_loss_type: str = "mse",
                 beta: float = 1.0,
                 cache_dir: Optional[str] = None):
        super().__init__()
        if vae is None:
            if hf_model_id is None:
                raise ValueError("Provide either `hf_model_id` or `vae`.")
            # Lazy import keeps `diffusers` an optional dependency.
            try:
                from diffusers import AutoencoderKL
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Using a HuggingFace pretrained VAE requires the "
                    "`diffusers` package. Install it with `pip install diffusers`."
                ) from exc
            vae = AutoencoderKL.from_pretrained(hf_model_id, cache_dir=cache_dir)

        self.vae = vae
        self.hf_model_id = hf_model_id
        self.image_channels = image_channels
        self.image_size = image_size
        self.recon_loss_type = recon_loss_type
        self.beta = beta

        # Latent channel count and spatial down-sampling factor from the config.
        cfg = self.vae.config
        self.latent_channels = getattr(cfg, "latent_channels", 4)
        n_blocks = len(getattr(cfg, "block_out_channels", (1, 2, 3, 4)))
        self.downscale = 2 ** (n_blocks - 1)

    # ------------------------------------------------------------------ #
    # Pixel-range helpers ([0, 1] <-> [-1, 1])
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_model_range(x: torch.Tensor) -> torch.Tensor:
        """Map images from the pipeline range [0, 1] to the VAE range [-1, 1]."""
        return x * 2.0 - 1.0

    @staticmethod
    def _to_pixel_range(x: torch.Tensor) -> torch.Tensor:
        """Map VAE outputs from [-1, 1] back to [0, 1] (clamped)."""
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input rank, channels and divisibility by the latent factor."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D (B, C, H, W) input, got {x.dim()}D.")
        if x.size(1) != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} channels, got {x.size(1)}.")
        if x.size(2) % self.downscale or x.size(3) % self.downscale:
            raise ValueError(
                f"Image H/W must be divisible by {self.downscale} for this VAE; "
                f"got {tuple(x.shape[2:])}. Use --resize accordingly."
            )

    # ------------------------------------------------------------------ #
    # Core VAE operations (mirrors model.vae.VAE)
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor):
        """Encode ``[0, 1]`` images into latent mean and log-variance."""
        self._validate_input(x)
        posterior = self.vae.encode(self._to_model_range(x)).latent_dist
        # Clamp logvar for numerical stability (diffusers also clamps to [-30, 20]).
        return posterior.mean, posterior.logvar.clamp(-30.0, 20.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick ``z = mu + eps * sigma`` (differentiable)."""
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents into ``[0, 1]`` images."""
        recon = self.vae.decode(z).sample
        return self._to_pixel_range(recon)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode-sample-decode, returning the project's structured output."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def loss_function(self, x: torch.Tensor,
                      output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the ELBO with decomposed reconstruction / KL terms."""
        recon_loss = reconstruction_loss(output["recon"], x, self.recon_loss_type)
        # Flatten the spatial latent so KL is summed over all latent dims/sample.
        mu = output["mu"].reshape(output["mu"].size(0), -1)
        logvar = output["logvar"].reshape(output["logvar"].size(0), -1)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate images by decoding random spatial latents.

        Args:
            num_samples: Number of images to generate.
            device: Device for the latent tensor. Must match the model device.
        """
        spatial = self.image_size // self.downscale
        z = torch.randn(num_samples, self.latent_channels, spatial, spatial,
                        device=device)
        return self.decode(z)
