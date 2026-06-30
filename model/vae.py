"""Standard (Vanilla) Variational Autoencoder.

Provides a single :class:`VAE` class that supports two interchangeable
backbones selected via configuration:

* ``mlp``: fully-connected encoder/decoder, suited for MNIST / Fashion-MNIST.
* ``cnn``: convolutional encoder/decoder, suited for CIFAR-10.

The model exposes a unified interface (``forward`` returns a structured dict,
``loss_function`` returns decomposed loss terms) so that it can be used as a
drop-in component by the training and validation pipelines.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from utils.losses import kl_divergence, reconstruction_loss

# Mapping from string identifiers to activation modules.
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    """Return an activation module instance from its string name."""
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation {name!r}. "
                         f"Choose from {list(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]()


class VAE(nn.Module):
    """Vanilla Variational Autoencoder with MLP or CNN backbone.

    Args:
        image_channels: Number of input image channels (1 for MNIST, 3 for CIFAR).
        image_size: Spatial resolution (assumes square images).
        backbone: Either ``"mlp"`` or ``"cnn"``.
        hidden_dims: Hidden layer widths (MLP) or channel widths (CNN).
        latent_dim: Dimensionality of the latent space.
        activation: Activation function name (see :data:`_ACTIVATIONS`).
        recon_loss_type: Reconstruction loss type, ``"bce"`` or ``"mse"``.
        beta: Weight applied to the KL divergence term (beta-VAE).
    """

    def __init__(self,
                 image_channels: int = 1,
                 image_size: int = 28,
                 backbone: str = "mlp",
                 hidden_dims: Sequence[int] = (512, 256),
                 latent_dim: int = 32,
                 activation: str = "relu",
                 recon_loss_type: str = "bce",
                 beta: float = 1.0):
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.backbone = backbone.lower()
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.activation = activation
        self.recon_loss_type = recon_loss_type
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")
        self.beta = beta
        self.input_dim = image_channels * image_size * image_size

        if self.backbone == "mlp":
            self._build_mlp()
        elif self.backbone == "cnn":
            self._build_cnn()
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}. "
                             f"Choose from {{'mlp', 'cnn'}}.")

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Architecture construction
    # ------------------------------------------------------------------ #
    def _build_mlp(self) -> None:
        """Build a fully-connected encoder / decoder."""
        encoder_layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(get_activation(self.activation))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        decoder_layers: List[nn.Module] = []
        prev_dim = self.latent_dim
        for h_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))
            decoder_layers.append(get_activation(self.activation))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def _build_cnn(self) -> None:
        """Build a convolutional encoder / decoder.

        The encoder halves the spatial resolution at every stage; the decoder
        mirrors this with transposed convolutions. ``hidden_dims`` defines the
        channel progression.
        """
        channels = self.hidden_dims
        encoder_layers: List[nn.Module] = []
        in_ch = self.image_channels
        for out_ch in channels:
            encoder_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                            stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm2d(out_ch))
            encoder_layers.append(get_activation(self.activation))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # Each stage halves the resolution, so the image size must be divisible
        # by 2**num_stages; otherwise the transposed-conv decoder cannot recover
        # the original resolution and the reconstruction shape would mismatch.
        downscale = 2 ** len(channels)
        if self.image_size % downscale != 0:
            raise ValueError(
                f"CNN backbone with {len(channels)} stages requires image_size "
                f"divisible by {downscale}, but got image_size={self.image_size}. "
                f"Adjust --hidden-dims length or --resize."
            )
        self.feature_size = self.image_size // downscale
        if self.feature_size < 1:
            raise ValueError(
                f"Too many CNN down-sampling stages ({len(channels)}) for "
                f"image size {self.image_size}. Reduce --hidden-dims length."
            )
        self.flatten_dim = channels[-1] * self.feature_size * self.feature_size

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)

        decoder_layers: List[nn.Module] = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            decoder_layers.append(
                nn.ConvTranspose2d(rev_channels[i], rev_channels[i + 1],
                                   kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm2d(rev_channels[i + 1]))
            decoder_layers.append(get_activation(self.activation))
        decoder_layers.append(
            nn.ConvTranspose2d(rev_channels[-1], self.image_channels,
                               kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def _init_weights(self) -> None:
        """Initialize linear / convolutional weights for stable training.

        Uses kaiming normal with a nonlinearity heuristic: ``"relu"`` or
        ``"leaky_relu"`` preserve the standard gain; ``"tanh"`` uses the
        tanh gain so gradients don't saturate too early.
        """
        # Map the configured activation to a kaiming-appropriate nonlinearity.
        nl = "leaky_relu" if self.activation == "leaky_relu" else \
             "tanh" if self.activation == "tanh" else "relu"
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nl)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------ #
    # Core VAE operations
    # ------------------------------------------------------------------ #
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate that the input batch matches the configured image shape.

        Args:
            x: Input tensor expected to be ``(B, C, H, W)``.

        Raises:
            ValueError: If the tensor rank or the channel / spatial dimensions
                do not match the values the model was constructed with.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected a 4D input tensor (B, C, H, W), got {x.dim()}D "
                f"with shape {tuple(x.shape)}."
            )
        _, c, h, w = x.shape
        if c != self.image_channels or h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Input shape mismatch: expected "
                f"(B, {self.image_channels}, {self.image_size}, {self.image_size}) "
                f"but got {tuple(x.shape)}."
            )

    def encode(self, x: torch.Tensor):
        """Encode an input batch into latent mean and log-variance."""
        self._validate_input(x)
        if self.backbone == "mlp":
            h = self.encoder(x.reshape(x.size(0), -1))
        else:
            h = self.encoder(x)
            h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterization trick: ``z = mu + eps * sigma``.

        Sampling ``eps`` from ``N(0, I)`` and scaling keeps the operation
        differentiable so gradients flow through ``mu`` and ``logvar``.
        """
        # Clamp for numerical stability before exponentiation.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back into the image space ``(B, C, H, W)``."""
        if self.backbone == "mlp":
            out = self.decoder(z)
            return out.reshape(-1, self.image_channels,
                               self.image_size, self.image_size)
        h = self.decoder_input(z)
        h = h.reshape(-1, self.hidden_dims[-1],
                      self.feature_size, self.feature_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a full encode-sample-decode pass.

        Returns:
            A dict with keys ``recon``, ``mu``, ``logvar`` and ``z``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    # ------------------------------------------------------------------ #
    # Loss and sampling
    # ------------------------------------------------------------------ #
    def loss_function(self, x: torch.Tensor,
                      output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the ELBO loss with decomposed terms.

        Args:
            x: Ground-truth input batch.
            output: The dict returned by :meth:`forward`.

        Returns:
            Dict containing ``total_loss``, ``recon_loss`` and ``kl_loss``
            (all scalar tensors).
        """
        recon_loss = reconstruction_loss(output["recon"], x, self.recon_loss_type)
        kl_loss = kl_divergence(output["mu"], output["logvar"])
        total_loss = recon_loss + self.beta * kl_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples by decoding random latent vectors.

        Args:
            num_samples: Number of images to generate.
            device: Device for the latent tensor. Must match the model device
                (calling ``self.decode`` on mismatched devices will fail).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
