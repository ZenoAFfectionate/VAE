"""Command-line argument definitions shared by training and validation.

A common base parser holds the arguments needed to *reconstruct a model and
data pipeline*; the train / valid parsers extend it with mode-specific
options. Sharing the base keeps the two entry points perfectly in sync.
"""

from __future__ import annotations

import argparse


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach arguments common to both training and validation."""
    # --- Model configuration ------------------------------------------- #
    group = parser.add_argument_group("model")
    group.add_argument("--model", type=str, default="VAE",
                       choices=["VAE", "SparseVAE"],
                       help="Model variant to use.")
    group.add_argument("--backbone", type=str, default="mlp",
                       choices=["mlp", "cnn"],
                       help="Encoder/decoder backbone type.")
    group.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256],
                       help="Hidden widths (MLP) or channel widths (CNN).")
    group.add_argument("--latent-dim", type=int, default=32,
                       help="Latent space dimensionality.")
    group.add_argument("--activation", type=str, default="relu",
                       choices=["relu", "elu", "leaky_relu", "gelu", "tanh"],
                       help="Activation function.")
    group.add_argument("--recon-loss-type", type=str, default="bce",
                       choices=["bce", "mse"],
                       help="Reconstruction loss type.")
    group.add_argument("--beta", type=float, default=1.0,
                       help="KL divergence weight (beta-VAE).")

    # --- Sparse VAE specific ------------------------------------------- #
    group = parser.add_argument_group("sparse")
    group.add_argument("--target-sparsity", type=float, default=0.05,
                       help="Target average latent activation (Sparse VAE).")
    group.add_argument("--sparse-weight", type=float, default=1.0,
                       help="Weight of the sparsity penalty (Sparse VAE).")

    # --- Data configuration -------------------------------------------- #
    group = parser.add_argument_group("data")
    group.add_argument("--dataset", type=str, default="MNIST",
                       choices=["MNIST", "FashionMNIST", "CIFAR10",
                                "CelebA", "STL10"],
                       help="Benchmark dataset.")
    group.add_argument("--data-root", type=str, default="./dataset",
                       help="Relative directory for dataset storage.")
    group.add_argument("--batch-size", type=int, default=128,
                       help="Mini-batch size.")
    group.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of train data used for validation.")
    group.add_argument("--num-workers", type=int, default=4,
                       help="Number of data-loading worker processes.")
    group.add_argument("--resize", type=int, default=None,
                       help="Optional image resize target.")

    # --- Runtime ------------------------------------------------------- #
    group = parser.add_argument_group("runtime")
    group.add_argument("--seed", type=int, default=42,
                       help="Global random seed for reproducibility.")
    group.add_argument("--no-cuda", action="store_true",
                       help="Disable CUDA even if available.")


def get_train_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a VAE / Sparse VAE.")
    _add_common_args(parser)

    group = parser.add_argument_group("optimization")
    group.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs.")
    group.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate.")
    group.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Optimizer weight decay.")
    group.add_argument("--grad-clip", type=float, default=5.0,
                       help="Max gradient norm for clipping (<=0 disables).")
    group.add_argument("--scheduler", action="store_true",
                       help="Enable ReduceLROnPlateau LR scheduling.")

    group = parser.add_argument_group("io")
    group.add_argument("--log-root", type=str, default="./logs",
                       help="Root directory for log files / configs.")
    group.add_argument("--output-root", type=str, default="./outputs",
                       help="Root directory for checkpoints / figures.")
    group.add_argument("--save-interval", type=int, default=5,
                       help="Epoch interval for periodic checkpoint saving.")
    return parser.parse_args(argv)


def get_valid_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the validation / evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VAE / Sparse VAE checkpoint.")
    _add_common_args(parser)

    group = parser.add_argument_group("evaluation")
    group.add_argument("--checkpoint", type=str, default=None,
                       help="Path to a locally trained checkpoint to evaluate. "
                            "Mutually exclusive with --hf-model.")
    group.add_argument("--output-dir", type=str, default="./outputs/eval",
                       help="Directory for evaluation figures.")
    group.add_argument("--num-vis", type=int, default=8,
                       help="Number of reconstruction pairs to visualize.")
    group.add_argument("--num-gen", type=int, default=64,
                       help="Number of random samples to generate.")

    group = parser.add_argument_group("pretrained")
    group.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace AutoencoderKL repo id or local path "
                            "(e.g. 'stabilityai/sd-vae-ft-mse'). Loads a mature, "
                            "ready-to-use pretrained VAE. Mutually exclusive "
                            "with --checkpoint.")
    group.add_argument("--hf-cache", type=str, default="./pretrained",
                       help="Directory to cache downloaded pretrained weights.")
    return parser.parse_args(argv)
