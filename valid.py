"""Validation / evaluation entry point.

Supports two evaluation sources behind a single unified pipeline:

1. A locally trained checkpoint (``--checkpoint``), rebuilt from its stored
   training configuration.
2. A mature, ready-to-use pretrained VAE downloaded from HuggingFace
   (``--hf-model``, e.g. ``stabilityai/sd-vae-ft-mse``), wrapped by
   :class:`~model.pretrained.PretrainedVAE`.

In both cases the script reports quantitative metrics and saves reconstruction
and generation visualizations to ``outputs/``.

Examples:
    # Evaluate a local checkpoint
    python valid.py --checkpoint outputs/VAE_MNIST_xxx/best.pth

    # Evaluate a pretrained HuggingFace VAE on CIFAR-10 (resized to 256)
    python valid.py --hf-model stabilityai/sd-vae-ft-mse \
        --dataset CIFAR10 --resize 256 --batch-size 8
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import torch

from data import get_dataloaders
from utils import (
    MetricTracker,
    build_model,
    get_device,
    latent_sparsity_ratio,
    load_checkpoint,
    log_config,
    save_generated_samples,
    save_reconstruction,
    set_seed,
    setup_logger,
)
from utils.options import get_valid_args


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Evaluate the model on a dataloader without gradient computation.

    Returns:
        Averaged metrics dict over the whole loader.
    """
    model.eval()
    tracker = MetricTracker()
    for data, _ in loader:
        data = data.to(device)
        output = model(data)
        losses = model.loss_function(data, output)

        metrics = {
            "total_loss": losses["total_loss"].item(),
            "recon_loss": losses["recon_loss"].item(),
            "kl_loss": losses["kl_loss"].item(),
            # Prefer the model's own sparsity_ratio (SparseVAE uses its
            # target_sparsity threshold); otherwise compute from latent mean.
            "sparsity_ratio": losses["sparsity_ratio"].item()
            if "sparsity_ratio" in losses
            else latent_sparsity_ratio(output["mu"]),
        }
        if "sparse_loss" in losses:
            metrics["sparse_loss"] = losses["sparse_loss"].item()
        tracker.update(metrics, n=data.size(0))
    return tracker.averages()


def _prepare_from_checkpoint(args, device):
    """Build model + data config from a locally trained checkpoint."""
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    source = f"checkpoint '{args.checkpoint}' (epoch {ckpt.get('epoch', '?')})"
    # Data settings come from the checkpoint to match training conditions.
    data_cfg = dict(dataset_name=config["dataset"], data_root=config["data_root"],
                    val_split=config["val_split"], resize=config.get("resize"),
                    seed=config["seed"])
    return model, config, data_cfg, source


def _prepare_from_hf(args, device):
    """Build a HuggingFace pretrained VAE + data config from CLI arguments."""
    # Imported here so `diffusers` is only required on this code path.
    from model import PretrainedVAE

    # Pretrained AutoencoderKL models operate on RGB images.
    image_channels = 3
    image_size = args.resize if args.resize is not None else 256
    model = PretrainedVAE(
        hf_model_id=args.hf_model,
        image_channels=image_channels,
        image_size=image_size,
        recon_loss_type=args.recon_loss_type,
        beta=args.beta,
        cache_dir=args.hf_cache,
    ).to(device)

    config = {
        "model": "PretrainedVAE", "hf_model": args.hf_model,
        "dataset": args.dataset, "data_root": args.data_root,
        "val_split": args.val_split, "resize": image_size,
        "seed": args.seed, "beta": args.beta,
        "recon_loss_type": args.recon_loss_type,
        "image_channels": image_channels, "image_size": image_size,
        "latent_channels": model.latent_channels, "downscale": model.downscale,
    }
    data_cfg = dict(dataset_name=args.dataset, data_root=args.data_root,
                    val_split=args.val_split, resize=image_size, seed=args.seed)
    source = f"HuggingFace pretrained VAE '{args.hf_model}'"
    return model, config, data_cfg, source


def main() -> None:
    """Program entry: evaluate a local checkpoint or a pretrained HF VAE."""
    args = get_valid_args()

    # Require exactly one evaluation source.
    if bool(args.checkpoint) == bool(args.hf_model):
        raise SystemExit(
            "Specify exactly one of --checkpoint or --hf-model "
            "(got both or neither)."
        )

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)

    # Timestamped output directory so evaluations never overwrite each other.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    logger = setup_logger(output_dir, log_name="eval.log")

    # --- Build the model and data configuration ----------------------- #
    if args.checkpoint:
        model, config, data_cfg, source = _prepare_from_checkpoint(args, device)
    else:
        logger.info(f"Loading pretrained VAE '{args.hf_model}' "
                    f"(cache: {args.hf_cache}) ...")
        model, config, data_cfg, source = _prepare_from_hf(args, device)
    model.eval()

    log_config(logger, "EVALUATION CONFIGURATION", config)
    logger.info(f"Source: {source}")
    logger.info(f"Device: {device}")

    # --- Data pipeline ------------------------------------------------- #
    _, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, **data_cfg)

    # --- Quantitative evaluation -------------------------------------- #
    metrics = evaluate(model, val_loader, device)
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"  Avg total loss        : {metrics['total_loss']:.4f}")
    logger.info(f"  Avg reconstruction    : {metrics['recon_loss']:.4f}")
    logger.info(f"  Avg KL divergence     : {metrics['kl_loss']:.4f}")
    logger.info(f"  Avg sparsity ratio    : {metrics['sparsity_ratio']:.4f}")
    if "sparse_loss" in metrics:
        logger.info(f"  Avg sparsity penalty  : {metrics['sparse_loss']:.4f}")
    logger.info("=" * 60)

    # Persist metrics + config for reproducibility / later inspection.
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)

    # --- Visualizations ----------------------------------------------- #
    vis_data, _ = next(iter(val_loader))
    recon_path = os.path.join(figures_dir, "reconstruction.png")
    samples_path = os.path.join(figures_dir, "generated_samples.png")
    save_reconstruction(model, vis_data.to(device), recon_path,
                        n_samples=args.num_vis)
    save_generated_samples(model, device, samples_path,
                          n_samples=args.num_gen)
    logger.info(f"Saved reconstruction -> {recon_path}")
    logger.info(f"Saved generated samples -> {samples_path}")
    logger.info(f"Saved metrics -> {os.path.join(output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
