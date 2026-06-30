"""Training entry point for VAE / Sparse VAE experiments.

This is the single executable script for all training runs. It builds the
data pipeline and model from command-line arguments, runs a full training /
validation loop with logging and checkpointing, and persists the experiment
configuration for reproducibility.

Example:
    python train.py --model VAE --dataset MNIST --backbone mlp --epochs 50
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict

import torch
import torch.optim as optim

from data import get_dataloaders
from utils import (
    MetricTracker,
    build_model,
    count_parameters,
    get_device,
    latent_sparsity_ratio,
    log_config,
    save_checkpoint,
    save_generated_samples,
    save_reconstruction,
    set_seed,
    setup_logger,
)
from utils.options import get_train_args


def _build_config(args, info) -> Dict:
    """Merge CLI arguments with dataset metadata into a single config dict."""
    config = vars(args).copy()
    config["image_channels"] = info.image_channels
    config["image_size"] = info.image_size
    config["input_dim"] = info.input_dim
    return config


def run_epoch(model, loader, device, optimizer=None, grad_clip: float = 0.0) -> Dict[str, float]:
    """Run a single training or evaluation epoch.

    When ``optimizer`` is provided the model is trained; otherwise it is
    evaluated under ``torch.no_grad``.

    Args:
        model: The VAE / SparseVAE model.
        loader: Iterable yielding ``(data, target)`` batches.
        device: Device to run computation on.
        optimizer: Optional optimizer; presence selects training mode.
        grad_clip: Max gradient norm for clipping (``<= 0`` disables it).

    Returns:
        A dict of averaged metrics for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for data, _ in loader:
            data = data.to(device)
            batch_size = data.size(0)

            output = model(data)
            losses = model.loss_function(data, output)
            total_loss = losses["total_loss"]

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping guards against occasional exploding grads.
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Collect scalar metrics for this batch.
            batch_metrics = {
                "total_loss": total_loss.item(),
                "recon_loss": losses["recon_loss"].item(),
                "kl_loss": losses["kl_loss"].item(),
                # Use the sparsity_ratio from the loss function if present
                # (SparseVAE uses its own target_sparsity threshold); otherwise
                # fall back to the default 0.05 via latent_sparsity_ratio.
                "sparsity_ratio": losses["sparsity_ratio"].item()
                if "sparsity_ratio" in losses
                else latent_sparsity_ratio(output["mu"]),
            }
            if "sparse_loss" in losses:
                batch_metrics["sparse_loss"] = losses["sparse_loss"].item()
            tracker.update(batch_metrics, n=batch_size)

    return tracker.averages()


def _format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dict into an aligned single-line summary."""
    parts = [
        f"loss={metrics['total_loss']:.4f}",
        f"recon={metrics['recon_loss']:.4f}",
        f"kl={metrics['kl_loss']:.4f}",
        f"sparsity={metrics['sparsity_ratio']:.3f}",
    ]
    if "sparse_loss" in metrics:
        parts.append(f"sparse={metrics['sparse_loss']:.4f}")
    return " | ".join(parts)


def main() -> None:
    """Program entry: parse args, set up experiment, and run training."""
    args = get_train_args()

    # --- Reproducibility & device ------------------------------------- #
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)

    # --- Timestamped experiment directories --------------------------- #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_{args.dataset}_{timestamp}"
    log_dir = os.path.join(args.log_root, exp_name)
    output_dir = os.path.join(args.output_root, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(log_dir, log_name="train.log")

    # --- Data --------------------------------------------------------- #
    train_loader, val_loader, info = get_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        resize=args.resize,
        seed=args.seed,
    )

    config = _build_config(args, info)
    # Persist the full configuration for reproducibility.
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --- Model & optimizer -------------------------------------------- #
    model = build_model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
        if args.scheduler else None)

    # --- Startup summary ---------------------------------------------- #
    log_config(logger, "EXPERIMENT CONFIGURATION", config)
    logger.info(f"Device: {device} | Trainable params: "
                f"{count_parameters(model):,}")
    logger.info(f"Train batches: {len(train_loader)} | "
                f"Val batches: {len(val_loader)}")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 60)

    # --- Training loop ------------------------------------------------ #
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer,
                                  grad_clip=args.grad_clip)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        elapsed = time.time() - start

        if scheduler is not None:
            scheduler.step(val_metrics["total_loss"])

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | {elapsed:5.1f}s | lr={lr:.2e}")
        logger.info(f"  [train] {_format_metrics(train_metrics)}")
        logger.info(f"  [valid] {_format_metrics(val_metrics)}")

        # --- Checkpointing -------------------------------------------- #
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        # Always keep the latest checkpoint.
        save_checkpoint(checkpoint, os.path.join(output_dir, "latest.pth"))
        # Track and persist the best checkpoint by validation loss.
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_checkpoint(checkpoint, os.path.join(output_dir, "best.pth"))
            logger.info(f"  -> new best (val_loss={best_val_loss:.4f}) saved")

        # Periodic reconstruction / generation snapshots.
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            model.eval()
            vis_data, _ = next(iter(val_loader))
            save_reconstruction(
                model, vis_data.to(device),
                os.path.join(output_dir, "figures",
                             f"recon_epoch_{epoch:03d}.png"))
            save_generated_samples(
                model, device,
                os.path.join(output_dir, "figures",
                             f"samples_epoch_{epoch:03d}.png"))

    logger.info("=" * 60)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved under: {output_dir}")


if __name__ == "__main__":
    main()
