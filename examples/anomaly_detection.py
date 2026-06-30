"""Demo: Anomaly / out-of-distribution detection via reconstruction error.

A VAE trained on a "normal" dataset reconstructs in-distribution images well but
struggles with out-of-distribution inputs. By scoring each image with its
reconstruction error, we can separate normal from anomalous samples.

The cleanest showcase is cross-dataset detection: e.g. a VAE trained on MNIST
treats MNIST as normal and FashionMNIST as anomalies (both are 1x28x28, so they
are shape-compatible). The script reports ROC-AUC and saves a score histogram.

Example:
    python examples/anomaly_detection.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --anomaly-dataset FashionMNIST
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from examples.common import (  # noqa: E402
    collect_samples, compute_auc, ensure_dir, get_eval_loader, load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VAE anomaly detection demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="VAE checkpoint trained on the NORMAL dataset.")
    parser.add_argument("--anomaly-dataset", type=str, default="FashionMNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"],
                        help="Dataset treated as anomalies (must match shape).")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of images per group (normal / anomaly).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/anomaly")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def reconstruction_scores(model, images: torch.Tensor) -> torch.Tensor:
    """Per-image reconstruction error (mean squared error) as anomaly score."""
    output = model(images)
    err = F.mse_loss(output["recon"], images, reduction="none")
    return err.reshape(images.size(0), -1).mean(dim=1)  # (N,)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config, device = load_model(args.checkpoint)
    normal_name = config["dataset"]

    # Normal samples come from the training dataset's validation split.
    normal_loader, n_info = get_eval_loader(config, batch_size=args.batch_size)
    anomaly_loader, a_info = get_eval_loader(
        config, batch_size=args.batch_size, dataset_name=args.anomaly_dataset)

    # The anomaly dataset must be shape-compatible with the trained model.
    if (n_info.image_channels, n_info.image_size) != \
            (a_info.image_channels, a_info.image_size):
        raise SystemExit(
            f"Shape mismatch: normal '{normal_name}' is "
            f"{n_info.image_channels}x{n_info.image_size}x{n_info.image_size} "
            f"but anomaly '{args.anomaly_dataset}' is "
            f"{a_info.image_channels}x{a_info.image_size}x{a_info.image_size}. "
            f"Choose a shape-compatible pair (e.g. MNIST <-> FashionMNIST)."
        )

    normal_imgs, _ = collect_samples(normal_loader, args.num_samples, device)
    anomaly_imgs, _ = collect_samples(anomaly_loader, args.num_samples, device)

    normal_scores = reconstruction_scores(model, normal_imgs).cpu()
    anomaly_scores = reconstruction_scores(model, anomaly_imgs).cpu()

    scores = torch.cat([normal_scores, anomaly_scores])
    labels = torch.cat([torch.zeros_like(normal_scores),
                        torch.ones_like(anomaly_scores)])  # 1 = anomaly
    auc = compute_auc(scores, labels)

    print(f"[anomaly] normal={normal_name}  anomaly={args.anomaly_dataset}")
    print(f"[anomaly] normal  mean error = {normal_scores.mean():.5f}")
    print(f"[anomaly] anomaly mean error = {anomaly_scores.mean():.5f}")
    print(f"[anomaly] ROC-AUC = {auc:.4f}")

    out_dir = ensure_dir(args.output_dir)
    # Save a histogram if matplotlib is available; always save raw scores.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(normal_scores.numpy(), bins=50, alpha=0.6,
                label=f"normal ({normal_name})", color="tab:blue")
        ax.hist(anomaly_scores.numpy(), bins=50, alpha=0.6,
                label=f"anomaly ({args.anomaly_dataset})", color="tab:red")
        ax.set_title(f"Reconstruction-error anomaly scores (AUC={auc:.3f})")
        ax.set_xlabel("per-image reconstruction MSE")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        hist_path = os.path.join(out_dir, "anomaly_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"[anomaly] saved histogram -> {hist_path}")
    except ImportError:
        print("[anomaly] matplotlib not found; skipping histogram plot.")

    torch.save({"scores": scores, "labels": labels, "auc": auc},
               os.path.join(out_dir, "scores.pt"))


if __name__ == "__main__":
    main()
