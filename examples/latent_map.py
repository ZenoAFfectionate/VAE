"""Demo: Latent-space map (2D visualization).

Encodes a subset of the validation set, projects the latent means to 2D
(PCA by default, or t-SNE when scikit-learn is installed), and draws a scatter
plot colored by class label. Clusters by class indicate that the VAE has learned
a semantically organized latent space.

Example:
    python examples/latent_map.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --num-samples 2000 --method pca
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from examples.common import (  # noqa: E402
    collect_samples, encode_mean, ensure_dir, get_eval_loader, load_model, pca_2d,
)


def parse_args():
    parser = argparse.ArgumentParser(description="2D latent-space map demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained VAE checkpoint.")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of validation images to encode.")
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne"],
                        help="Dimensionality reduction method.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/latent_map")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def reduce_to_2d(mu: torch.Tensor, method: str, seed: int) -> torch.Tensor:
    """Reduce latent means to 2D using PCA or (optionally) t-SNE."""
    if mu.size(1) == 1:
        # Degenerate 1D latent: pad a zero second axis so it can be plotted.
        return torch.cat([mu, torch.zeros_like(mu)], dim=1)
    if mu.size(1) == 2:
        return mu
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("[latent_map] scikit-learn not found; falling back to PCA.")
        else:
            emb = TSNE(n_components=2, random_state=seed,
                       init="pca").fit_transform(mu.cpu().numpy())
            return torch.from_numpy(emb)
    return pca_2d(mu)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Plotting requires matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("This demo requires matplotlib: pip install matplotlib")

    model, config, device = load_model(args.checkpoint)
    loader, _ = get_eval_loader(config, batch_size=args.batch_size)

    imgs, labels = collect_samples(loader, n=args.num_samples, device=device)
    mu = encode_mean(model, imgs)
    coords = reduce_to_2d(mu, args.method, args.seed).cpu().numpy()
    labels_np = labels.numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels_np,
                         cmap="tab10", s=8, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="class",
                       loc="best", fontsize=8)
    ax.add_artist(legend)
    ax.set_title(f"Latent space ({args.method.upper()}) — {config['dataset']}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, f"latent_map_{args.method}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[latent_map] encoded {len(labels_np)} samples -> {out_path}")


if __name__ == "__main__":
    main()
