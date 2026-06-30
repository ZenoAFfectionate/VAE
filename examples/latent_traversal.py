"""Demo: Latent-space traversal.

Encodes a single image, then sweeps one latent dimension at a time across a
range of values (keeping the others fixed) and decodes the result. Each row
shows the effect of one latent dimension, revealing what factor it controls --
a simple way to inspect disentanglement (especially for Sparse / beta-VAE).

Example:
    python examples/latent_traversal.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --num-dims 10 --steps 9
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from examples.common import (  # noqa: E402
    build_traversal_grid, collect_samples, ensure_dir, get_eval_loader,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Latent-space traversal demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained VAE checkpoint.")
    parser.add_argument("--num-dims", type=int, default=10,
                        help="Number of latent dimensions to traverse (rows).")
    parser.add_argument("--steps", type=int, default=9,
                        help="Number of values per dimension (columns).")
    parser.add_argument("--span", type=float, default=3.0,
                        help="Traversal range: each dim is swept over [-span, span].")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/traversal")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config, device = load_model(args.checkpoint)
    loader, _ = get_eval_loader(config, batch_size=args.batch_size)

    # Use a single reference image as the traversal anchor.
    imgs, _ = collect_samples(loader, n=1, device=device)
    steps = max(args.steps, 2)
    grid, n_dims = build_traversal_grid(model, imgs[:1], args.num_dims,
                                        steps, args.span)

    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, "traversal.png")
    save_image(grid, out_path, nrow=steps)
    print(f"[traversal] swept {n_dims} dims x {steps} steps -> {out_path}")


if __name__ == "__main__":
    main()
