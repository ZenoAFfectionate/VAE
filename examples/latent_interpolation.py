"""Demo: Latent-space interpolation (image morphing).

Encodes pairs of images into the latent space, interpolates between their latent
codes, and decodes each intermediate code to produce a smooth morphing sequence.
This visualizes the continuity and semantic structure of the VAE latent space.

Example:
    python examples/latent_interpolation.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --pairs 6 --steps 10
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
    build_interpolation_grid, collect_samples, ensure_dir, get_eval_loader,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Latent-space interpolation demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained VAE checkpoint.")
    parser.add_argument("--pairs", type=int, default=6,
                        help="Number of image pairs to morph (rows).")
    parser.add_argument("--steps", type=int, default=10,
                        help="Interpolation steps per pair (columns).")
    parser.add_argument("--slerp", action="store_true",
                        help="Use spherical interpolation instead of linear.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/interpolation")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config, device = load_model(args.checkpoint)
    loader, _ = get_eval_loader(config, batch_size=args.batch_size)

    # Two independent sets of images form the interpolation endpoints.
    imgs, _ = collect_samples(loader, n=2 * args.pairs, device=device)
    n_pairs = imgs.size(0) // 2
    starts, ends = imgs[:n_pairs], imgs[n_pairs:2 * n_pairs]

    steps = max(args.steps, 2)
    grid = build_interpolation_grid(model, starts, ends, steps,
                                    use_slerp=args.slerp)

    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, "interpolation.png")
    save_image(grid, out_path, nrow=steps)
    print(f"[interpolation] saved {n_pairs}x{steps} morphing grid -> {out_path}")


if __name__ == "__main__":
    main()
