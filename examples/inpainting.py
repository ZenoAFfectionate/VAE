"""Demo: Image inpainting via latent-space optimization.

A region of each image is masked out. We then optimize a latent code ``z`` so
that the decoded image matches the *observed* (unmasked) pixels, and use the
decoder to hallucinate plausible content for the hole. The final image keeps the
original observed pixels and fills the masked region with the generated content.

Example:
    python examples/inpainting.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --mask-frac 0.5 --steps 300
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
from torchvision.utils import save_image  # noqa: E402

from examples.common import (  # noqa: E402
    collect_samples, encode_mean, ensure_dir, get_eval_loader, load_model,
    make_center_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VAE inpainting demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained VAE checkpoint.")
    parser.add_argument("--num", type=int, default=8,
                        help="Number of images to inpaint.")
    parser.add_argument("--mask-frac", type=float, default=0.5,
                        help="Side length of the square hole (fraction of image).")
    parser.add_argument("--steps", type=int, default=300,
                        help="Latent optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Latent optimization learning rate.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/inpainting")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def inpaint(model, images: torch.Tensor, mask: torch.Tensor,
            steps: int, lr: float) -> torch.Tensor:
    """Recover masked regions by optimizing latent codes on observed pixels.

    Args:
        model: A trained VAE (frozen).
        images: Clean images ``(B, C, H, W)`` in ``[0, 1]``.
        mask: Mask ``(1, H, W)`` with 1=observed, 0=hole (broadcast over batch).
        steps: Number of optimization iterations.
        lr: Learning rate for the latent optimizer.

    Returns:
        The composited inpainted images.
    """
    device = images.device
    mask = mask.to(device)
    observed = images * mask

    # Freeze the decoder; only the latent code is optimized.
    for p in model.parameters():
        p.requires_grad_(False)

    # Initialize the latent from the masked image's encoding.
    with torch.no_grad():
        z = encode_mean(model, observed).clone()
    z = z.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        recon = model.decode(z)
        # Data-consistency loss on observed pixels only.
        loss = F.mse_loss(recon * mask, observed)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = model.decode(z)
        # Keep the known pixels, fill the hole with generated content.
        filled = observed + recon * (1.0 - mask)
    return filled


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config, device = load_model(args.checkpoint)
    loader, info = get_eval_loader(config, batch_size=args.batch_size)
    clean, _ = collect_samples(loader, n=args.num, device=device)

    mask = make_center_mask(info.image_size, info.image_size, frac=args.mask_frac)
    masked = clean * mask.to(device)
    filled = inpaint(model, clean, mask, steps=args.steps, lr=args.lr)

    # Rows: original (top), masked input (middle), inpainted (bottom).
    grid = torch.cat([clean.cpu(), masked.cpu(), filled.cpu()], dim=0)
    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, "inpainting.png")
    save_image(grid, out_path, nrow=args.num)
    print(f"[inpainting] optimized {args.steps} steps -> {out_path}")


if __name__ == "__main__":
    main()
