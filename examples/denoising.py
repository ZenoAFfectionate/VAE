"""Demo: Image denoising with a VAE.

Corrupts clean images with Gaussian or salt-and-pepper noise, then passes them
through the VAE. Because the decoder maps latent codes onto the learned data
manifold, the reconstruction tends to suppress noise. The script saves a
side-by-side comparison (clean / noisy / denoised) and reports PSNR gain.

Example:
    python examples/denoising.py \
        --checkpoint outputs/VAE_MNIST_xxx/best.pth --noise gaussian --std 0.4
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
    add_gaussian_noise, add_salt_pepper, collect_samples, ensure_dir,
    get_eval_loader, load_model, psnr,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VAE denoising demo.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained VAE checkpoint.")
    parser.add_argument("--noise", type=str, default="gaussian",
                        choices=["gaussian", "salt_pepper"],
                        help="Corruption type.")
    parser.add_argument("--std", type=float, default=0.4,
                        help="Gaussian noise std (for --noise gaussian).")
    parser.add_argument("--amount", type=float, default=0.1,
                        help="Corruption fraction (for --noise salt_pepper).")
    parser.add_argument("--num", type=int, default=8,
                        help="Number of images to display.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/examples/denoising")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config, device = load_model(args.checkpoint)
    loader, _ = get_eval_loader(config, batch_size=args.batch_size)
    clean, _ = collect_samples(loader, n=args.num, device=device)

    if args.noise == "gaussian":
        noisy = add_gaussian_noise(clean, std=args.std)
    else:
        noisy = add_salt_pepper(clean, amount=args.amount)

    denoised = model(noisy)["recon"]

    psnr_noisy = psnr(clean, noisy)
    psnr_denoised = psnr(clean, denoised)
    print(f"[denoising] PSNR noisy = {psnr_noisy:.2f} dB | "
          f"denoised = {psnr_denoised:.2f} dB | "
          f"gain = {psnr_denoised - psnr_noisy:+.2f} dB")

    # Three stacked rows: clean (top), noisy (middle), denoised (bottom).
    grid = torch.cat([clean.cpu(), noisy.cpu(), denoised.cpu()], dim=0)
    out_dir = ensure_dir(args.output_dir)
    out_path = os.path.join(out_dir, f"denoising_{args.noise}.png")
    save_image(grid, out_path, nrow=args.num)
    print(f"[denoising] saved clean/noisy/denoised comparison -> {out_path}")


if __name__ == "__main__":
    main()
