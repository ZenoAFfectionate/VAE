"""FastAPI backend for VAE Studio.

Replaces the Gradio frontend with a standard REST API + static-file-served
SPA. All inference logic reuses ``examples/studio_backend.py`` (fully unit
tested) — this file is purely the API layer.

Run:
    pip install fastapi uvicorn python-multipart
    python server.py --checkpoint outputs/VAE_MNIST_xxx/best.pth
    # open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.common import load_model  # noqa: E402
from examples import studio_backend as be  # noqa: E402
from utils.helpers import get_device  # noqa: E402

# --------------------------------------------------------------------- #
# Global model singleton (loaded once at startup)
# --------------------------------------------------------------------- #
_state: dict = {"model": None, "config": None, "device": None}


def _load_hf_mnist_vae(cache_dir: str = "./pretrained"):
    """Load the HuggingFace MNIST VAE (uday9k/Binarized_MNIST_VAE)."""
    import glob
    import torch
    import torch.nn as nn

    from huggingface_hub import hf_hub_download

    repo = "uday9k/Binarized_MNIST_VAE"
    weights_path = hf_hub_download(repo_id=repo,
                                   filename="customVAE_model2.pth",
                                   cache_dir=cache_dir)

    class _MNISTVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 20
            self.image_channels = 1
            self.image_size = 28
            self.encodeLayers = nn.Sequential(
                nn.Linear(784, 1024), nn.Tanh(),
                nn.Linear(1024, 512), nn.Tanh(),
            )
            self.fc_mu = nn.Linear(512, 20)
            self.fc_logvar = nn.Linear(512, 20)
            self.decoder_bernoulli = nn.Sequential(
                nn.Linear(20, 1024), nn.Tanh(),
                nn.Linear(1024, 784), nn.Sigmoid(),
            )

        def encode(self, x):
            if x.dim() == 4:
                x = x.reshape(x.size(0), -1)
            h = self.encodeLayers(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        def decode(self, z):
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return self.decoder_bernoulli(z).reshape(-1, 1, 28, 28)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return {"recon": self.decode(z), "mu": mu, "logvar": logvar, "z": z}

        def loss_function(self, x, output):
            import torch.nn.functional as F
            recon = output["recon"].reshape(x.size(0), -1).clamp(1e-8, 1 - 1e-8)
            x_flat = x.reshape(x.size(0), -1)
            rc = F.binary_cross_entropy(recon, x_flat, reduction="sum") / x.size(0)
            mu, lv = output["mu"], output["logvar"]
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
            return {"total_loss": rc + kl, "recon_loss": rc, "kl_loss": kl}

        @torch.no_grad()
        def sample(self, n, device):
            return self.decode(torch.randn(n, 20, device=device))

    ckpt = torch.load(weights_path, map_location="cpu")
    model = _MNISTVAE()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    config = {
        "model": "HF_MNIST_VAE", "dataset": "MNIST", "data_root": "./dataset",
        "val_split": 0.1, "resize": None, "seed": 42, "beta": 1.0,
        "recon_loss_type": "bce", "image_channels": 1, "image_size": 28,
        "latent_dim": 20, "backbone": "mlp", "hidden_dims": [1024, 512],
        "activation": "tanh",
    }
    return model, config


def _load_hf_celeba_vae(cache_dir: str = "./pretrained"):
    """Load the CelebA beta-VAE with facial attribute direction vectors.

    This model (ayushshah/beta-vae-capacity-annealing-celeba) was trained on
    CelebA faces and includes 20 latent direction vectors (Smiling, Eyeglasses,
    Blond_Hair, Young, etc.) for controllable face generation.
    """
    import glob
    import torch
    import torch.nn as nn

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo = "ayushshah/beta-vae-capacity-annealing-celeba"
    model_path = hf_hub_download(repo_id=repo, filename="model.safetensors",
                                 cache_dir=cache_dir)

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
            self.gn1 = nn.GroupNorm(8, out_ch)
            self.gn2 = nn.GroupNorm(8, out_ch)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.GroupNorm(8, out_ch))

        def forward(self, x):
            h = torch.nn.functional.leaky_relu(self.gn1(self.conv1(x)))
            h = self.gn2(self.conv2(h))
            return torch.nn.functional.leaky_relu(h + self.shortcut(x))

    class Upsample(nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2,
                                                   mode="nearest")

    class CelebAVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 32
            self.image_channels = 3
            self.image_size = 64
            self.downscale = 8
            self.latent_channels = 32
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1), nn.GroupNorm(8, 64),
                nn.LeakyReLU(), ResBlock(64, 128, stride=2),
                ResBlock(128, 256, stride=2),
                ResBlock(256, 256, stride=2))
            self.fc_mu = nn.Linear(4096, 32)
            self.fc_logvar = nn.Linear(4096, 32)
            self.fc = nn.Linear(32, 4096)
            self.decoder = nn.Sequential(
                Upsample(), ResBlock(256, 128), Upsample(), ResBlock(128, 64),
                Upsample(), ResBlock(64, 64), Upsample(), ResBlock(64, 64),
                nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid())

        def encode(self, x):
            h = self.encoder(x).reshape(x.size(0), -1)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        def decode(self, z):
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return self.decoder(self.fc(z).reshape(-1, 256, 4, 4))

        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.reparameterize(mu, lv)
            return {"recon": self.decode(z), "mu": mu, "logvar": lv, "z": z}

        @torch.no_grad()
        def sample(self, n, device):
            return self.decode(torch.randn(n, 32, device=device))

    state = load_file(model_path)
    model = CelebAVAE()
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load available direction vectors.
    directions = {}
    snapshot_dir = os.path.dirname(model_path)
    dir_folder = os.path.join(snapshot_dir, "directions")
    if os.path.isdir(dir_folder):
        for fname in os.listdir(dir_folder):
            if fname.endswith(".pt"):
                key = fname.replace(".pt", "")
                directions[key] = torch.load(
                    os.path.join(dir_folder, fname), map_location="cpu")
    else:
        # Download a few key directions on demand.
        for attr in ["Smiling", "Eyeglasses", "Blond_Hair", "Young",
                      "Male", "Black_Hair", "Bangs", "Pale_Skin"]:
            try:
                path = hf_hub_download(repo_id=repo,
                                      filename=f"directions/{attr}.pt",
                                      cache_dir=cache_dir)
                directions[attr] = torch.load(path, map_location="cpu")
            except Exception:
                pass

    model._directions = directions

    config = {
        "model": "CelebA_BetaVAE", "dataset": "CelebA", "data_root": "./dataset",
        "val_split": 0.1, "resize": 64, "seed": 42, "beta": 1.0,
        "recon_loss_type": "mse", "image_channels": 3, "image_size": 64,
        "latent_dim": 32, "backbone": "cnn", "hidden_dims": [64, 128, 256],
        "directions": list(directions.keys()),
    }
    return model, config


def _ensure_model():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _state["model"], _state["config"]


def _upload_to_numpy(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image file into a NumPy array (HWC or HW)."""
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    img = Image.open(io.BytesIO(raw))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def _numpy_to_png_base64(arr: np.ndarray) -> str:
    """Encode a float [0,1] or uint8 NumPy image as a base64 PNG string."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _success(data: dict) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


# --------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------- #
def create_app(checkpoint: str | None = None,
               hf_model: str | None = None,
               hf_cache: str = "./pretrained",
               hf_mnist_vae: bool = False,
               hf_celeba_vae: bool = False) -> FastAPI:
    """Create and configure the FastAPI app with a loaded model.

    Either a local ``checkpoint``, a HuggingFace ``hf_model`` repo id, or
    ``hf_mnist_vae=True`` must be provided.
    """
    if hf_celeba_vae:
        model, config = _load_hf_celeba_vae(hf_cache)
        device = get_device()
        model = model.to(device)
    elif hf_mnist_vae:
        model, config = _load_hf_mnist_vae(hf_cache)
        device = get_device()
        model = model.to(device)
    elif hf_model:
        from model import PretrainedVAE
        device = get_device()
        image_size = 256
        model = PretrainedVAE(
            hf_model_id=hf_model, image_channels=3, image_size=image_size,
            recon_loss_type="mse", beta=1.0, cache_dir=hf_cache,
        ).to(device)
        config = {
            "model": "PretrainedVAE", "hf_model": hf_model,
            "dataset": "custom", "data_root": "./dataset",
            "val_split": 0.1, "resize": image_size, "seed": 42,
            "beta": 1.0, "recon_loss_type": "mse",
            "image_channels": 3, "image_size": image_size,
            "latent_dim": model.latent_channels,
        }
    else:
        model, config, device = load_model(checkpoint)
    _state.update(model=model, config=config, device=device)

    app = FastAPI(title="VAE Studio API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                      allow_methods=["*"], allow_headers=["*"])

    # ---- Metadata ---- #
    @app.get("/api/info")
    def info():
        m, cfg = _ensure_model()
        return _success({
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "latent_dim": cfg["latent_dim"],
            "image_channels": cfg["image_channels"],
            "image_size": cfg["image_size"],
        })

    # ---- Reconstruct ---- #
    @app.post("/api/reconstruct")
    async def reconstruct(file: UploadFile = File(...)):
        m, cfg = _ensure_model()
        arr = _upload_to_numpy(file)
        _, recon, stats = be.reconstruct(m, arr, cfg["image_channels"],
                                         cfg["image_size"])
        return _success({
            "reconstruction": _numpy_to_png_base64(recon),
            "stats": stats,
        })

    # ---- Generate ---- #
    @app.get("/api/generate")
    def generate(n: int = 16, seed: int = 0):
        m, cfg = _ensure_model()
        grid = be.generate_samples(m, n, seed)
        return _success({"image": _numpy_to_png_base64(grid)})

    # ---- Denoise ---- #
    @app.post("/api/denoise")
    async def denoise(file: UploadFile = File(...),
                      noise_type: str = Form("gaussian"),
                      level: float = Form(0.4)):
        m, cfg = _ensure_model()
        arr = _upload_to_numpy(file)
        noisy, denoised, info = be.denoise(m, arr, noise_type, level,
                                           cfg["image_channels"],
                                           cfg["image_size"])
        return _success({
            "noisy": _numpy_to_png_base64(noisy),
            "denoised": _numpy_to_png_base64(denoised),
            "info": info,
        })

    # ---- Inpaint ---- #
    @app.post("/api/inpaint")
    async def inpaint(file: UploadFile = File(...),
                      mask_frac: float = Form(0.5),
                      steps: int = Form(200)):
        m, cfg = _ensure_model()
        arr = _upload_to_numpy(file)
        masked, filled = be.inpaint_image(m, arr, mask_frac, steps, 0.05,
                                          cfg["image_channels"],
                                          cfg["image_size"])
        return _success({
            "masked": _numpy_to_png_base64(masked),
            "inpainted": _numpy_to_png_base64(filled),
        })

    # ---- Interpolate ---- #
    @app.post("/api/interpolate")
    async def interpolate(file_a: UploadFile = File(...),
                          file_b: UploadFile = File(...),
                          steps: int = Form(10),
                          use_slerp: bool = Form(False)):
        m, cfg = _ensure_model()
        a = _upload_to_numpy(file_a)
        b = _upload_to_numpy(file_b)
        grid = be.interpolate_images(m, a, b, steps, use_slerp,
                                     cfg["image_channels"], cfg["image_size"])
        return _success({"image": _numpy_to_png_base64(grid)})

    # ---- Latent generate ---- #
    @app.get("/api/latent")
    def latent(values: str = ""):
        m, cfg = _ensure_model()
        latent_dim = cfg["latent_dim"]
        slider_vals = [float(v) for v in values.split(",") if v.strip()] if values else []
        img = be.latent_generate(m, slider_vals, latent_dim)
        return _success({"image": _numpy_to_png_base64(img)})

    # ---- Attribute-controlled face generation (CelebA VAE) ---- #
    @app.get("/api/attributes")
    def list_attributes():
        """List available facial attribute directions (CelebA beta-VAE)."""
        m, cfg = _ensure_model()
        dirs = getattr(m, "_directions", {})
        return _success({"attributes": list(dirs.keys()),
                         "has_directions": len(dirs) > 0})

    @app.get("/api/face")
    def generate_face(seed: int = 0, attribute: str = "",
                      strength: float = 0.0):
        """Generate a face with optional attribute control.

        If ``attribute`` is given (e.g. "Smiling"), the latent is shifted
        along that attribute's direction vector by ``strength``.
        """
        m, cfg = _ensure_model()
        torch.manual_seed(int(seed))
        device = next(m.parameters()).device
        z = torch.randn(1, cfg["latent_dim"], device=device)

        dirs = getattr(m, "_directions", {})
        if attribute and attribute in dirs:
            direction = dirs[attribute].to(device)
            z = z + direction * float(strength)

        with torch.no_grad():
            img = m.decode(z)
        return _success({"image": _numpy_to_png_base64(be.to_display(img))})

    # ---- Serve frontend static files ---- #
    web_dir = os.path.join(ROOT, "web")
    if os.path.isdir(web_dir):
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="VAE Studio FastAPI server.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a locally trained checkpoint.")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="HuggingFace AutoencoderKL repo id (e.g. stabilityai/sd-vae-ft-mse).")
    parser.add_argument("--hf-celeba-vae", action="store_true",
                        help="Load the CelebA beta-VAE with face attribute control "
                             "(smiling, eyeglasses, hair color, etc.).")
    parser.add_argument("--hf-mnist-vae", action="store_true",
                        help="Load the HuggingFace MNIST VAE (uday9k/Binarized_MNIST_VAE) "
                             "— a standalone VAE that generates real digits.")
    parser.add_argument("--hf-cache", type=str, default="./pretrained")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.checkpoint and not args.hf_model and not args.hf_mnist_vae and not args.hf_celeba_vae:
        parser.error("Provide one of --checkpoint, --hf-model, --hf-mnist-vae, or --hf-celeba-vae.")

    app = create_app(checkpoint=args.checkpoint, hf_model=args.hf_model,
                     hf_cache=args.hf_cache, hf_mnist_vae=args.hf_mnist_vae,
                     hf_celeba_vae=args.hf_celeba_vae)
    print(f"VAE Studio running on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
