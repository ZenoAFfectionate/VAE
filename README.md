# 🧠 VAE Research Repository

> A clean, reproducible and production-quality PyTorch implementation of
> **Variational Autoencoders (VAE)** and **Sparse VAE** for generative modelling
> research on standard image benchmarks.

---

## 📖 1. Project Overview

### 🔮 Variational Autoencoder (VAE)

A VAE is a deep latent-variable generative model. An **encoder** maps an input
`x` to a distribution over a latent variable `z`, parameterized by a mean `μ`
and a log-variance `logσ²`. A latent code is drawn with the **reparameterization
trick** `z = μ + σ · ε`, `ε ~ N(0, I)`, which keeps sampling differentiable so
the network can be trained with backpropagation. A **decoder** reconstructs the
input from `z`. Training maximizes the Evidence Lower Bound (ELBO), equivalent to
minimizing:

```
L = Reconstruction(x, x̂) + β · KL( N(μ, σ²) || N(0, I) )
```

### 🌫️ Sparse VAE

The Sparse VAE augments the standard objective with a **sparsity constraint** on
the latent activations. The average activation of each latent unit (mapped to
`(0, 1)` via a sigmoid) is encouraged to approach a small target value `ρ` using
a Bernoulli KL-divergence penalty:

```
L_sparse = L_VAE + λ · Σ_j KL( ρ || ρ̂_j )
```

This yields a more **disentangled, interpretable** latent space where only a few
units are active for any given input.

### ✨ Implemented Features

| | Feature |
|---|---------|
| 🏗️ | Standard **Vanilla VAE** and **Sparse VAE** with a unified, drop-in interface |
| 🔧 | Two selectable backbones: **MLP** (MNIST / Fashion-MNIST) and **CNN** (CIFAR-10 / CelebA / STL-10) |
| 🎛️ | Configurable reconstruction loss (**BCE** / **MSE**), `β`-weighting, latent dim, hidden dims and activation functions |
| 🔄 | Full training + validation pipelines with logging, checkpointing and visualization |
| 🎲 | Fully reproducible (fixed seeds across Python / NumPy / PyTorch) |
| 🛡️ | Numerically stable KL / loss computation with clamping |
| 💻 | Runs unmodified on both **CPU** and **CUDA** |
| 🤗 | **HuggingFace integration**: load pretrained VAEs — CelebA face VAE (with 😊 attribute control), MNIST digit VAE, Stable Diffusion VAE |
| 🌐 | **VAE Studio** web app (FastAPI + native frontend) with 6 interactive pages |
| 📊 | Command-line demos: interpolation, traversal, latent maps, anomaly detection, denoising, inpainting |

### 📦 Supported Datasets

| Dataset | Size | Channels | Native Size | Best for |
|---------|------|----------|-------------|----------|
| 📝 `MNIST` | 60k | 1 | 28×28 | Quick prototyping, digits |
| 👕 `FashionMNIST` | 60k | 1 | 28×28 | Clothing categories |
| 🖼️ `CIFAR10` | 50k | 3 | 32×32 | Natural images (small) |
| 😊 `CelebA` | ~200k | 3 | 178×218 | **Large-scale** face generation |
| 🔍 `STL10` | 100k unlabeled | 3 | 96×96 | **Unsupervised** representation learning |

> Datasets are fetched **automatically at runtime** 🔄 — on launch the code
> checks the local `dataset/` directory and only downloads what is missing,
> reusing the cached copy on subsequent runs. For CelebA and STL-10, pass
> `--resize` to control the training resolution.

---

## ⚙️ 2. Environment Setup

- 🐍 **Python**: 3.8+
- 📚 **Core dependencies**: PyTorch, torchvision, NumPy, FastAPI

```bash
# (optional) create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

> 💡 Install the CUDA-enabled PyTorch build matching your driver from
> <https://pytorch.org> if you intend to train on GPU.

---

## 🚀 3. Quick Start

### 🏗️ 3.1 Train a standard VAE

```bash
# MNIST with the MLP backbone
python train.py --model VAE --dataset MNIST --backbone mlp \
    --latent-dim 32 --epochs 50 --batch-size 128 --lr 1e-3

# CIFAR-10 with the CNN backbone (use MSE reconstruction for RGB)
python train.py --model VAE --dataset CIFAR10 --backbone cnn \
    --hidden-dims 32 64 128 --latent-dim 128 --recon-loss-type mse \
    --epochs 100 --batch-size 128

# CelebA (larger dataset, ~200k faces) — resize to 64x64 for efficiency
python train.py --model VAE --dataset CelebA --backbone cnn \
    --hidden-dims 32 64 128 256 --latent-dim 256 --recon-loss-type mse \
    --resize 64 --epochs 50 --batch-size 64

# STL-10 (100k unlabeled images) — good for unsupervised VAE training
python train.py --model VAE --dataset STL10 --backbone cnn \
    --hidden-dims 32 64 128 --latent-dim 128 --recon-loss-type mse \
    --resize 48 --epochs 50 --batch-size 128
```

### 🌫️ 3.2 Train a Sparse VAE

```bash
python train.py --model SparseVAE --dataset FashionMNIST --backbone mlp \
    --latent-dim 64 --target-sparsity 0.05 --sparse-weight 1.0 \
    --epochs 50 --batch-size 128
```

### 📊 3.3 Evaluate a trained checkpoint & generate samples

```bash
python valid.py --checkpoint outputs/VAE_MNIST_<timestamp>/best.pth \
    --batch-size 128 --num-vis 8 --num-gen 64
```

Reconstruction comparisons and generated-sample grids are written to a
timestamped folder under `outputs/eval/`.

### 🤗 3.4 Use a mature pretrained VAE from HuggingFace

The project can directly load and evaluate ready-to-use, well-trained VAEs
published on the HuggingFace Hub — specifically Stable Diffusion–style
`AutoencoderKL` models — without any local training. The pretrained weights are
**downloaded and cached automatically** on first use (into `./pretrained/` by
default, configurable via `--hf-cache`).

```bash
pip install diffusers huggingface_hub

# Stable Diffusion VAE (fine-tuned, MSE) on CIFAR-10 resized to 256x256
python valid.py --hf-model stabilityai/sd-vae-ft-mse \
    --dataset CIFAR10 --resize 256 --batch-size 8 --num-vis 8 --num-gen 16
```

| HuggingFace repo id | Notes |
|---------------------|-------|
| `stabilityai/sd-vae-ft-mse` | SD 1.x VAE, MSE fine-tuned — smooth reconstructions |
| `stabilityai/sd-vae-ft-ema` | SD 1.x VAE, EMA fine-tuned — sharper details |
| `madebyollin/sdxl-vae-fp16-fix` | SDXL VAE, fp16-safe variant |

> ⚠️ Pretrained VAEs are **RGB** models. Input H/W must be **divisible by 8**;
> pass `--resize 256`. The adapter handles `[0,1] ↔ [-1,1]` pixel-range
> conversion internally. `--checkpoint` and `--hf-model` are mutually exclusive.

### 🎛️ 3.5 Key Hyperparameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | `VAE` or `SparseVAE` | `VAE` |
| `--backbone` | `mlp` or `cnn` | `mlp` |
| `--dataset` | `MNIST` / `FashionMNIST` / `CIFAR10` / `CelebA` / `STL10` | `MNIST` |
| `--hidden-dims` | Hidden widths (MLP) / channel widths (CNN) | `512 256` |
| `--latent-dim` | Latent space dimensionality | `32` |
| `--activation` | `relu`/`elu`/`leaky_relu`/`gelu`/`tanh` | `relu` |
| `--recon-loss-type` | `bce` or `mse` | `bce` |
| `--beta` | KL divergence weight (β-VAE) | `1.0` |
| `--target-sparsity` | Target latent activation ρ (Sparse VAE) | `0.05` |
| `--sparse-weight` | Sparsity penalty weight λ (Sparse VAE) | `1.0` |
| `--lr` | Learning rate | `1e-3` |
| `--epochs` | Training epochs | `50` |
| `--batch-size` | Mini-batch size | `128` |
| `--scheduler` | Enable ReduceLROnPlateau | off |
| `--seed` | Global random seed | `42` |
| `--no-cuda` | Force CPU execution | off |
| `--resize` | Resize images to N×N | none |
| `--checkpoint` | (valid.py) Local checkpoint to evaluate | none |
| `--hf-model` | (valid.py) HuggingFace `AutoencoderKL` repo id | none |

---

## 📈 4. Expected Results

Reference validation behaviour (per-sample BCE/MSE-sum reconstruction; exact
numbers vary with configuration and hardware):

| Dataset | Backbone | Recon loss | KL | Notes |
|---------|----------|-----------|----|-------|
| 📝 MNIST | MLP | ~80–110 | ~20–30 | Crisp digit reconstructions after ~30 epochs |
| 👕 Fashion-MNIST | MLP | ~210–240 | ~20–30 | Recognizable garment silhouettes |
| 🖼️ CIFAR-10 | CNN (MSE) | low MSE-sum | ~30–60 | Blurry but coherent object structure |

> 🌫️ For the **Sparse VAE**, the reported `sparsity_ratio` should rise across
> epochs as more latent units become inactive, while reconstruction quality
> remains comparable to the standard VAE.

---

## 🤗 5. Pretrained VAE Models from HuggingFace

The project can directly load and use mature, ready-to-use VAE models
downloaded from the HuggingFace Hub — **no local training required**. Weights are
cached in `./pretrained/` (git-ignored) and reused on subsequent runs.

### 😊 CelebA Face VAE (Beta-VAE with attribute control)

**Model**: `ayushshah/beta-vae-capacity-annealing-celeba`
- Trained on ~200k CelebA celebrity faces (64×64 RGB)
- CNN architecture with ResNet residual blocks, latent_dim=32
- Includes **8 facial attribute direction vectors** for controllable generation

| Attribute | Effect |
|-----------|--------|
| 😊 `Smiling` | Add/remove a smile |
| 👓 `Eyeglasses` | Add/remove glasses |
| 👱 `Blond_Hair` | Change hair to blond |
| 👦 `Young` | Make the face look younger |
| 👨 `Male` | Shift gender appearance |
| 💇 `Bangs` | Add/remove bangs |
| 🖤 `Black_Hair` | Change hair to black |
| 🧊 `Pale_Skin` | Lighten skin tone |

```bash
# Download (automatic on first run) and launch VAE Studio with face model
python server.py --hf-celeba-vae
# open http://127.0.0.1:8000

# Generate a random face
curl "http://127.0.0.1:8000/api/generate?n=4&seed=42" -o face.png

# Generate a smiling face (attribute control via API)
curl "http://127.0.0.1:8000/api/face?seed=42&attribute=Smiling&strength=3.0" -o smiling.png

# List all available attributes
curl "http://127.0.0.1:8000/api/attributes"
```

### 📝 MNIST Digit VAE

**Model**: `uday9k/Binarized_MNIST_VAE`
- Standard MLP VAE trained on MNIST for 44 epochs
- latent_dim=20, generates 28×28 grayscale digits from N(0, I)

```bash
python server.py --hf-mnist-vae
```

### 🖼️ Stable Diffusion VAE (image codec)

**Model**: `stabilityai/sd-vae-ft-mse`
- AutoencoderKL from Stable Diffusion (256×256 RGB)
- **Not a standalone generator** — its latent space is not aligned with N(0,I)
- Best used for high-fidelity reconstruction / denoising of RGB images

```bash
python server.py --hf-model stabilityai/sd-vae-ft-mse
```

> ⚠️ SD VAEs produce noise-like images when sampling from N(0,I) because the
> latent space is scaled by a `scaling_factor`. For generation use the CelebA
> or MNIST VAE instead.

---

## 🌐 6. Applications & Demos

### 🖥️ 6.1 VAE Studio — Interactive Web App

A production-grade **FastAPI + native frontend** web app with a dark *Developer
Tool* theme (Inter / JetBrains Mono, indigo accent `#6366F1`), drag-and-drop
image upload, and six functional pages:

| Page | What it does |
|------|-------------|
| 🖼️ Reconstruct | Upload any image → auto-resized, encoded & reconstructed |
| ✨ Generate | Sample brand-new images from the latent prior |
| 🎛️ Latent Lab | Drag sliders to explore latent dimensions live |
| 🧹 Denoise | Add noise to an upload and let the VAE clean it (PSNR gain) |
| 🩹 Inpaint | Mask a region and recover it via latent optimization |
| 🔀 Interpolate | Morph between two uploaded images |

```bash
pip install fastapi uvicorn python-multipart Pillow
python server.py --checkpoint outputs/VAE_MNIST_<ts>/best.pth
# open http://127.0.0.1:8000
```

### 📊 6.2 Command-line Demos

Scriptable demos under `examples/` that save figures to `outputs/examples/`:

| Demo | Application |
|------|-------------|
| `latent_interpolation.py` | 🔀 Latent-space morphing between image pairs |
| `latent_traversal.py` | 🔍 Per-dimension factor inspection (disentanglement) |
| `latent_map.py` | 🗺️ 2D latent map (PCA / t-SNE) colored by class |
| `anomaly_detection.py` | 🚨 Out-of-distribution detection via reconstruction error (ROC-AUC) |
| `denoising.py` | 🧹 Image denoising with PSNR reporting |
| `inpainting.py` | 🩹 Region recovery via latent optimization |

```bash
python examples/latent_interpolation.py --checkpoint outputs/VAE_MNIST_<ts>/best.pth
python examples/anomaly_detection.py    --checkpoint outputs/VAE_MNIST_<ts>/best.pth \
    --anomaly-dataset FashionMNIST
```

---

## 🧪 7. Testing Guide

The repository ships with a fast, self-contained `pytest` suite (169 tests!).
Tests use dummy in-memory data and monkeypatching — **no dataset download
required**, full suite runs in seconds.

```bash
pip install pytest

python -m pytest                    # run everything
python -m pytest -v                 # verbose
python -m pytest tests/test_models.py    # single file
```

| Test file | Scope |
|-----------|-------|
| `test_models.py` | 🏗️ Output shapes, reparameterization, gradient flow, ELBO / sparsity loss validity, edge cases |
| `test_data.py` | 📦 Batch shapes, value ranges, train/val splits, loader iteration |
| `test_utils.py` | 🔧 Loss functions, metrics, seed fixing, checkpoint I/O, argument parsing |
| `test_pipeline.py` | 🔄 End-to-end training & evaluation, checkpoint save → reload flow |
| `test_pretrained.py` | 🤗 HuggingFace adapter (skipped if `diffusers` absent) |
| `test_examples.py` | 📊 Demo helpers: interpolation, noising, masking, AUC, PSNR, PCA |
| `test_studio.py` | 🌐 VAE Studio backend: preprocessing, reconstruct, generate, denoise, inpaint, interpolate |

---

## 💡 8. Notes & FAQ

- 📝 **BCE vs MSE**: Use BCE for MNIST/Fashion-MNIST (near-binary grayscale), MSE for CIFAR-10/CelebA (natural RGB).
- 📐 **CNN image size**: Each `--hidden-dims` entry halves the spatial resolution, so `image_size` must be divisible by `2^(len(hidden-dims))` (e.g. 3 stages for 32×32, 2 stages for 28×28). The model raises a clear error otherwise.
- 🎲 **Reproducibility**: All seeds fixed at startup; cuDNN runs in deterministic mode — identical configs produce identical results.
- 💾 **Out-of-memory**: Lower `--batch-size`, reduce `--latent-dim`/`--hidden-dims`, or pass `--no-cuda`.
- 🔧 **Custom dataset**: Add an entry to `_DATASET_REGISTRY` in `data/dataset.py` — no other changes needed.
- 📁 **Results location**: Logs & configs → `logs/<exp_name>/`; checkpoints & figures → `outputs/<exp_name>/`. Both git-ignored.
- 💽 **Dataset cache**: Raw data → `./dataset/` (configurable via `--data-root`), auto-created and git-ignored.
