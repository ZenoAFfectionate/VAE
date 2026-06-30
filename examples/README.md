# VAE Application Demos

A collection of small, self-contained demos showcasing practical applications of
the VAE / Sparse VAE models trained in this repository. Each demo loads a
checkpoint produced by `train.py` and reuses the project's own modules
(`model/`, `data/`, `utils/`).

> All demos share reusable logic in `examples/common.py` (model/data loading and
> the math helpers used for interpolation, noising, masking, AUC and PCA).

## Prerequisites

1. Train a model first (or reuse an existing checkpoint):
   ```bash
   python train.py --model VAE --dataset MNIST --backbone mlp \
       --latent-dim 32 --epochs 30
   ```
   The checkpoint is saved under `outputs/VAE_MNIST_<timestamp>/best.pth`.

2. Install optional dependencies used by some demos:
   ```bash
   pip install matplotlib      # latent_map, anomaly_detection (plots)
   pip install scikit-learn    # latent_map --method tsne (optional)
   ```

Outputs are written under `outputs/examples/<demo>/` (git-ignored).

---

## Demos

### 1. Latent Interpolation (image morphing)
Smoothly morph between pairs of images through the latent space.
```bash
python examples/latent_interpolation.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --pairs 6 --steps 10
# add --slerp for spherical interpolation
```
Shows the **continuity** of the latent space. Each row is one pair; columns are
interpolation steps.

### 2. Latent Traversal (factor discovery)
Sweep one latent dimension at a time and observe what it controls.
```bash
python examples/latent_traversal.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --num-dims 10 --steps 9 --span 3
```
Useful for inspecting **disentanglement** (try a Sparse VAE or larger `--beta`).

### 3. Latent Map (2D visualization)
Encode the validation set and plot a 2D scatter colored by class.
```bash
python examples/latent_map.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --num-samples 2000 --method pca
# or --method tsne (requires scikit-learn)
```
Class clusters indicate a **semantically organized** latent space.

### 4. Anomaly Detection (out-of-distribution)
Use reconstruction error to separate normal from anomalous images. The cleanest
setup is cross-dataset: a MNIST-trained VAE treats FashionMNIST as anomalies.
```bash
python examples/anomaly_detection.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --anomaly-dataset FashionMNIST
```
Reports **ROC-AUC** and saves a score histogram. (Normal and anomaly datasets
must share image shape, e.g. MNIST ↔ FashionMNIST, both 1×28×28.)

### 5. Denoising
Corrupt images with noise and let the VAE project them back onto the data
manifold.
```bash
python examples/denoising.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --noise gaussian --std 0.4
# or --noise salt_pepper --amount 0.1
```
Saves a clean / noisy / denoised comparison and reports **PSNR gain**.

### 6. Inpainting (latent optimization)
Mask a region, then optimize a latent code to match the observed pixels and
hallucinate the missing content.
```bash
python examples/inpainting.py \
    --checkpoint outputs/VAE_MNIST_<ts>/best.pth --mask-frac 0.5 --steps 300
```
Rows: original / masked / inpainted.

### 7. VAE Studio — Interactive Web App
A production-grade FastAPI + native frontend web app (dark Developer Tool theme)
with six functional pages and drag-and-drop image upload.

**Not in `examples/`** — it's the project root entry point:
```bash
pip install fastapi uvicorn python-multipart Pillow
python server.py --checkpoint outputs/VAE_MNIST_<ts>/best.pth
# open http://127.0.0.1:8000
```
Pages: Reconstruct / Generate / Latent Lab / Denoise / Inpaint / Interpolate.
All inference logic lives in `examples/studio_backend.py` (unit tested in
`tests/test_studio.py`); `server.py` is a thin FastAPI API layer and
`web/index.html` is a native HTML/CSS/JS SPA.

Options: `--num-sliders` (latent dims exposed), `--span` (slider range),
`--server-name` / `--server-port`, `--share` (public link).

---

## Notes
- Demos auto-detect CPU/CUDA. They run on CPU but a GPU speeds up larger models
  (and the inpainting optimization).
- For CNN-backbone / CIFAR-10 checkpoints, the same commands apply; pass the
  corresponding checkpoint. Color images render in RGB automatically.
- The pure logic in `examples/common.py` is covered by `tests/test_examples.py`.
