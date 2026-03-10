# SegFlow4D

Propagate sparse segmentations across all time points of a 4D image using GPU-accelerated deformable image registration.

Given one or more segmented reference time points, SegFlow4D registers every target frame to its nearest reference and warps both the label map and any associated surface meshes, producing a consistent 4D segmentation series.

---

## Requirements

### GPU Installation (FireANTs backend)

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| CUDA toolkit (`nvcc`) | ≥ 11.8 (must match PyTorch CUDA build) |
| PyTorch | ≥ 1.8 (GPU build) |
| Conda / Miniconda | any recent version |

### CPU Installation (Greedy backend)

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| Conda / Miniconda | any recent version |

No GPU or CUDA toolkit required. Registration runs on CPU via `picsl-greedy`.

---

## Setup

Choose the installation track that matches your hardware:

- **[GPU Installation](#gpu-installation-fireants-backend-1)** — CUDA-capable GPU required; faster registration using FireANTs
- **[CPU Installation](#cpu-installation-greedy-backend-1)** — no GPU required; simpler setup using the Greedy backend

---

### GPU Installation (FireANTs backend)

#### 1. Create a conda environment

```bash
conda create -n segflow4d python=3.10
conda activate segflow4d
```

#### 2. Install PyTorch (GPU) — do this first

PyPI only distributes CPU-only PyTorch wheels. If you let FireANTs or segflow4d
pull in `torch` as a dependency automatically, you will get a CPU build.
Install the GPU build explicitly *before* any other package to prevent this.

Check your CUDA version first:
```bash
nvcc --version        # CUDA toolkit version (used to compile CUDA extensions)
nvidia-smi            # driver's maximum supported CUDA version (top-right corner)

# Parse the version number directly:
nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+"   # e.g. 12.0
```
Use the `nvcc` version when choosing the PyTorch wheel. If `nvcc` is not found,
install the CUDA toolkit or add it to `PATH`.

Visit https://pytorch.org/get-started/locally/ to find the right command for your CUDA version.

PyTorch publishes wheels only for specific CUDA versions. Choose the one that
matches your `nvcc` output:

| nvcc version | PyTorch wheel tag |
|---|---|
| 11.8 | `cu118` |
| 12.1 | `cu121` |
| 12.4 | `cu124` |
| 12.6 | `cu126` |
| 12.8 | `cu128` |

> **Note:** there is no `cu120`, `cu122`, etc. — only the tags above are available.
> If your toolkit version is not listed, use the closest lower entry, or upgrade
> the toolkit to a listed version (see step 2 in the troubleshooting section).

Example for CUDA 12.8 (required for Blackwell / RTX 50-series GPUs):
```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128 \
  --trusted-host download.pytorch.org
```

Example for CUDA 12.1:
```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121 \
  --trusted-host download.pytorch.org
```

> **Blackwell GPUs (RTX 50-series, compute capability 10.0):** PyTorch built
> against CUDA 12.1 does not support these GPUs. Use CUDA 12.8 instead:
> ```bash
> pip install torch torchvision \
>   --index-url https://download.pytorch.org/whl/cu128 \
>   --trusted-host download.pytorch.org
> ```

> **Corporate/institutional networks:** If you see an SSL certificate error, your
> network proxy is intercepting HTTPS traffic. Add `--trusted-host` to bypass it:
> ```bash
> pip install torch torchvision \
>   --index-url https://download.pytorch.org/whl/cu121 \
>   --trusted-host download.pytorch.org
> ```
> For a permanent fix, ask your IT team for the corporate CA certificate and run:
> `pip config set global.cert /path/to/ca-bundle.pem`

#### 3. Install FireANTs with fused CUDA ops

The fused CUDA operations are required for best performance. Run the provided setup script:

```bash
bash scripts/setup_fireants.sh
```

This will:
- Clone the [FireANTs repository](https://github.com/rohitrango/FireANTs)
- Install the `fireants` Python package
- Compile and install the `fireants_fused_ops` CUDA extension

> **Note:** If the system `nvcc` version doesn't match the CUDA version PyTorch
> was compiled against, the script will automatically install the matching
> `cuda-toolkit` into the active conda environment via `conda install -c nvidia`.
> No manual toolkit management is needed in most cases.

#### 4. Install SegFlow4D

```bash
pip install -e .
```

This installs `segflow4d` in editable (development) mode, so local code changes
take effect immediately without reinstalling. All remaining Python dependencies
listed in `pyproject.toml` are installed automatically.

> **Upgrading from a previous version?** If you installed an older version that
> used a different package layout, stale top-level packages may linger in
> `site-packages` and cause import errors (e.g.
> `RegistrationManager is not initialized yet`). Clean them up with:
> ```bash
> SITE=$(python -c "import site; print(site.getsitepackages()[0])")
> rm -rf "$SITE"/{common,processing,propagation,registration,utility}
> pip install -e .
> ```

---

### CPU Installation (Greedy backend)

No GPU or CUDA toolkit required. Uses [picsl-greedy](https://pypi.org/project/picsl-greedy/) for CPU-based deformable registration.

#### 1. Create a conda environment

```bash
conda create -n segflow4d python=3.10
conda activate segflow4d
```

#### 2. Install SegFlow4D with the Greedy extra

```bash
pip install -e ".[greedy]"
```

This installs `segflow4d` along with `picsl-greedy` and all other required
dependencies. PyTorch is pulled in automatically as a CPU build — no manual
PyTorch installation step is needed.

> **Note:** Registration will run entirely on CPU. Expect longer runtimes
> compared to the GPU (FireANTs) backend. Set `registration_backend: greedy`
> in your config file (see Configuration below).

---

## Configuration

Copy and edit the example configuration file:

```bash
cp example_config.yaml config.yaml
```

Key fields:

| Field | Description |
|---|---|
| `image4d` | Path to the 4D image (e.g. `.nii.gz`, `.tif`) |
| `output` | Directory where propagated segmentations are written |
| `registration_backend` | `fireants` (default) or `greedy` |
| `lowres_factor` | Downsample factor for registration (0.5 = half resolution) |
| `dilation_radius` | Mask dilation radius in voxels |
| `tp_input_groups` | List of `{tp_ref, tp_targets, seg_ref, additional_meshes}` entries |

See `example_config.yaml` for the full set of registration options (`scales`, `affine_iterations`, `deformable_iterations`, `loss_type`, etc.).

---

## Usage

### Run from a config file (recommended)

```bash
segflow4d --config config.yaml
```

### Run from the command line

```bash
segflow4d \
  --image4d /path/to/4d_image.nii.gz \
  --output  /path/to/output \
  --tp-ref  0 \
  --tp-targets 1 2 3 \
  --seg-ref /path/to/seg_tp0.nii.gz \
  --lowres-factor 0.5 \
  --dilation-radius 2
```

Pass `--help` to see all available options:

```bash
segflow4d --help
```

---

## Additional CLI Tools

| Command | Description |
|---|---|
| `create-reference-mask` | Create a reference mask from a segmentation |
| `create-tp-images` | Extract individual time-point images from a 4D volume |
| `resample-to-reference` | Resample an image to match a reference geometry |
| `mesh-gen` | Generate surface meshes from a label map |
| `gpu-mesh-warper` | Warp a surface mesh using a GPU deformation field |
| `run-fireants` | Run FireANTs registration directly |
| `segmentation-validation` | Compute Dice / surface-distance metrics for a single case |
| `batch-segmentation-validation` | Run validation across a batch of cases |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fireants_fused_ops'`**  
The fused ops were not installed. Re-run `bash setup_fireants.sh`.

**`ImportError: libc10.so: cannot open shared object file`**  
PyTorch libraries are not on `LD_LIBRARY_PATH`. Make sure you are running inside the correct conda environment.

**`fatal error: cuda_runtime.h: No such file or directory`**  
The CUDA runtime development headers are missing. Install them into the active
conda environment:
```bash
# Replace 12.1 with your PyTorch CUDA version (python -c "import torch; print(torch.version.cuda)")
conda install -c nvidia cuda-cudart-dev=12.1
```
Then re-run `bash scripts/setup_fireants.sh`.

**`nvcc warning: incompatible redefinition for option 'compiler-bindir'` / wrong `-ccbin` path**  
Stale environment variables from a previously activated conda environment (one
that had `gcc_linux-64` or `gxx_linux-64` installed) are injecting a wrong host
compiler into `nvcc`. Fix by unsetting them before building:
```bash
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDAHOSTCXX
bash scripts/setup_fireants.sh
```
Or start a fresh shell and activate only your target environment before running
the script.

**`error: #error -- unsupported GNU version! gcc versions later than 12 are not supported!`**  
Your system GCC is too new for the installed CUDA toolkit (CUDA 12.x requires GCC ≤ 12,
CUDA 11.x requires GCC ≤ 11). The setup script handles this automatically, but if you
are building manually, install a compatible GCC and set it as the host compiler:
```bash
sudo apt install gcc-12 g++-12
export CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12
```

**CUDA out of memory**  
Increase `lowres_factor` (e.g. `0.5` → `0.25`) or reduce the number of scales / iterations in your config.
