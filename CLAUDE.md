# CLAUDE.md — AI Assistant Guide for SegFlow4D

## Project Overview

**SegFlow4D** is a Python library for 4D medical image segmentation propagation. Given a reference timepoint with a manual segmentation, it propagates that segmentation across all timepoints of a 4D (3D + time) image using GPU-accelerated deformable image registration.

- **Author**: Jilei Hao
- **Version**: 1.0.0a2 (alpha)
- **Python**: 3.10+
- **GPU Required**: CUDA 11.8+ with PyTorch GPU build

---

## Repository Layout

```
segflow4d/
├── pyproject.toml               # Package metadata and dependencies
├── example_config.yaml          # Example configuration (all parameters documented)
├── README.md                    # Setup instructions, usage, troubleshooting
├── scripts/
│   └── setup_fireants.sh        # Installs FireANTs with CUDA fused ops
├── FireANTs/                    # Cloned FireANTs registration library (git submodule-like)
└── src/segflow4d/               # Main package root
    ├── main.py                  # CLI entry point
    ├── common/
    │   └── types/               # Core data structures (wrappers and containers)
    ├── processing/              # Image resampling, dilation, mesh generation
    ├── propagation/             # Pipeline orchestration and strategies
    ├── registration/            # Registration backends and GPU job management
    └── utility/                 # Async I/O, validation, image/mesh helpers
```

### Key Source Files

| File | Purpose |
|------|---------|
| `src/segflow4d/main.py` | CLI arg parsing, YAML config loading, pipeline entry point |
| `src/segflow4d/propagation/propagation_pipeline.py` | Main execution engine; orchestrates all timepoints |
| `src/segflow4d/propagation/propagation_strategy/` | `Sequential` and `Star` propagation strategies |
| `src/segflow4d/registration/registration_manager/gpu_registration_manager.py` | GPU VRAM management, job dispatch via multiprocessing |
| `src/segflow4d/registration/registration_manager/gpu_device_manager.py` | Per-device VRAM monitoring |
| `src/segflow4d/registration/registration_manager/gpu_worker.py` | Subprocess worker for CUDA-isolated job execution |
| `src/segflow4d/registration/registration_manager/factory.py` | Singleton factory: CPU vs GPU manager selection |
| `src/segflow4d/registration/registration_handler/fireants/` | FireANTs registration backend implementation |
| `src/segflow4d/common/types/image_wrapper.py` | `ImageWrapper` — wraps `SimpleITK.Image` with deepcopy and metadata |
| `src/segflow4d/common/types/mesh_wrapper.py` | `MeshWrapper` — wraps `vtk.vtkPolyData` with pickling for multiprocessing |
| `src/segflow4d/common/types/tp_data.py` | `TPData` — per-timepoint container (image, seg, mesh, registration data) |
| `src/segflow4d/common/types/propagation_input.py` | `PropagationInput` and its factory |
| `src/segflow4d/utility/file_writer/async_writer.py` | Thread-based async file I/O writer |
| `src/segflow4d/utility/validation/batch_validation.py` | Batch segmentation evaluation |
| `src/segflow4d/utility/validation/segmentation_validation.py` | Dice + surface distance metrics |

---

## Architecture and Data Flow

### Two-Stage Propagation

1. **Mask stage** (low-resolution, fast): Propagates a dilated binary mask using the `Sequential` strategy (frame-to-frame). Used to focus high-res registration.
2. **Segmentation stage** (high-resolution): Propagates the full multi-label segmentation using the `Star` strategy (all timepoints registered in parallel relative to reference).

### Data Flow

```
Config (YAML or CLI args)
  → PropagationInputFactory.create()  → PropagationInput
  → PropagationPipeline.run()
      → TPPartition  (splits timepoints into forward/backward groups)
      → TPData[]     (per-timepoint: image, seg, mesh, fields)
      → PropagationStrategy (Sequential | Star)
          → RegistrationManager (GPU | CPU)
              → GPUWorker subprocess (CUDA-isolated)
                  → RegistrationHandler (FireANTs | Greedy)
  → AsyncWriter     (non-blocking file I/O)
```

### Concurrency Model

- **Threading**: Parallel forward/backward propagation; `AsyncWriter` for I/O; job dispatch coordination via `ThreadPoolExecutor`.
- **Multiprocessing**: Each GPU job runs in an isolated subprocess (`ProcessPoolExecutor`) to prevent CUDA context conflicts.
- **VRAM Management**: `GPUDeviceManager` monitors free VRAM per device; `GPURegistrationManager` holds device locks and enforces a 2 GB safety margin before dispatching jobs.

---

## Design Patterns

| Pattern | Where Used |
|---------|-----------|
| **Singleton** | `RegistrationManager.get_instance()` |
| **Factory** | `RegistrationManagerFactory`, `PropagationStrategyFactory`, `PropagationInputFactory`, `ImageHelperFactory` |
| **Abstract Base Classes** | `AbstractRegistrationHandler`, `AbstractRegistrationManager`, `AbstractPropagationStrategy`, `AbstractImageHelper` |
| **Wrapper/Adapter** | `ImageWrapper` (SimpleITK), `MeshWrapper` (VTK) — adds pickling, deepcopy, and metadata |
| **Strategy** | Swap `SequentialPropagationStrategy` vs `StarPropagationStrategy` at runtime |
| **Builder** | `PropagationInputFactory.create()` constructs complex input objects step-by-step |

---

## Code Conventions

### Naming
- **Classes**: `PascalCase` (e.g., `PropagationPipeline`, `ImageWrapper`)
- **Functions/methods**: `snake_case` (e.g., `create_reference_mask`, `extract_timepoint_image`)
- **Private members**: single underscore prefix (e.g., `_data`, `_manager`)
- **Constants**: `SCREAMING_SNAKE_CASE`

### Type Hints
- Use Python 3.10+ union syntax (`X | Y` instead of `Optional[X]` or `Union[X, Y]`)
- Type hints are used extensively throughout — maintain this on new code

### Logging
- Use `logging.getLogger(__name__)` at the top of each module
- Do not use `print()` for pipeline output — use the logger

### Error Handling
- Raise meaningful exceptions with descriptive messages
- GPU operations should handle CUDA OOM gracefully

### Wrappers
- Always use `ImageWrapper` instead of raw `SimpleITK.Image` inside the pipeline
- Always use `MeshWrapper` instead of raw `vtkPolyData` when data crosses process boundaries (required for pickling)

---

## CLI Entry Points

Defined in `pyproject.toml` `[project.scripts]`:

| Command | Module | Purpose |
|---------|--------|---------|
| `segflow4d` | `segflow4d.main:main` | Main propagation pipeline |
| `run-fireants` | `segflow4d.registration.registration_handler.fireants.fireants_registration_handler:run` | Run FireANTs registration directly |
| `create-reference-mask` | `segflow4d.processing.reference_mask_creator:main` | Generate reference mask |
| `create-tp-images` | `segflow4d.utility.image_helper.tp_image_creator:main` | Extract 3D frames from 4D image |
| `resample-to-reference` | `segflow4d.utility.image_helper.resampler:main` | Resample image to reference geometry |
| `mesh-gen` | `segflow4d.processing.segmentation_mesh_generator.multi_label_seg_mesh_generator:main` | Generate surface meshes from segmentations |
| `gpu-mesh-warper` | `segflow4d.utility.mesh_helper.gpu_mesh_warper:main` | Warp meshes with a GPU deformation field |
| `segmentation-validation` | `segflow4d.utility.validation.segmentation_validation:main` | Compute Dice + surface distances |
| `batch-segmentation-validation` | `segflow4d.utility.validation.batch_validation:main` | Batch validation across cases |
| `test-registration-manager` | `segflow4d.registration.tests.test_registration_manager:main` | Registration manager integration test |

---

## Dependencies

### Runtime (from `pyproject.toml`)

| Package | Role |
|---------|------|
| `torch` | GPU computation backbone |
| `SimpleITK` | Medical image I/O and processing |
| `fireants` | Primary registration backend (GPU, CUDA fused ops) |
| `vtk` | 3D surface mesh I/O and processing |
| `pyyaml` | YAML config file parsing |
| `nvidia-ml-py3` | GPU VRAM monitoring |
| `medpy` | Medical image utilities |
| `matplotlib` | Visualization (optional diagnostics) |

### External Projects

- **FireANTs** (`./FireANTs/`): Installed by `scripts/setup_fireants.sh`. Do not delete or modify this directory manually.

---

## Setup and Installation

```bash
# 1. Create and activate conda env with Python 3.10
conda create -n segflow4d python=3.10
conda activate segflow4d

# 2. Install PyTorch (GPU build — match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install FireANTs (requires NVCC)
bash scripts/setup_fireants.sh

# 4. Install the package
pip install -e .
```

For CUDA/PyTorch compatibility details, see README.md.

---

## Running the Pipeline

### Via YAML config

```bash
segflow4d --config example_config.yaml
```

### Via CLI flags

```bash
segflow4d \
  --image4d /path/to/4d_image.nii.gz \
  --seg /path/to/reference_seg.nii.gz \
  --reference-tp 0 \
  --output-dir /path/to/output \
  --registration-backend fireants
```

### Key config parameters (`example_config.yaml`)

- `image4d`: 4D input image path
- `seg`: Reference segmentation path
- `reference_tp`: Index of reference timepoint (0-based)
- `output_dir`: Where to write propagated segmentations and meshes
- `registration_backend`: `fireants` (default) or `greedy`
- `scales`, `iterations`, `loss_type`: FireANTs registration parameters

---

## Testing

Test coverage is currently minimal. The primary validation mechanism is the CLI validation tools:

```bash
# Single case
segmentation-validation --pred /path/pred.nii.gz --gt /path/gt.nii.gz

# Batch
batch-segmentation-validation --config batch_config.yaml
```

The only formal test file is `src/segflow4d/registration/tests/test_mesh_warping.py` (placeholder). When adding new features, add corresponding tests here or in a new file under `tests/`.

---

## Common Gotchas

1. **CUDA context conflicts**: GPU registration jobs run in subprocess workers deliberately. Do not import `torch` or initialize CUDA before forking — it will cause deadlocks.
2. **`MeshWrapper` pickling**: `vtkPolyData` is not natively picklable. Always use `MeshWrapper` when passing mesh data across process boundaries.
3. **`ImageWrapper` deepcopy**: `SimpleITK.Image` has unexpected deepcopy semantics. `ImageWrapper` handles this correctly — use it.
4. **FireANTs path**: The `setup_fireants.sh` script clones into `./FireANTs/`. If FireANTs import fails, re-run the script.
5. **VRAM safety margin**: The GPU manager reserves 2 GB of headroom; if jobs stall waiting for VRAM, reduce registration resolution or batch size.
6. **Python 3.10 required**: The codebase uses `X | Y` union syntax which requires Python 3.10 minimum.

---

## Adding a New Registration Backend

1. Create a new directory under `src/segflow4d/registration/registration_handler/<backend_name>/`
2. Implement a class inheriting from `AbstractRegistrationHandler`
3. Register it in the factory (`registration_manager/factory.py`) via a new string key
4. Expose a CLI entry point in `pyproject.toml` if needed

## Adding a New Propagation Strategy

1. Create a new file under `src/segflow4d/propagation/propagation_strategy/`
2. Inherit from `AbstractPropagationStrategy` and implement `propagate()`
3. Register the strategy key in `PropagationStrategyFactory`
