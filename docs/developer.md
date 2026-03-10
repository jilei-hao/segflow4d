# Developer Guide

## Running Tests

SegFlow4D has three installation profiles that determine which tests can run.
Choose the section that matches your environment.

---

### Installation Profiles

| Profile | Command | What runs |
|---------|---------|-----------|
| **Minimal** (no GPU, no greedy) | `pip install -e ".[test]"` | Unit + integration tests only |
| **Greedy** (CPU-only backend) | `pip install -e ".[test,greedy]"` | + Greedy handler & e2e tests |
| **Full** (GPU + FireANTs) | See [README.md](../README.md) setup | All tests including GPU e2e |

---

### Minimal install (no GPU, no greedy)

Install test dependencies only:

```bash
pip install -e ".[test]"
```

Run all tests that do not require GPU or picsl-greedy:

```bash
pytest tests/ -v
```

This executes:
- All unit tests (`tests/unit/`)
- All integration tests (`tests/integration/`)
- Skips any test marked `gpu` or `greedy` (no error, just skipped)

---

### Greedy install (CPU-only backend)

Install the greedy extra:

```bash
pip install -e ".[test,greedy]"
# or equivalently:
pip install -e ".[test]" picsl-greedy
```

Run all CPU tests including greedy:

```bash
# Run everything that doesn't need a GPU
pytest tests/ -v

# Run only greedy-specific tests
pytest tests/ -m greedy -v

# Run the greedy unit tests
pytest tests/unit/test_greedy_handler.py -v

# Run the greedy e2e tests
pytest tests/e2e/test_pipeline_synthetic.py -m greedy -v
```

---

### Full install (GPU + FireANTs)

Follow the full setup in [README.md](../README.md) to install PyTorch with CUDA
and FireANTs.

Run all GPU tests (FireANTs backend):

```bash
# Run only GPU tests
pytest tests/ -m gpu -v

# Run the FireANTs e2e synthetic pipeline tests
pytest tests/e2e/test_pipeline_synthetic.py -m gpu -v

# Run the real-data e2e tests (also requires real data files on disk)
pytest tests/e2e/test_pipeline_real.py -m "gpu and requires_real_data" -v
```

---

### Running the full test suite

With both GPU and greedy installed:

```bash
pytest tests/ -v
```

To see which tests were skipped and why:

```bash
pytest tests/ -v --tb=short -r s
```

---

### Pytest markers

| Marker | Meaning | Skip condition |
|--------|---------|----------------|
| `gpu` | Requires a CUDA-capable GPU | No CUDA device detected |
| `greedy` | Requires `picsl-greedy` package | `picsl_greedy` not importable |
| `slow` | Long-running test | Pass `-m "not slow"` to skip |
| `requires_real_data` | Needs real patient data files | Missing data files on disk |

Skip a marker explicitly:

```bash
pytest tests/ -m "not slow" -v
pytest tests/ -m "not gpu and not greedy" -v
```

---

### Test layout

```
tests/
├── conftest.py               # Shared CPU-only fixtures (synthetic images, meshes)
├── unit/
│   ├── test_async_writer.py
│   ├── test_greedy_handler.py  # GreedyRegistrationOptions + handler (greedy marker)
│   ├── test_image_processing.py
│   ├── test_image_wrapper.py
│   ├── test_mesh_warper.py
│   ├── test_tp_data.py
│   └── test_validation_metrics.py
├── integration/
│   ├── test_propagation_input_factory.py
│   ├── test_propagation_strategies.py
│   └── test_tp_partition.py
└── e2e/
    ├── test_pipeline_synthetic.py  # FireANTs (gpu) + Greedy (greedy) backends
    └── test_pipeline_real.py       # Real patient data (gpu + requires_real_data)
```
