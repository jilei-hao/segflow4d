# ANTs Backend — Manual Test Plan

Temporary checklist for validating the classic ANTs (`antspyx`) integration
added on `feature/ants-handler`. Delete this file once the backend is
shipped.

All commands assume the `segflow4d` conda env:

```bash
conda activate segflow4d
```

---

## 0. Pre-flight

```bash
# Install the new optional extra
pip install -e ".[ants]"

# Verify antspyx is importable
python -c "import ants; print('antspyx', ants.__version__)"
```

**Expected:** version prints (any 0.4+ is fine).
**Fails if:** ImportError → confirm pip picked up the new `[ants]` extra
from the local checkout, not from PyPI.

---

## 1. Automated tests (gated suite)

```bash
# Full ANTs test file — unit + integration
pytest tests/unit/test_ants_handler.py -v

# Just the integration tests (the ones that need antspyx)
pytest tests/unit/test_ants_handler.py -m ants -v

# Regression: full unit suite, including greedy
pytest tests/unit/ -v
```

**Expected:**
- `tests/unit/test_ants_handler.py`: 32 passed, 0 skipped.
- `tests/unit/`: 97 passed, 1 skipped (pre-existing mesh_warper skip).
- Identity-registration Dice ≥ 0.75 on the synthetic sphere test.

**Watch for:**
- `test_sitk_ants_roundtrip_preserves_metadata` failing → axis transpose
  in `_sitk_to_ants` / `_ants_to_sitk` is wrong; geometry won't survive
  the round-trip.
- `test_affine_with_syn_returns_matrix` failing → the
  `_extract_affine_path` heuristic missed the `.mat` file in
  `reg['fwdtransforms']`.

---

## 2. CLI smoke test (synthetic / tiny real case)

Pick the smallest real 4D dataset you have on disk. Then:

```bash
segflow4d \
  --image4d /path/to/tiny_4d.nii.gz \
  --seg-ref /path/to/ref_seg.nii.gz \
  --tp-ref 0 \
  --tp-targets 1 \
  --output /tmp/segflow_ants_smoke \
  --registration-backend ants \
  --ants-transform-type SyN \
  --ants-aff-iterations 100 50 \
  --ants-reg-iterations 20 10 0 \
  --ants-threads 4 \
  --log-level INFO
```

**Expected:**
- "Creating CPU registration manager ('ants' is a CPU-only backend)" log.
- "Initialized AntsRegistrationHandler" log per worker.
- Output dir contains a propagated segmentation per target timepoint.
- No CUDA usage (check `nvidia-smi` — should stay flat).

**Common failure modes:**
- "antspyx is required..." → `[ants]` extra didn't install. Re-run pre-flight.
- KeyError on `transform_type` inside `AntsRegistrationOptions(...)` →
  a stray non-ANTs option (e.g. `metric` from a leftover greedy flag)
  was passed; remember `--metric NCC` is for greedy, not ANTs.

---

## 3. YAML config path

Create `/tmp/ants_config.yaml`:

```yaml
image4d: /path/to/tiny_4d.nii.gz
output: /tmp/segflow_ants_yaml
registration_backend: ants
lowres_factor: 0.5
dilation_radius: 2
propagation_strategy_combo: sequential_star

registration_backend_options:
  transform_type: SyN
  metric: mattes
  aff_iterations: [100, 50]
  reg_iterations: [20, 10, 0]
  grad_step: 0.1
  flow_sigma: 3.0
  label_interpolation: genericLabel
  threads: 4

tp_input_groups:
  - tp_ref: 0
    tp_targets: [1]
    seg_ref: /path/to/ref_seg.nii.gz
```

```bash
segflow4d --config /tmp/ants_config.yaml
```

**Expected:** same as CLI smoke test, plus verify that the YAML list
schedules (`[100, 50]`, `[20, 10, 0]`) are coerced to tuples in the
options dataclass without error.

---

## 4. Mesh-warping direction (correctness check)

This is the single most-likely correctness bug. ANTs writes both
`*Warp.nii.gz` (forward) and `*InverseWarp.nii.gz`; we feed the forward
one to `cpu_mesh_warper.warp_mesh_vertices_cpu`. Verify on a case
where you can eyeball the result.

```bash
segflow4d \
  --image4d /path/to/tiny_4d.nii.gz \
  --seg-ref /path/to/ref_seg.nii.gz \
  --tp-ref 0 \
  --tp-targets 1 \
  --output /tmp/segflow_ants_mesh \
  --registration-backend ants \
  --ants-transform-type SyN \
  --add-mesh ref:/path/to/ref_mesh.vtk
```

**Open the output mesh in ParaView / 3D Slicer and compare against the
same case run with Greedy:**

```bash
segflow4d \
  --image4d /path/to/tiny_4d.nii.gz \
  --seg-ref /path/to/ref_seg.nii.gz \
  --tp-ref 0 --tp-targets 1 \
  --output /tmp/segflow_greedy_mesh \
  --registration-backend greedy \
  --add-mesh ref:/path/to/ref_mesh.vtk
```

**Expected:** Both meshes deform in the same direction. Shapes may
differ in detail (different registration algorithms) but they should not
be mirror images or move in opposite directions.

**If the ANTs mesh moves the wrong way:** flip the forward/inverse
selection in
[ants_registration_handler.py](src/segflow4d/registration/registration_handler/ants/ants_registration_handler.py)
`_extract_warp_path` — change the condition to match `InverseWarp` files
instead.

---

## 5. Mask-aware registration

```bash
# Add --debug to dump the mask actually used
segflow4d \
  --image4d /path/to/tiny_4d.nii.gz \
  --seg-ref /path/to/ref_seg.nii.gz \
  --tp-ref 0 --tp-targets 1 \
  --output /tmp/segflow_ants_mask \
  --registration-backend ants \
  --dilation-radius 3 \
  --debug --debug-dir /tmp/segflow_ants_mask_debug
```

**Expected:** Registration runs without error. ANTs receives the mask
via `ants.registration(..., mask=...)` — confirm by adding `--ants-*`
flags don't blow up.

---

## 6. Threading behavior

```bash
# Without --ants-threads → ITK uses its default (all cores).
# Watch CPU usage in Activity Monitor / htop while running.
segflow4d ... --registration-backend ants

# With --ants-threads 2 and 4 workers running in parallel,
# CPU usage per worker should be capped near 200%.
segflow4d ... --registration-backend ants --ants-threads 2
```

**Expected:** With `--ants-threads N`, each subprocess worker stays
near `N * 100%` CPU. Without it, the first worker grabs all cores and
others starve until it finishes.

**Note:** `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` is set per-call inside
the handler and restored in a `finally`. Verify with a quick
`env | grep ITK` from a worker if behavior is surprising.

---

## 7. Regression sanity — Greedy & FireANTs unaffected

```bash
# Greedy (should be unchanged)
segflow4d ... --registration-backend greedy

# FireANTs on GPU box (should still pick GPU manager)
segflow4d ... --registration-backend fireants
```

**Expected:**
- Greedy: "'greedy' is a CPU-only backend" log; identical behavior to
  before this branch.
- FireANTs: "Creating GPU registration manager for backend 'fireants'"
  log; CUDA workers spin up as before.

---

## 8. Edge cases worth poking at least once

- [ ] `--ants-transform-type Affine` (no deformable stage) → `warp_image`
      should be `None` in the resulting `TPData`; mesh warping is skipped
      with a warning instead of crashing.
- [ ] `--ants-transform-type Rigid` (also no deformable) → same as above.
- [ ] `--ants-transform-type SyNRA` (rigid+affine+SyN) → `affine_matrix`
      populated.
- [ ] Very few iterations `--ants-aff-iterations 10 --ants-reg-iterations 5`
      → completes quickly, no crashes (quality will be poor; that's fine).
- [ ] Run twice with `--ants-random-seed 1` → byte-identical outputs.

---

## Sign-off checklist

- [ ] Section 1: all automated tests green.
- [ ] Section 2: CLI smoke test completes end-to-end.
- [ ] Section 3: YAML config path works.
- [ ] Section 4: mesh deforms in the correct direction (vs. Greedy reference).
- [ ] Section 5: mask path runs without error.
- [ ] Section 6: thread cap honored.
- [ ] Section 7: no regression in Greedy or FireANTs.
- [ ] Section 8: at least the Affine/Rigid edge cases poked.

Once all checked, this file can be deleted.
