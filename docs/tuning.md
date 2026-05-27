# Registration Parameter Tuning

`scripts/tune_registration.py` finds good hyperparameters for a registration
backend (`fireants`, `greedy`, or `ants`) on a specific set of image pairs,
using [Optuna](https://optuna.org/)'s TPE Bayesian sampler.

It is intended as the parameter-search step that precedes a fair comparison
between SegFlow4D's full pipeline and direct ref → target registration
(the `direct_star` propagation combo).

---

## How it works

### Objective

For one set of hyperparameters the tuner produces a **composite score**:

```
composite = mean_dice − λ · mean_msd_mm
```

* `mean_dice` — macro-average Dice across all labels and all cases in the
  search subset.
* `mean_msd_mm` — macro-average mean (symmetric) surface distance in
  millimetres, also averaged across labels and cases. Computed by MedPy's
  ASSD.
* `λ` — the `--lambda-surface` weight. The default `0.05` makes 1 mm of MSD
  worth ~0.05 Dice — adjust based on the typical MSD magnitude of your
  data.

Higher composite is better; Optuna maximises it.

### What one trial does

1. The TPE sampler proposes a value for every parameter in the backend's
   search space (see [search spaces](#search-spaces) below).
2. The values are packed into a `*RegistrationOptions` instance and wrapped
   in a minimal `PropagationOptions`.
3. For each case in the dataset:
   1. The moving image, fixed image, and moving segmentation are loaded.
   2. The trial submits a single
      `RUN_REGISTRATION_AND_RESLICE` job through the singleton
      `RegistrationManager` — **the SegFlow4D propagation pipeline is not
      involved**. This is a deliberate baseline: the tuner answers
      "what params make direct ref → target registration best?", which is
      the right operating point to benchmark `direct_star` against.
   3. The resliced segmentation is compared to the ground-truth
      segmentation on the fixed side. Per-case Dice and MSD are recorded.
   4. The running composite (`mean over cases so far`) is reported to
      Optuna as an intermediate value at `step=i`.
4. After every case the trial calls `trial.should_prune()`. The
   `MedianPruner` kills the trial early if the running composite is
   already worse than the median of completed trials at the same step.
   A trial with a bad first case typically dies in one case-time instead
   of `N`.
5. The final composite is returned. Trial user-attrs also record
   `mean_dice`, `mean_msd_mm`, per-case lists, and the resolved backend
   options dict.

### Search spaces

Only **core** parameters are tuned (loss/metric and smoothing sigmas) —
the knobs that move Dice meaningfully on typical 4D medical data. Lower-
tier knobs like learning rates, jitter, shrink factors, threads, and seeds
are left at backend defaults.

Schedule lists and scales are **pinned** (not tuned) so trial cost is
predictable. They follow the prior log-uniform sweep at
`iter_multiplier=50`:

* FireANTs: `scales=[4, 2, 1]`, `deformable_iterations = affine_iterations
  = [200, 100, 50]`.
* Greedy: `deformable_iterations = affine_iterations = [200, 100, 50]`.
* ANTs: `reg_iterations = (50, 25, 0)` (the finest level terminates at 0
  by ANTs convention).

If you need a different cost/quality trade-off, edit the
`_FIREANTS_ITERATIONS`, `_GREEDY_ITERATIONS`, or `_ANTS_ITERATIONS`
constants near the top of `sample_*_params` in
`scripts/tune_registration.py`.

| Backend | Parameter | Range / choices |
|---|---|---|
| **fireants** | `loss_type` | {`mse`, `cc`} |
|              | `cc_kernel_size` | {3, 5, 7} (conditional on `loss_type=cc`) |
|              | `smooth_grad_sigma_mm` | [0.5, 6.0], log-uniform |
|              | `smooth_warp_sigma_mm` | [0.25, 3.0], log-uniform |
| **greedy**   | `metric` | {`NCC`, `SSD`, `NMI`} |
|              | `metric_radius` | {1, 2, 3, 4} (conditional on `metric=NCC`) |
|              | `smooth_sigma_pre_mm` | [0.5, 4.0], log-uniform |
|              | `smooth_sigma_post_mm` | [0.1, 2.0], log-uniform |
| **ants**     | `transform_type` | {`SyN`, `SyNRA`, `SyNOnly`, `SyNCC`} |
|              | `metric` | {`CC`, `MI`, `mattes`, `GC`} |
|              | `grad_step` | [0.05, 0.5], log-uniform |
|              | `flow_sigma` | [1.0, 6.0] |

### Sampler and pruner

* **Sampler:** `optuna.samplers.TPESampler(seed=42)` — Tree-structured
  Parzen Estimator. Sample-efficient on mixed continuous/categorical
  spaces with conditional parameters (e.g. `cc_kernel_size` only sampled
  when `loss_type=cc`).
* **Pruner:** `optuna.pruners.MedianPruner(n_warmup_steps=1)` — needs at
  least one case completed before it considers pruning.

### Persistence

The study is stored as a SQLite database at `<output-dir>/study.db`.
Re-running with `--resume` reloads the study and continues adding
trials, so you can spread a search across multiple sessions or grow it
once you have an initial answer.

---

## How to run

### 1. Install

```bash
pip install -e ".[tune]"
```

This installs Optuna alongside SegFlow4D. The registration backend
itself must already work in your environment — the tuner does not
install backend dependencies (see `[greedy]`, `[ants]`, or the FireANTs
install steps in [README.md](../README.md)).

### 2. Prepare a cases YAML

The tuner needs 3–5 representative `(moving, fixed)` image pairs with a
ground-truth segmentation on the fixed side. Use cases that span the
range of motion / deformation in your dataset — tuning on a single pair
overfits, tuning on the whole dataset is wastefully slow.

```yaml
# my_cases.yaml
cases:
  - name: subj01_t00_to_t05
    moving_image: /data/subj01/image_tp00.nii.gz
    fixed_image:  /data/subj01/image_tp05.nii.gz
    moving_seg:   /data/subj01/seg_tp00.nii.gz
    fixed_seg_gt: /data/subj01/seg_tp05_gt.nii.gz
  - name: subj01_t00_to_t10
    ...
```

The four image paths per case are required; `name` defaults to
`case_NNN` if omitted. See `scripts/example_tune_config.yaml` for the
full template.

### 3. Launch the search

```bash
python scripts/tune_registration.py \
    --backend fireants \
    --config my_cases.yaml \
    --output-dir ./tuner_runs/fireants \
    --n-trials 50
```

A typical first pass: 30–50 trials per backend. With 3 cases per trial
and pruning enabled, this is usually well under a day on one GPU for
FireANTs, less for the CPU backends (Greedy / ANTs).

### Common flags

| Flag | Default | Notes |
|---|---|---|
| `--backend` | required | `fireants` \| `greedy` \| `ants` |
| `--config` | required | YAML with the `cases` list |
| `--output-dir` | required | Study DB, best params, and trial log land here |
| `--n-trials` | `50` | Total trials to attempt this run |
| `--lambda-surface` | `0.05` | Weight on MSD in the composite |
| `--seed` | `42` | TPE sampler seed |
| `--study-name` | `tune_<backend>` | Custom name; lets multiple studies share one DB dir |
| `--resume` | off | Reload an existing study from `--output-dir` and keep going |
| `--required-vram-gb` | `10` | GPU VRAM safety margin (FireANTs only) |
| `--log-level` | `INFO` |  |

### 4. Inspect outputs

After the run the output directory contains:

```
tuner_runs/fireants/
├── study.db            # Optuna SQLite study (open with optuna-dashboard)
├── trials.csv          # One row per trial: state, composite, dice, msd, params
├── best_params.yaml    # Best trial's params, value, and user-attrs (Dice/MSD)
└── best_params.json    # Same as YAML, JSON-encoded
```

`best_params.yaml` is the artefact you'll use to launch the comparison
run — the `user_attrs.backend_options` dict drops directly into the
`registration_backend_options` block of a SegFlow4D YAML config.

### 5. Apply the best params to a SegFlow4D run

Take `user_attrs.backend_options` from `best_params.yaml` and paste it
under `registration_backend_options:` in your propagation config. For
example, for the comparison run:

```yaml
registration_backend: fireants
registration_backend_options:
  # paste from best_params.yaml -> user_attrs.backend_options
  loss_type: cc
  cc_kernel_size: 5
  smooth_grad_sigma_mm: 2.3
  smooth_warp_sigma_mm: 0.8
  scales: [4, 2, 1]
  affine_iterations: [200, 100, 50]
  deformable_iterations: [200, 100, 50]

propagation_strategy_combo: direct_star      # or sequential_star / sasd_star
```

Re-run the same config with each combo (`direct_star`,
`sequential_star`, `sasd_star`) and compare the resulting Dice/MSD on a
held-out subset. The tuner-selected params are now fair to both sides
of the comparison.

### Resuming or extending a study

To add more trials to an existing study:

```bash
python scripts/tune_registration.py \
    --backend fireants \
    --config my_cases.yaml \
    --output-dir ./tuner_runs/fireants \
    --n-trials 30 \
    --resume
```

The 30 new trials build on the TPE model fitted to the previous trials.

---

## Tips

* **Pick the held-out test set first.** The tuner's case list should be
  disjoint from the cases you'll report results on, or you're tuning on
  the test set.
* **Cap MSD blow-ups.** When a trial produces a label that is empty on
  one side (e.g. registration collapses), MSD becomes `inf`. The tuner
  clamps it to `50.0 mm` so Optuna can still rank that trial as bad
  rather than dropping it.
* **Conditional parameters waste trials if you over-sample.** TPE
  handles them, but if `loss_type=mse` is consistently winning you can
  prune the search space by hard-coding the categorical and re-running
  — fewer dimensions = faster convergence.
* **Inspect with the Optuna dashboard.** `pip install optuna-dashboard`
  and run `optuna-dashboard sqlite:///tuner_runs/fireants/study.db` to
  see param importances, parallel-coordinate plots, and the live
  pruner state.
