"""Bayesian hyperparameter search for SegFlow4D registration backends.

Tunes a single registration backend (fireants / greedy / ants) on a fixed
set of (moving, fixed, moving_seg, fixed_seg_gt) cases using Optuna's TPE
sampler.  Objective is a composite of Dice and mean surface distance:

    composite = mean_dice - lambda_surface * mean_msd_mm        (maximize)

Only the *high-tier* parameters are tuned (see CLAUDE notes / discussion
in the design doc).  Schedule lists are tuned via a single integer
``iter_multiplier`` applied to a fixed schedule shape; FireANTs
``scales`` is fixed at ``[4, 2, 1]``.

Usage
-----
    pip install segflow4d[tune]
    python scripts/tune_registration.py \\
        --backend fireants \\
        --config scripts/example_tune_config.yaml \\
        --output-dir ./tuner_runs/fireants \\
        --n-trials 50

Each trial directly invokes ``RegistrationHandler.run_registration_and_reslice``
through the singleton ``RegistrationManager``; no SegFlow4D propagation
machinery is involved (the tuner answers "what params work best for direct
ref->target registration?", which is the right baseline for comparing the
SegFlow4D pipeline against direct registration).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
import yaml

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.registration_methods import REGISTRATION_METHODS
from segflow4d.registration.registration_manager import RegistrationManager
from segflow4d.utility.validation.segmentation_validation import (
    evaluate_segmentation,
    load_segmentation,
)

logger = logging.getLogger("tune_registration")


# ---------------------------------------------------------------------------
# Case dataset
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """One (moving, fixed) pair with a ground-truth segmentation on the fixed side."""

    name: str
    moving_image: str
    fixed_image: str
    moving_seg: str
    fixed_seg_gt: str


def load_cases(config_path: str) -> list[Case]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_cases = cfg.get("cases")
    if not raw_cases:
        raise ValueError(f"No 'cases' list in {config_path}")

    cases: list[Case] = []
    for i, c in enumerate(raw_cases):
        try:
            cases.append(
                Case(
                    name=c.get("name", f"case_{i:03d}"),
                    moving_image=c["moving_image"],
                    fixed_image=c["fixed_image"],
                    moving_seg=c["moving_seg"],
                    fixed_seg_gt=c["fixed_seg_gt"],
                )
            )
        except KeyError as e:
            raise ValueError(f"Case {i} missing required key: {e}") from e

    for case in cases:
        for label, path in [
            ("moving_image", case.moving_image),
            ("fixed_image", case.fixed_image),
            ("moving_seg", case.moving_seg),
            ("fixed_seg_gt", case.fixed_seg_gt),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{case.name}: {label} not found: {path}")

    return cases


# ---------------------------------------------------------------------------
# Per-backend search spaces
# ---------------------------------------------------------------------------
def _fireants_schedule(iter_multiplier: int) -> list[int]:
    # Fixed shape [4x, 2x, x] over scales=[4, 2, 1] (coarse->fine).
    return [iter_multiplier * 4, iter_multiplier * 2, iter_multiplier]


def _greedy_schedule(iter_multiplier: int) -> list[int]:
    return [iter_multiplier * 4, iter_multiplier * 2, iter_multiplier]


def _ants_schedule(iter_multiplier: int) -> tuple[int, ...]:
    # ANTs schedules conventionally end at 0 at the finest level.
    return (iter_multiplier, iter_multiplier // 2, 0)


def sample_fireants_params(trial) -> dict[str, Any]:
    loss_type = trial.suggest_categorical("loss_type", ["mse", "cc"])
    params: dict[str, Any] = {
        "loss_type": loss_type,
        "smooth_grad_sigma_mm": trial.suggest_float("smooth_grad_sigma_mm", 0.5, 6.0, log=True),
        "smooth_warp_sigma_mm": trial.suggest_float("smooth_warp_sigma_mm", 0.25, 3.0, log=True),
        "scales": [4, 2, 1],
    }
    iter_mult = trial.suggest_int("iter_multiplier", 25, 200, log=True)
    params["deformable_iterations"] = _fireants_schedule(iter_mult)
    # Affine schedule is not tuned; mirror the deformable shape so list lengths
    # match scales (FireantsRegistrationOptions enforces this).
    params["affine_iterations"] = _fireants_schedule(iter_mult)
    if loss_type == "cc":
        params["cc_kernel_size"] = trial.suggest_categorical("cc_kernel_size", [3, 5, 7])
    return params


def sample_greedy_params(trial) -> dict[str, Any]:
    metric = trial.suggest_categorical("metric", ["NCC", "SSD", "NMI"])
    params: dict[str, Any] = {
        "metric": metric,
        "smooth_sigma_pre_mm": trial.suggest_float("smooth_sigma_pre_mm", 0.5, 4.0, log=True),
        "smooth_sigma_post_mm": trial.suggest_float("smooth_sigma_post_mm", 0.1, 2.0, log=True),
    }
    iter_mult = trial.suggest_int("iter_multiplier", 25, 200, log=True)
    params["deformable_iterations"] = _greedy_schedule(iter_mult)
    params["affine_iterations"] = _greedy_schedule(iter_mult)
    if metric == "NCC":
        r = trial.suggest_categorical("metric_radius", [1, 2, 3, 4])
        params["metric_radius"] = [r, r, r]
    return params


def sample_ants_params(trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "transform_type": trial.suggest_categorical(
            "transform_type", ["SyN", "SyNRA", "SyNOnly", "SyNCC"]
        ),
        "metric": trial.suggest_categorical("metric", ["CC", "MI", "mattes", "GC"]),
        "grad_step": trial.suggest_float("grad_step", 0.05, 0.5, log=True),
        "flow_sigma": trial.suggest_float("flow_sigma", 1.0, 6.0),
    }
    iter_mult = trial.suggest_int("iter_multiplier", 10, 80, log=True)
    params["reg_iterations"] = _ants_schedule(iter_mult)
    return params


SAMPLERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "fireants": sample_fireants_params,
    "greedy": sample_greedy_params,
    "ants": sample_ants_params,
}


# ---------------------------------------------------------------------------
# Per-trial evaluation
# ---------------------------------------------------------------------------
def _make_propagation_options(backend: str, backend_options: dict[str, Any]) -> PropagationOptions:
    """Build a minimal PropagationOptions for a single direct registration call.

    The fields below (lowres_scale_factor, dilation_radius, propagation_strategy_combo)
    are not consumed by the registration handler itself, but PropagationOptions
    requires them.  We use sensible inert defaults.
    """
    return PropagationOptions(
        lowres_scale_factor=1.0,
        dilation_radius=0,
        registration_backend=backend,
        registration_backend_options=backend_options,
        write_result_to_disk=False,
        output_directory="",
        debug=False,
        debug_output_directory="",
        minimum_required_vram_gb=0,
        propagation_strategy_combo="direct_star",
    )


def _resliced_to_array_and_spacing(resliced: ImageWrapper) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = resliced.get_data()
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    sx, sy, sz = img.GetSpacing()
    # numpy gives (Z, Y, X); spacing in matching order.
    return arr, (float(sz), float(sy), float(sx))


def evaluate_case(
    case: Case,
    registration_manager: RegistrationManager,
    propagation_options: PropagationOptions,
) -> tuple[float, float]:
    """Run direct ref→target registration for one case, return (dice, msd_mm)."""
    moving_img = ImageWrapper(sitk.ReadImage(case.moving_image))
    fixed_img = ImageWrapper(sitk.ReadImage(case.fixed_image))
    moving_seg = ImageWrapper(sitk.ReadImage(case.moving_seg))

    future = registration_manager.submit(
        REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE,
        img_fixed=fixed_img,
        img_moving=moving_img,
        img_to_reslice=moving_seg,
        mesh_to_reslice=None,
        options=propagation_options,
        mask_fixed=None,
        mask_moving=None,
    )
    result = future.result()

    if result.resliced_image is None:
        raise RuntimeError(f"{case.name}: registration produced no resliced segmentation")

    pred_arr, pred_spacing = _resliced_to_array_and_spacing(result.resliced_image)
    gt_arr, gt_spacing = load_segmentation(case.fixed_seg_gt)

    if pred_arr.shape != gt_arr.shape:
        raise RuntimeError(
            f"{case.name}: prediction shape {pred_arr.shape} != ground truth shape {gt_arr.shape}"
        )

    val = evaluate_segmentation(target=pred_arr, ref=gt_arr, spacing=pred_spacing)
    dice = val.macro_avg.dice
    msd = val.macro_avg.msd
    if not np.isfinite(msd):
        # One side empty — penalize hard but keep finite so Optuna can compare.
        msd = 50.0
    return float(dice), float(msd)


def make_objective(
    cases: list[Case],
    backend: str,
    registration_manager: RegistrationManager,
    lambda_surface: float,
):
    sampler = SAMPLERS[backend]

    def objective(trial) -> float:
        params = sampler(trial)
        options = _make_propagation_options(backend, params)

        per_case_dice: list[float] = []
        per_case_msd: list[float] = []

        for i, case in enumerate(cases):
            try:
                dice, msd = evaluate_case(case, registration_manager, options)
            except Exception as exc:
                logger.exception(f"Trial {trial.number} case {case.name} failed: {exc}")
                # Mark the trial as failed so the sampler doesn't pick this region again.
                raise

            per_case_dice.append(dice)
            per_case_msd.append(msd)
            logger.info(
                f"Trial {trial.number} case {case.name}: dice={dice:.4f} msd={msd:.3f} mm"
            )

            # Report intermediate so MedianPruner can kill weak trials early.
            running_composite = float(np.mean(per_case_dice) - lambda_surface * np.mean(per_case_msd))
            trial.report(running_composite, step=i)
            if trial.should_prune():
                import optuna  # imported lazily so pyflakes is happy
                raise optuna.TrialPruned()

        mean_dice = float(np.mean(per_case_dice))
        mean_msd = float(np.mean(per_case_msd))
        composite = mean_dice - lambda_surface * mean_msd

        trial.set_user_attr("mean_dice", mean_dice)
        trial.set_user_attr("mean_msd_mm", mean_msd)
        trial.set_user_attr("per_case_dice", per_case_dice)
        trial.set_user_attr("per_case_msd_mm", per_case_msd)
        trial.set_user_attr("backend_options", params)

        logger.info(
            f"Trial {trial.number} done: mean_dice={mean_dice:.4f} "
            f"mean_msd={mean_msd:.3f} mm composite={composite:.4f}"
        )
        return composite

    return objective


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s - %(name)s - %(levelname)s] - %(message)s",
        force=True,
    )


def _save_best(study, output_dir: Path, backend: str) -> None:
    best = study.best_trial
    payload = {
        "backend": backend,
        "value": best.value,
        "params": best.params,
        "user_attrs": dict(best.user_attrs),
    }
    (output_dir / "best_params.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (output_dir / "best_params.json").write_text(json.dumps(payload, indent=2, default=str))


def _dump_trials_csv(study, output_dir: Path) -> None:
    import csv

    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "mean_dice": t.user_attrs.get("mean_dice"),
            "mean_msd_mm": t.user_attrs.get("mean_msd_mm"),
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        rows.append(row)

    if not rows:
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=["fireants", "greedy", "ants"],
                        help="Registration backend to tune.")
    parser.add_argument("--config", required=True,
                        help="YAML with a 'cases' list of (moving_image, fixed_image, moving_seg, fixed_seg_gt).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for study DB, best params, and trial log.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--lambda-surface", type=float, default=0.05,
                        help="Weight on mean surface distance (mm) in composite objective. "
                             "composite = mean_dice - lambda * mean_msd. Default 0.05 makes "
                             "1 mm of MSD worth ~0.05 Dice.")
    parser.add_argument("--seed", type=int, default=42, help="TPE sampler seed.")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (default: tune_<backend>).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study from --output-dir.")
    parser.add_argument("--required-vram-gb", type=int, default=10,
                        help="Per-job VRAM safety margin (GPU backends only).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.config)
    logger.info(f"Loaded {len(cases)} cases from {args.config}")

    # Import optuna lazily so users without it get a clean error.
    try:
        import optuna
    except ImportError:
        print("optuna is required.  pip install segflow4d[tune]", file=sys.stderr)
        return 2

    study_name = args.study_name or f"tune_{args.backend}"
    storage = f"sqlite:///{output_dir / 'study.db'}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=args.resume,
    )

    # Initialize the singleton manager with the chosen backend.
    registration_manager = RegistrationManager(
        registration_backend=args.backend,
        required_vram_mb=args.required_vram_gb * 1024,
    )

    objective = make_objective(cases, args.backend, registration_manager, args.lambda_surface)

    try:
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    finally:
        try:
            registration_manager.shutdown(wait=True)
        except Exception:
            pass

        if len(study.trials) > 0:
            _dump_trials_csv(study, output_dir)
            try:
                _save_best(study, output_dir, args.backend)
                logger.info(f"Best composite={study.best_value:.4f} params={study.best_params}")
                logger.info(f"Best params written to {output_dir / 'best_params.yaml'}")
            except ValueError:
                logger.warning("No completed trials; skipping best-params dump.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
