"""
Segmentation validation (Dice, MSD/ASSD, HD95) for 3D images using MedPy.

Works for:
- 3D masks (Z,Y,X) or (Y,X,Z) etc.
- Multi-label segmentations (compute per-label + macro average)

Install:
  pip install medpy nibabel numpy

Notes:
- Dice is computed on binary masks per label.
- MSD is computed as ASSD (average symmetric surface distance) from medpy.metric.binary.assd
  (commonly reported as MSD/ASSD in papers).
- HD95 is medpy.metric.binary.hd95
- Provide voxel spacing in mm as (sz, sy, sx) matching the spatial axes order of your arrays.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from utility.image_helper.image_helper_factory import create_image_helper
from medpy.metric.binary import dc, assd, hd95


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class Metrics:
    dice: float
    msd: float   # using MedPy ASSD
    hd95: float


@dataclass
class ValidationResult:
    per_label: Dict[int, Metrics]
    macro_avg: Metrics
    # optional: label sizes for debugging/weighting
    ref_voxels: Dict[int, int]
    target_voxels: Dict[int, int]


# ----------------------------
# Loading helpers
# ----------------------------
def load_segmentation(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a segmentation image using SITK and return:
      - data as numpy array (integer labels) in (Z, Y, X) order
      - voxel spacing (sz, sy, sx) in mm

    SITK uses (X, Y, Z) image order internally, but we convert to numpy array
    which gives (Z, Y, X) order for compatibility with MedPy metrics.
    """
    ih = create_image_helper()
    img_wrapper = ih.read_image(path)  # Returns ImageWrapper
    img = img_wrapper.get_data()  # Get underlying sitk.Image
    
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    
    # Convert SITK image to numpy array (automatically gives Z, Y, X order)
    data = sitk.GetArrayFromImage(img).astype(np.int32)
    
    # Get spacing from ImageWrapper (returns as tuple in X, Y, Z order)
    spacing_xyz = img_wrapper.get_spacing()
    if spacing_xyz is None or len(spacing_xyz) < 3:
        raise ValueError(f"Expected at least 3D image. Got spacing={spacing_xyz}")
    
    # Convert from (sx, sy, sz) to (sz, sy, sx) to match numpy array order (Z, Y, X)
    spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return data, spacing


# ----------------------------
# Metric computation
# ----------------------------
def _safe_binary_metrics(
    target_bin: np.ndarray,
    ref_bin: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Metrics:
    """
    Compute Dice, MSD(ASSD), HD95 for one binary label.

    Edge cases:
      - If both empty => perfect (Dice=1, distances=0)
      - If only one empty => Dice=0, distances=inf (or large). We return inf.
    """
    target_any = bool(np.any(target_bin))
    ref_any = bool(np.any(ref_bin))

    if (not target_any) and (not ref_any):
        return Metrics(dice=1.0, msd=0.0, hd95=0.0)
    if target_any != ref_any:
        return Metrics(dice=0.0, msd=float("inf"), hd95=float("inf"))

    # MedPy expects voxelspacing aligned with array axes order.
    # Here we assume target_bin and ref_bin are in (X,Y,Z) order w.r.t spacing_xyz.
    # If your arrays are (Z,Y,X), pass spacing accordingly (see below).
    dice_val = float(dc(target_bin, ref_bin))
    msd_val = float(assd(target_bin, ref_bin, voxelspacing=spacing_xyz))
    hd95_val = float(hd95(target_bin, ref_bin, voxelspacing=spacing_xyz))
    return Metrics(dice=dice_val, msd=msd_val, hd95=hd95_val)


def evaluate_segmentation(
    target: np.ndarray,
    ref: np.ndarray,
    *,
    spacing: Union[Tuple[float, float, float], Sequence[float]] = (1.0, 1.0, 1.0),
    labels: Optional[Sequence[int]] = None,
    background_label: int = 0,
) -> ValidationResult:
    """
    Evaluate segmentation metrics for two 3D images.

    Parameters
    ----------
    target, ref:
      Integer-labeled 3D arrays. Must have same shape.

    spacing:
      Voxel spacing for spatial axes in the SAME ORDER as the spatial axes in the arrays.
      Example:
        - If arrays are (Z,Y,X): spacing=(sz,sy,sx)
      If you loaded with SITK and converted to numpy, data is in (Z,Y,X) order.

    labels:
      Which labels to evaluate. If None, uses union of labels in target and ref (excluding background_label).

    background_label:
      Label value to exclude from evaluation (typically 0 for background).

    Returns
    -------
    ValidationResult with per-label metrics and macro average
    """
    target = np.asarray(target)
    ref = np.asarray(ref)

    if target.shape != ref.shape:
        raise ValueError(f"target and ref shapes differ: {target.shape} vs {ref.shape}")

    if target.ndim != 3:
        raise ValueError(f"Expected 3D arrays. Got {target.ndim}D")

    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3:
        raise ValueError("spacing must have 3 values (for spatial axes).")

    # Determine labels
    if labels is None:
        labs = set(np.unique(target)).union(set(np.unique(ref)))
        labs.discard(background_label)
        labs = sorted(int(x) for x in labs)
    else:
        labs = [int(x) for x in labels if int(x) != background_label]

    per_label: Dict[int, Metrics] = {}
    ref_sizes: Dict[int, int] = {}
    target_sizes: Dict[int, int] = {}

    # Compute per label
    for lab in labs:
        target_bin = (target == lab)
        ref_bin = (ref == lab)
        ref_sizes[lab] = int(ref_bin.sum())
        target_sizes[lab] = int(target_bin.sum())
        per_label[lab] = _safe_binary_metrics(target_bin, ref_bin, spacing)

    # Macro average (unweighted)
    if len(labs) == 0:
        macro = Metrics(dice=1.0, msd=0.0, hd95=0.0)
    else:
        dice_vals = [per_label[l].dice for l in labs]
        msd_vals = [per_label[l].msd for l in labs]
        hd_vals = [per_label[l].hd95 for l in labs]

        macro = Metrics(
            dice=float(np.mean(dice_vals)),
            msd=float(np.mean(msd_vals)),
            hd95=float(np.mean(hd_vals)),
        )

    return ValidationResult(
        per_label=per_label,
        macro_avg=macro,
        ref_voxels=ref_sizes,
        target_voxels=target_sizes,
    )


# ----------------------------
# Pretty printing / CSV export
# ----------------------------
def print_results(result: ValidationResult) -> None:
    print("\nValidation Results")
    print("  Macro avg:", result.macro_avg)
    for lab, m in result.per_label.items():
        print(f"  Label {lab}: Dice={m.dice:.4f}, MSD={m.msd:.3f} mm, HD95={m.hd95:.3f} mm "
              f"(ref={result.ref_voxels[lab]}, target={result.target_voxels[lab]})")


# ----------------------------
# CLI Interface
# ----------------------------
def main():
    import argparse
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Compare two 3D segmentations and compute validation metrics (Dice, MSD, HD95)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to reference segmentation image (e.g., target.nii.gz)"
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to reference segmentation (ground truth) image (e.g., reference.nii.gz)"
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="Labels to evaluate (e.g., 1 2 3). If not provided, uses all non-background labels."
    )
    parser.add_argument(
        "--background-label",
        type=int,
        default=0,
        help="Background label to exclude from evaluation (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Load images
    logger.info(f"Loading target segmentation from {args.target}")
    target_arr, spacing_target = load_segmentation(args.target)
    logger.info(f"target image shape: {target_arr.shape}, spacing: {spacing_target} mm")

    logger.info(f"Loading reference segmentation from {args.ref}")
    ref_arr, spacing_ref = load_segmentation(args.ref)
    logger.info(f"Reference image shape: {ref_arr.shape}, spacing: {spacing_ref} mm")
    
    # Validate spacing matches
    if spacing_target != spacing_ref:
        logger.warning(
            f"⚠️  Spacing differs between images!\n"
            f"  target: {spacing_target} mm\n"
            f"  Reference: {spacing_ref} mm\n"
            f"  Using target image spacing for metrics."
        )
    
    # Run evaluation using spacing from target image
    logger.info(f"Evaluating segmentation with background_label={args.background_label}")
    result = evaluate_segmentation(
        target=target_arr,
        ref=ref_arr,
        spacing=spacing_target,
        labels=args.labels,
        background_label=args.background_label
    )
    
    # Print results
    print_results(result)


if __name__ == "__main__":
    main()
