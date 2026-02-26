"""
Generate real-data fixtures for SegFlow4D E2E tests.

Crops a small sub-volume from the bav16 dataset and saves it as test fixtures.
Also runs a quick pipeline on the crop and records the baseline Dice score
in expected_metrics.json.

Usage:
    python scripts/generate_real_fixtures.py [--data-root /path/to/data]

Outputs (written to tests/fixtures/real_data/):
    image_4d_crop.nii.gz        — 3-TP cropped 4D image
    seg_ref_crop.nii.gz         — matching 3D segmentation (TP 48 region)
    expected_metrics.json       — baseline mean macro Dice for regression tests
"""

import argparse
import json
import os
import sys
import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE_OUT_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "real_data")

DATA_ROOT_DEFAULT = "/home/jileihao/data/segflow4d/bav16"
IMAGE_4D_SRC = os.path.join(DATA_ROOT_DEFAULT, "i4.nii.gz")
SEG_REF_SRC = os.path.join(DATA_ROOT_DEFAULT, "sr_tp-48.nii.gz")

# Reference TP index in the 4D volume (0-based within the file) and two targets
# bav16 has TPs 48–79 stored; TP 48 = index 0 in the file
REF_IDX = 0   # file index for TP 48
TGT_IDXS = [1, 2]  # file indices for TP 49, 50

# Crop size in voxels (X, Y, Z) — centred on the segmentation's bounding box
# Z must be >= 64 so that after lowres_factor=2.0 the downsampled Z (32) stays
# at or above FireANTs' MIN_IMG_SIZE=32; smaller values cause downsample_fft to
# produce zero-length slices and raise "Invalid number of data points (0)".
CROP_XYZ = (64, 64, 64)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounding_box_center(seg_arr):
    """Return the (z, y, x) centre of the bounding box of non-zero voxels (ZYX array)."""
    coords = np.argwhere(seg_arr > 0)
    if len(coords) == 0:
        return tuple(s // 2 for s in seg_arr.shape)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centre = ((mins + maxs) // 2).tolist()
    return tuple(centre)


def _crop_3d(img_sitk, start_xyz, size_xyz):
    """Crop a 3D SITK image starting at (x,y,z) with given size (W, H, D)."""
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize(list(size_xyz))
    extractor.SetIndex(list(start_xyz))
    return extractor.Execute(img_sitk)


def _crop_4d_tp(img4d_sitk, tp_idx, start_xyz, size_xyz):
    """Extract timepoint `tp_idx` from a 4D image and crop it."""
    size_4d = list(img4d_sitk.GetSize())
    extract_size = list(size_xyz) + [0]  # size[3]=0 collapses the T dimension
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(extract_size)
    extractor.SetIndex(list(start_xyz) + [tp_idx])
    return extractor.Execute(img4d_sitk)


def _combine_to_4d(volumes_sitk):
    """Stack a list of 3D SITK images into a 4D image along a new T axis."""
    # Use JoinSeries
    joiner = sitk.JoinSeriesImageFilter()
    img_4d = joiner.Execute(*volumes_sitk)
    return img_4d


def _compute_crop_origin(center_zyx, crop_xyz, shape_zyx):
    """
    Given centre in ZYX order and crop size in XYZ order, return safe start
    indices in XYZ order (clamped to image bounds).
    """
    cx, cy, cz = center_zyx[2], center_zyx[1], center_zyx[0]  # ZYX → XYZ
    wx, wy, wz = crop_xyz

    start_x = max(0, min(cx - wx // 2, shape_zyx[2] - wx))
    start_y = max(0, min(cy - wy // 2, shape_zyx[1] - wy))
    start_z = max(0, min(cz - wz // 2, shape_zyx[0] - wz))
    return (start_x, start_y, start_z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT_DEFAULT,
        help=f"Root data directory for bav16 (default: {DATA_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the pipeline baseline; only write image fixtures.",
    )
    args = parser.parse_args()

    img4d_path = os.path.join(args.data_root, "i4.nii.gz")
    seg_ref_path = os.path.join(args.data_root, "sr_tp-48.nii.gz")

    for p in [img4d_path, seg_ref_path]:
        if not os.path.isfile(p):
            print(f"ERROR: source file not found: {p}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(FIXTURE_OUT_DIR, exist_ok=True)

    print("Loading source images…")
    img4d = sitk.ReadImage(img4d_path)
    seg_ref = sitk.ReadImage(seg_ref_path)

    seg_arr = sitk.GetArrayFromImage(seg_ref)  # ZYX
    shape_zyx = seg_arr.shape

    print(f"  4D image size (XYZT): {img4d.GetSize()}")
    print(f"  Seg ref size  (XYZ):  {seg_ref.GetSize()}")

    # Find crop centre from segmentation bounding box
    centre_zyx = _bounding_box_center(seg_arr)
    print(f"  Segmentation centre (ZYX): {centre_zyx}")

    start_xyz = _compute_crop_origin(centre_zyx, CROP_XYZ, shape_zyx)
    print(f"  Crop start (XYZ): {start_xyz}, size (XYZ): {CROP_XYZ}")

    # Crop reference segmentation
    seg_crop = _crop_3d(seg_ref, start_xyz, CROP_XYZ)
    seg_crop_path = os.path.join(FIXTURE_OUT_DIR, "seg_ref_crop.nii.gz")
    sitk.WriteImage(seg_crop, seg_crop_path)
    print(f"  Written: {seg_crop_path}")

    # Crop 3 consecutive TPs and combine into a 4D image
    tp_indices = [REF_IDX] + TGT_IDXS
    volumes = []
    for tp_idx in tp_indices:
        tp_vol = _crop_4d_tp(img4d, tp_idx, start_xyz, CROP_XYZ)
        volumes.append(tp_vol)
    img4d_crop = _combine_to_4d(volumes)

    img4d_crop_path = os.path.join(FIXTURE_OUT_DIR, "image_4d_crop.nii.gz")
    sitk.WriteImage(img4d_crop, img4d_crop_path)
    print(f"  Written: {img4d_crop_path}")

    if args.skip_baseline:
        print("--skip-baseline set; skipping pipeline baseline computation.")
        return

    # -----------------------------------------------------------------------
    # Run pipeline on the crop to compute baseline Dice
    # -----------------------------------------------------------------------
    print("\nRunning pipeline on cropped fixtures to compute baseline Dice…")
    import tempfile

    # Check for CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available; skipping baseline computation.")
            print("  Set minimum_required_vram_gb=0 or run on a GPU machine.")
            _write_empty_baseline()
            return
    except ImportError:
        print("WARNING: torch not importable; skipping baseline computation.")
        _write_empty_baseline()
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

        import sys as _sys
        from segflow4d.common.types.propagation_input import PropagationInputFactory
        from segflow4d.propagation.propagation_pipeline import PropagationPipeline
        from segflow4d.utility.validation.segmentation_validation import evaluate_segmentation
        # Import the module via sys.modules to avoid getting the re-exported instance
        # from file_writer/__init__.py (which shadows the submodule name).
        import segflow4d.utility.file_writer.async_writer  # ensure it's in sys.modules
        _aw_module = _sys.modules['segflow4d.utility.file_writer.async_writer']

        prop_input = (
            PropagationInputFactory()
            .set_image_4d_from_disk(img4d_crop_path)
            .add_tp_input_group_from_disk(
                tp_ref=0,
                tp_target=list(range(1, len(tp_indices))),
                seg_ref_path=seg_crop_path,
                additional_meshes_ref=None,
            )
            .set_options(
                lowres_factor=1.0,  # keep at 1.0 for small crops: lowres_factor=2.0 would halve
                                    # 64-voxel dims to 32, then downsample_fft(32→32) produces
                                    # 0-length FFT slices and crashes with "Invalid number of data points"
                registration_backend="FIREANTS",
                dilation_radius=3,
                write_result_to_disk=True,
                output_directory=out_dir,
                minimum_required_vram_gb=0,
            )
            .build()
        )

        pipeline = PropagationPipeline(prop_input)
        pipeline.run()

        # Flush async writer — use flush() (not shutdown) so the worker thread
        # stays alive; the installed pipeline code holds a direct reference to
        # the singleton instance.
        _aw_module.async_writer.flush()

        # Compute Dice against reference segmentation
        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        ref_arr = sitk.GetArrayFromImage(seg_crop).astype(int)
        ref_spacing_xyz = seg_crop.GetSpacing()
        spacing_zyx = (ref_spacing_xyz[2], ref_spacing_xyz[1], ref_spacing_xyz[0])

        extractor = sitk.ExtractImageFilter()
        size = list(seg_4d.GetSize())
        size[3] = 0
        extractor.SetSize(size)

        dices = []
        for tp in range(1, len(tp_indices)):
            extractor.SetIndex([0, 0, 0, tp])
            tp_seg = extractor.Execute(seg_4d)
            pred_arr = sitk.GetArrayFromImage(tp_seg).astype(int)
            result = evaluate_segmentation(pred_arr, ref_arr, spacing=spacing_zyx)
            dices.append(result.macro_avg.dice)
            print(f"  TP {tp} macro Dice: {result.macro_avg.dice:.4f}")

        mean_dice = float(np.mean(dices))
        print(f"\n  Mean macro Dice: {mean_dice:.4f}")

    metrics = {
        "mean_macro_dice": mean_dice,
        "per_tp_dice": dices,
        "n_tp": len(tp_indices),
        "note": (
            "Baseline generated automatically. "
            "Tests will fail if mean_macro_dice drops more than 0.02 below this value."
        ),
    }
    metrics_path = os.path.join(FIXTURE_OUT_DIR, "expected_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Written: {metrics_path}")
    print("\nFixture generation complete.")


def _write_empty_baseline():
    metrics = {
        "mean_macro_dice": 0.0,
        "note": "Baseline not computed (no GPU available at generation time). "
                "Re-run generate_real_fixtures.py on a GPU machine.",
    }
    metrics_path = os.path.join(FIXTURE_OUT_DIR, "expected_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Written placeholder: {metrics_path}")


if __name__ == "__main__":
    main()
