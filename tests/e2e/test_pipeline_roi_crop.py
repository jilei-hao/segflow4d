"""End-to-end test for ROI cropping wired into the propagation pipeline.

Exercises the new ``roi_crop_padding_voxels`` option together with the
device-aware FireANTs backend. Skipped automatically when no accelerator
(CUDA or MPS) is present.

The test sets ``roi_crop_padding_voxels`` > 0 so the pipeline:
  1. Runs low-res mask propagation as usual.
  2. Computes the union bbox of all propagated high-res masks.
  3. Crops the per-TP image, mask, and the reference segmentation to the bbox.
  4. Runs FireANTs registration on the cropped data.
  5. Uncrops the resliced segmentation back to the reference frame.

Assertions cover:
  - Output dimensions match the original (i.e. uncrop reconstituted the frame).
  - Each non-reference TP carries the propagated label (Dice >= 0.5).
"""

import os
import pytest
import numpy as np
import SimpleITK as sitk

from segflow4d.common.types.propagation_input import PropagationInputFactory
from segflow4d.propagation.propagation_pipeline import PropagationPipeline


SHAPE_ZYX = (96, 96, 96)
N_TP = 3
SPHERE_RADIUS = 12
# Picked so the cropped bbox is >= FireANTs' MIN_IMG_SIZE (32 voxels per dim):
# radius 12 + padding 8 on each side => ~40 voxels per dim.
ROI_PADDING = 8


def _make_4d_image(n_tp=N_TP, shape_zyx=SHAPE_ZYX, shift_px=1):
    volumes_3d = []
    for tp in range(n_tp):
        arr = np.zeros(shape_zyx, dtype=np.float32)
        cz = shape_zyx[0] // 2
        cy = shape_zyx[1] // 2
        cx = shape_zyx[2] // 2 + tp * shift_px
        zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
        dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        arr[dist <= SPHERE_RADIUS] = 1.0
        arr = arr + np.random.default_rng(tp).uniform(0, 0.05, shape_zyx).astype(np.float32)
        vol = sitk.GetImageFromArray(arr)
        vol.SetSpacing((1.0, 1.0, 1.0))
        vol.SetOrigin((0.0, 0.0, 0.0))
        volumes_3d.append(vol)
    return sitk.JoinSeries(volumes_3d)


def _make_seg_ref(shape_zyx=SHAPE_ZYX):
    arr = np.zeros(shape_zyx, dtype=np.int16)
    cz, cy, cx = shape_zyx[0] // 2, shape_zyx[1] // 2, shape_zyx[2] // 2
    zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    arr[dist <= SPHERE_RADIUS] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _make_seg_gt(tp, shape_zyx=SHAPE_ZYX, shift_px=1):
    arr = np.zeros(shape_zyx, dtype=np.int16)
    cz = shape_zyx[0] // 2
    cy = shape_zyx[1] // 2
    cx = shape_zyx[2] // 2 + tp * shift_px
    zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    arr[dist <= SPHERE_RADIUS] = 1
    return arr


def _write_images(tmp_path):
    img_path = str(tmp_path / "image_4d.nii.gz")
    seg_path = str(tmp_path / "seg_ref.nii.gz")
    sitk.WriteImage(_make_4d_image(), img_path)
    sitk.WriteImage(_make_seg_ref(), seg_path)
    return img_path, seg_path


def _flush_async_writer():
    import sys, importlib
    aw_mod = sys.modules.get('segflow4d.utility.file_writer.async_writer')
    if aw_mod is None:
        aw_mod = importlib.import_module('segflow4d.utility.file_writer.async_writer')
    aw_mod.async_writer.flush()


def _build_input_fireants_roi(img_path, seg_path, out_dir, roi_padding):
    os.makedirs(out_dir, exist_ok=True)
    return (
        PropagationInputFactory()
        .set_image_4d_from_disk(img_path)
        .add_tp_input_group_from_disk(
            tp_ref=1,
            tp_target=[2, 3],
            seg_ref_path=seg_path,
            additional_meshes_ref=None,
        )
        .set_options(
            lowres_factor=2.0,
            registration_backend="FIREANTS",
            dilation_radius=4,
            write_result_to_disk=True,
            output_directory=out_dir,
            minimum_required_vram_gb=0,
            roi_crop_padding_voxels=roi_padding,
            scales=[1],
            affine_iterations=[2],
            deformable_iterations=[2],
        )
        .build()
    )


@pytest.mark.gpu
class TestPipelineRoiCropFireantsMps:
    """Pipeline with ROI cropping enabled, running FireANTs on the active accelerator."""

    def test_pipeline_output_seg_shape_matches_input(self, tmp_path):
        """ROI crop must round-trip — the output 4D segmentation must have the full reference shape."""
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_fireants_roi(img_path, seg_path, out_dir, roi_padding=ROI_PADDING)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        size = seg_4d.GetSize()  # (X, Y, Z, T)
        assert size[3] == N_TP, f"Expected {N_TP} timepoints, got {size[3]}"
        assert size[0] == SHAPE_ZYX[2], f"X dim mismatch: {size[0]} vs {SHAPE_ZYX[2]}"
        assert size[1] == SHAPE_ZYX[1], f"Y dim mismatch: {size[1]} vs {SHAPE_ZYX[1]}"
        assert size[2] == SHAPE_ZYX[0], f"Z dim mismatch: {size[2]} vs {SHAPE_ZYX[0]}"

    def test_propagated_dice_above_threshold_with_roi(self, tmp_path):
        """Even with ROI cropping the propagated segmentation should have Dice >= 0.5 against GT."""
        from segflow4d.utility.validation.segmentation_validation import evaluate_segmentation

        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_fireants_roi(img_path, seg_path, out_dir, roi_padding=ROI_PADDING)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        n_tps = seg_4d.GetSize()[3]
        for tp in range(1, n_tps):
            extractor = sitk.ExtractImageFilter()
            size = list(seg_4d.GetSize())
            size[3] = 0
            extractor.SetSize(size)
            extractor.SetIndex([0, 0, 0, tp])
            tp_seg = extractor.Execute(seg_4d)

            gt_arr = _make_seg_gt(tp).astype(np.int32)
            pred_arr = sitk.GetArrayFromImage(tp_seg).astype(np.int32)

            result = evaluate_segmentation(pred_arr, gt_arr, spacing=(1.0, 1.0, 1.0))
            dice = result.macro_avg.dice
            assert dice >= 0.50, (
                f"TP {tp}: propagated Dice {dice:.3f} < 0.50 threshold with ROI crop"
            )

    def test_outside_bbox_voxels_are_zero(self, tmp_path):
        """Voxels outside the union ROI bbox must remain 0 after uncrop.

        Since the GT sphere is well inside the centre and ROI padding is small,
        the corners of the output frame should be empty.
        """
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_fireants_roi(img_path, seg_path, out_dir, roi_padding=ROI_PADDING)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        # Inspect a non-reference timepoint (e.g. TP 2) and confirm the corners are zero.
        extractor = sitk.ExtractImageFilter()
        size = list(seg_4d.GetSize())
        size[3] = 0
        extractor.SetSize(size)
        extractor.SetIndex([0, 0, 0, 2])
        tp_seg = sitk.GetArrayFromImage(extractor.Execute(seg_4d))  # ZYX
        # Any 4-voxel cube at a corner should be all zeros after uncrop.
        for z_slice, y_slice, x_slice in [
            (slice(0, 4),  slice(0, 4),  slice(0, 4)),
            (slice(-4, None), slice(-4, None), slice(-4, None)),
            (slice(0, 4),  slice(-4, None), slice(-4, None)),
        ]:
            corner = tp_seg[z_slice, y_slice, x_slice]
            assert np.all(corner == 0), (
                f"Corner block z={z_slice}, y={y_slice}, x={x_slice} has non-zero "
                f"values after uncrop: {np.unique(corner)}"
            )
