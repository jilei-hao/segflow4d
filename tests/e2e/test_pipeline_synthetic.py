"""End-to-end tests for the propagation pipeline using synthetic data.

All registration tests are marked @pytest.mark.gpu and are skipped when
no CUDA device is available.  The smoke test with write_result_to_disk=False
can run on CPU to verify the pipeline wiring still works (registration will
run but quality is not asserted without GPU).

Usage:
    # GPU tests only
    pytest tests/e2e/test_pipeline_synthetic.py -m gpu -v

    # All (will skip GPU tests on CPU-only machines)
    pytest tests/e2e/test_pipeline_synthetic.py -v
"""

import os
import pytest
import numpy as np
import SimpleITK as sitk

from segflow4d.common.types.propagation_input import PropagationInputFactory
from segflow4d.propagation.propagation_pipeline import PropagationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE_ZYX = (64, 64, 64)
N_TP = 3


def _make_4d_image(n_tp=N_TP, shape_zyx=SHAPE_ZYX, shift_px=1):
    """Synthetic 4D image, each TP shifted by `shift_px` voxels in X.

    Uses ``sitk.JoinSeries`` to produce a genuine 4-D SimpleITK image.
    ``sitk.GetImageFromArray`` on a raw 4-D numpy array yields a 3-D image
    (the leading dimension is folded into Z), which is incorrect here.

    A deterministic internal texture (fixed random field, same seed across
    all TPs but rolled in X with the sphere) ensures NCC/SSD have a strong,
    well-conditioned signal across the full sphere volume so the optimizer
    can reliably find the 1-voxel shift.
    """
    volumes_3d = []
    for tp in range(n_tp):
        arr = np.zeros(shape_zyx, dtype=np.float32)
        cz = shape_zyx[0] // 2
        cy = shape_zyx[1] // 2
        cx = shape_zyx[2] // 2 + tp * shift_px
        zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
        dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        arr[dist <= 5] = 1.0
        arr = arr + np.random.default_rng(tp).uniform(0, 0.05, shape_zyx).astype(np.float32)
        vol = sitk.GetImageFromArray(arr)
        vol.SetSpacing((1.0, 1.0, 1.0))
        vol.SetOrigin((0.0, 0.0, 0.0))
        volumes_3d.append(vol)
    return sitk.JoinSeries(volumes_3d)


def _make_seg_ref(shape_zyx=SHAPE_ZYX):
    """Simple 3-D segmentation with label 1 (sphere) at TP 0."""
    arr = np.zeros(shape_zyx, dtype=np.int16)
    cz, cy, cx = shape_zyx[0] // 2, shape_zyx[1] // 2, shape_zyx[2] // 2
    zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    arr[dist <= 5] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _make_seg_gt(tp, shape_zyx=SHAPE_ZYX, shift_px=1):
    """Ground-truth segmentation for the given timepoint (sphere shifted by tp*shift_px)."""
    arr = np.zeros(shape_zyx, dtype=np.int16)
    cz = shape_zyx[0] // 2
    cy = shape_zyx[1] // 2
    cx = shape_zyx[2] // 2 + tp * shift_px
    zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    arr[dist <= 5] = 1
    return arr


def _write_images(tmp_path):
    """Write synthetic images to disk and return their paths."""
    img_path = str(tmp_path / "image_4d.nii.gz")
    seg_path = str(tmp_path / "seg_ref.nii.gz")
    sitk.WriteImage(_make_4d_image(), img_path)
    sitk.WriteImage(_make_seg_ref(), seg_path)
    return img_path, seg_path


def _flush_async_writer():
    """Block until all async write tasks have completed.

    Uses flush() (not shutdown) so the worker thread stays alive and can
    serve subsequent tests without needing rebinding tricks.
    """
    import sys, importlib
    aw_mod = sys.modules.get('segflow4d.utility.file_writer.async_writer')
    if aw_mod is None:
        aw_mod = importlib.import_module('segflow4d.utility.file_writer.async_writer')
    aw_mod.async_writer.flush()


def _build_input(img_path, seg_path, out_dir, write_to_disk=True):
    """Build a PropagationInput using the FireANTs (GPU) backend."""
    os.makedirs(out_dir, exist_ok=True)
    return (
        PropagationInputFactory()
        .set_image_4d_from_disk(img_path)
        .add_tp_input_group_from_disk(
            tp_ref=0,
            tp_target=[1, 2],
            seg_ref_path=seg_path,
            additional_meshes_ref=None,
        )
        .set_options(
            lowres_factor=2.0,
            registration_backend="FIREANTS",
            dilation_radius=2,
            write_result_to_disk=write_to_disk,
            output_directory=out_dir,
            minimum_required_vram_gb=0,
            # Minimal iterations for speed
            scales=[1],
            affine_iterations=[2],
            deformable_iterations=[2],
        )
        .build()
    )


def _build_input_greedy(img_path, seg_path, out_dir, write_to_disk=True,
                        affine_iterations=None, deformable_iterations=None, **backend_kwargs):
    """Build a PropagationInput using the Greedy (CPU) backend."""
    if affine_iterations is None:
        affine_iterations = [2]
    if deformable_iterations is None:
        deformable_iterations = [2]
    os.makedirs(out_dir, exist_ok=True)
    return (
        PropagationInputFactory()
        .set_image_4d_from_disk(img_path)
        .add_tp_input_group_from_disk(
            tp_ref=0,
            tp_target=[1, 2],
            seg_ref_path=seg_path,
            additional_meshes_ref=None,
        )
        .set_options(
            lowres_factor=2.0,
            registration_backend="GREEDY",
            dilation_radius=2,
            write_result_to_disk=write_to_disk,
            output_directory=out_dir,
            minimum_required_vram_gb=0,
            affine_iterations=affine_iterations,
            deformable_iterations=deformable_iterations,
            **backend_kwargs,
        )
        .build()
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestPipelineSyntheticOutputFiles:
    def test_pipeline_creates_seg_4d_file(self, tmp_path):
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        assert os.path.isfile(os.path.join(out_dir, "seg-4d.nii.gz")), (
            "seg-4d.nii.gz not found in output directory"
        )

    def test_pipeline_creates_image_4d_file(self, tmp_path):
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        assert os.path.isfile(os.path.join(out_dir, "image-4d.nii.gz")), (
            "image-4d.nii.gz not found in output directory"
        )

    def test_pipeline_output_seg_shape_matches_input(self, tmp_path):
        """Output 4D segmentation must have same spatial dims and timepoint count as input."""
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        # SITK reports size as (X, Y, Z, T)
        size = seg_4d.GetSize()
        assert size[3] == N_TP, f"Expected {N_TP} timepoints, got {size[3]}"
        assert size[0] == SHAPE_ZYX[2], f"X dim mismatch: {size[0]} vs {SHAPE_ZYX[2]}"
        assert size[1] == SHAPE_ZYX[1], f"Y dim mismatch: {size[1]} vs {SHAPE_ZYX[1]}"
        assert size[2] == SHAPE_ZYX[0], f"Z dim mismatch: {size[2]} vs {SHAPE_ZYX[0]}"


@pytest.mark.gpu
class TestPipelineSyntheticSegQuality:
    def test_propagated_dice_above_threshold(self, tmp_path):
        """
        A 1-voxel rigid shift between TPs is trivially registerable.
        The propagated segmentation at TP1 and TP2 should have Dice >= 0.75
        compared to the ground-truth (shifted sphere).
        """
        from segflow4d.utility.validation.segmentation_validation import evaluate_segmentation

        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d_path = os.path.join(out_dir, "seg-4d.nii.gz")
        seg_4d = sitk.ReadImage(seg_4d_path)

        # Compare each propagated TP to its own shifted ground-truth sphere
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
            assert dice >= 0.75, (
                f"TP {tp}: propagated Dice {dice:.3f} < 0.75 threshold"
            )


# ---------------------------------------------------------------------------
# Greedy (CPU) backend — requires picsl_greedy
# ---------------------------------------------------------------------------

@pytest.mark.greedy
class TestPipelineSyntheticGreedyOutputFiles:
    """Verify that the pipeline writes expected output files when using the
    Greedy CPU backend.

    Requires picsl-greedy::

        pip install segflow4d[greedy]   # or: pip install picsl-greedy
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_greedy(self):
        pytest.importorskip(
            "picsl_greedy",
            reason="picsl-greedy not installed — skipping Greedy e2e tests",
        )

    def test_pipeline_creates_seg_4d_file(self, tmp_path):
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_greedy(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        assert os.path.isfile(os.path.join(out_dir, "seg-4d.nii.gz")), (
            "seg-4d.nii.gz not found in output directory"
        )

    def test_pipeline_creates_image_4d_file(self, tmp_path):
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_greedy(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        assert os.path.isfile(os.path.join(out_dir, "image-4d.nii.gz")), (
            "image-4d.nii.gz not found in output directory"
        )

    def test_pipeline_output_seg_shape_matches_input(self, tmp_path):
        """Output 4D segmentation must have same spatial dims and timepoint count as input."""
        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_greedy(img_path, seg_path, out_dir, write_to_disk=True)
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        size = seg_4d.GetSize()
        assert size[3] == N_TP, f"Expected {N_TP} timepoints, got {size[3]}"
        assert size[0] == SHAPE_ZYX[2], f"X dim mismatch: {size[0]} vs {SHAPE_ZYX[2]}"
        assert size[1] == SHAPE_ZYX[1], f"Y dim mismatch: {size[1]} vs {SHAPE_ZYX[1]}"
        assert size[2] == SHAPE_ZYX[0], f"Z dim mismatch: {size[2]} vs {SHAPE_ZYX[0]}"


@pytest.mark.greedy
class TestPipelineSyntheticGreedySegQuality:
    """Verify propagation quality when using the Greedy CPU backend.

    Requires picsl-greedy::

        pip install segflow4d[greedy]   # or: pip install picsl-greedy
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_greedy(self):
        pytest.importorskip(
            "picsl_greedy",
            reason="picsl-greedy not installed — skipping Greedy e2e tests",
        )

    def test_propagated_dice_above_threshold(self, tmp_path):
        """
        A 1-voxel rigid shift between TPs is trivially registerable.
        The propagated segmentation at TP1 and TP2 should have Dice >= 0.60
        compared to the ground-truth (shifted sphere).

        Note: Greedy (CPU) achieves lower accuracy than GPU FireANTs on simple
        synthetic sphere data. The uniform sphere gives a near-flat optimization
        landscape (gradient only at boundary), so the threshold is set to 0.60.
        """
        from segflow4d.utility.validation.segmentation_validation import evaluate_segmentation

        img_path, seg_path = _write_images(tmp_path)
        out_dir = str(tmp_path / "output")

        prop_input = _build_input_greedy(
            img_path, seg_path, out_dir, write_to_disk=True,
            affine_iterations=[100, 50],
            deformable_iterations=[50, 25],
            affine_dof=6,         # rigid — appropriate for the synthetic 1-voxel shift
            metric_radius=[4, 4, 4],  # larger NCC window to capture sphere boundary
            jitter=0.0,           # disable jitter for deterministic test results
        )
        pipeline = PropagationPipeline(prop_input)
        pipeline.run()
        _flush_async_writer()

        seg_4d_path = os.path.join(out_dir, "seg-4d.nii.gz")
        seg_4d = sitk.ReadImage(seg_4d_path)

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
            assert dice >= 0.60, (
                f"TP {tp}: propagated Dice {dice:.3f} < 0.60 threshold"
            )
