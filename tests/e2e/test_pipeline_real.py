"""End-to-end tests using real (cropped) patient data fixtures.

These tests require:
  1. A CUDA-capable GPU  (@pytest.mark.gpu)
  2. The committed fixture data in tests/fixtures/real_data/
     (@pytest.mark.requires_real_data)

Skip conditions:
  - No GPU → entire class is skipped automatically via the gpu marker
  - Fixture files absent → skipped with an informative message

Run:
    pytest tests/e2e/test_pipeline_real.py -m "gpu and requires_real_data" -v
"""

import json
import os
import pytest
import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "real_data")
IMAGE_4D_PATH = os.path.join(FIXTURE_DIR, "image_4d_crop.nii.gz")
SEG_REF_PATH = os.path.join(FIXTURE_DIR, "seg_ref_crop.nii.gz")
EXPECTED_METRICS_PATH = os.path.join(FIXTURE_DIR, "expected_metrics.json")

# ---------------------------------------------------------------------------
# Skip guard: check that required fixture files exist
# ---------------------------------------------------------------------------

_FIXTURES_PRESENT = (
    os.path.isfile(IMAGE_4D_PATH)
    and os.path.isfile(SEG_REF_PATH)
)


def _skip_if_no_fixtures():
    if not _FIXTURES_PRESENT:
        pytest.skip(
            "Real-data fixtures not found. "
            "Run scripts/generate_real_fixtures.py to create them."
        )


def _flush_async_writer():
    """Flush and reinitialise the global async writer."""
    from segflow4d.utility.file_writer.async_writer import AsyncWriter
    import segflow4d.utility.file_writer.async_writer as _aw_module
    _aw_module.async_writer.shutdown(wait=True)
    new_writer = AsyncWriter()
    _aw_module.async_writer = new_writer
    try:
        import segflow4d.utility.file_writer as _fw
        _fw.async_writer = new_writer
    except Exception:
        pass
    try:
        import segflow4d.propagation.tp_partition as _tp
        _tp.async_writer = new_writer
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper: determine timepoints from the 4D image
# ---------------------------------------------------------------------------

def _get_n_tp(image_4d_path):
    img = sitk.ReadImage(image_4d_path)
    return img.GetSize()[3]  # SITK: (X, Y, Z, T)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.requires_real_data
class TestPipelineRealSmoke:
    def test_real_pipeline_completes_without_error(self, tmp_path):
        """Pipeline must run to completion on the real fixture data."""
        _skip_if_no_fixtures()
        from segflow4d.common.types.propagation_input import PropagationInputFactory
        from segflow4d.propagation.propagation_pipeline import PropagationPipeline

        n_tp = _get_n_tp(IMAGE_4D_PATH)
        tp_targets = list(range(1, n_tp))
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)

        prop_input = (
            PropagationInputFactory()
            .set_image_4d_from_disk(IMAGE_4D_PATH)
            .add_tp_input_group_from_disk(
                tp_ref=0,
                tp_target=tp_targets,
                seg_ref_path=SEG_REF_PATH,
                additional_meshes_ref=None,
            )
            .set_options(
                lowres_factor=2.0,
                registration_backend="FIREANTS",
                dilation_radius=3,
                write_result_to_disk=True,
                output_directory=out_dir,
                minimum_required_vram_gb=0,
                scales=[4, 2, 1],
                affine_iterations=[100, 50, 25],
                deformable_iterations=[100, 50, 25],
            )
            .build()
        )

        pipeline = PropagationPipeline(prop_input)
        pipeline.run()  # must not raise
        _flush_async_writer()

        assert os.path.isfile(os.path.join(out_dir, "seg-4d.nii.gz"))
        assert os.path.isfile(os.path.join(out_dir, "image-4d.nii.gz"))

    def test_real_pipeline_output_timepoint_count(self, tmp_path):
        """Output 4D seg must contain the same number of TPs as the input."""
        _skip_if_no_fixtures()
        from segflow4d.common.types.propagation_input import PropagationInputFactory
        from segflow4d.propagation.propagation_pipeline import PropagationPipeline

        n_tp = _get_n_tp(IMAGE_4D_PATH)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)

        prop_input = (
            PropagationInputFactory()
            .set_image_4d_from_disk(IMAGE_4D_PATH)
            .add_tp_input_group_from_disk(
                tp_ref=0,
                tp_target=list(range(1, n_tp)),
                seg_ref_path=SEG_REF_PATH,
                additional_meshes_ref=None,
            )
            .set_options(
                lowres_factor=2.0,
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
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
        assert seg_4d.GetSize()[3] == n_tp


@pytest.mark.gpu
@pytest.mark.requires_real_data
class TestPipelineRealDiceRegression:
    def test_real_pipeline_dice_above_baseline(self, tmp_path):
        """
        Dice on the real fixture case must stay above the stored baseline in
        expected_metrics.json.  This is a regression guard: if algorithm
        changes degrade quality, this test will catch it.

        The expected_metrics.json is generated by the script
        scripts/generate_real_fixtures.py after the first successful run.
        """
        _skip_if_no_fixtures()

        if not os.path.isfile(EXPECTED_METRICS_PATH):
            pytest.skip(
                "expected_metrics.json not found. "
                "Run scripts/generate_real_fixtures.py to create baseline metrics."
            )

        from segflow4d.common.types.propagation_input import PropagationInputFactory
        from segflow4d.propagation.propagation_pipeline import PropagationPipeline
        from segflow4d.utility.validation.segmentation_validation import evaluate_segmentation

        with open(EXPECTED_METRICS_PATH) as f:
            expected = json.load(f)

        n_tp = _get_n_tp(IMAGE_4D_PATH)
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)

        prop_input = (
            PropagationInputFactory()
            .set_image_4d_from_disk(IMAGE_4D_PATH)
            .add_tp_input_group_from_disk(
                tp_ref=0,
                tp_target=list(range(1, n_tp)),
                seg_ref_path=SEG_REF_PATH,
                additional_meshes_ref=None,
            )
            .set_options(
                lowres_factor=2.0,
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
        _flush_async_writer()

        seg_4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))

        ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(SEG_REF_PATH)).astype(np.int32)
        ref_spacing = sitk.ReadImage(SEG_REF_PATH).GetSpacing()
        spacing_zyx = (ref_spacing[2], ref_spacing[1], ref_spacing[0])

        extractor = sitk.ExtractImageFilter()
        size = list(seg_4d.GetSize())
        size[3] = 0
        extractor.SetSize(size)

        macro_dices = []
        for tp in range(1, n_tp):
            extractor.SetIndex([0, 0, 0, tp])
            tp_seg = extractor.Execute(seg_4d)
            pred_arr = sitk.GetArrayFromImage(tp_seg).astype(np.int32)

            result = evaluate_segmentation(pred_arr, ref_arr, spacing=spacing_zyx)
            macro_dices.append(result.macro_avg.dice)

        mean_dice = float(np.mean(macro_dices))
        baseline_dice = float(expected.get("mean_macro_dice", 0.0))
        # Allow a small tolerance (2%) below baseline before failing
        tolerance = 0.02
        assert mean_dice >= baseline_dice - tolerance, (
            f"Mean macro Dice {mean_dice:.4f} dropped more than {tolerance} "
            f"below baseline {baseline_dice:.4f}"
        )
