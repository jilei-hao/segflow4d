"""Unit tests for image processing utilities."""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.processing.image_processing import create_reference_mask, create_high_res_mask


class TestCreateReferenceMask:
    def test_output_is_binary(self, synthetic_seg_3d):
        """create_reference_mask must produce only 0/1 values."""
        mask = create_reference_mask(synthetic_seg_3d, scale_factor=2.0, dilation_radius=2)
        arr = sitk.GetArrayFromImage(mask.get_data())
        unique = set(np.unique(arr).tolist())
        assert unique.issubset({0, 1}), f"Non-binary values found: {unique}"

    def test_dilated_mask_volume_exceeds_binary_input(self, synthetic_seg_3d):
        """Dilation must expand the mask (more foreground voxels than the thresholded input)."""
        # Build binary reference without dilation (radius=0)
        mask_no_dilate = create_reference_mask(synthetic_seg_3d, scale_factor=1.0, dilation_radius=0)
        vol_no_dilate = int(
            sitk.GetArrayFromImage(mask_no_dilate.get_data()).sum()
        )

        # Build mask with dilation
        mask_dilated = create_reference_mask(synthetic_seg_3d, scale_factor=1.0, dilation_radius=3)
        vol_dilated = int(
            sitk.GetArrayFromImage(mask_dilated.get_data()).sum()
        )

        assert vol_dilated > vol_no_dilate, (
            f"Dilated volume ({vol_dilated}) should exceed undilated ({vol_no_dilate})"
        )

    def test_output_spacing_matches_scale_factor(self, synthetic_seg_3d):
        """Output spacing must be approximately input_spacing / scale_factor.

        scale_factor=0.5 halves the voxel count, which doubles the spacing.
        """
        factor = 0.5
        input_spacing = synthetic_seg_3d.get_spacing()  # (1, 1, 1)
        mask = create_reference_mask(synthetic_seg_3d, scale_factor=factor, dilation_radius=1)
        out_spacing = mask.get_spacing()
        for i in range(3):
            expected = input_spacing[i] / factor
            assert out_spacing[i] == pytest.approx(expected, rel=0.05), (
                f"Axis {i}: expected {expected}, got {out_spacing[i]}"
            )

    def test_returns_image_wrapper(self, synthetic_seg_3d):
        mask = create_reference_mask(synthetic_seg_3d, scale_factor=1.0, dilation_radius=1)
        assert isinstance(mask, ImageWrapper)
        assert mask.get_data() is not None


class TestCreateHighResMask:
    def test_output_dimensions_match_ref_seg(self, synthetic_seg_3d):
        """create_high_res_mask must produce an image with the same size as seg_ref."""
        low_res_mask = create_reference_mask(synthetic_seg_3d, scale_factor=2.0, dilation_radius=1)
        high_res_mask = create_high_res_mask(
            ref_seg_image=synthetic_seg_3d,
            low_res_mask=low_res_mask
        )
        assert high_res_mask.get_dimensions() == synthetic_seg_3d.get_dimensions()

    def test_high_res_mask_is_binary(self, synthetic_seg_3d):
        low_res_mask = create_reference_mask(synthetic_seg_3d, scale_factor=2.0, dilation_radius=1)
        high_res_mask = create_high_res_mask(
            ref_seg_image=synthetic_seg_3d,
            low_res_mask=low_res_mask
        )
        arr = sitk.GetArrayFromImage(high_res_mask.get_data())
        unique = set(np.unique(arr).tolist())
        assert unique.issubset({0, 1}), f"Non-binary values: {unique}"
