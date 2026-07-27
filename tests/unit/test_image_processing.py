"""Unit tests for image processing utilities."""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.processing.image_processing import (
    create_reference_mask,
    create_high_res_mask,
    clamp_scale_factor_for_min_size,
    pad_image_to_min_size,
    FIREANTS_MIN_IMG_SIZE,
)


def _resampled_size(sz: int, factor: float) -> int:
    """Mirror CPUImageHelper.resample's size rule for assertions."""
    return max(1, int(sz * factor))


class TestClampScaleFactorForMinSize:
    def test_no_op_when_all_dims_stay_above_floor(self):
        """A factor that keeps every dim >= the floor is returned unchanged."""
        # 64 * 0.5 = 32 == floor
        assert clamp_scale_factor_for_min_size((64, 64, 64), 0.5, 32) == 0.5

    def test_upsampling_factor_is_never_lowered(self):
        """Upsampling can't underflow, so the requested factor passes through."""
        assert clamp_scale_factor_for_min_size((40, 48, 48), 2.0, 32) == 2.0

    def test_raises_binding_dim_to_floor(self):
        """The smallest dim drives the clamp; the result keeps it at >= floor."""
        # z=57 at 0.5 -> 28 < 32 (the reported repro). Expect a larger factor.
        eff = clamp_scale_factor_for_min_size((512, 512, 57), 0.5, 32)
        assert eff > 0.5
        assert _resampled_size(57, eff) >= 32

    def test_small_z_repro_keeps_every_dim_at_or_above_floor(self):
        for sz in [(40, 48, 48), (512, 512, 57), (33, 64, 64)]:
            eff = clamp_scale_factor_for_min_size(sz, 0.5, 32)
            assert eff >= 0.5
            for d in sz:
                assert _resampled_size(d, eff) >= 32, (
                    f"dim {d} resampled to {_resampled_size(d, eff)} < 32 at factor {eff}"
                )

    def test_native_dim_below_floor_is_not_upsampled_past_native(self):
        """A dim already below the floor can't be rescued; factor caps at 1.0
        and the dim stays at its native size (mirrors compute_union_bbox)."""
        eff = clamp_scale_factor_for_min_size((20, 64, 64), 0.5, 32)
        assert eff == 1.0
        assert _resampled_size(20, eff) == 20  # still below floor, but not upsampled

    def test_floor_zero_disables_clamping(self):
        assert clamp_scale_factor_for_min_size((10, 10, 10), 0.5, 0) == 0.5

    def test_nonpositive_factor_raises(self):
        with pytest.raises(ValueError):
            clamp_scale_factor_for_min_size((64, 64, 64), 0.0, 32)

    def test_default_floor_is_fireants_min(self):
        # Uses FIREANTS_MIN_IMG_SIZE by default.
        eff = clamp_scale_factor_for_min_size((57, 64, 64), 0.5)
        assert _resampled_size(57, eff) >= FIREANTS_MIN_IMG_SIZE


class TestPadImageToMinSize:
    def _img(self, size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        # size is (x, y, z); numpy array is (z, y, x)
        arr = np.ones(tuple(reversed(size)), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        return ImageWrapper(img)

    def test_no_op_when_all_dims_meet_floor(self):
        img = self._img((40, 48, 64))
        out = pad_image_to_min_size(img, FIREANTS_MIN_IMG_SIZE)
        assert list(out.get_data().GetSize()) == [40, 48, 64]

    def test_thin_dim_is_padded_to_floor(self):
        img = self._img((64, 64, 28))  # z below the 32 floor
        out = pad_image_to_min_size(img, 32)
        assert list(out.get_data().GetSize()) == [64, 64, 32]

    def test_all_dims_padded_when_all_below_floor(self):
        img = self._img((10, 12, 8))
        out = pad_image_to_min_size(img, 32)
        assert list(out.get_data().GetSize()) == [32, 32, 32]

    def test_symmetric_padding_preserves_physical_position(self):
        # ConstantPad shifts the origin so existing voxels keep their physical
        # location: a lower pad of L voxels moves the origin back by L*spacing.
        spacing = (2.0, 2.0, 2.0)
        img = self._img((64, 64, 28), spacing=spacing, origin=(0.0, 0.0, 0.0))
        out = pad_image_to_min_size(img, 32)
        need = 32 - 28
        lower = need // 2  # = 2
        assert out.get_data().GetOrigin()[2] == pytest.approx(-lower * spacing[2])
        # spacing/direction unchanged
        assert out.get_data().GetSpacing() == spacing

    def test_padding_uses_constant_fill(self):
        img = self._img((64, 64, 28))
        out = pad_image_to_min_size(img, 32, constant=0.0)
        arr = sitk.GetArrayFromImage(out.get_data())  # (z, y, x)
        # first padded z-slice is all background, interior is the original ones
        assert np.all(arr[0] == 0.0)
        assert np.all(arr[2:30] == 1.0)

    def test_floor_zero_disables_padding(self):
        img = self._img((8, 8, 8))
        out = pad_image_to_min_size(img, 0)
        assert list(out.get_data().GetSize()) == [8, 8, 8]


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
