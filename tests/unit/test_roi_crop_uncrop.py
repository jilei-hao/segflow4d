"""Unit tests for the ROI crop / uncrop helpers in image_processing."""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.processing.image_processing import (
    compute_union_bbox,
    crop_to_bbox,
    uncrop_to_reference,
)


def _make_mask(shape_zyx, fg_zyx_slices=None, spacing=(1.0, 1.0, 1.0),
               origin=(0.0, 0.0, 0.0), dtype=np.uint8, fg_value=1):
    """Build a binary ImageWrapper. Pass ``fg_zyx_slices=None`` for an empty mask."""
    arr = np.zeros(shape_zyx, dtype=dtype)
    if fg_zyx_slices is not None:
        arr[fg_zyx_slices] = fg_value
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return ImageWrapper(img)


class TestComputeUnionBbox:
    def test_single_mask_no_padding(self):
        # ZYX foreground block [4:8, 10:14, 12:16] → ITK XYZ start (12, 10, 4), size (4, 4, 4).
        mask = _make_mask((16, 32, 32), (slice(4, 8), slice(10, 14), slice(12, 16)))
        start, size = compute_union_bbox([mask], padding_voxels=0)
        assert start == [12, 10, 4]
        assert size == [4, 4, 4]

    def test_union_of_two_disjoint_masks(self):
        a = _make_mask((16, 32, 32), (slice(2, 4), slice(2, 4), slice(2, 4)))
        b = _make_mask((16, 32, 32), (slice(10, 12), slice(20, 22), slice(24, 26)))
        start, size = compute_union_bbox([a, b], padding_voxels=0)
        # XYZ union: start (2, 2, 2), end (26, 22, 12) → size (24, 20, 10).
        assert start == [2, 2, 2]
        assert size == [24, 20, 10]

    def test_padding_expands_bbox(self):
        mask = _make_mask((16, 32, 32), (slice(8, 10), slice(15, 17), slice(15, 17)))
        start, size = compute_union_bbox([mask], padding_voxels=3)
        # XYZ start (15-3, 15-3, 8-3) = (12, 12, 5); size (2+6, 2+6, 2+6) = (8, 8, 8).
        assert start == [12, 12, 5]
        assert size == [8, 8, 8]

    def test_padding_clamps_at_boundaries(self):
        # Foreground touches three image edges; padding must clamp to [0, image_size].
        mask = _make_mask((16, 32, 32), (slice(0, 2), slice(0, 2), slice(30, 32)))
        start, size = compute_union_bbox([mask], padding_voxels=4)
        assert start == [26, 0, 0]
        assert size == [6, 6, 6]

    def test_padding_larger_than_image_returns_full_image(self):
        mask = _make_mask((16, 32, 32), (slice(7, 9), slice(15, 17), slice(15, 17)))
        start, size = compute_union_bbox([mask], padding_voxels=1000)
        assert start == [0, 0, 0]
        assert size == [32, 32, 16]

    def test_single_voxel_mask(self):
        mask = _make_mask((16, 32, 32), (slice(8, 9), slice(16, 17), slice(16, 17)))
        start, size = compute_union_bbox([mask], padding_voxels=2)
        # XYZ start (16-2, 16-2, 8-2) = (14, 14, 6); size (1+4, 1+4, 1+4) = (5, 5, 5).
        assert start == [14, 14, 6]
        assert size == [5, 5, 5]

    def test_empty_mask_in_list_is_skipped(self):
        non_empty = _make_mask((16, 32, 32), (slice(4, 6), slice(4, 6), slice(4, 6)))
        empty = _make_mask((16, 32, 32), None)
        start, size = compute_union_bbox([non_empty, empty], padding_voxels=0)
        assert start == [4, 4, 4]
        assert size == [2, 2, 2]

    def test_all_empty_masks_raise(self):
        e1 = _make_mask((16, 32, 32), None)
        e2 = _make_mask((16, 32, 32), None)
        with pytest.raises(ValueError, match="every mask"):
            compute_union_bbox([e1, e2], padding_voxels=0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_union_bbox([], padding_voxels=0)

    def test_mismatched_grids_raise(self):
        a = _make_mask((16, 32, 32), (slice(0, 4), slice(0, 4), slice(0, 4)))
        b = _make_mask((8, 16, 16), (slice(0, 4), slice(0, 4), slice(0, 4)))
        with pytest.raises(ValueError, match="size"):
            compute_union_bbox([a, b], padding_voxels=0)

    def test_negative_padding_raises(self):
        m = _make_mask((16, 32, 32), (slice(4, 6), slice(4, 6), slice(4, 6)))
        with pytest.raises(ValueError, match="padding_voxels"):
            compute_union_bbox([m], padding_voxels=-1)

    def test_min_size_expands_small_bbox_symmetrically(self):
        # 1-voxel foreground at the centre of a 64-voxel cube; padding=0,
        # min_size=10 should grow to size 10 in each dim centred on the voxel.
        mask = _make_mask((64, 64, 64), (slice(32, 33), slice(32, 33), slice(32, 33)))
        start, size = compute_union_bbox([mask], padding_voxels=0, min_size_voxels=10)
        assert size == [10, 10, 10]
        # The single foreground voxel at index 32 must still lie inside the bbox.
        for k in range(3):
            assert start[k] <= 32 < start[k] + size[k]

    def test_min_size_pushes_off_corner(self):
        # Foreground touches the lower corner; symmetric expansion would go
        # negative, so the extra must spill onto the opposite side.
        mask = _make_mask((64, 64, 64), (slice(0, 1), slice(0, 1), slice(0, 1)))
        start, size = compute_union_bbox([mask], padding_voxels=0, min_size_voxels=8)
        assert start == [0, 0, 0]
        assert size == [8, 8, 8]

    def test_min_size_pushes_off_upper_edge(self):
        # Foreground at the upper edge; symmetric expansion past size-1 must
        # spill back toward the origin.
        mask = _make_mask((32, 32, 32), (slice(31, 32), slice(31, 32), slice(31, 32)))
        start, size = compute_union_bbox([mask], padding_voxels=0, min_size_voxels=6)
        assert size == [6, 6, 6]
        for k in range(3):
            assert start[k] + size[k] == 32

    def test_min_size_clamped_when_image_smaller(self):
        # The image itself is 8 voxels in Z; min_size=16 cannot be satisfied
        # there. We should get the full Z extent (8) and the requested 16 in
        # the larger dims.
        mask = _make_mask((8, 32, 32), (slice(2, 5), slice(10, 13), slice(10, 13)))
        start, size = compute_union_bbox([mask], padding_voxels=0, min_size_voxels=16)
        # ITK (x, y, z) ordering — image_size_zyx = (8, 32, 32) -> ITK (32, 32, 8)
        assert size[0] == 16
        assert size[1] == 16
        assert size[2] == 8  # capped by image

    def test_min_size_no_op_when_bbox_already_large_enough(self):
        # Padding alone already reaches 12 voxels per dim; min_size_voxels=8
        # should change nothing.
        mask = _make_mask((64, 64, 64), (slice(20, 24), slice(20, 24), slice(20, 24)))
        start_no_min, size_no_min = compute_union_bbox([mask], padding_voxels=4)
        start_min, size_min = compute_union_bbox([mask], padding_voxels=4, min_size_voxels=8)
        assert start_no_min == start_min
        assert size_no_min == size_min

    def test_negative_min_size_raises(self):
        m = _make_mask((16, 32, 32), (slice(4, 6), slice(4, 6), slice(4, 6)))
        with pytest.raises(ValueError, match="min_size_voxels"):
            compute_union_bbox([m], padding_voxels=0, min_size_voxels=-1)

    def test_works_on_multi_label_mask(self):
        """Union must cover all non-zero labels, not just label 1."""
        arr = np.zeros((16, 32, 32), dtype=np.int16)
        arr[2:4, 2:4, 2:4] = 1
        arr[10:12, 20:22, 24:26] = 5
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        wrapper = ImageWrapper(img)
        start, size = compute_union_bbox([wrapper], padding_voxels=0)
        assert start == [2, 2, 2]
        assert size == [24, 20, 10]

    def test_handles_non_one_foreground_value(self):
        """A binary mask with foreground value != 1 must still produce the correct bbox."""
        mask = _make_mask(
            (16, 32, 32),
            (slice(4, 6), slice(8, 10), slice(12, 14)),
            fg_value=255,
        )
        start, size = compute_union_bbox([mask], padding_voxels=0)
        assert start == [12, 8, 4]
        assert size == [2, 2, 2]


class TestCropToBbox:
    def test_size_matches_request(self, synthetic_seg_3d):
        start, size = [4, 4, 2], [8, 8, 4]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        assert list(cropped.get_dimensions()) == size

    def test_spacing_and_direction_preserved(self, synthetic_seg_3d):
        cropped = crop_to_bbox(synthetic_seg_3d, [4, 4, 2], [8, 8, 4])
        assert cropped.get_spacing() == synthetic_seg_3d.get_spacing()
        assert cropped.get_direction() == synthetic_seg_3d.get_direction()

    def test_origin_shifts_to_cropped_subvolume(self, synthetic_seg_3d):
        """Origin must shift so the sub-volume keeps its physical-space location."""
        start = [4, 4, 2]
        cropped = crop_to_bbox(synthetic_seg_3d, start, [8, 8, 4])
        old_origin = synthetic_seg_3d.get_origin()
        spacing = synthetic_seg_3d.get_spacing()
        expected = tuple(old_origin[i] + start[i] * spacing[i] for i in range(3))
        new_origin = cropped.get_origin()
        for i in range(3):
            assert new_origin[i] == pytest.approx(expected[i])

    def test_voxel_values_match_corresponding_region(self, synthetic_seg_3d):
        start, size = [12, 10, 6], [8, 12, 4]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        full = sitk.GetArrayFromImage(synthetic_seg_3d.get_data())  # ZYX
        crop_arr = sitk.GetArrayFromImage(cropped.get_data())
        z0, y0, x0 = start[2], start[1], start[0]
        zs, ys, xs = size[2], size[1], size[0]
        np.testing.assert_array_equal(
            crop_arr, full[z0:z0 + zs, y0:y0 + ys, x0:x0 + xs]
        )

    def test_pixel_type_preserved(self, synthetic_seg_3d):
        cropped = crop_to_bbox(synthetic_seg_3d, [4, 4, 2], [8, 8, 4])
        assert cropped.get_data().GetPixelID() == synthetic_seg_3d.get_data().GetPixelID()


class TestUncropToReference:
    def test_output_size_matches_reference(self, synthetic_seg_3d):
        start, size = [4, 4, 2], [8, 8, 4]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        assert restored.get_dimensions() == synthetic_seg_3d.get_dimensions()

    def test_metadata_matches_reference(self, synthetic_seg_3d):
        start = [4, 4, 2]
        cropped = crop_to_bbox(synthetic_seg_3d, start, [8, 8, 4])
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        assert restored.get_origin() == synthetic_seg_3d.get_origin()
        assert restored.get_spacing() == synthetic_seg_3d.get_spacing()
        assert restored.get_direction() == synthetic_seg_3d.get_direction()

    def test_outside_region_is_fill_value(self, synthetic_seg_3d):
        """A crop that covers the sphere centre; outside the pasted region must be zero."""
        start, size = [8, 8, 4], [16, 16, 8]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        arr = sitk.GetArrayFromImage(restored.get_data())
        z0, y0, x0 = start[2], start[1], start[0]
        zs, ys, xs = size[2], size[1], size[0]
        outside = arr.copy()
        outside[z0:z0 + zs, y0:y0 + ys, x0:x0 + xs] = 0
        assert np.all(outside == 0)

    def test_pasted_region_matches_cropped_content(self, synthetic_seg_3d):
        start, size = [8, 8, 4], [16, 16, 8]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        full = sitk.GetArrayFromImage(restored.get_data())
        crop_arr = sitk.GetArrayFromImage(cropped.get_data())
        z0, y0, x0 = start[2], start[1], start[0]
        zs, ys, xs = size[2], size[1], size[0]
        np.testing.assert_array_equal(
            full[z0:z0 + zs, y0:y0 + ys, x0:x0 + xs], crop_arr
        )

    def test_custom_fill_value(self, synthetic_seg_3d):
        start, size = [8, 8, 4], [16, 16, 8]
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start, fill_value=7)
        arr = sitk.GetArrayFromImage(restored.get_data())
        outside = arr.copy()
        z0, y0, x0 = start[2], start[1], start[0]
        zs, ys, xs = size[2], size[1], size[0]
        outside[z0:z0 + zs, y0:y0 + ys, x0:x0 + xs] = 7
        assert np.all(outside == 7)


class TestRoundTrip:
    def test_roundtrip_preserves_full_image(self, synthetic_seg_3d):
        """compute_bbox → crop → uncrop reproduces the original everywhere, because
        the bbox by construction covers every foreground voxel."""
        start, size = compute_union_bbox([synthetic_seg_3d], padding_voxels=2)
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(restored.get_data()),
            sitk.GetArrayFromImage(synthetic_seg_3d.get_data()),
        )

    def test_roundtrip_preserves_metadata(self, synthetic_seg_3d):
        start, size = compute_union_bbox([synthetic_seg_3d], padding_voxels=2)
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        assert restored.get_dimensions() == synthetic_seg_3d.get_dimensions()
        assert restored.get_origin() == synthetic_seg_3d.get_origin()
        assert restored.get_spacing() == synthetic_seg_3d.get_spacing()
        assert restored.get_direction() == synthetic_seg_3d.get_direction()

    def test_roundtrip_preserves_pixel_type(self, synthetic_seg_3d):
        start, size = compute_union_bbox([synthetic_seg_3d], padding_voxels=2)
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        assert restored.get_data().GetPixelID() == synthetic_seg_3d.get_data().GetPixelID()

    def test_roundtrip_with_non_unit_spacing_and_origin(self):
        """Round-trip must work for images with non-trivial spacing/origin."""
        arr = np.zeros((16, 32, 32), dtype=np.int16)
        arr[6:10, 12:18, 14:20] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 0.5, 0.8))
        img.SetOrigin((-12.5, 7.25, 3.0))
        wrapper = ImageWrapper(img)

        start, size = compute_union_bbox([wrapper], padding_voxels=2)
        cropped = crop_to_bbox(wrapper, start, size)
        restored = uncrop_to_reference(cropped, wrapper, start)

        # Cropped origin lands at the physical position of `start` in the original.
        for i in range(3):
            expected = img.GetOrigin()[i] + start[i] * img.GetSpacing()[i]
            assert cropped.get_origin()[i] == pytest.approx(expected)

        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(restored.get_data()),
            sitk.GetArrayFromImage(wrapper.get_data()),
        )

    def test_roundtrip_padding_zero(self):
        """Padding 0 must still produce an exact round-trip across the bbox."""
        arr = np.zeros((10, 20, 20), dtype=np.uint8)
        arr[3:6, 7:12, 9:13] = 1
        wrapper = ImageWrapper(sitk.GetImageFromArray(arr))

        start, size = compute_union_bbox([wrapper], padding_voxels=0)
        cropped = crop_to_bbox(wrapper, start, size)
        restored = uncrop_to_reference(cropped, wrapper, start)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(restored.get_data()),
            sitk.GetArrayFromImage(wrapper.get_data()),
        )

    def test_roundtrip_with_multilabel_segmentation(self, synthetic_seg_3d):
        """All label values must survive the round trip unchanged."""
        start, size = compute_union_bbox([synthetic_seg_3d], padding_voxels=4)
        cropped = crop_to_bbox(synthetic_seg_3d, start, size)
        restored = uncrop_to_reference(cropped, synthetic_seg_3d, start)
        orig = sitk.GetArrayFromImage(synthetic_seg_3d.get_data())
        out = sitk.GetArrayFromImage(restored.get_data())
        assert set(np.unique(out).tolist()) == set(np.unique(orig).tolist())
        np.testing.assert_array_equal(out, orig)
