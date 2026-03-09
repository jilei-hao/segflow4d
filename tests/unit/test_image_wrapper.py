"""Unit tests for ImageWrapper."""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper


class TestImageWrapperDeepCopy:
    def test_deepcopy_preserves_origin(self, synthetic_image_3d):
        copy = synthetic_image_3d.deepcopy()
        assert copy.get_origin() == pytest.approx(synthetic_image_3d.get_origin())

    def test_deepcopy_preserves_spacing(self, synthetic_image_3d):
        copy = synthetic_image_3d.deepcopy()
        assert copy.get_spacing() == pytest.approx(synthetic_image_3d.get_spacing())

    def test_deepcopy_preserves_direction(self, synthetic_image_3d):
        copy = synthetic_image_3d.deepcopy()
        assert copy.get_direction() == pytest.approx(synthetic_image_3d.get_direction())

    def test_deepcopy_pixel_data_is_independent(self, synthetic_image_3d):
        """Mutating copy pixel data must not affect the original."""
        copy = synthetic_image_3d.deepcopy()
        orig_arr = sitk.GetArrayFromImage(synthetic_image_3d.get_data()).copy()

        arr = sitk.GetArrayFromImage(copy.get_data())
        arr[:] = 0.0
        mutated_img = sitk.GetImageFromArray(arr)
        mutated_img.CopyInformation(copy.get_data())
        copy.set_data(mutated_img)

        result_arr = sitk.GetArrayFromImage(synthetic_image_3d.get_data())
        np.testing.assert_array_equal(result_arr, orig_arr)

    def test_deepcopy_of_none_returns_none_data(self):
        wrapper = ImageWrapper(None)
        copy = wrapper.deepcopy()
        assert copy.get_data() is None


class TestImageWrapperGetters:
    def test_get_dimensions_matches_sitk(self, synthetic_image_3d):
        """get_dimensions() must equal the underlying SimpleITK GetSize()."""
        expected = synthetic_image_3d.get_data().GetSize()
        assert synthetic_image_3d.get_dimensions() == expected

    def test_getters_return_none_for_empty_wrapper(self):
        wrapper = ImageWrapper(None)
        assert wrapper.get_origin() is None
        assert wrapper.get_spacing() is None
        assert wrapper.get_direction() is None
        assert wrapper.get_dimensions() is None
