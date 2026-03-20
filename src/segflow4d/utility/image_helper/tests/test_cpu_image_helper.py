"""
Tests for CPUImageHelper.

Covers the 1-based timepoint convention: user-facing timepoints are 1-based,
and extract_timepoint_image converts to 0-based internally before calling ITK.
"""
import pytest
import SimpleITK as sitk
import numpy as np

from segflow4d.utility.image_helper.cpu_image_helper import CPUImageHelper
from segflow4d.common.types.image_wrapper import ImageWrapper


def _make_4d_image(num_timepoints: int) -> ImageWrapper:
    """
    Build a synthetic 4D image with `num_timepoints` volumes.
    Each volume is uniformly filled with its 1-based timepoint index so
    content is easy to verify after extraction.
    """
    volumes = []
    for tp in range(1, num_timepoints + 1):
        arr = np.full((5, 6, 7), float(tp), dtype=np.float32)  # shape: D x H x W
        volumes.append(sitk.GetImageFromArray(arr))
    image_4d = sitk.JoinSeriesImageFilter().Execute(volumes)
    return ImageWrapper(image_4d)


class TestExtractTimepointImage:
    def setup_method(self):
        self.helper = CPUImageHelper()

    def test_extract_first_timepoint(self):
        """tp=1 (first frame) should return volume filled with 1.0."""
        num_tps = 5
        image_4d = _make_4d_image(num_tps)

        result = self.helper.extract_timepoint_image(image_4d, 1)

        arr = sitk.GetArrayFromImage(result.get_data())
        assert arr.ndim == 3, "Extracted image should be 3D"
        assert np.all(arr == 1.0), f"Expected all voxels == 1.0, got {arr.flat[0]}"

    def test_extract_last_timepoint(self):
        """tp=N (last frame) should return volume filled with float(N)."""
        num_tps = 5
        image_4d = _make_4d_image(num_tps)

        result = self.helper.extract_timepoint_image(image_4d, num_tps)

        arr = sitk.GetArrayFromImage(result.get_data())
        assert arr.ndim == 3, "Extracted image should be 3D"
        assert np.all(arr == float(num_tps)), f"Expected all voxels == {float(num_tps)}, got {arr.flat[0]}"

    def test_each_timepoint_has_correct_content(self):
        """Every timepoint should extract the correct volume (not shifted by ±1)."""
        num_tps = 4
        image_4d = _make_4d_image(num_tps)

        for tp in range(1, num_tps + 1):
            result = self.helper.extract_timepoint_image(image_4d, tp)
            arr = sitk.GetArrayFromImage(result.get_data())
            assert np.all(arr == float(tp)), (
                f"tp={tp}: expected all voxels == {float(tp)}, got {arr.flat[0]}"
            )

    def test_out_of_bounds_raises(self):
        """Requesting a timepoint beyond the image extent should raise ValueError."""
        num_tps = 3
        image_4d = _make_4d_image(num_tps)

        with pytest.raises(ValueError):
            self.helper.extract_timepoint_image(image_4d, num_tps + 1)

    def test_zero_timepoint_raises(self):
        """Passing tp=0 should raise ValueError (1-based; 0 is invalid)."""
        image_4d = _make_4d_image(3)

        with pytest.raises(ValueError):
            self.helper.extract_timepoint_image(image_4d, 0)

    def test_negative_timepoint_raises(self):
        """Passing a negative timepoint should raise ValueError."""
        image_4d = _make_4d_image(3)

        with pytest.raises(ValueError):
            self.helper.extract_timepoint_image(image_4d, -1)
