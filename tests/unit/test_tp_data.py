"""Unit tests for TPData."""

import numpy as np
import pytest

from segflow4d.common.types.tp_data import TPData
from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper


class TestTPDataClear:
    def test_clear_resets_image(self, synthetic_image_3d):
        tp = TPData(image=synthetic_image_3d)
        tp.clear()
        assert tp.image is None

    def test_clear_resets_all_fields(self, synthetic_image_3d, synthetic_seg_3d):
        tp = TPData(
            image=synthetic_image_3d,
            image_low_res=synthetic_image_3d.deepcopy(),
            segmentation=synthetic_seg_3d,
            affine_matrix=np.eye(4),
            resliced_image=synthetic_image_3d.deepcopy(),
            warp_image=synthetic_image_3d.deepcopy(),
        )
        tp.clear()

        assert tp.image is None
        assert tp.image_low_res is None
        assert tp.segmentation is None
        assert tp.segmentation_mesh is None
        assert tp.mask is None
        assert tp.mask_low_res is None
        assert tp.mask_high_res is None
        assert tp.additional_meshes is None
        assert tp.affine_matrix is None
        assert tp.resliced_image is None
        assert tp.resliced_segmentation_mesh is None
        assert tp.resliced_meshes is None
        assert tp.warp_image is None
        assert tp.affine_from_prev is None
        assert tp.affine_from_ref is None
        assert tp.deformable_from_ref is None
        assert tp.deformable_from_ref_low_res is None


class TestTPDataDeepCopy:
    def test_deepcopy_is_independent_image(self, synthetic_image_3d):
        """Clearing the original must not affect the copy's image field."""
        tp = TPData(image=synthetic_image_3d)
        copy = tp.deepcopy()
        tp.clear()
        assert copy.image is not None
        assert copy.image.get_data() is not None

    def test_deepcopy_affine_matrix_is_independent(self):
        mat = np.eye(4, dtype=np.float64)
        tp = TPData(affine_matrix=mat)
        copy = tp.deepcopy()
        mat[0, 0] = 999.0
        assert copy.affine_matrix[0, 0] == pytest.approx(1.0)

    def test_deepcopy_none_fields_stay_none(self):
        tp = TPData()
        copy = tp.deepcopy()
        assert copy.image is None
        assert copy.segmentation is None
        assert copy.warp_image is None
