"""Integration tests for TPPartition.

These tests exercise the partition initialisation logic (TP extraction,
low-res resampling, mask creation) using purely synthetic in-memory data.
No GPU, no registration, no disk I/O (debug=False).
"""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.propagation.tp_partition import TPPartition
from segflow4d.propagation.tp_partition_input import TPPartitionInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_options(lowres_factor=2.0, dilation_radius=2):
    return PropagationOptions(
        lowres_resample_factor=lowres_factor,
        dilation_radius=dilation_radius,
        registration_backend="FIREANTS",
        registration_backend_options={},
        write_result_to_disk=False,
        output_directory="",
        debug=False,
        debug_output_directory="",
        minimum_required_vram_gb=0,
    )


def _make_4d_image(n_tp=5, shape_zyx=(16, 32, 32)):
    """Construct a 4D SimpleITK image with `n_tp` timepoints."""
    volumes = []
    for tp in range(n_tp):
        arr = np.zeros(shape_zyx, dtype=np.float32)
        # Put a small bright sphere at slightly different location per TP
        cz, cy, cx = shape_zyx[0]//2, shape_zyx[1]//2, shape_zyx[2]//2 + tp
        zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
        dist = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2)
        arr[dist <= 4] = 1.0
        volumes.append(arr)

    stack = np.stack(volumes, axis=0)  # (T, Z, Y, X)
    img_4d = sitk.GetImageFromArray(stack)
    img_4d.SetSpacing(tuple([1.0] * 4))
    img_4d.SetOrigin(tuple([0.0] * 4))
    return ImageWrapper(img_4d)


def _make_seg_ref(shape_zyx=(16, 32, 32)):
    """Small 3-D label image with two filled spheres (labels 1, 2)."""
    arr = np.zeros(shape_zyx, dtype=np.int16)
    cz, cy, cx = shape_zyx[0]//2, shape_zyx[1]//2, shape_zyx[2]//2
    zz, yy, xx = np.mgrid[0:shape_zyx[0], 0:shape_zyx[1], 0:shape_zyx[2]]
    dist = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2)
    arr[dist <= 5] = 1
    arr[dist <= 2] = 2
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return ImageWrapper(img)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_4d_5tp():
    return _make_4d_image(n_tp=5)


@pytest.fixture
def seg_ref_3d():
    return _make_seg_ref()


@pytest.fixture
def tp_partition_input(seg_ref_3d):
    return TPPartitionInput(
        seg_ref=seg_ref_3d,
        additional_meshes_ref=None,
        tp_ref=0,
        tp_target=[1, 2, 3, 4],
    )


@pytest.fixture
def options():
    return _make_options()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTPPartitionInitialisation:
    def test_tp_data_count_matches_all_timepoints(self, tp_partition_input, image_4d_5tp, options):
        """TPPartition must hold one TPData entry for every timepoint (ref + targets)."""
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        assert len(partition._tp_data) == 5

    def test_tp_data_keys_match_expected_timepoints(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        assert set(partition._tp_data.keys()) == {0, 1, 2, 3, 4}

    def test_all_tp_data_have_image(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        for tp, data in partition._tp_data.items():
            assert data.image is not None, f"TP {tp} missing image"
            assert data.image.get_data() is not None

    def test_all_tp_data_have_low_res_image(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        for tp, data in partition._tp_data.items():
            assert data.image_low_res is not None, f"TP {tp} missing image_low_res"

    def test_low_res_images_are_smaller_than_full_res(self, tp_partition_input, image_4d_5tp, options):
        """Low-res images must have larger spacing (coarser resolution) than full-res."""
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        for tp, data in partition._tp_data.items():
            full_spacing = data.image.get_spacing()
            lr_spacing = data.image_low_res.get_spacing()
            for i in range(3):
                assert lr_spacing[i] >= full_spacing[i], (
                    f"TP {tp}, axis {i}: low-res spacing {lr_spacing[i]} "
                    f"should be >= full-res {full_spacing[i]}"
                )

    def test_ref_tp_has_segmentation(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        ref_data = partition._tp_data[tp_partition_input.tp_ref]
        assert ref_data.segmentation is not None

    def test_ref_tp_has_mask_low_res(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        ref_data = partition._tp_data[tp_partition_input.tp_ref]
        assert ref_data.mask_low_res is not None

    def test_ref_tp_mask_low_res_is_binary(self, tp_partition_input, image_4d_5tp, options):
        partition = TPPartition(tp_partition_input, image_4d_5tp, options)
        ref_data = partition._tp_data[tp_partition_input.tp_ref]
        arr = sitk.GetArrayFromImage(ref_data.mask_low_res.get_data())
        unique = set(np.unique(arr).tolist())
        assert unique.issubset({0, 1}), f"Non-binary values: {unique}"
