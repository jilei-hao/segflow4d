"""Integration tests for PropagationInputFactory."""

import os
import pytest
import numpy as np
import SimpleITK as sitk

from segflow4d.common.types.propagation_input import PropagationInputFactory, PropagationInput
from segflow4d.common.types.image_wrapper import ImageWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_4d_image(path, n_tp=3, shape_zyx=(8, 16, 16)):
    """Write a minimal 4D NIfTI to `path`."""
    stack = np.zeros((n_tp,) + shape_zyx, dtype=np.float32)
    img = sitk.GetImageFromArray(stack)
    img.SetSpacing(tuple([1.0] * 4))
    img.SetOrigin(tuple([0.0] * 4))
    sitk.WriteImage(img, path)


def _write_seg_ref(path, shape_zyx=(8, 16, 16)):
    """Write a minimal 3D segmentation (one label) to `path`."""
    arr = np.zeros(shape_zyx, dtype=np.int16)
    arr[2:6, 4:12, 4:12] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPropagationInputFactoryFromDisk:
    def test_factory_builds_valid_input(self, tmp_path):
        img_path = str(tmp_path / "image_4d.nii.gz")
        seg_path = str(tmp_path / "seg_ref.nii.gz")
        out_dir = str(tmp_path / "output")

        _write_4d_image(img_path)
        _write_seg_ref(seg_path)

        prop_input = (
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
                write_result_to_disk=False,
                output_directory=out_dir,
            )
            .build()
        )

        assert isinstance(prop_input, PropagationInput)
        assert prop_input.image_4d is not None
        assert prop_input.image_4d.get_data() is not None

    def test_factory_creates_correct_number_of_groups(self, tmp_path):
        img_path = str(tmp_path / "image_4d.nii.gz")
        seg_path = str(tmp_path / "seg_ref.nii.gz")
        out_dir = str(tmp_path / "output")

        _write_4d_image(img_path, n_tp=5)
        _write_seg_ref(seg_path)

        factory = (
            PropagationInputFactory()
            .set_image_4d_from_disk(img_path)
            .set_options(
                lowres_factor=2.0,
                registration_backend="FIREANTS",
                dilation_radius=2,
                output_directory=out_dir,
            )
        )
        for _ in range(3):
            factory.add_tp_input_group_from_disk(
                tp_ref=0,
                tp_target=[1, 2],
                seg_ref_path=seg_path,
                additional_meshes_ref=None,
            )

        result = factory.build()
        assert len(result.tp_input_groups) == 3

    def test_factory_missing_image_raises(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.nii.gz")
        seg_path = str(tmp_path / "seg.nii.gz")
        _write_seg_ref(seg_path)

        with pytest.raises(Exception):
            (
                PropagationInputFactory()
                .set_image_4d_from_disk(missing_path)
                .add_tp_input_group_from_disk(
                    tp_ref=0,
                    tp_target=[1],
                    seg_ref_path=seg_path,
                    additional_meshes_ref=None,
                )
                .set_options(
                    lowres_factor=2.0,
                    registration_backend="FIREANTS",
                    dilation_radius=2,
                    output_directory=str(tmp_path / "out"),
                )
                .build()
            )

    def test_factory_missing_seg_raises(self, tmp_path):
        img_path = str(tmp_path / "image_4d.nii.gz")
        _write_4d_image(img_path)
        missing_seg = str(tmp_path / "missing_seg.nii.gz")

        with pytest.raises(Exception):
            (
                PropagationInputFactory()
                .set_image_4d_from_disk(img_path)
                .add_tp_input_group_from_disk(
                    tp_ref=0,
                    tp_target=[1],
                    seg_ref_path=missing_seg,
                    additional_meshes_ref=None,
                )
                .set_options(
                    lowres_factor=2.0,
                    registration_backend="FIREANTS",
                    dilation_radius=2,
                    output_directory=str(tmp_path / "out"),
                )
                .build()
            )
