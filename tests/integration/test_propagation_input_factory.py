"""Integration tests for PropagationInputFactory."""

import os
import pytest
import numpy as np
import SimpleITK as sitk

import vtk

from segflow4d.common.types.propagation_input import PropagationInputFactory, PropagationInput
from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper


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


# ---------------------------------------------------------------------------
# In-memory factory path (add_tp_input_group / set_image_4d)
#
# This is the path the AVRP avrp-handler uses: it generates reference meshes in
# process and hands them to the factory as objects (no disk round-trip). The
# from_disk tests above don't exercise it, so it gets its own coverage here.
# ---------------------------------------------------------------------------

def _make_4d_image_obj(n_tp=7, shape_zyx=(8, 16, 16)):
    """A genuine 4-D SimpleITK image (via JoinSeries) held in memory."""
    volumes = []
    for _ in range(n_tp):
        vol = sitk.GetImageFromArray(np.zeros(shape_zyx, dtype=np.float32))
        vol.SetSpacing((1.0, 1.0, 1.0))
        vol.SetOrigin((0.0, 0.0, 0.0))
        volumes.append(vol)
    return sitk.JoinSeries(volumes)


def _make_seg_ref_obj(shape_zyx=(8, 16, 16)):
    """A 3-D label image (label 1 cube) held in memory."""
    arr = np.zeros(shape_zyx, dtype=np.int16)
    arr[2:6, 4:12, 4:12] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _make_polydata_sphere(center=(8.0, 8.0, 4.0), radius=3.0):
    """A vtkPolyData surface held in memory."""
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.SetThetaResolution(8)
    src.SetPhiResolution(8)
    src.Update()
    return src.GetOutput()


class TestPropagationInputFactoryInMemory:
    def test_factory_builds_valid_input_from_objects(self, tmp_path):
        prop_input = (
            PropagationInputFactory()
            .set_image_4d(_make_4d_image_obj())
            .add_tp_input_group(
                tp_ref=0,
                tp_target=[1, 2],
                seg_ref=_make_seg_ref_obj(),
                additional_meshes_ref=None,
            )
            .set_options(
                lowres_factor=2.0,
                registration_backend="FIREANTS",
                dilation_radius=2,
                write_result_to_disk=False,
                output_directory=str(tmp_path / "out"),
            )
            .build()
        )
        assert isinstance(prop_input, PropagationInput)
        assert prop_input.image_4d is not None
        assert prop_input.image_4d.get_data() is not None
        assert len(prop_input.tp_input_groups) == 1

    def test_factory_wraps_in_memory_additional_meshes(self, tmp_path):
        meshes = {
            "model-sl": _make_polydata_sphere(),
            "model-ml_pi-01": _make_polydata_sphere(radius=2.0),
        }
        prop_input = (
            PropagationInputFactory()
            .set_image_4d(_make_4d_image_obj())
            .add_tp_input_group(
                tp_ref=0,
                tp_target=[1, 2],
                seg_ref=_make_seg_ref_obj(),
                additional_meshes_ref=meshes,
            )
            .set_options(
                lowres_factor=2.0,
                registration_backend="FIREANTS",
                dilation_radius=2,
                output_directory=str(tmp_path / "out"),
            )
            .build()
        )
        group = prop_input.tp_input_groups[0]
        assert group.additional_meshes_ref is not None
        assert set(group.additional_meshes_ref.keys()) == {"model-sl", "model-ml_pi-01"}
        # The factory must wrap raw vtkPolyData as MeshWrapper.
        assert all(
            isinstance(m, MeshWrapper) for m in group.additional_meshes_ref.values()
        )

    def test_factory_builds_two_in_memory_groups_avrp_shape(self, tmp_path):
        """Mirror the avrp-handler usage: a systolic and a diastolic group, each
        with in-memory ref meshes, on one shared 4-D image."""
        factory = (
            PropagationInputFactory()
            .set_image_4d(_make_4d_image_obj(n_tp=7))
            .set_options(
                lowres_factor=2.0,
                registration_backend="FIREANTS",
                dilation_radius=2,
                output_directory=str(tmp_path / "out"),
            )
        )
        factory.add_tp_input_group(
            tp_ref=1, tp_target=[2, 3], seg_ref=_make_seg_ref_obj(),
            additional_meshes_ref={"model-sl": _make_polydata_sphere()},
        )
        factory.add_tp_input_group(
            tp_ref=4, tp_target=[5, 6], seg_ref=_make_seg_ref_obj(),
            additional_meshes_ref={"model-sl": _make_polydata_sphere()},
        )
        result = factory.build()
        assert len(result.tp_input_groups) == 2
        assert result.tp_input_groups[0].tp_ref == 1
        assert result.tp_input_groups[0].tp_target == [2, 3]
        assert result.tp_input_groups[1].tp_ref == 4
        assert result.tp_input_groups[1].tp_target == [5, 6]
