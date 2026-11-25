"""Additional meshes must sit inside the reference image (sf4d-200 KNOWN_ISSUES H1/B11).

A mesh outside the image volume has no deformation to follow, so every backend
copied it unchanged to every timepoint without a word. The usual cause is a RAS
mesh handed to segflow4d, which works in ITK/LPS physical coordinates.
"""

import logging

import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from segflow4d.common.types.propagation_input import PropagationInputFactory
from segflow4d.utility.mesh_helper.mesh_placement import check_mesh_placement, fraction_inside
from segflow4d.registration.registration_handler.greedy.greedy_mesh_warp import inverse_chain


def _image(origin=(-80.0, 40.0, 5.0), direction=(1, 0, 0, 0, -1, 0, 0, 0, 1)):
    """40 mm cube of 1 mm voxels; the y axis runs backwards, like many scanner frames.

    Physical extent: x -80.5..-40.5, y 0.5..40.5 (backwards), z 4.5..44.5. It is far
    from the origin, so a RAS-flipped copy of an inside mesh lands well outside."""
    img = sitk.Image(40, 40, 40, sitk.sitkInt16)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img


def _centre(img):
    return np.array(img.TransformContinuousIndexToPhysicalPoint((19.5, 19.5, 19.5)))


def _sphere_points(center, radius=5.0):
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.Update()
    return src.GetOutput()


def _points(poly):
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def _with_points(poly, pts):
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    out.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    return out


class TestFractionInside:
    def test_mesh_at_the_image_centre_is_inside(self):
        img = _image()
        assert fraction_inside(_points(_sphere_points(_centre(img))), img) == 1.0

    def test_direction_matrix_is_honoured(self):
        # The backwards y axis puts the volume at y 0.5..40.5. With an identity
        # direction it would be at y 39.5..79.5 instead.
        img = _image()
        assert fraction_inside(np.array([[-60.0, 30.0, 25.0]]), img) == 1.0
        assert fraction_inside(np.array([[-60.0, 50.0, 25.0]]), img) == 0.0


class TestCheckMeshPlacement:
    def test_inside_mesh_passes_silently(self, caplog):
        img = _image()
        with caplog.at_level(logging.WARNING):
            check_mesh_placement("root", _points(_sphere_points(_centre(img))), img)
        assert caplog.records == []

    def test_ras_mesh_is_rejected_with_a_hint(self):
        img = _image()
        ras = _points(_sphere_points(_centre(img))) * np.array([-1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="looks like a RAS mesh"):
            check_mesh_placement("root", ras, img)

    def test_far_away_mesh_is_rejected_without_the_ras_hint(self):
        img = _image()
        far = _points(_sphere_points(_centre(img) + np.array([500.0, 0.0, 0.0])))
        with pytest.raises(ValueError, match="entirely outside") as exc:
            check_mesh_placement("root", far, img)
        assert "RAS" not in str(exc.value)

    def test_partly_outside_mesh_warns(self, caplog):
        img = _image()
        # centred on the image's x edge: about half the vertices are outside
        edge = np.array(img.TransformContinuousIndexToPhysicalPoint((-0.5, 19.5, 19.5)))
        with caplog.at_level(logging.WARNING):
            check_mesh_placement("root", _points(_sphere_points(edge)), img)
        assert "lie outside the reference image" in caplog.text


class TestFactoryBuild:
    def _factory(self, img, mesh_poly, out_dir):
        factory = PropagationInputFactory()
        factory.set_image_4d(sitk.JoinSeries([img, img]))
        seg_arr = np.zeros((40, 40, 40), dtype=np.int16)
        seg_arr[15:25, 15:25, 15:25] = 1
        seg = sitk.GetImageFromArray(seg_arr)
        seg.CopyInformation(img)
        factory.add_tp_input_group(tp_ref=1, tp_target=[2], seg_ref=seg,
                                   additional_meshes_ref={"root": mesh_poly})
        factory.set_options(lowres_factor=0.5, registration_backend="greedy", dilation_radius=1,
                            write_result_to_disk=False, output_directory=str(out_dir),
                            debug=False, debug_output_directory="", minimum_required_vram_gb=0)
        return factory

    def test_build_accepts_a_mesh_inside_the_image(self, tmp_path):
        img = _image()
        self._factory(img, _sphere_points(_centre(img)), tmp_path).build()

    def test_build_rejects_a_ras_mesh(self, tmp_path):
        img = _image()
        poly = _sphere_points(_centre(img))
        ras = _with_points(poly, _points(poly) * np.array([-1.0, -1.0, 1.0]))
        with pytest.raises(ValueError, match="RAS"):
            self._factory(img, ras, tmp_path).build()

    def test_build_rejects_an_empty_mesh(self, tmp_path):
        with pytest.raises(ValueError, match="has no vertices"):
            self._factory(_image(), vtk.vtkPolyData(), tmp_path).build()


class TestGreedyInverseChain:
    def test_warp_then_affine_inverts_to_affine_inverse_then_inverse_warp(self):
        assert inverse_chain(["my_warp", "my_affine"]) == "my_affine,-1 my_warp_inv"

    def test_deformable_only(self):
        assert inverse_chain(["my_warp"]) == "my_warp_inv"

    def test_affine_file_path(self):
        assert inverse_chain(["my_warp", "/tmp/a.txt"]) == "/tmp/a.txt,-1 my_warp_inv"

    def test_empty_chain_is_rejected(self):
        with pytest.raises(ValueError):
            inverse_chain([])
