"""Warp surface meshes with greedy's own transform chain.

Direction. greedy's ``-r`` chain maps FIXED-space points to MOVING space, which is
what image reslicing needs: each fixed voxel pulls its value from moving space. A
mesh lives on the moving (reference) image and has to be pushed the other way,
into fixed (target) space, so it needs the INVERSE chain. If the segmentation is
resliced with ``-r W A`` (deformable W, then affine A), the mesh gets
``-r A,-1 W_inv``, where ``W_inv`` comes from ``-oinv`` in the deformable stage.

Before this module, the greedy handler applied the forward deformable field alone.
That dropped the affine (every rigid component of the motion) and applied the
small deformable residual in the wrong direction, so meshes barely moved
(sf4d-200 KNOWN_ISSUES B11).

Frame. greedy reads mesh vertices as RAS; segflow4d meshes are in ITK LPS physical
coordinates, so x and y are negated on the way in and back out. Only the warped
vertex coordinates are taken from greedy's output. Topology and point/cell data
come from a deep copy of the input mesh, so nothing else about the mesh changes.

Both conventions were pinned empirically against an analytic moving-and-stretching
ellipsoid (tests/e2e/test_mesh_follows_anatomy.py).
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from segflow4d.common.types.mesh_wrapper import MeshWrapper

logger = logging.getLogger(__name__)

_LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def inverse_chain(transforms: list[str]) -> str:
    """The ``-r`` chain that pushes points the opposite way to ``-r <transforms>``.

    ``transforms`` is the image-reslice chain in greedy's order, e.g.
    ``["my_warp", "my_affine"]``. Every element after the first must be an affine:
    greedy inverts affines with the ``,-1`` suffix. The first element must be the
    deformable warp, whose inverse the caller computed with ``-oinv`` and names
    ``<warp>_inv`` (``my_warp`` -> ``my_warp_inv``).
    """
    if not transforms:
        raise ValueError("empty transform chain")
    warp, affines = transforms[0], transforms[1:]
    return " ".join([f"{a},-1" for a in reversed(affines)] + [f"{warp}_inv"])


def _write_ras_vtk(mesh: MeshWrapper, path: str) -> None:
    poly = vtk.vtkPolyData()
    poly.ShallowCopy(mesh.get_data())
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64) * _LPS_TO_RAS
    ras_points = vtk.vtkPoints()
    ras_points.SetData(numpy_to_vtk(points, deep=True))
    poly.SetPoints(ras_points)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetFileTypeToBinary()  # ASCII would round vertex coordinates
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"could not write temporary mesh {path}")


def _read_lps_points(path: str) -> np.ndarray:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    points = reader.GetOutput().GetPoints()
    if points is None:
        raise RuntimeError(f"greedy wrote no points to {path}")
    return vtk_to_numpy(points.GetData()).astype(np.float64) * _LPS_TO_RAS


def warp_meshes_with_greedy(
    greedy,
    meshes: dict[str, MeshWrapper],
    moving_image: sitk.Image,
    reslice_chain: list[str],
    common_flags: str = "",
) -> dict[str, MeshWrapper]:
    """Push ``meshes`` from moving (reference) space into fixed (target) space.

    ``greedy`` is the ``Greedy3D`` instance that ran the registration; the
    transforms named in ``reslice_chain`` (and the inverse warp) must be cached in
    it, or be paths on disk. ``reslice_chain`` is the exact chain used to reslice
    the segmentation, so the meshes and the segmentation agree by construction.
    """
    chain = inverse_chain(reslice_chain)
    warped: dict[str, MeshWrapper] = {}
    with tempfile.TemporaryDirectory(prefix="segflow4d-greedy-mesh-") as tmp:
        for i, (name, mesh) in enumerate(meshes.items()):
            src = os.path.join(tmp, f"in_{i}.vtk")
            dst = os.path.join(tmp, f"out_{i}.vtk")
            _write_ras_vtk(mesh, src)
            cmd = f"-rf my_mesh_reference -rs {src} {dst} -r {chain} {common_flags}".strip()
            logger.debug(f"Greedy mesh warp command: {cmd}")
            greedy.execute(cmd, my_mesh_reference=moving_image)
            points = _read_lps_points(dst)
            n_expected = mesh.get_data().GetNumberOfPoints()
            if points.shape[0] != n_expected:
                raise RuntimeError(
                    f"greedy returned {points.shape[0]} vertices for mesh {name!r}, "
                    f"expected {n_expected}"
                )
            warped[name] = mesh.deepcopy().update_vertices(points)
    return warped
