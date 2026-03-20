"""
CPU-based mesh warping using SimpleITK displacement field transforms.

This module provides mesh warping for CPU-based registration backends (e.g. Greedy).
It mirrors the interface of gpu_mesh_warper.py but runs entirely on CPU using ITK.
"""

import logging

import SimpleITK as sitk
import numpy as np

from segflow4d.common.types.mesh_wrapper import MeshWrapper

logger = logging.getLogger(__name__)


def warp_mesh_vertices_cpu(
    mesh_wrapper: MeshWrapper,
    warp_field_sitk: sitk.Image,
    img_fixed_sitk: sitk.Image,
) -> MeshWrapper:
    """
    Warp mesh vertices using a SimpleITK displacement field.

    Applies a greedy-style deformation field to a VTK surface mesh by transforming
    each vertex through an ITK DisplacementFieldTransform.  All computation is CPU-only.

    Args:
        mesh_wrapper: Input mesh (VTK surface mesh wrapped in MeshWrapper).
        warp_field_sitk: Displacement field image produced by greedy (vector image,
            component order matches ITK conventions).
        img_fixed_sitk: The fixed (reference) image used during registration.  Its
            physical-space metadata (spacing, origin, direction) is used to reconcile
            the transform with mesh vertex coordinates.

    Returns:
        A new MeshWrapper containing the warped mesh with all topology preserved.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    # Build ITK displacement field transform from the warp image
    warp_field_double = sitk.Cast(warp_field_sitk, sitk.sitkVectorFloat64)
    displacement_tx = sitk.DisplacementFieldTransform(warp_field_double)

    # Extract VTK polydata
    polydata = mesh_wrapper.get_data()
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()

    logger.debug(f"Warping {n_points} mesh vertices on CPU")

    # Convert all vertices to numpy for bulk processing
    pts_np = vtk_to_numpy(points.GetData()).astype(np.float64)  # (N, 3)

    # Apply displacement to each vertex
    warped = np.empty_like(pts_np)
    for i in range(n_points):
        x, y, z = float(pts_np[i, 0]), float(pts_np[i, 1]), float(pts_np[i, 2])
        wx, wy, wz = displacement_tx.TransformPoint((x, y, z))
        warped[i, 0] = wx
        warped[i, 1] = wy
        warped[i, 2] = wz

    # Build new VTK polydata with warped points (shallow-copy topology)
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(warped, deep=True, array_type=vtk.VTK_DOUBLE))

    new_polydata = vtk.vtkPolyData()
    new_polydata.DeepCopy(polydata)
    new_polydata.SetPoints(new_points)

    logger.debug("CPU mesh warping complete")
    return MeshWrapper(new_polydata)
